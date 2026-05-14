"""Paper trader: fires simulated entries at verified signal buckets and logs PnL against live resolutions."""

import asyncio
import csv
import json
import logging
import signal as os_signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import websockets

from src.analysis import DEFAULT_FEE_RATE, compute_taker_fee
from src.binance_price_feed import BinancePriceFeed
from src.config import config
from src.live_executor import LiveExecutor, compute_clean_order_amounts
from src.realistic_executor import RealisticExecutor, SIMULATED_STAKES_USDC
from src.safety import SafetyChecker
from src.scanner import Market, scan

logger = logging.getLogger(__name__)

UTC = timezone.utc

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_CLOB_MARKETS_URL = "https://clob.polymarket.com/markets"

# Simulated stake per trade (USDC). Chosen for clean math in logs.
STAKE_USDC = 1.0

# Resolution timing constants.
RESOLUTION_TIMEOUT_SECONDS = 1800  # 30 minutes — Polymarket CLOB closure can lag 5-15 min after market end
SWEEP_INTERVAL_SECONDS = 300       # 5 minutes between sweep passes
SWEEP_MAX_AGE_SECONDS = 6 * 3600   # 6 hours — after this, give up permanently on unknown trades

# Edge 4 signal params — final validated 2026-05-10.
# Backtest (real fees): WR=92.7%, Sharpe=0.095, Max DD=-$5.68 at $1/trade over 20 days.
EDGE4_MID_THRESHOLD = 0.75          # orderbook mid ≥ 75%
EDGE4_MAX_MID = 0.90                # orderbook mid < 90%: entries ≥0.90 are EV-negative at live fills
EDGE4_TTL_MIN_S = 90.0              # seconds to resolution window
EDGE4_TTL_MAX_S = 110.0
EDGE4_ASSETS = frozenset({"BTC"})   # BTC only; ETH edge is negative
EDGE4_SKIP_UTC_HOURS = frozenset({2, 7, 9, 14, 18})  # extended skip set (improves Sharpe)
EDGE4_LABEL = "E4_TTL90-110s_0.75-0.90"

_CSV_HEADER = [
    "trade_id",
    "entry_timestamp_utc",
    "market_slug",
    "asset",
    "condition_id",
    "side",
    "signal_bucket_label",
    "signal_target_time_s",
    "seconds_to_resolution_at_entry",
    "entry_price",
    "simulated_shares",
    "simulated_stake_usdc",
    "fee_usdc",
    "resolution_timestamp_utc",
    "winner",
    "payout_usdc",
    "pnl_usdc",
    # Realistic execution columns
    "realistic_entry_price_1",
    "realistic_entry_price_5",
    "realistic_entry_price_25",
    "realistic_out_of_bucket",
    # Live execution columns
    "live_order_id",
    "live_fill_status",
    "live_fill_price",
    "live_filled_shares",
    "live_pnl_usdc",
]


@dataclass
class PaperTrade:
    """A simulated entry. PnL fields filled in after market resolves."""

    trade_id: str
    entry_timestamp_utc: datetime
    market: Market
    side: str  # "Up" or "Down"
    signal_bucket_label: str
    signal_target_time_s: float
    seconds_to_resolution_at_entry: float
    entry_price: float  # best_ask at time of entry
    simulated_shares: float  # STAKE_USDC / entry_price
    simulated_stake_usdc: float  # STAKE_USDC
    fee_usdc: float  # computed via compute_taker_fee

    # Filled in at resolution
    resolution_timestamp_utc: Optional[datetime] = None
    winner: Optional[str] = None  # "Up", "Down", or "unknown"
    payout_usdc: Optional[float] = None  # shares × $1 on win, $0 on loss
    pnl_usdc: Optional[float] = None  # payout - stake - fee

    # Filled in by RealisticExecutor after simulated latency + book re-fetch
    realistic_entry_price_1: Optional[float] = None
    realistic_entry_price_5: Optional[float] = None
    realistic_entry_price_25: Optional[float] = None
    realistic_out_of_bucket: Optional[bool] = None

    # Filled in by LiveExecutor if LIVE_TRADING=True
    live_order_id: Optional[str] = None
    live_fill_status: Optional[str] = None
    live_fill_price: Optional[float] = None
    live_filled_shares: Optional[float] = None
    live_pnl_usdc: Optional[float] = None


class PaperTrader:
    """Fires simulated entries and tracks PnL against live resolution."""

    def __init__(self, max_markets: int = 20, refresh_interval: int = 60) -> None:
        self.max_markets = max_markets
        self.refresh_interval = refresh_interval

        # Tracked markets (subset of scan() output)
        self._tracked: dict[str, Market] = {}  # keyed by condition_id
        self._token_to_market: dict[str, tuple[Market, str]] = {}

        # Per-token orderbook state (same as orderbook_observer)
        self._book_bids: dict[str, dict[float, float]] = {}
        self._book_asks: dict[str, dict[float, float]] = {}

        # Deduplication: (condition_id, side) → True if we've already entered
        self._entered: set[tuple[str, str]] = set()

        # Open trades awaiting resolution
        self._open_trades: list[PaperTrade] = []
        # Finalized trades (ready to write to CSV)
        self._buffer: list[PaperTrade] = []

        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None  # type: ignore[type-arg]

        # Binance spot price feed for directional momentum filter
        self._price_feed = BinancePriceFeed()

        # Realistic execution simulator
        self._realistic_executor = RealisticExecutor()

        # Live trading components
        self._live_executor: Optional[LiveExecutor] = None
        self._safety_checker = SafetyChecker()
        self._cached_balance: float = 0.0
        self._balance_last_check: Optional[datetime] = None

        if config.LIVE_TRADING:
            if not config.WALLET_PRIVATE_KEY:
                raise RuntimeError("LIVE_TRADING=True but WALLET_PRIVATE_KEY not set in env")
            if not config.WALLET_ADDRESS or not config.WALLET_FUNDER:
                raise RuntimeError("LIVE_TRADING=True requires WALLET_ADDRESS and WALLET_FUNDER")
            self._live_executor = LiveExecutor()
            logger.info(
                "LIVE TRADING ENABLED — wallet=%s, funder=%s, stake=$%.2f, daily_loss_limit=$%.2f",
                config.WALLET_ADDRESS, config.WALLET_FUNDER,
                config.LIVE_STAKE_USDC, config.LIVE_DAILY_LOSS_LIMIT_USDC,
            )
        else:
            logger.info("PAPER MODE (LIVE_TRADING=False)")

        # Session stats
        self._total_entries = 0
        self._total_wins = 0
        self._total_losses = 0
        self._total_pnl_usdc = 0.0
        self._trade_id_counter = 0
        self._binance_filtered_skips = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        await self._refresh_markets()

        await asyncio.gather(
            self._subscription_loop(),
            self._refresh_loop(),
            self._resolution_loop(),
            self._flush_loop(),
            self._sweep_loop(),
            self._heartbeat_loop(),
            self._price_feed.run(),
            self._balance_check_loop(),
        )

    # ------------------------------------------------------------------
    # Market refresh (mirrors orderbook_observer)
    # ------------------------------------------------------------------

    async def _refresh_markets(self) -> None:
        markets = scan()
        now = datetime.now(UTC)
        imminent = [m for m in markets if m.end_time > now][: self.max_markets]

        new_ids = {m.condition_id for m in imminent}
        old_ids = set(self._tracked.keys())

        added = new_ids - old_ids
        removed = old_ids - new_ids

        for cid in removed:
            slug = self._tracked[cid].slug
            for tid in (self._tracked[cid].up_token_id, self._tracked[cid].down_token_id):
                self._token_to_market.pop(tid, None)
                self._book_bids.pop(tid, None)
                self._book_asks.pop(tid, None)
            del self._tracked[cid]
            logger.info("Removed resolved market: %s", slug)

        new_tokens: list[str] = []
        for m in imminent:
            if m.condition_id in added:
                self._tracked[m.condition_id] = m
                self._token_to_market[m.up_token_id] = (m, "Up")
                self._token_to_market[m.down_token_id] = (m, "Down")
                new_tokens.extend([m.up_token_id, m.down_token_id])
                logger.info("Added market: %s (resolves in %.0fs)",
                            m.slug, (m.end_time - now).total_seconds())

        if new_tokens and self._ws is not None:
            try:
                await self._ws.send(
                    json.dumps({"type": "market", "assets_ids": new_tokens})
                )
            except Exception as exc:
                logger.warning("Could not subscribe new tokens: %s", exc)

    async def _refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.refresh_interval)
            await self._refresh_markets()

    # ------------------------------------------------------------------
    # WebSocket (mirrors orderbook_observer)
    # ------------------------------------------------------------------

    async def _subscription_loop(self) -> None:
        backoff = 3
        while self._running:
            try:
                await self._connect_and_stream()
                backoff = 3
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("WS error: %s — reconnecting in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(
            _WS_URL,
            open_timeout=15,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            logger.info("WebSocket connected: %s", _WS_URL)

            all_tokens = list(self._token_to_market.keys())
            if all_tokens:
                await ws.send(json.dumps({"type": "market", "assets_ids": all_tokens}))
                logger.info("Subscribed to %d tokens", len(all_tokens))

            async for raw_msg in ws:
                if not self._running:
                    break
                self._handle_ws_message(raw_msg)

        self._ws = None

    def _handle_ws_message(self, raw_msg: str) -> None:
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            return

        now = datetime.now(UTC)
        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue
            event_type = item.get("event_type")
            if event_type == "book":
                self._apply_book_snapshot(item, now)
            elif event_type == "price_change":
                self._apply_price_change(item, now)

    def _apply_book_snapshot(self, item: dict, now: datetime) -> None:
        asset_id = item.get("asset_id", "")
        if asset_id not in self._token_to_market:
            return
        bids_raw = item.get("bids") or []
        asks_raw = item.get("asks") or []
        self._book_bids[asset_id] = {float(b["price"]): float(b["size"]) for b in bids_raw}
        self._book_asks[asset_id] = {float(a["price"]): float(a["size"]) for a in asks_raw}
        self._evaluate_signals(asset_id, now)

    def _apply_price_change(self, item: dict, now: datetime) -> None:
        asset_id = item.get("asset_id", "")
        if asset_id not in self._token_to_market:
            return
        bids = self._book_bids.setdefault(asset_id, {})
        asks = self._book_asks.setdefault(asset_id, {})
        for change in item.get("changes", []):
            price = float(change["price"])
            size = float(change["size"])
            side = change.get("side", "").upper()
            book = bids if side == "BUY" else asks
            if size == 0:
                book.pop(price, None)
            else:
                book[price] = size
        self._evaluate_signals(asset_id, now)

    def _best_ask(self, asset_id: str) -> Optional[float]:
        asks = self._book_asks.get(asset_id, {})
        active = [p for p, s in asks.items() if s > 0]
        return min(active) if active else None

    def _best_bid(self, asset_id: str) -> Optional[float]:
        bids = self._book_bids.get(asset_id, {})
        active = [p for p, s in bids.items() if s > 0]
        return max(active) if active else None

    # ------------------------------------------------------------------
    # Signal evaluation — Edge 4
    # ------------------------------------------------------------------

    def _evaluate_signals(self, asset_id: str, now: datetime) -> None:
        entry = self._token_to_market.get(asset_id)
        if entry is None:
            return
        market, side = entry

        # BTC-only filter
        if market.asset not in EDGE4_ASSETS:
            return

        # UTC hour skip filter
        if now.hour in EDGE4_SKIP_UTC_HOURS:
            return

        # Dedup: one entry per (condition_id, side)
        key = (market.condition_id, side)
        if key in self._entered:
            return

        best_ask = self._best_ask(asset_id)
        best_bid = self._best_bid(asset_id)
        if best_ask is None or best_bid is None:
            return

        mid = (best_ask + best_bid) / 2.0
        seconds_to_resolution = (market.end_time - now).total_seconds()
        if seconds_to_resolution < 0:
            return

        if (EDGE4_TTL_MIN_S <= seconds_to_resolution <= EDGE4_TTL_MAX_S
                and EDGE4_MID_THRESHOLD <= mid < EDGE4_MAX_MID):
            target_s = (EDGE4_TTL_MIN_S + EDGE4_TTL_MAX_S) / 2.0
            self._fire_entry(market, side, best_ask, seconds_to_resolution,
                             target_s, EDGE4_LABEL, now)

    def _fire_entry(
        self,
        market: Market,
        side: str,
        best_ask: float,
        seconds_to_resolution: float,
        target_s: float,
        label: str,
        now: datetime,
    ) -> None:
        key = (market.condition_id, side)
        self._entered.add(key)
        self._trade_id_counter += 1
        trade_id = f"{now:%Y%m%d}_{self._trade_id_counter:06d}"

        simulated_shares = STAKE_USDC / best_ask
        fee = compute_taker_fee(best_ask, simulated_shares, fee_rate=DEFAULT_FEE_RATE)

        trade = PaperTrade(
            trade_id=trade_id,
            entry_timestamp_utc=now,
            market=market,
            side=side,
            signal_bucket_label=label,
            signal_target_time_s=target_s,
            seconds_to_resolution_at_entry=seconds_to_resolution,
            entry_price=best_ask,
            simulated_shares=simulated_shares,
            simulated_stake_usdc=STAKE_USDC,
            fee_usdc=fee,
        )
        self._open_trades.append(trade)
        self._total_entries += 1

        logger.info(
            "ENTRY %s: %s %s @ %.4f  T-%.0fs  bucket=%s  trade_id=%s",
            market.asset, market.slug, side, best_ask,
            seconds_to_resolution, label, trade_id,
        )

        # Schedule realistic execution simulation (runs concurrently, fills trade fields).
        # Guard against no running loop in sync test contexts — close coro to avoid warnings.
        _coro = self._simulate_realistic_fill(trade, market, side)
        try:
            asyncio.create_task(_coro)
        except RuntimeError:
            _coro.close()

        # Live order placement (only when LIVE_TRADING=True)
        if config.LIVE_TRADING and self._live_executor is not None:
            allowed, reason = self._safety_checker.can_place_order(
                balance_usdc=self._cached_balance,
                open_positions=len(self._open_trades),
                stake_usdc=config.LIVE_STAKE_USDC,
            )
            if not allowed:
                logger.warning(
                    "Live order BLOCKED for %s %s: %s",
                    market.slug, side, reason,
                )
                trade.live_fill_status = f"blocked:{reason}"
            else:
                token_id = market.up_token_id if side == "Up" else market.down_token_id
                # Add 2-cent buffer above signal ask so FAK fills at current market price.
                # Signal ask is captured at evaluation time; by order arrival the book may
                # have ticked up 1-2 cents, causing the FAK to find no match and be killed.
                live_price = min(round(best_ask + 0.05, 2), 0.99)
                size_shares, actual_notional = compute_clean_order_amounts(
                    config.LIVE_STAKE_USDC, live_price
                )
                logger.debug(
                    "Clean amounts: %.4f shares × $%.4f = $%.2f notional (signal ask=%.4f + 0.02 buffer)",
                    size_shares, live_price, actual_notional, best_ask,
                )
                _live_coro = self._place_live_order(trade, token_id, live_price, size_shares)
                try:
                    asyncio.create_task(_live_coro)
                except RuntimeError:
                    _live_coro.close()
                self._safety_checker.record_order(config.LIVE_STAKE_USDC)

    async def _simulate_realistic_fill(
        self,
        trade: PaperTrade,
        market: Market,
        side: str,
    ) -> None:
        """Run the realistic executor for this trade and store results on the trade object."""
        token_id = market.up_token_id if side == "Up" else market.down_token_id

        # Parse price range from label suffix, e.g. "T=270s_0.70-0.85" → (0.70, 0.85)
        try:
            price_part = trade.signal_bucket_label.split("_")[-1]
            min_str, max_str = price_part.split("-")
            signal_min_ask = float(min_str)
            signal_max_ask = float(max_str)
        except (ValueError, IndexError):
            logger.warning(
                "Could not parse bucket label %s for realistic fill", trade.signal_bucket_label
            )
            return

        try:
            fills = await self._realistic_executor.simulate_fill(
                token_id=token_id,
                signal_min_ask=signal_min_ask,
                signal_max_ask=signal_max_ask,
            )
        except Exception as exc:
            logger.warning("Realistic fill simulation failed: %s", exc)
            return

        trade.realistic_entry_price_1 = fills[1.0].weighted_avg_price if 1.0 in fills else None
        trade.realistic_entry_price_5 = fills[5.0].weighted_avg_price if 5.0 in fills else None
        trade.realistic_entry_price_25 = fills[25.0].weighted_avg_price if 25.0 in fills else None
        trade.realistic_out_of_bucket = fills[1.0].out_of_bucket if 1.0 in fills else None

        logger.info(
            "REALISTIC %s: paper=%.4f, fills(1/5/25)=%s/%s/%s, out_of_bucket=%s, trade=%s",
            market.slug,
            trade.entry_price,
            f"{trade.realistic_entry_price_1:.4f}" if trade.realistic_entry_price_1 is not None else "N/A",
            f"{trade.realistic_entry_price_5:.4f}" if trade.realistic_entry_price_5 is not None else "N/A",
            f"{trade.realistic_entry_price_25:.4f}" if trade.realistic_entry_price_25 is not None else "N/A",
            trade.realistic_out_of_bucket,
            trade.trade_id,
        )

    async def _place_live_order(
        self,
        trade: PaperTrade,
        token_id: str,
        price: float,
        size_shares: float,
    ) -> None:
        """Place a real Polymarket FAK order and store the result on the trade."""
        if self._live_executor is None:
            return

        try:
            result = await self._live_executor.place_order(
                token_id=token_id,
                price=price,
                size_shares=size_shares,
                side="BUY",
            )
        except Exception as exc:
            logger.warning("Live order task failed: %s", exc)
            trade.live_fill_status = f"task_error:{exc}"
            self._safety_checker.record_resolution(config.LIVE_STAKE_USDC)
            return

        trade.live_order_id = result.order_id
        trade.live_fill_status = result.fill_status
        trade.live_fill_price = result.avg_fill_price
        trade.live_filled_shares = result.filled_shares

        if result.fill_status != "filled":
            # Order didn't fill (FAK rejected) — money never at risk, release pending stake.
            self._safety_checker.record_resolution(config.LIVE_STAKE_USDC)

        logger.info(
            "LIVE %s: status=%s, fill=%s, shares=%.2f, order_id=%s",
            trade.market.slug,
            result.fill_status,
            f"{result.avg_fill_price:.4f}" if result.avg_fill_price else "N/A",
            result.filled_shares,
            result.order_id,
        )

    async def _balance_check_loop(self) -> None:
        """Periodically refresh the cached wallet balance (no-op in paper mode)."""
        if not config.LIVE_TRADING or self._live_executor is None:
            return

        try:
            self._cached_balance = await self._live_executor.get_balance()
            self._balance_last_check = datetime.now(UTC)
            logger.info("Initial wallet balance: $%.4f USDC", self._cached_balance)
        except Exception as exc:
            logger.warning("Initial balance fetch failed: %s", exc)

        while self._running:
            await asyncio.sleep(60)
            try:
                self._cached_balance = await self._live_executor.get_balance()
                self._balance_last_check = datetime.now(UTC)
            except Exception as exc:
                logger.warning("Balance check failed: %s", exc)

    # ------------------------------------------------------------------
    # Resolution matching
    # ------------------------------------------------------------------

    async def _resolution_loop(self) -> None:
        """Every 15s, check open trades whose markets have resolved."""
        while self._running:
            await asyncio.sleep(15)
            await self._resolve_open_trades()

    async def _resolve_open_trades(self, now: Optional[datetime] = None) -> None:
        """For open trades past end_time by at least 30s, query API and finalize."""
        if now is None:
            now = datetime.now(UTC)
        still_open: list[PaperTrade] = []

        for trade in self._open_trades:
            secs_past_end = (now - trade.market.end_time).total_seconds()
            if secs_past_end < 30:
                still_open.append(trade)
                continue

            # Query API for authoritative resolution
            winner = await self._fetch_winner_from_api(trade.market.condition_id)
            trade.resolution_timestamp_utc = now

            if winner is None:
                # Not yet resolved on API side; keep waiting up to 30 minutes past end
                if secs_past_end < RESOLUTION_TIMEOUT_SECONDS:
                    still_open.append(trade)
                    continue
                # Give up — mark as unknown (sweep loop will retry for 6 hours)
                trade.winner = "unknown"
                trade.payout_usdc = 0.0
                trade.pnl_usdc = -trade.simulated_stake_usdc - trade.fee_usdc
            else:
                trade.winner = winner
                if trade.side == winner:
                    trade.payout_usdc = trade.simulated_shares * 1.0
                    trade.pnl_usdc = trade.payout_usdc - trade.simulated_stake_usdc - trade.fee_usdc
                    self._total_wins += 1
                else:
                    trade.payout_usdc = 0.0
                    trade.pnl_usdc = -trade.simulated_stake_usdc - trade.fee_usdc
                    self._total_losses += 1

            self._total_pnl_usdc += trade.pnl_usdc

            # Compute live PnL if a real order was filled and market resolved
            if (
                config.LIVE_TRADING
                and trade.live_filled_shares
                and trade.live_fill_price
                and trade.winner != "unknown"
            ):
                cost = trade.live_filled_shares * trade.live_fill_price
                fee = trade.live_filled_shares * DEFAULT_FEE_RATE * trade.live_fill_price * (1 - trade.live_fill_price)
                if trade.winner == trade.side:
                    trade.live_pnl_usdc = trade.live_filled_shares - cost - fee
                else:
                    trade.live_pnl_usdc = -cost - fee
                # Outcome is now in CSV on next flush — release pending stake.
                self._safety_checker.record_resolution(config.LIVE_STAKE_USDC)

            self._buffer.append(trade)

            logger.info(
                "RESOLVED %s: side=%s winner=%s pnl=%+.4f  trade_id=%s",
                trade.market.slug, trade.side, trade.winner, trade.pnl_usdc, trade.trade_id,
            )

        self._open_trades = still_open

    async def _fetch_winner_from_api(self, condition_id: str) -> Optional[str]:
        """Query Polymarket's CLOB API for authoritative market resolution.

        Returns "Up", "Down", or None if the market is not yet resolved
        (or if any error occurred during the request).
        """
        url = f"{_CLOB_MARKETS_URL}/{condition_id}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("API winner lookup failed for %s: %s", condition_id, exc)
            return None

        # Market not yet resolved
        if not data.get("closed", False):
            return None

        # Inspect tokens for the winner flag
        tokens = data.get("tokens", [])
        for token in tokens:
            if token.get("winner") is True:
                outcome = token.get("outcome")
                if outcome in ("Up", "Down"):
                    return outcome

        # Market is closed but no winner flagged — shouldn't happen but log it
        logger.warning("Market %s is closed but no winner token found", condition_id)
        return None

    # ------------------------------------------------------------------
    # Sweep retry: recover late-resolved unknowns
    # ------------------------------------------------------------------

    async def _sweep_loop(self) -> None:
        """Periodically re-check unknown trades; some markets resolve >5 min late."""
        while self._running:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            await self._sweep_unknowns()

    async def _sweep_unknowns(self) -> None:
        """Walk today's CSV and re-resolve any row marked 'unknown' if not too old."""
        today = datetime.now(UTC).strftime("%Y%m%d")
        path = Path(config.DATA_DIR) / f"paper_trades_{today}.csv"
        if not path.exists():
            return

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        if not rows or not fieldnames:
            return

        now = datetime.now(UTC)
        fixed_count = 0

        for row in rows:
            if row["winner"] != "unknown":
                continue

            try:
                entry_ts = datetime.fromisoformat(row["entry_timestamp_utc"])
            except (ValueError, KeyError):
                continue
            age_seconds = (now - entry_ts).total_seconds()
            if age_seconds > SWEEP_MAX_AGE_SECONDS:
                continue

            winner = await self._fetch_winner_from_api(row["condition_id"])
            if winner is None:
                continue

            # Reverse the phantom-loss accounting
            previous_pnl = float(row["pnl_usdc"])
            self._total_pnl_usdc -= previous_pnl
            self._total_losses -= 1

            side = row["side"]
            shares = float(row["simulated_shares"])
            stake = float(row["simulated_stake_usdc"])
            fee = float(row["fee_usdc"])

            if side == winner:
                payout = shares * 1.0
                pnl = payout - stake - fee
                self._total_wins += 1
            else:
                payout = 0.0
                pnl = -stake - fee
                self._total_losses += 1

            self._total_pnl_usdc += pnl

            row["winner"] = winner
            row["payout_usdc"] = f"{payout:.4f}"
            row["pnl_usdc"] = f"{pnl:.4f}"
            row["resolution_timestamp_utc"] = now.isoformat(timespec="milliseconds")
            fixed_count += 1

            logger.info(
                "BACKFILL %s: side=%s winner=%s pnl=%+.4f  trade_id=%s",
                row["market_slug"], side, winner, pnl, row["trade_id"],
            )

        if fixed_count == 0:
            return

        # Atomic rewrite: write to .tmp then rename
        tmp_path = path.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
        logger.info("Sweep complete: fixed %d unknown trades", fixed_count)

    # ------------------------------------------------------------------
    # CSV flush
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(10)
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        today = datetime.now(UTC).strftime("%Y%m%d")
        path = Path(config.DATA_DIR) / f"paper_trades_{today}.csv"
        is_new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(_CSV_HEADER)
            for t in self._buffer:
                writer.writerow([
                    t.trade_id,
                    t.entry_timestamp_utc.isoformat(timespec="milliseconds"),
                    t.market.slug,
                    t.market.asset,
                    t.market.condition_id,
                    t.side,
                    t.signal_bucket_label,
                    f"{t.signal_target_time_s:.0f}",
                    f"{t.seconds_to_resolution_at_entry:.1f}",
                    f"{t.entry_price:.4f}",
                    f"{t.simulated_shares:.6f}",
                    f"{t.simulated_stake_usdc:.4f}",
                    f"{t.fee_usdc:.5f}",
                    t.resolution_timestamp_utc.isoformat(timespec="milliseconds") if t.resolution_timestamp_utc else "",
                    t.winner or "",
                    f"{t.payout_usdc:.4f}" if t.payout_usdc is not None else "",
                    f"{t.pnl_usdc:.4f}" if t.pnl_usdc is not None else "",
                    # Realistic execution columns
                    f"{t.realistic_entry_price_1:.4f}" if t.realistic_entry_price_1 is not None else "",
                    f"{t.realistic_entry_price_5:.4f}" if t.realistic_entry_price_5 is not None else "",
                    f"{t.realistic_entry_price_25:.4f}" if t.realistic_entry_price_25 is not None else "",
                    str(t.realistic_out_of_bucket) if t.realistic_out_of_bucket is not None else "",
                    # Live execution columns
                    t.live_order_id or "",
                    t.live_fill_status or "",
                    f"{t.live_fill_price:.4f}" if t.live_fill_price is not None else "",
                    f"{t.live_filled_shares:.4f}" if t.live_filled_shares is not None else "",
                    f"{t.live_pnl_usdc:.4f}" if t.live_pnl_usdc is not None else "",
                ])
        logger.debug("Flushed %d trades to %s", len(self._buffer), path)
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            total_closed = self._total_wins + self._total_losses
            win_rate_str = f"{self._total_wins / total_closed * 100:.1f}%" if total_closed else "n/a"
            logger.info(
                "Paper trader: %d markets, %d entries, %d open, %d wins, %d losses (%s), "
                "PnL=%+.4f USDC, binance_skips=%d",
                len(self._tracked),
                self._total_entries,
                len(self._open_trades),
                self._total_wins,
                self._total_losses,
                win_rate_str,
                self._total_pnl_usdc,
                self._binance_filtered_skips,
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("Shutting down — flushing buffer...")
        self._running = False
        self._flush()
        sys.exit(0)


if __name__ == "__main__":
    trader = PaperTrader(max_markets=20, refresh_interval=60)
    asyncio.run(trader.run())
