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

import websockets

from src.analysis import DEFAULT_FEE_RATE, compute_taker_fee
from src.config import config
from src.scanner import Market, scan

logger = logging.getLogger(__name__)

UTC = timezone.utc

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Simulated stake per trade (USDC). Chosen for clean math in logs.
STAKE_USDC = 1.0

# Signal rules: each is (target_seconds, seconds_tolerance, min_ask, max_ask, label).
# Target times and price buckets are validated by analysis PRs #6-8 on 33h of data.
SIGNAL_RULES: list[tuple[float, float, float, float, str]] = [
    (60.0, 5.0, 0.90, 0.95, "T=60s_0.90-0.95"),
    (60.0, 5.0, 0.95, 1.00, "T=60s_0.95-1.00"),
    (10.0, 5.0, 0.95, 1.00, "T=10s_0.95-1.00"),
]

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

        # Session stats
        self._total_entries = 0
        self._total_wins = 0
        self._total_losses = 0
        self._total_pnl_usdc = 0.0
        self._trade_id_counter = 0

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
            self._heartbeat_loop(),
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

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _evaluate_signals(self, asset_id: str, now: datetime) -> None:
        entry = self._token_to_market.get(asset_id)
        if entry is None:
            return
        market, side = entry

        # Skip if we already entered this market/side
        key = (market.condition_id, side)
        if key in self._entered:
            return

        best_ask = self._best_ask(asset_id)
        if best_ask is None:
            return

        seconds_to_resolution = (market.end_time - now).total_seconds()
        if seconds_to_resolution < 0:
            return

        for target_s, tol_s, min_ask, max_ask, label in SIGNAL_RULES:
            in_time_window = abs(seconds_to_resolution - target_s) <= tol_s
            in_price_bucket = min_ask <= best_ask < max_ask
            if in_time_window and in_price_bucket:
                self._fire_entry(
                    market, side, best_ask, seconds_to_resolution,
                    target_s, label, now,
                )
                return  # one entry fires at most per tick

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

    # ------------------------------------------------------------------
    # Resolution matching
    # ------------------------------------------------------------------

    async def _resolution_loop(self) -> None:
        """Every 15s, check open trades whose markets have resolved."""
        while self._running:
            await asyncio.sleep(15)
            self._resolve_open_trades()

    def _resolve_open_trades(self, now: Optional[datetime] = None) -> None:
        """For open trades past end_time by at least 30s, determine winner and finalize."""
        if now is None:
            now = datetime.now(UTC)
        still_open: list[PaperTrade] = []

        for trade in self._open_trades:
            secs_past_end = (now - trade.market.end_time).total_seconds()
            if secs_past_end < 30:
                still_open.append(trade)
                continue

            # Determine winner from current book state
            winner = self._determine_winner_from_book(trade.market)
            trade.resolution_timestamp_utc = now
            trade.winner = winner

            if winner == "unknown":
                # Can't finalize yet; keep waiting up to 5 minutes past end
                if secs_past_end < 300:
                    still_open.append(trade)
                    continue
                # Give up and write with winner=unknown
                trade.payout_usdc = 0.0
                trade.pnl_usdc = -trade.simulated_stake_usdc - trade.fee_usdc
            else:
                if trade.side == winner:
                    trade.payout_usdc = trade.simulated_shares * 1.0
                    trade.pnl_usdc = trade.payout_usdc - trade.simulated_stake_usdc - trade.fee_usdc
                    self._total_wins += 1
                else:
                    trade.payout_usdc = 0.0
                    trade.pnl_usdc = -trade.simulated_stake_usdc - trade.fee_usdc
                    self._total_losses += 1

            self._total_pnl_usdc += trade.pnl_usdc
            self._buffer.append(trade)

            logger.info(
                "RESOLVED %s: side=%s winner=%s pnl=%+.4f  trade_id=%s",
                trade.market.slug, trade.side, winner, trade.pnl_usdc, trade.trade_id,
            )

        self._open_trades = still_open

    def _determine_winner_from_book(self, market: Market) -> str:
        """Inspect the current best_ask on each side to infer the winner.

        At resolution, the winning side has best_ask near 1.0 (and best_bid near 1.0);
        the losing side has best_ask near 0.0. If the book hasn't settled, returns "unknown".
        """
        up_ask = self._best_ask(market.up_token_id)
        down_ask = self._best_ask(market.down_token_id)

        up_bids = self._book_bids.get(market.up_token_id, {})
        down_bids = self._book_bids.get(market.down_token_id, {})
        up_best_bid = max((p for p, s in up_bids.items() if s > 0), default=None)
        down_best_bid = max((p for p, s in down_bids.items() if s > 0), default=None)

        def side_state(ask: Optional[float], bid: Optional[float]) -> str:
            val = ask if ask is not None else bid
            if val is None:
                return "unknown"
            if val >= 0.95:
                return "high"
            if val <= 0.05:
                return "low"
            return "mid"

        up_state = side_state(up_ask, up_best_bid)
        down_state = side_state(down_ask, down_best_bid)

        if up_state == "high" and down_state == "low":
            return "Up"
        if down_state == "high" and up_state == "low":
            return "Down"
        return "unknown"

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
                "Paper trader: %d markets, %d entries, %d open, %d wins, %d losses (%s), PnL=%+.4f USDC",
                len(self._tracked),
                self._total_entries,
                len(self._open_trades),
                self._total_wins,
                self._total_losses,
                win_rate_str,
                self._total_pnl_usdc,
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
