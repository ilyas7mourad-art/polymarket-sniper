"""
Bonereaper-style dynamic accumulation paper trader.

Strategy (reverse-engineered from 196k trades, Apr 27 – May 7 2026):
  1. At candle open (TTL ~265-300s), enter BOTH sides with a base stake.
  2. As one side's mid climbs above 0.50, accumulate that side in steps.
  3. Stake per entry scales with conviction: bigger bets the more certain the outcome.
  4. Never sell — hold everything to resolution.

Capital model:
  - TOTAL_CAPITAL USDC is the simulated budget.
  - Each entry deducts from the virtual balance; resolution credits it back.
  - New entries are blocked if total live exposure exceeds MAX_TOTAL_EXPOSURE.
"""

import argparse
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
from src.scanner import Market, scan

logger = logging.getLogger(__name__)
UTC = timezone.utc

# ── Capital parameters ────────────────────────────────────────────────────────
TOTAL_CAPITAL        = 94.0   # simulated USDC budget
MAX_TOTAL_EXPOSURE   = 70.0   # max simultaneously deployed across all candles
MAX_CANDLE_EXPOSURE  = 18.0   # max deployed into a single candle
BASE_STAKE           = 0.25   # stake on each first entry (both sides at open)

# ── Accumulation parameters ───────────────────────────────────────────────────
CANDLE_ENTER_TTL_HI  = 300    # don't track a market we see for the first time past this
CANDLE_ENTER_TTL_LO  = 250    # enter both sides once TTL drops below this
STOP_ENTRY_TTL       = 15     # stop accumulating with this many seconds left
OPEN_MID_BAND        = 0.06   # |mid - 0.50| must be < this to trigger the opening entry
PRICE_STEP           = 0.04   # minimum mid increase since last entry to add more

# ── Binance signal parameters ─────────────────────────────────────────────────
BINANCE_LOOKBACK     = 30     # seconds of BTC price history to measure momentum
BINANCE_THRESHOLD    = 0.0002 # minimum |move| to trigger accumulation (0.02%)

# ── Resolution ────────────────────────────────────────────────────────────────
RESOLUTION_TIMEOUT   = 1800   # give up on a market after this many seconds past end
SWEEP_INTERVAL       = 300

_CLOB_MARKETS_URL = "https://clob.polymarket.com/markets"
_WS_URL           = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_CSV_HEADER = [
    "trade_id", "entry_timestamp_utc", "market_slug", "asset",
    "condition_id", "side", "entry_type",
    "seconds_to_resolution_at_entry", "entry_price", "entry_mid",
    "shares", "stake_usdc", "fee_usdc",
    "binance_move_pct", "prev_ask_tick",
    "resolution_timestamp_utc", "winner", "payout_usdc", "pnl_usdc",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Entry:
    """One accumulation buy into a market side."""
    trade_id:    str
    side:        str          # "Up" | "Down"
    entry_type:  str          # "open" | "accum"
    price:       float        # best_ask at entry
    mid:         float        # mid at entry
    stake:       float
    shares:      float
    fee:         float
    entry_time:  datetime

    binance_move_pct: float        = 0.0   # Binance 30s move at signal time
    prev_ask_tick:    Optional[float] = None  # ask from prior WS tick

    winner:   Optional[str]   = None
    pnl:      Optional[float] = None


@dataclass
class MarketState:
    """Running accumulation state for one 5-min BTC candle."""
    market:      Market
    first_seen:  datetime

    # Both-sides opening done?
    opened_up:   bool = False
    opened_down: bool = False

    # Last mid at which we added each side (to enforce PRICE_STEP)
    last_mid:    dict = field(default_factory=lambda: {"Up": 0.0, "Down": 0.0})

    entries:     list[Entry] = field(default_factory=list)
    total_staked: float = 0.0

    resolved:   bool          = False
    winner:     Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def conviction_stake(mid: float) -> float:
    """Scale stake quadratically with distance from 0.50.

    At mid=0.50 → BASE_STAKE.  At mid=0.90 → ~5× BASE_STAKE.
    Capped at $5 to protect capital on any single entry.
    """
    deviation = abs(mid - 0.50)          # 0 … 0.50
    multiplier = (1.0 + deviation / 0.08) ** 1.8
    return round(min(BASE_STAKE * multiplier, 5.0), 2)


# ── Trader ────────────────────────────────────────────────────────────────────

class BonereaperTrader:

    def __init__(self, up_only: bool = False, slippage: float = 0.0, accum_only: bool = False) -> None:
        self._up_only    = up_only
        self._slippage   = slippage
        self._accum_only = accum_only
        self._binance    = BinancePriceFeed()
        if accum_only:
            sides_tag = "accum"
        elif up_only:
            sides_tag = "up"
        else:
            sides_tag = "both"
        slip_tag  = f"_slip{int(slippage*100):02d}" if slippage > 0 else ""
        self._mode_tag  = f"{sides_tag}{slip_tag}"

        self._tracked:          dict[str, Market]      = {}   # cid → Market
        self._token_to_market:  dict[str, tuple[Market, str]] = {}
        self._book_bids:        dict[str, dict[float, float]] = {}
        self._book_asks:        dict[str, dict[float, float]] = {}
        self._prev_ask:         dict[str, float]        = {}   # ask from prior WS tick
        self._market_states:    dict[str, MarketState]  = {}

        # Pending resolution
        self._open_entries:  list[Entry]       = []   # all un-resolved entries
        self._open_states:   list[MarketState] = []   # states with open entries
        self._flush_buffer:  list[tuple[MarketState, Entry]] = []

        self._running  = False
        self._ws:      Optional[websockets.WebSocketClientProtocol] = None  # type: ignore

        # Virtual capital tracking
        self._balance      = TOTAL_CAPITAL
        self._total_pnl    = 0.0
        self._total_entries = 0
        self._wins = self._losses = 0
        self._trade_counter = 0

    # ── Entry point ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        await self._refresh_markets()
        await asyncio.gather(
            self._binance.run(),
            self._ws_loop(),
            self._refresh_loop(),
            self._resolution_loop(),
            self._flush_loop(),
            self._sweep_loop(),
            self._heartbeat_loop(),
        )

    # ── Market management ────────────────────────────────────────────────────

    async def _refresh_markets(self) -> None:
        markets = scan()
        now = datetime.now(UTC)
        imminent = [m for m in markets if m.asset == "BTC" and m.end_time > now]

        new_ids = {m.condition_id for m in imminent}
        old_ids = set(self._tracked.keys())

        for cid in old_ids - new_ids:
            m = self._tracked.pop(cid)
            for tid in (m.up_token_id, m.down_token_id):
                self._token_to_market.pop(tid, None)
                self._book_bids.pop(tid, None)
                self._book_asks.pop(tid, None)

        new_tokens: list[str] = []
        for m in imminent:
            if m.condition_id not in self._tracked:
                self._tracked[m.condition_id] = m
                self._token_to_market[m.up_token_id]   = (m, "Up")
                self._token_to_market[m.down_token_id] = (m, "Down")
                new_tokens.extend([m.up_token_id, m.down_token_id])

        if new_tokens and self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "market", "assets_ids": new_tokens}))
            except Exception as exc:
                logger.warning("WS subscribe failed: %s", exc)

    async def _refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            await self._refresh_markets()

    # ── WebSocket ────────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        backoff = 3
        while self._running:
            try:
                await self._connect_and_stream()
                backoff = 3
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("WS error: %s — retry in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(_WS_URL, open_timeout=15, close_timeout=5,
                                      max_size=10 * 1024 * 1024) as ws:
            self._ws = ws
            all_tokens = list(self._token_to_market.keys())
            if all_tokens:
                await ws.send(json.dumps({"type": "market", "assets_ids": all_tokens}))
            async for raw in ws:
                if not self._running:
                    break
                self._handle_ws(raw)
        self._ws = None

    def _handle_ws(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        now = datetime.now(UTC)
        for item in (data if isinstance(data, list) else [data]):
            if not isinstance(item, dict):
                continue
            t = item.get("event_type")
            if t == "book":
                self._apply_snapshot(item, now)
            elif t == "price_change":
                self._apply_change(item, now)

    def _apply_snapshot(self, item: dict, now: datetime) -> None:
        aid = item.get("asset_id", "")
        if aid not in self._token_to_market:
            return
        self._book_bids[aid] = {float(b["price"]): float(b["size"]) for b in item.get("bids") or []}
        self._book_asks[aid] = {float(a["price"]): float(a["size"]) for a in item.get("asks") or []}
        self._on_book_update(aid, now)

    def _apply_change(self, item: dict, now: datetime) -> None:
        aid = item.get("asset_id", "")
        if aid not in self._token_to_market:
            return
        bids = self._book_bids.setdefault(aid, {})
        asks = self._book_asks.setdefault(aid, {})
        for c in item.get("changes", []):
            p, s = float(c["price"]), float(c["size"])
            book = bids if c.get("side", "").upper() == "BUY" else asks
            if s == 0:
                book.pop(p, None)
            else:
                book[p] = s
        self._on_book_update(aid, now)

    def _best_ask(self, aid: str) -> Optional[float]:
        active = [p for p, s in self._book_asks.get(aid, {}).items() if s > 0]
        return min(active) if active else None

    def _best_bid(self, aid: str) -> Optional[float]:
        active = [p for p, s in self._book_bids.get(aid, {}).items() if s > 0]
        return max(active) if active else None

    # ── Core signal logic ────────────────────────────────────────────────────

    def _on_book_update(self, aid: str, now: datetime) -> None:
        entry = self._token_to_market.get(aid)
        if entry is None:
            return
        market, side = entry

        prev_ask = self._prev_ask.get(aid)

        best_ask = self._best_ask(aid)
        best_bid = self._best_bid(aid)
        if best_ask is None or best_bid is None:
            return
        self._prev_ask[aid] = best_ask

        mid = (best_ask + best_bid) / 2.0
        ttl = (market.end_time - now).total_seconds()

        if ttl < 0 or ttl > CANDLE_ENTER_TTL_HI:
            return

        cid = market.condition_id

        # ── Initialise state on first sighting ──
        if cid not in self._market_states:
            if ttl < CANDLE_ENTER_TTL_LO:
                return  # too late to open a fresh candle
            self._market_states[cid] = MarketState(market=market, first_seen=now)
            logger.debug("Tracking new candle: %s  TTL=%.0fs", market.slug, ttl)

        state = self._market_states[cid]

        if state.resolved or ttl < STOP_ENTRY_TTL:
            return

        if self._up_only and side == "Down":
            return

        # Capital guards
        live_exposure = TOTAL_CAPITAL - self._balance
        if live_exposure >= MAX_TOTAL_EXPOSURE:
            return
        if state.total_staked >= MAX_CANDLE_EXPOSURE:
            return

        # ── Phase 1: opening — enter BOTH sides when market is flat ──
        if not self._accum_only:
            opened_this_side = (side == "Up" and state.opened_up) or (side == "Down" and state.opened_down)
            if not opened_this_side:
                if ttl < CANDLE_ENTER_TTL_LO and abs(mid - 0.50) <= OPEN_MID_BAND:
                    self._add_entry(state, side, best_ask, mid, BASE_STAKE, "open", now,
                                    binance_move_pct=0.0, prev_ask=prev_ask)
                    if side == "Up":
                        state.opened_up = True
                    else:
                        state.opened_down = True
                return

        # ── Phase 2: accumulate the Binance-aligned side ──
        # Only enter when Binance confirms a real move in this direction,
        # regardless of where Polymarket mid currently sits (it may lag).
        move_pct = 0.0
        if not self._accum_only:
            move_pct = self._binance.get_move_pct("BTC", BINANCE_LOOKBACK)
            if abs(move_pct) < BINANCE_THRESHOLD:
                return  # no meaningful Binance momentum yet

            binance_dir = "Up" if move_pct > 0 else "Down"
            if side != binance_dir:
                return  # only accumulate the Binance-aligned side

        # In accum_only mode: skip the disfavored side — only follow the crowd
        if self._accum_only and mid < 0.50:
            return

        last = state.last_mid[side]
        if mid < last + PRICE_STEP:
            return  # rate-limit: wait for Polymarket to tick further

        stake = conviction_stake(mid)
        # Cap so we don't blow the per-candle limit or available balance
        remaining = MAX_CANDLE_EXPOSURE - state.total_staked
        stake = min(stake, remaining, self._balance)
        if stake < 0.05:
            return

        self._add_entry(state, side, best_ask, mid, stake, "accum", now,
                        binance_move_pct=move_pct, prev_ask=prev_ask)

    def _add_entry(
        self,
        state: MarketState,
        side: str,
        ask: float,
        mid: float,
        stake: float,
        entry_type: str,
        now: datetime,
        binance_move_pct: float = 0.0,
        prev_ask: Optional[float] = None,
    ) -> None:
        self._trade_counter += 1
        tid = f"{now:%Y%m%d}_{self._trade_counter:06d}"

        if self._slippage > 0:
            ask = min(ask + self._slippage, 0.99)

        shares = stake / ask
        fee    = compute_taker_fee(ask, shares, fee_rate=DEFAULT_FEE_RATE)
        ttl    = (state.market.end_time - now).total_seconds()

        e = Entry(
            trade_id=tid, side=side, entry_type=entry_type,
            price=ask, mid=mid, stake=stake, shares=shares, fee=fee,
            entry_time=now,
            binance_move_pct=binance_move_pct,
            prev_ask_tick=prev_ask,
        )
        state.entries.append(e)
        state.last_mid[side] = mid
        state.total_staked += stake
        self._balance -= stake
        self._total_entries += 1

        if state not in self._open_states:
            self._open_states.append(state)

        logger.info(
            "ENTRY [%s] %s %-4s @ %.3f  mid=%.3f  stake=$%.2f  TTL=%.0fs  bal=$%.2f",
            entry_type, state.market.slug, side, ask, mid, stake, ttl, self._balance,
        )

    # ── Resolution ───────────────────────────────────────────────────────────

    async def _resolution_loop(self) -> None:
        while self._running:
            await asyncio.sleep(15)
            await self._resolve()

    async def _resolve(self, now: Optional[datetime] = None) -> None:
        if now is None:
            now = datetime.now(UTC)
        still_open: list[MarketState] = []

        for state in self._open_states:
            secs_past = (now - state.market.end_time).total_seconds()
            if secs_past < 30:
                still_open.append(state)
                continue

            winner = await self._fetch_winner(state.market.condition_id)

            if winner is None:
                if secs_past < RESOLUTION_TIMEOUT:
                    still_open.append(state)
                else:
                    winner = "unknown"
                    self._settle_state(state, winner, now)
                continue

            self._settle_state(state, winner, now)

        self._open_states = still_open

    def _settle_state(self, state: MarketState, winner: str, now: datetime) -> None:
        state.resolved = True
        state.winner   = winner

        for e in state.entries:
            e.winner = winner
            if winner == "unknown":
                e.pnl = -e.stake - e.fee
                self._balance -= e.fee   # stake gone at entry; fee also lost
            elif e.side == winner:
                payout = e.shares * 1.0
                e.pnl  = payout - e.stake - e.fee
                self._wins    += 1
                self._balance += e.stake + e.pnl   # return stake + profit
            else:
                e.pnl  = -e.stake - e.fee
                self._losses  += 1
                self._balance -= e.fee   # stake gone at entry; fee also lost

            self._total_pnl += e.pnl
            self._flush_buffer.append((state, e))

            logger.info(
                "RESOLVED %s: side=%-4s winner=%-4s type=%-5s pnl=%+.3f  (bal=$%.2f)",
                state.market.slug, e.side, winner, e.entry_type, e.pnl, self._balance,
            )

    async def _fetch_winner(self, cid: str) -> Optional[str]:
        url = f"{_CLOB_MARKETS_URL}/{cid}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Winner fetch failed %s: %s", cid, exc)
            return None
        if not data.get("closed", False):
            return None
        for tok in data.get("tokens", []):
            if tok.get("winner") is True:
                outcome = tok.get("outcome")
                if outcome in ("Up", "Down"):
                    return outcome
        return None

    # ── Sweep (retry unknown resolutions) ────────────────────────────────────

    async def _sweep_loop(self) -> None:
        while self._running:
            await asyncio.sleep(SWEEP_INTERVAL)
            await self._sweep_unknowns()

    async def _sweep_unknowns(self) -> None:
        today = datetime.now(UTC).strftime("%Y%m%d")
        path  = Path("data") / f"bonereaper_paper_{self._mode_tag}_{today}.csv"
        if not path.exists():
            return

        with path.open("r", newline="") as f:
            reader     = csv.DictReader(f)
            rows       = list(reader)
            fieldnames = reader.fieldnames

        if not rows or not fieldnames:
            return

        now, fixed = datetime.now(UTC), 0
        for row in rows:
            if row["winner"] != "unknown":
                continue
            winner = await self._fetch_winner(row["condition_id"])
            if winner is None:
                continue

            stake  = float(row["stake_usdc"])
            shares = float(row["shares"])
            fee    = float(row["fee_usdc"])
            side   = row["side"]

            if side == winner:
                payout = shares
                pnl    = payout - stake - fee
                self._wins   += 1
                self._losses -= 1
                # Fee already deducted when settled as unknown; only credit payout
                self._balance += payout
            else:
                pnl  = -stake - fee
                payout = 0.0
                # Balance already correct from unknown settle (stake + fee both gone)

            self._total_pnl = self._total_pnl - float(row["pnl_usdc"]) + pnl
            row.update(winner=winner, payout_usdc=f"{payout:.4f}",
                       pnl_usdc=f"{pnl:.4f}",
                       resolution_timestamp_utc=now.isoformat(timespec="milliseconds"))
            fixed += 1

        if not fixed:
            return

        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
        logger.info("Sweep: fixed %d unknown entries", fixed)

    # ── CSV flush ─────────────────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(10)
            self._flush()

    def _flush(self) -> None:
        if not self._flush_buffer:
            return
        today = datetime.now(UTC).strftime("%Y%m%d")
        path  = Path("data") / f"bonereaper_paper_{self._mode_tag}_{today}.csv"
        is_new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(_CSV_HEADER)
            for state, e in self._flush_buffer:
                w.writerow([
                    e.trade_id,
                    e.entry_time.isoformat(timespec="milliseconds"),
                    state.market.slug,
                    state.market.asset,
                    state.market.condition_id,
                    e.side,
                    e.entry_type,
                    f"{(state.market.end_time - e.entry_time).total_seconds():.1f}",
                    f"{e.price:.4f}",
                    f"{e.mid:.4f}",
                    f"{e.shares:.6f}",
                    f"{e.stake:.4f}",
                    f"{e.fee:.5f}",
                    f"{e.binance_move_pct:.6f}",
                    f"{e.prev_ask_tick:.4f}" if e.prev_ask_tick is not None else "",
                    datetime.now(UTC).isoformat(timespec="milliseconds"),
                    e.winner or "",
                    f"{(e.shares if e.side == e.winner else 0.0):.4f}",
                    f"{e.pnl:.4f}" if e.pnl is not None else "",
                ])
        self._flush_buffer.clear()

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            closed = self._wins + self._losses
            wr     = f"{self._wins/closed:.1%}" if closed else "n/a"
            exposure = TOTAL_CAPITAL - self._balance
            bnc_move = self._binance.get_move_pct("BTC", BINANCE_LOOKBACK)
            bnc_dir  = self._binance.get_direction("BTC", BINANCE_LOOKBACK) or "flat"
            logger.info(
                "HEARTBEAT | bal=$%.2f  exposure=$%.2f  entries=%d  "
                "wins=%d losses=%d wr=%s  PnL=%+.2f  "
                "bnc=%s(%.3f%%)  slip=%.3f",
                self._balance, exposure, self._total_entries,
                self._wins, self._losses, wr, self._total_pnl,
                bnc_dir, bnc_move * 100, self._slippage,
            )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        logger.info("Shutting down — flushing...")
        self._running = False
        self._flush()
        sys.exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--up-only", action="store_true", help="Trade Up side only")
    ap.add_argument("--accum-only", action="store_true",
                    help="Skip the flat-open phase; accumulate Binance-aligned side on every candle")
    ap.add_argument("--slippage", type=float, default=0.0,
                    help="Base slippage in price units (e.g. 0.02 = 2¢), scaled by mid conviction")
    args = ap.parse_args()

    if args.accum_only:
        sides_tag = "accum"
    elif args.up_only:
        sides_tag = "up"
    else:
        sides_tag = "both"
    slip_tag  = f"_slip{int(args.slippage*100):02d}" if args.slippage > 0 else ""
    mode_tag  = f"{sides_tag}{slip_tag}"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path("data") / f"bonereaper_paper_{mode_tag}_{datetime.now(UTC):%Y%m%d}.log"
        ),
    ]:
        h.setFormatter(fmt)
        root.addHandler(h)
    trader = BonereaperTrader(up_only=args.up_only, slippage=args.slippage, accum_only=args.accum_only)
    asyncio.run(trader.run())
