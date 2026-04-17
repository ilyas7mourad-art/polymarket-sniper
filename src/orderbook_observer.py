"""Real-time orderbook observer for Polymarket BTC/ETH 5-minute markets.

Connects to the Polymarket CLOB WebSocket, subscribes to the 20 markets
closest to resolution, and logs every orderbook update to a daily CSV.
See docs/polymarket_orderbook.md for WebSocket protocol details.
"""

import asyncio
import csv
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import websockets

from src.config import config
from src.scanner import Market, scan

logger = logging.getLogger(__name__)

UTC = timezone.utc

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_CSV_HEADER = [
    "timestamp_utc", "market_slug", "asset", "condition_id",
    "side", "best_bid", "best_ask", "mid", "seconds_to_resolution",
]


@dataclass
class OrderbookTick:
    """A single orderbook snapshot for one side of one market."""

    timestamp_utc: datetime
    market_slug: str
    asset: str
    condition_id: str
    side: str        # "Up" or "Down"
    best_bid: float
    best_ask: float
    seconds_to_resolution: float

    @property
    def mid(self) -> float:
        """Midpoint price."""
        return (self.best_bid + self.best_ask) / 2


class OrderbookObserver:
    """Observes CLOB orderbooks and writes ticks to a daily CSV.

    Uses the Polymarket WebSocket market channel (no auth required).
    Maintains an in-memory order book per token_id and emits a tick on
    every update. Refreshes the tracked market list every refresh_interval
    seconds.
    """

    def __init__(self, max_markets: int = 20, refresh_interval: int = 60) -> None:
        self.max_markets = max_markets
        self.refresh_interval = refresh_interval

        # condition_id → Market
        self._tracked: dict[str, Market] = {}
        # asset_id (token) → (Market, "Up" or "Down")
        self._token_to_market: dict[str, tuple[Market, str]] = {}
        # In-memory book: asset_id → {price_float: size_float}
        self._book_bids: dict[str, dict[float, float]] = {}
        self._book_asks: dict[str, dict[float, float]] = {}

        self._buffer: list[OrderbookTick] = []
        self._running: bool = False
        self._tick_count_since_last_heartbeat: int = 0

        # Active WebSocket connection (set by _subscription_loop)
        self._ws: Optional[websockets.WebSocketClientProtocol] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: refresh market list, stream orderbook, flush CSV."""
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        await self._refresh_markets()

        await asyncio.gather(
            self._subscription_loop(),
            self._refresh_loop(),
            self._flush_loop(),
            self._heartbeat_loop(),
        )

    # ------------------------------------------------------------------
    # Market refresh
    # ------------------------------------------------------------------

    async def _refresh_markets(self) -> None:
        """Scan for active markets and update the tracked set."""
        markets = scan()
        now = datetime.now(UTC)
        imminent = [m for m in markets if m.end_time > now][: self.max_markets]

        new_ids = {m.condition_id for m in imminent}
        old_ids = set(self._tracked.keys())

        added = new_ids - old_ids
        removed = old_ids - new_ids

        for cid in removed:
            slug = self._tracked[cid].slug
            # Remove token mappings for this market
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
                logger.info(
                    "Added market: %s (resolves in %.0fs)",
                    m.slug, (m.end_time - now).total_seconds(),
                )

        # Subscribe the new tokens on the existing WS connection if available
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
    # WebSocket subscription
    # ------------------------------------------------------------------

    async def _subscription_loop(self) -> None:
        """Connect to WS, subscribe, stream. Reconnects on any error."""
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
        """Open one WS connection and process messages until disconnect."""
        async with websockets.connect(  # type: ignore[attr-defined]
            _WS_URL,
            open_timeout=15,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            logger.info("WebSocket connected: %s", _WS_URL)

            # Subscribe to all currently tracked tokens
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
        """Parse and process one WebSocket message."""
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError as exc:
            logger.debug("Unparseable WS message: %s", exc)
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
            # Other event types (market_resolved, etc.) are silently ignored

    def _apply_book_snapshot(self, item: dict, now: datetime) -> None:
        """Replace in-memory book with snapshot and emit a tick."""
        asset_id = item.get("asset_id", "")
        if asset_id not in self._token_to_market:
            return

        bids_raw = item.get("bids") or []
        asks_raw = item.get("asks") or []

        bids: dict[float, float] = {float(b["price"]): float(b["size"]) for b in bids_raw}
        asks: dict[float, float] = {float(a["price"]): float(a["size"]) for a in asks_raw}

        self._book_bids[asset_id] = bids
        self._book_asks[asset_id] = asks

        self._emit_tick(asset_id, now)

    def _apply_price_change(self, item: dict, now: datetime) -> None:
        """Apply incremental level changes and emit a tick."""
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

        self._emit_tick(asset_id, now)

    def _best_bid(self, asset_id: str) -> float:
        bids = self._book_bids.get(asset_id, {})
        active = [p for p, s in bids.items() if s > 0]
        return max(active) if active else 0.0

    def _best_ask(self, asset_id: str) -> float:
        asks = self._book_asks.get(asset_id, {})
        active = [p for p, s in asks.items() if s > 0]
        return min(active) if active else 1.0

    def _emit_tick(self, asset_id: str, now: datetime) -> None:
        entry = self._token_to_market.get(asset_id)
        if entry is None:
            return
        market, side = entry
        secs = (market.end_time - now).total_seconds()
        tick = OrderbookTick(
            timestamp_utc=now,
            market_slug=market.slug,
            asset=market.asset,
            condition_id=market.condition_id,
            side=side,
            best_bid=self._best_bid(asset_id),
            best_ask=self._best_ask(asset_id),
            seconds_to_resolution=secs,
        )
        self._record_tick(tick)

    # ------------------------------------------------------------------
    # Buffering and CSV flush
    # ------------------------------------------------------------------

    def _record_tick(self, tick: OrderbookTick) -> None:
        self._buffer.append(tick)
        self._tick_count_since_last_heartbeat += 1
        if len(self._buffer) >= 50:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        today = datetime.now(UTC).strftime("%Y%m%d")
        path = Path(config.DATA_DIR) / f"orderbook_{today}.csv"
        is_new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(_CSV_HEADER)
            for t in self._buffer:
                writer.writerow([
                    t.timestamp_utc.isoformat(timespec="milliseconds"),
                    t.market_slug,
                    t.asset,
                    t.condition_id,
                    t.side,
                    f"{t.best_bid:.4f}",
                    f"{t.best_ask:.4f}",
                    f"{t.mid:.4f}",
                    f"{t.seconds_to_resolution:.1f}",
                ])
        logger.debug("Flushed %d ticks to %s", len(self._buffer), path)
        self._buffer.clear()

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(5)
            self._flush()

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            logger.info(
                "Observer: tracking %d markets, logged %d rows in last 60s",
                len(self._tracked),
                self._tick_count_since_last_heartbeat,
            )
            self._tick_count_since_last_heartbeat = 0

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("Shutting down — flushing buffer...")
        self._running = False
        self._flush()
        sys.exit(0)


if __name__ == "__main__":
    observer = OrderbookObserver(max_markets=20, refresh_interval=60)
    asyncio.run(observer.run())
