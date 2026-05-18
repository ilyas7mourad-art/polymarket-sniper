"""Streams real-time BTC/ETH spot prices from Binance and exposes directional momentum."""

import asyncio
import json
import logging
import signal as os_signal
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import websockets

logger = logging.getLogger(__name__)

UTC = timezone.utc

_BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade"
)

# How long to keep price history (seconds).
HISTORY_RETENTION_SECONDS = 60

# Default lookback window for directional signal (seconds).
SIGNAL_LOOKBACK_SECONDS = 30

# Price is considered stale if last update was this many seconds ago.
MAX_STALE_SECONDS = 10

_STREAM_TO_ASSET = {
    "btcusdt@trade": "BTC",
    "ethusdt@trade": "ETH",
}


class BinancePriceFeed:
    """Streams Binance spot trades and computes directional momentum."""

    def __init__(self) -> None:
        self._history: dict[str, deque[tuple[datetime, float]]] = {
            "BTC": deque(),
            "ETH": deque(),
        }
        self._last_update: dict[str, Optional[datetime]] = {
            "BTC": None,
            "ETH": None,
        }
        self._running = False

    def get_move_pct(
        self,
        asset: str,
        lookback_seconds: float = SIGNAL_LOOKBACK_SECONDS,
    ) -> float:
        """Return price change as a fraction over the lookback window (positive = up).

        Returns 0.0 if fewer than two data points exist in the window.
        """
        history = self._history.get(asset)
        if not history:
            return 0.0
        now    = datetime.now(UTC)
        cutoff = now.timestamp() - lookback_seconds
        window = [price for ts, price in history if ts.timestamp() >= cutoff]
        if len(window) < 2:
            return 0.0
        return (window[-1] - window[0]) / window[0]

    def get_direction(
        self,
        asset: str,
        lookback_seconds: float = SIGNAL_LOOKBACK_SECONDS,
    ) -> Optional[str]:
        """Return 'Up', 'Down', or None based on price movement over the lookback window.

        Compares the oldest price in the lookback window to the most recent price.
        Returns None if fewer than two data points exist in the window.
        """
        history = self._history.get(asset)
        if not history:
            return None

        now = datetime.now(UTC)
        cutoff = now.timestamp() - lookback_seconds

        window = [(ts, price) for ts, price in history if ts.timestamp() >= cutoff]
        if len(window) < 2:
            return None

        oldest_price = window[0][1]
        newest_price = window[-1][1]

        if newest_price > oldest_price:
            return "Up"
        if newest_price < oldest_price:
            return "Down"
        return None

    def get_price(self, asset: str) -> Optional[float]:
        """Return the most recent price for the asset, or None if stale/unavailable."""
        last_update = self._last_update.get(asset)
        if last_update is None:
            return None

        age = (datetime.now(UTC) - last_update).total_seconds()
        if age > MAX_STALE_SECONDS:
            return None

        history = self._history.get(asset)
        if not history:
            return None
        return history[-1][1]

    def _record_price(self, asset: str, price: float, ts: datetime) -> None:
        """Append a price point and evict entries older than HISTORY_RETENTION_SECONDS."""
        history = self._history[asset]
        history.append((ts, price))

        cutoff = ts.timestamp() - HISTORY_RETENTION_SECONDS
        while history and history[0][0].timestamp() < cutoff:
            history.popleft()

        self._last_update[asset] = ts

    def _process_message(self, raw: str) -> None:
        """Parse a Binance combined stream message and record the trade price."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        stream = msg.get("stream", "")
        asset = _STREAM_TO_ASSET.get(stream)
        if asset is None:
            return

        data = msg.get("data", {})
        try:
            price = float(data["p"])
            # Binance trade timestamps are in milliseconds
            ts = datetime.fromtimestamp(data["T"] / 1000, tz=UTC)
        except (KeyError, TypeError, ValueError):
            return

        self._record_price(asset, price, ts)

    async def run(self) -> None:
        """Connect to Binance WS and stream trades indefinitely."""
        self._running = True
        backoff = 3
        while self._running:
            try:
                async with websockets.connect(
                    _BINANCE_WS_URL,
                    open_timeout=15,
                    close_timeout=5,
                ) as ws:
                    logger.info("Binance price feed connected")
                    backoff = 3
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        self._process_message(raw_msg)
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("Binance WS error: %s — reconnecting in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    feed = BinancePriceFeed()

    async def _main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: (setattr(feed, "_running", False), sys.exit(0)))
        await feed.run()

    asyncio.run(_main())
