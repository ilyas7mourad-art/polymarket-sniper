"""Polls Polymarket data API for Bonereaper's trades and logs to daily CSV."""

import asyncio
import csv
import logging
import signal as os_signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from src.config import config

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Bonereaper's proxy wallet (Gnosis Safe).
BONEREAPER_PROXY = "0x3F9ffDC79719B8B47543461C197c2689FF5051b0"

# Polymarket's data API endpoint for trades by wallet.
_TRADES_URL = "https://data-api.polymarket.com/trades"

# How often to poll (seconds). Polymarket's API does not appear to rate-limit
# at 5s intervals for this endpoint.
POLL_INTERVAL_SECONDS = 5

# How many trades to fetch per poll. We only care about new ones since last
# poll, but we fetch enough to handle bursts.
TRADES_PER_POLL = 50

_CSV_HEADER = [
    "timestamp_utc",
    "transaction_hash",
    "side",
    "outcome",
    "price",
    "size",
    "usdc_size",
    "condition_id",
    "asset",
    "market_slug",
    "market_title",
]


class BonereaperTracker:
    """Polls Polymarket data API and appends new Bonereaper trades to daily CSV."""

    def __init__(self, wallet: str = BONEREAPER_PROXY) -> None:
        self.wallet = wallet
        # Set of transaction_hashes we've already written, to avoid dupes
        self._seen_tx: set[str] = set()
        self._running = False
        self._total_logged = 0

    async def run(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        # On startup, prime _seen_tx from today's CSV so we don't re-log past trades
        self._prime_seen_from_csv()

        await asyncio.gather(
            self._poll_loop(),
            self._heartbeat_loop(),
        )

    def _prime_seen_from_csv(self) -> None:
        """Read today's CSV (if exists) to populate _seen_tx, preventing dupes after restart."""
        today = datetime.now(UTC).strftime("%Y%m%d")
        path = Path(config.DATA_DIR) / f"bonereaper_trades_{today}.csv"
        if not path.exists():
            return
        try:
            with path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tx = row.get("transaction_hash", "")
                    if tx:
                        self._seen_tx.add(tx)
            logger.info("Primed seen-set with %d existing trades from today", len(self._seen_tx))
        except Exception as exc:
            logger.warning("Could not prime from CSV: %s", exc)

    async def _poll_loop(self) -> None:
        """Poll API every POLL_INTERVAL_SECONDS and log new trades."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                try:
                    new_trades = await self._fetch_new_trades(client)
                    if new_trades:
                        self._append_trades(new_trades)
                        self._total_logged += len(new_trades)
                        for t in new_trades:
                            logger.info(
                                "TRADE %s @ %.4f size=%.2f (%s) — %s",
                                t["side"], t["price"], t["size"],
                                t["outcome"], t["market_slug"],
                            )
                except Exception as exc:
                    logger.warning("Poll error: %s", exc)

                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _fetch_new_trades(self, client: httpx.AsyncClient) -> list[dict]:
        """Hit the API and return only trades we haven't seen yet."""
        params = {"user": self.wallet, "limit": TRADES_PER_POLL}
        resp = await client.get(_TRADES_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            logger.warning("Unexpected response shape: %s", type(data).__name__)
            return []

        new_trades = []
        for raw in data:
            tx = raw.get("transactionHash")
            if not tx or tx in self._seen_tx:
                continue
            self._seen_tx.add(tx)

            # Compute USDC notional size = shares × price
            try:
                price = float(raw.get("price", 0))
                size = float(raw.get("size", 0))
            except (TypeError, ValueError):
                continue

            usdc_size = price * size

            new_trades.append({
                "timestamp_utc": datetime.fromtimestamp(
                    raw.get("timestamp", 0), tz=UTC
                ).isoformat(timespec="seconds"),
                "transaction_hash": tx,
                "side": raw.get("side", ""),
                "outcome": raw.get("outcome", ""),
                "price": price,
                "size": size,
                "usdc_size": usdc_size,
                "condition_id": raw.get("conditionId", ""),
                "asset": raw.get("asset", ""),
                "market_slug": raw.get("slug", ""),
                "market_title": raw.get("title", ""),
            })

        # Reverse so we write oldest-first within the batch
        return list(reversed(new_trades))

    def _append_trades(self, trades: list[dict]) -> None:
        """Append trades to today's daily CSV."""
        today = datetime.now(UTC).strftime("%Y%m%d")
        path = Path(config.DATA_DIR) / f"bonereaper_trades_{today}.csv"
        is_new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
            if is_new:
                writer.writeheader()
            for t in trades:
                writer.writerow(t)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            logger.info(
                "Bonereaper tracker: %d total trades logged, %d unique tx tracked",
                self._total_logged, len(self._seen_tx),
            )

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        self._running = False
        sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    tracker = BonereaperTracker()
    asyncio.run(tracker.run())
