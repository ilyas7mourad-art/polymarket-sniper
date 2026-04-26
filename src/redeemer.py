"""Redemption loop: redeems winning positions on resolved markets."""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from py_clob_client.client import ClobClient

from src.config import config

logger = logging.getLogger(__name__)

UTC = timezone.utc

REDEMPTION_INTERVAL_SECONDS = 300  # 5 minutes


class Redeemer:
    """Periodically redeems winning positions for resolved markets."""

    def __init__(self) -> None:
        self._client = ClobClient(
            host=config.CLOB_HOST,
            chain_id=config.CHAIN_ID,
            key=config.WALLET_PRIVATE_KEY,
            signature_type=2,
            funder=config.WALLET_FUNDER,
        )
        self._redeemed_conditions: set[str] = set()
        self._running = False
        self._state_path = Path(config.DATA_DIR) / "redemption_state.txt"
        self._load_state()

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with self._state_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._redeemed_conditions.add(line)
            logger.info("Loaded %d previously-redeemed conditions", len(self._redeemed_conditions))
        except Exception as exc:
            logger.warning("Could not load redemption state: %s", exc)

    def _save_state(self) -> None:
        try:
            with self._state_path.open("w") as f:
                for cid in sorted(self._redeemed_conditions):
                    f.write(f"{cid}\n")
        except Exception as exc:
            logger.warning("Could not save redemption state: %s", exc)

    async def run(self) -> None:
        """Main loop: every REDEMPTION_INTERVAL_SECONDS, check and redeem."""
        self._running = True
        await asyncio.sleep(10)  # let paper_trader start up first
        while self._running:
            try:
                await self._check_and_redeem()
            except Exception as exc:
                logger.warning("Redemption loop error: %s", exc)
            await asyncio.sleep(REDEMPTION_INTERVAL_SECONDS)

    async def _check_and_redeem(self) -> None:
        """Find resolved markets where we have winning tokens and redeem them."""
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get(
                    "https://data-api.polymarket.com/positions",
                    params={"user": config.WALLET_ADDRESS, "limit": 100},
                )
                resp.raise_for_status()
                positions = resp.json()
            except Exception as exc:
                logger.warning("Position fetch failed: %s", exc)
                return

        if not isinstance(positions, list):
            return

        to_redeem = []
        for pos in positions:
            cid = pos.get("conditionId", "")
            if not cid or cid in self._redeemed_conditions:
                continue
            redeemable = pos.get("redeemable", False)
            current_value = float(pos.get("currentValue", 0))
            if redeemable and current_value > 0:
                to_redeem.append((cid, current_value))

        if not to_redeem:
            return

        logger.info(
            "Found %d positions to redeem (total $%.2f)",
            len(to_redeem),
            sum(v for _, v in to_redeem),
        )

        for cid, value in to_redeem:
            try:
                await asyncio.to_thread(self._redeem_one, cid)
                self._redeemed_conditions.add(cid)
                self._save_state()
                logger.info("REDEEMED %s for ~$%.2f", cid, value)
            except Exception as exc:
                logger.warning("Redemption failed for %s: %s", cid, exc)

    def _redeem_one(self, condition_id: str) -> None:
        """Synchronous redemption call — runs in thread executor.

        py-clob-client (as of 0.20) does not expose redemption directly.
        If a future SDK version adds it, the hasattr checks below will pick it
        up automatically.  Until then, a NotImplementedError is raised so the
        caller can log a warning and fall back to manual UI redemption.
        """
        if hasattr(self._client, "redeem_positions"):
            self._client.redeem_positions(condition_id)
        elif hasattr(self._client, "redeem"):
            self._client.redeem(condition_id)
        else:
            raise NotImplementedError(
                "py-clob-client does not expose redemption directly. "
                "Need web3.py implementation calling CTF contract redeemPositions(). "
                "Manual redemption required via Polymarket UI for now."
            )

    def stop(self) -> None:
        self._running = False
