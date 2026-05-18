"""Safety checks for live trading.

Wraps all pre-flight checks behind one method (`can_place_order`) that returns
(allowed, reason). Uses the live wallet balance delta vs the start-of-day balance
to enforce the daily loss cap — no CSV reading, no formula errors.
"""

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.config import config

logger = logging.getLogger(__name__)

UTC = timezone.utc


class SafetyChecker:
    """Pre-flight and ongoing safety checks for live trading."""

    def __init__(self) -> None:
        self._kill_switch_path = Path(config.KILL_SWITCH_PATH)
        self._recent_order_timestamps: deque[datetime] = deque(maxlen=500)
        self._start_balance: Optional[float] = None

    def set_start_balance(self, balance: float) -> None:
        """Record the wallet balance at the start of the day.

        Call once at startup. The daily loss cap is enforced as:
            start_balance - current_balance >= LIVE_DAILY_LOSS_LIMIT_USDC → block.
        Persisted externally (start_balance_YYYYMMDD.txt) so restarts within the
        same day reuse the original morning balance.
        """
        self._start_balance = balance
        logger.info(
            "Daily loss cap anchored: start_balance=$%.4f, limit=$%.2f",
            balance, config.LIVE_DAILY_LOSS_LIMIT_USDC,
        )

    def check_kill_switch(self) -> bool:
        """Return True if kill switch is active (file exists)."""
        return self._kill_switch_path.exists()

    def check_balance_sufficient(self, current_balance_usdc: float) -> bool:
        """Return True if balance >= LIVE_MIN_BALANCE_USDC."""
        return current_balance_usdc >= config.LIVE_MIN_BALANCE_USDC

    def check_position_count(self, open_positions: int) -> bool:
        """Return True if open positions < LIVE_MAX_OPEN_POSITIONS."""
        return open_positions < config.LIVE_MAX_OPEN_POSITIONS

    def check_rate_limit(self) -> bool:
        """Return True if recent order rate is under LIVE_MAX_ORDERS_PER_HOUR."""
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        while self._recent_order_timestamps and self._recent_order_timestamps[0] < cutoff:
            self._recent_order_timestamps.popleft()
        return len(self._recent_order_timestamps) < config.LIVE_MAX_ORDERS_PER_HOUR

    def check_balance_loss(self, current_balance: float) -> tuple[bool, float]:
        """Return (within_limit, drop) where drop = start_balance - current_balance.

        Returns (True, 0.0) if start_balance has not been set yet (e.g. paper mode).
        """
        if self._start_balance is None:
            return (True, 0.0)
        drop = self._start_balance - current_balance
        within = drop < config.LIVE_DAILY_LOSS_LIMIT_USDC
        return (within, drop)

    def record_order(self) -> None:
        """Call after a live order is dispatched (tracks rate limiting)."""
        self._recent_order_timestamps.append(datetime.now(UTC))

    def can_place_order(
        self,
        balance_usdc: float,
        open_positions: int,
        stake_usdc: float = 0.0,
    ) -> tuple[bool, Optional[str]]:
        """Composite safety check.

        Returns:
            (allowed, reason_if_blocked).
        """
        if self.check_kill_switch():
            return (False, "kill_switch_active")

        if not self.check_balance_sufficient(balance_usdc):
            return (
                False,
                f"balance_below_min (${balance_usdc:.2f} < ${config.LIVE_MIN_BALANCE_USDC:.2f})",
            )

        if not self.check_position_count(open_positions):
            return (
                False,
                f"max_positions_reached ({open_positions} >= {config.LIVE_MAX_OPEN_POSITIONS})",
            )

        if not self.check_rate_limit():
            return (False, f"rate_limit ({len(self._recent_order_timestamps)} orders in last hour)")

        within_limit, drop = self.check_balance_loss(balance_usdc)
        if not within_limit:
            return (
                False,
                f"daily_loss_limit (balance dropped ${drop:.2f} >= limit ${config.LIVE_DAILY_LOSS_LIMIT_USDC:.2f})",
            )

        return (True, None)
