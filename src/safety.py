"""Safety checks for live trading.

Wraps all pre-flight checks behind one method (`can_place_order`) that returns
(allowed, reason). Uses the trades CSV to compute today's PnL and recent rate.
"""

import csv
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

    def check_daily_loss_limit(self) -> tuple[bool, float]:
        """Compute today's live PnL from CSV. Returns (within_limit, todays_pnl)."""
        today = datetime.now(UTC).strftime("%Y%m%d")
        csv_path = Path(config.DATA_DIR) / f"paper_trades_{today}.csv"

        if not csv_path.exists():
            return (True, 0.0)

        todays_pnl = 0.0
        try:
            with csv_path.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pnl_str = row.get("live_pnl_usdc", "")
                    if not pnl_str:
                        continue
                    try:
                        todays_pnl += float(pnl_str)
                    except (ValueError, TypeError):
                        continue
        except Exception as exc:
            logger.warning("Could not read today's CSV for loss limit check: %s", exc)
            return (True, 0.0)

        within = todays_pnl > -config.LIVE_DAILY_LOSS_LIMIT_USDC
        return (within, todays_pnl)

    def record_order(self) -> None:
        """Call after a live order is placed (for rate limiting)."""
        self._recent_order_timestamps.append(datetime.now(UTC))

    def can_place_order(
        self,
        balance_usdc: float,
        open_positions: int,
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

        within_limit, todays_pnl = self.check_daily_loss_limit()
        if not within_limit:
            return (
                False,
                f"daily_loss_limit (PnL=${todays_pnl:.2f} <= -${config.LIVE_DAILY_LOSS_LIMIT_USDC:.2f})",
            )

        return (True, None)
