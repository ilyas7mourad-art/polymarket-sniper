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
        # Worst-case open exposure from orders placed but not yet resolved in the CSV.
        # Prevents concurrent signals from both slipping past the daily loss cap before
        # either appears in the CSV (the breach mode that occurred on 2026-05-14).
        self._pending_stake: float = 0.0

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

    def check_daily_loss_at_startup(self) -> tuple[bool, float]:
        """Read today's CSV PnL on bot startup and block trading if limit already hit.

        Call this once at the top of run() before any order loops. Prevents a
        restart from resetting the in-memory loss counter while the CSV already
        shows losses beyond the daily cap.
        """
        within, todays_pnl = self.check_daily_loss_limit(stake_usdc=0.0)
        if not within:
            logger.error(
                "Daily loss limit already hit at startup (CSV PnL=%.2f <= -%.2f). "
                "Trading blocked for the rest of the day.",
                todays_pnl,
                config.LIVE_DAILY_LOSS_LIMIT_USDC,
            )
        else:
            logger.info("Startup loss check: CSV PnL=%.2f (limit=-%.2f)", todays_pnl, config.LIVE_DAILY_LOSS_LIMIT_USDC)
        return (within, todays_pnl)

    def check_daily_loss_limit(self, stake_usdc: float) -> tuple[bool, float]:
        """Compute today's live PnL from CSV. Returns (within_limit, todays_pnl).

        Includes _pending_stake (orders placed but not yet resolved/in-CSV) plus
        stake_usdc (this new order) in the worst-case check, so two concurrent
        signals can't both slip past the cap before either appears in the CSV.
        """
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

        worst_case = todays_pnl - self._pending_stake - stake_usdc
        within = worst_case > -config.LIVE_DAILY_LOSS_LIMIT_USDC
        return (within, todays_pnl)

    def record_order(self, stake_usdc: float) -> None:
        """Call after a live order is dispatched (rate limiting + pending stake tracking)."""
        self._recent_order_timestamps.append(datetime.now(UTC))
        self._pending_stake += stake_usdc

    def record_resolution(self, stake_usdc: float) -> None:
        """Call when a live order's outcome is known (filled+resolved, rejected, or error).

        Releases the pending stake so future cap checks don't double-count it.
        """
        self._pending_stake = max(0.0, self._pending_stake - stake_usdc)

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

        within_limit, todays_pnl = self.check_daily_loss_limit(stake_usdc)
        if not within_limit:
            return (
                False,
                f"daily_loss_limit (PnL=${todays_pnl:.2f} <= -${config.LIVE_DAILY_LOSS_LIMIT_USDC:.2f})",
            )

        return (True, None)
