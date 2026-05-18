"""Unit tests for src/safety.py."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.safety import SafetyChecker

UTC = timezone.utc


def _patch_config(tmp_path: Path, **overrides):
    """Patch src.safety.config with sane defaults for tests."""
    defaults = {
        "KILL_SWITCH_PATH": str(tmp_path / "killswitch_does_not_exist"),
        "LIVE_MIN_BALANCE_USDC": 1.0,
        "LIVE_MAX_OPEN_POSITIONS": 5,
        "LIVE_DAILY_LOSS_LIMIT_USDC": 5.0,
        "LIVE_MAX_ORDERS_PER_HOUR": 30,
    }
    defaults.update(overrides)
    mock_config = type("MockConfig", (), defaults)
    return patch("src.safety.config", mock_config)


def test_kill_switch_detected(tmp_path: Path) -> None:
    kill_path = tmp_path / "kill"
    kill_path.touch()
    with _patch_config(tmp_path, KILL_SWITCH_PATH=str(kill_path)):
        checker = SafetyChecker()
        assert checker.check_kill_switch() is True


def test_kill_switch_not_active(tmp_path: Path) -> None:
    with _patch_config(tmp_path):
        checker = SafetyChecker()
        assert checker.check_kill_switch() is False


def test_balance_sufficient(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_MIN_BALANCE_USDC=1.0):
        checker = SafetyChecker()
        assert checker.check_balance_sufficient(0.5) is False
        assert checker.check_balance_sufficient(1.0) is True
        assert checker.check_balance_sufficient(10.0) is True


def test_position_count(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_MAX_OPEN_POSITIONS=5):
        checker = SafetyChecker()
        assert checker.check_position_count(0) is True
        assert checker.check_position_count(4) is True
        assert checker.check_position_count(5) is False
        assert checker.check_position_count(10) is False


def test_rate_limit_under_threshold(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_MAX_ORDERS_PER_HOUR=30):
        checker = SafetyChecker()
        for _ in range(20):
            checker.record_order()
        assert checker.check_rate_limit() is True


def test_rate_limit_at_threshold(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_MAX_ORDERS_PER_HOUR=30):
        checker = SafetyChecker()
        for _ in range(30):
            checker.record_order()
        assert checker.check_rate_limit() is False


def test_rate_limit_old_entries_dropped(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_MAX_ORDERS_PER_HOUR=30):
        checker = SafetyChecker()
        old_ts = datetime.now(UTC) - timedelta(hours=2)
        for _ in range(25):
            checker._recent_order_timestamps.append(old_ts)
        for _ in range(5):
            checker.record_order()
        assert checker.check_rate_limit() is True


def test_balance_loss_no_start_balance(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_DAILY_LOSS_LIMIT_USDC=5.0):
        checker = SafetyChecker()
        within, drop = checker.check_balance_loss(100.0)
        assert within is True
        assert drop == 0.0


def test_balance_loss_within_limit(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_DAILY_LOSS_LIMIT_USDC=5.0):
        checker = SafetyChecker()
        checker.set_start_balance(100.0)
        within, drop = checker.check_balance_loss(96.0)
        assert within is True
        assert drop == pytest.approx(4.0)


def test_balance_loss_at_limit_blocks(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_DAILY_LOSS_LIMIT_USDC=5.0):
        checker = SafetyChecker()
        checker.set_start_balance(100.0)
        within, drop = checker.check_balance_loss(95.0)
        assert within is False
        assert drop == pytest.approx(5.0)


def test_balance_loss_exceeds_limit(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_DAILY_LOSS_LIMIT_USDC=5.0):
        checker = SafetyChecker()
        checker.set_start_balance(100.0)
        within, drop = checker.check_balance_loss(90.0)
        assert within is False
        assert drop == pytest.approx(10.0)


def test_can_place_order_blocks_on_kill_switch(tmp_path: Path) -> None:
    kill_path = tmp_path / "kill"
    kill_path.touch()
    with _patch_config(tmp_path, KILL_SWITCH_PATH=str(kill_path)):
        checker = SafetyChecker()
        allowed, reason = checker.can_place_order(balance_usdc=10.0, open_positions=0, stake_usdc=5.0)
        assert allowed is False
        assert "kill_switch" in reason


def test_can_place_order_blocks_on_balance(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_MIN_BALANCE_USDC=5.0):
        checker = SafetyChecker()
        allowed, reason = checker.can_place_order(balance_usdc=2.0, open_positions=0, stake_usdc=5.0)
        assert allowed is False
        assert "balance" in reason


def test_can_place_order_blocks_on_positions(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_MAX_OPEN_POSITIONS=3):
        checker = SafetyChecker()
        allowed, reason = checker.can_place_order(balance_usdc=10.0, open_positions=3, stake_usdc=5.0)
        assert allowed is False
        assert "positions" in reason


def test_can_place_order_blocks_on_loss_limit(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_DAILY_LOSS_LIMIT_USDC=5.0):
        checker = SafetyChecker()
        checker.set_start_balance(100.0)
        # Balance dropped $10 >= limit of $5 → block
        allowed, reason = checker.can_place_order(balance_usdc=90.0, open_positions=0, stake_usdc=1.0)
        assert allowed is False
        assert "daily_loss" in reason


def test_can_place_order_allows_normal_state(tmp_path: Path) -> None:
    with _patch_config(tmp_path):
        checker = SafetyChecker()
        # No start balance set → balance loss check always passes
        allowed, reason = checker.can_place_order(balance_usdc=10.0, open_positions=0, stake_usdc=5.0)
        assert allowed is True
        assert reason is None


def test_can_place_order_allows_with_start_balance_within_limit(tmp_path: Path) -> None:
    with _patch_config(tmp_path, LIVE_DAILY_LOSS_LIMIT_USDC=40.0):
        checker = SafetyChecker()
        checker.set_start_balance(126.0)
        # Balance at 116 → dropped $10 < $40 limit
        allowed, reason = checker.can_place_order(balance_usdc=116.0, open_positions=0, stake_usdc=5.0)
        assert allowed is True
        assert reason is None
