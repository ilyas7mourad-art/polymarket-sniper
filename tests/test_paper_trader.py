"""Unit tests for src/paper_trader.py — all offline."""

import asyncio
import csv
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.paper_trader import (
    PaperTrade,
    PaperTrader,
    SIGNAL_RULES,
    STAKE_USDC,
    _CSV_HEADER,
)
from src.scanner import Market

UTC = timezone.utc


def _make_market(end_offset_s: float = 60.0, condition_id: str = "0xabc") -> Market:
    """Build a Market for tests."""
    now = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)
    return Market(
        condition_id=condition_id,
        question="Bitcoin Up or Down - April 20, 8:00AM-8:05AM ET",
        asset="BTC",
        start_time=now + timedelta(seconds=end_offset_s - 300),
        end_time=now + timedelta(seconds=end_offset_s),
        up_token_id="token_up_" + condition_id,
        down_token_id="token_down_" + condition_id,
        slug="btc-updown-5m-test",
        raw={},
    )


def test_signal_rules_cover_two_validated_buckets() -> None:
    assert len(SIGNAL_RULES) == 2
    labels = [r[4] for r in SIGNAL_RULES]
    assert "T=60s_0.95-1.00" in labels
    assert "T=270s_0.70-0.85" in labels
    # These were dropped (negative realistic EV)
    assert "T=10s_0.95-1.00" not in labels
    assert "T=60s_0.90-0.95" not in labels


def test_stake_usdc_is_one_dollar() -> None:
    assert STAKE_USDC == 1.0


def test_evaluate_signals_fires_in_time_and_price_window() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=60.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    # Build a book with ask at 0.97 (in 0.95-1.00 bucket)
    trader._book_asks[market.up_token_id] = {0.97: 100.0}

    now = market.end_time - timedelta(seconds=60)  # exactly T-60s
    trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 1
    trade = trader._open_trades[0]
    assert trade.entry_price == 0.97
    assert trade.signal_bucket_label == "T=60s_0.95-1.00"


def test_evaluate_signals_respects_deduplication() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=60.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.97: 100.0}

    now = market.end_time - timedelta(seconds=60)
    trader._evaluate_signals(market.up_token_id, now)
    trader._evaluate_signals(market.up_token_id, now)  # second call should be ignored

    assert len(trader._open_trades) == 1


def test_evaluate_signals_no_fire_outside_time_window() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=120.0)  # T-120s
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.97: 100.0}

    now = market.end_time - timedelta(seconds=120)
    trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 0


def test_evaluate_signals_no_fire_outside_price_bucket() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=60.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.85: 100.0}  # below 0.90 minimum

    now = market.end_time - timedelta(seconds=60)
    trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 0


def test_fire_entry_computes_fee_correctly() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=60.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.99: 100.0}

    now = market.end_time - timedelta(seconds=60)
    trader._evaluate_signals(market.up_token_id, now)

    trade = trader._open_trades[0]
    # stake=$1, shares = 1/0.99 ≈ 1.0101
    # fee = shares × 0.072 × 0.99 × 0.01 = 1.0101 × 0.0007128 ≈ 0.00072 USDC
    expected_fee = (1.0 / 0.99) * 0.072 * 0.99 * 0.01
    assert abs(trade.fee_usdc - expected_fee) < 1e-4
    assert abs(trade.simulated_shares - 1.0 / 0.99) < 1e-6


def test_resolve_open_trades_marks_win_correctly() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=-60.0)
    # Pass now = end_time + 120s so secs_past_end = 120 (> 30 threshold), clock-independent
    resolve_now = market.end_time + timedelta(seconds=120)
    trade = PaperTrade(
        trade_id="20260420_000001",
        entry_timestamp_utc=market.end_time - timedelta(seconds=60),
        market=market,
        side="Up",
        signal_bucket_label="T=60s_0.95-1.00",
        signal_target_time_s=60.0,
        seconds_to_resolution_at_entry=60.0,
        entry_price=0.97,
        simulated_shares=1.0 / 0.97,
        simulated_stake_usdc=1.0,
        fee_usdc=0.002,
    )
    trader._open_trades.append(trade)

    with patch.object(trader, "_fetch_winner_from_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "Up"
        asyncio.run(trader._resolve_open_trades(now=resolve_now))

    assert trader._total_wins == 1
    assert trader._total_losses == 0
    assert trade.winner == "Up"
    assert trade.payout_usdc is not None and trade.payout_usdc > 1.0
    assert trade.pnl_usdc is not None and trade.pnl_usdc > 0


def test_resolve_open_trades_marks_loss_correctly() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=-60.0)
    resolve_now = market.end_time + timedelta(seconds=120)
    trade = PaperTrade(
        trade_id="20260420_000002",
        entry_timestamp_utc=market.end_time - timedelta(seconds=60),
        market=market,
        side="Up",
        signal_bucket_label="T=60s_0.95-1.00",
        signal_target_time_s=60.0,
        seconds_to_resolution_at_entry=60.0,
        entry_price=0.97,
        simulated_shares=1.0 / 0.97,
        simulated_stake_usdc=1.0,
        fee_usdc=0.002,
    )
    trader._open_trades.append(trade)

    with patch.object(trader, "_fetch_winner_from_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "Down"
        asyncio.run(trader._resolve_open_trades(now=resolve_now))

    assert trader._total_losses == 1
    assert trader._total_wins == 0
    assert trade.winner == "Down"
    assert trade.payout_usdc == 0.0
    assert trade.pnl_usdc is not None and trade.pnl_usdc < 0


def test_resolve_open_trades_stays_open_when_api_returns_none() -> None:
    """Market not yet resolved on API — trade stays open if within 5-minute timeout."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=-60.0)
    # 60s past end is within the 300s timeout
    resolve_now = market.end_time + timedelta(seconds=60)
    trade = PaperTrade(
        trade_id="20260420_000003",
        entry_timestamp_utc=market.end_time - timedelta(seconds=60),
        market=market,
        side="Up",
        signal_bucket_label="T=60s_0.95-1.00",
        signal_target_time_s=60.0,
        seconds_to_resolution_at_entry=60.0,
        entry_price=0.97,
        simulated_shares=1.0 / 0.97,
        simulated_stake_usdc=1.0,
        fee_usdc=0.002,
    )
    trader._open_trades.append(trade)

    with patch.object(trader, "_fetch_winner_from_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = None  # API says not resolved yet
        asyncio.run(trader._resolve_open_trades(now=resolve_now))

    assert len(trader._open_trades) == 1  # still open
    assert trader._total_wins == 0
    assert trader._total_losses == 0


def test_resolve_open_trades_times_out_as_unknown_after_30min() -> None:
    """Market hasn't resolved on API after 30 minutes → mark unknown and give up."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=-2000.0)
    # 2000s past end exceeds the 1800s (30-min) timeout
    resolve_now = market.end_time + timedelta(seconds=2000)
    trade = PaperTrade(
        trade_id="20260420_000004",
        entry_timestamp_utc=market.end_time - timedelta(seconds=60),
        market=market,
        side="Up",
        signal_bucket_label="T=60s_0.95-1.00",
        signal_target_time_s=60.0,
        seconds_to_resolution_at_entry=60.0,
        entry_price=0.97,
        simulated_shares=1.0 / 0.97,
        simulated_stake_usdc=1.0,
        fee_usdc=0.002,
    )
    trader._open_trades.append(trade)

    with patch.object(trader, "_fetch_winner_from_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = None
        asyncio.run(trader._resolve_open_trades(now=resolve_now))

    assert len(trader._open_trades) == 0
    assert trade.winner == "unknown"
    assert trade.pnl_usdc is not None and trade.pnl_usdc < 0


# ---------------------------------------------------------------------------
# Resolution timeout / sweep constants
# ---------------------------------------------------------------------------


def test_resolution_timeout_constant_is_30_min() -> None:
    from src.paper_trader import RESOLUTION_TIMEOUT_SECONDS
    assert RESOLUTION_TIMEOUT_SECONDS == 1800


def test_sweep_constants() -> None:
    from src.paper_trader import SWEEP_INTERVAL_SECONDS, SWEEP_MAX_AGE_SECONDS
    assert SWEEP_INTERVAL_SECONDS == 300
    assert SWEEP_MAX_AGE_SECONDS == 6 * 3600


# ---------------------------------------------------------------------------
# _sweep_unknowns — recovers late resolution
# ---------------------------------------------------------------------------


def test_sweep_unknowns_recovers_late_resolution(tmp_path) -> None:
    """Sweep finds an unknown row, queries API, and corrects it in the CSV."""
    from src.paper_trader import PaperTrader, _CSV_HEADER

    now = datetime.now(UTC)
    entry_ts = now - timedelta(minutes=30)
    csv_path = tmp_path / f"paper_trades_{now.strftime('%Y%m%d')}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        writer.writerow([
            "20260420_000001",
            entry_ts.isoformat(timespec="milliseconds"),
            "btc-updown-5m-test",
            "BTC", "0xabc", "Down",
            "T=10s_0.95-1.00", "10", "10.0",
            "0.9500", "1.052632", "1.0000", "0.00360",
            (entry_ts + timedelta(minutes=5)).isoformat(timespec="milliseconds"),
            "unknown", "0.0000", "-1.0036",
        ])

    trader = PaperTrader()
    # Simulate prior phantom-loss accounting
    trader._total_losses = 1
    trader._total_pnl_usdc = -1.0036

    with patch("src.paper_trader.config") as mock_config, \
         patch.object(trader, "_fetch_winner_from_api", new_callable=AsyncMock) as mock_api:
        mock_config.DATA_DIR = str(tmp_path)
        mock_api.return_value = "Down"  # Down wins — our Down bet is a win
        asyncio.run(trader._sweep_unknowns())

    with csv_path.open("r") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["winner"] == "Down"
    assert float(rows[0]["payout_usdc"]) == pytest.approx(1.052632, abs=1e-4)
    assert float(rows[0]["pnl_usdc"]) > 0

    # Phantom loss removed, real win added
    assert trader._total_wins == 1
    assert trader._total_losses == 0
    assert trader._total_pnl_usdc > 0


def test_sweep_unknowns_skips_old_trades(tmp_path) -> None:
    """Sweep ignores trades older than SWEEP_MAX_AGE_SECONDS."""
    from src.paper_trader import PaperTrader, _CSV_HEADER, SWEEP_MAX_AGE_SECONDS

    now = datetime.now(UTC)
    # Entry from 7 hours ago — past the 6-hour cutoff
    entry_ts = now - timedelta(seconds=SWEEP_MAX_AGE_SECONDS + 3600)
    csv_path = tmp_path / f"paper_trades_{now.strftime('%Y%m%d')}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        writer.writerow([
            "20260420_000099",
            entry_ts.isoformat(timespec="milliseconds"),
            "btc-updown-5m-old",
            "BTC", "0xold", "Up",
            "T=10s_0.95-1.00", "10", "10.0",
            "0.9500", "1.052632", "1.0000", "0.00360",
            (entry_ts + timedelta(minutes=5)).isoformat(timespec="milliseconds"),
            "unknown", "0.0000", "-1.0036",
        ])

    trader = PaperTrader()

    with patch("src.paper_trader.config") as mock_config, \
         patch.object(trader, "_fetch_winner_from_api", new_callable=AsyncMock) as mock_api:
        mock_config.DATA_DIR = str(tmp_path)
        mock_api.return_value = "Up"
        asyncio.run(trader._sweep_unknowns())

    mock_api.assert_not_called()


def test_sweep_unknowns_skips_already_resolved(tmp_path) -> None:
    """Sweep ignores rows where winner is already Up or Down."""
    from src.paper_trader import PaperTrader, _CSV_HEADER

    now = datetime.now(UTC)
    entry_ts = now - timedelta(minutes=10)
    csv_path = tmp_path / f"paper_trades_{now.strftime('%Y%m%d')}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        writer.writerow([
            "20260420_000050",
            entry_ts.isoformat(timespec="milliseconds"),
            "btc-updown-5m-resolved",
            "BTC", "0xres", "Down",
            "T=10s_0.95-1.00", "10", "10.0",
            "0.9500", "1.052632", "1.0000", "0.00360",
            (entry_ts + timedelta(minutes=5)).isoformat(timespec="milliseconds"),
            "Down", "1.0526", "0.0490",
        ])

    trader = PaperTrader()

    with patch("src.paper_trader.config") as mock_config, \
         patch.object(trader, "_fetch_winner_from_api", new_callable=AsyncMock) as mock_api:
        mock_config.DATA_DIR = str(tmp_path)
        asyncio.run(trader._sweep_unknowns())

    mock_api.assert_not_called()


# ---------------------------------------------------------------------------
# T=270s bucket tests
# ---------------------------------------------------------------------------


def test_t270_bucket_fires_at_mid_price() -> None:
    """At T-270s with best_ask=0.75, the T=270s_0.70-0.85 rule should fire."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=270.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.75: 100.0}

    with patch.object(trader._price_feed, "get_direction", return_value="Up"):
        now = market.end_time - timedelta(seconds=270)
        trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 1
    trade = trader._open_trades[0]
    assert trade.entry_price == 0.75
    assert trade.signal_bucket_label == "T=270s_0.70-0.85"


def test_t270_bucket_respects_tolerance_boundary() -> None:
    """At T-241s (within 30s tolerance of 270s) the rule should fire."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=241.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.78: 100.0}

    with patch.object(trader._price_feed, "get_direction", return_value="Up"):
        now = market.end_time - timedelta(seconds=241)
        trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 1


def test_t270_bucket_does_not_fire_outside_tolerance() -> None:
    """At T-200s (outside 30s tolerance of 270s) the rule should NOT fire."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=200.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.78: 100.0}

    now = market.end_time - timedelta(seconds=200)
    trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 0


# ---------------------------------------------------------------------------
# Binance momentum filter tests
# ---------------------------------------------------------------------------


def test_t270_fires_when_binance_direction_matches() -> None:
    """T=270s entry fires when Binance direction agrees with the side."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=270.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.75: 100.0}

    with patch.object(trader._price_feed, "get_direction", return_value="Up"):
        now = market.end_time - timedelta(seconds=270)
        trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 1
    assert trader._binance_filtered_skips == 0


def test_t270_skips_when_binance_direction_is_none() -> None:
    """T=270s entry is skipped when Binance has no direction data."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=270.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.75: 100.0}

    with patch.object(trader._price_feed, "get_direction", return_value=None):
        now = market.end_time - timedelta(seconds=270)
        trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 0
    assert trader._binance_filtered_skips == 1


def test_t270_skips_when_binance_direction_opposes_side() -> None:
    """T=270s Up entry is skipped when Binance says Down."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=270.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.75: 100.0}

    with patch.object(trader._price_feed, "get_direction", return_value="Down"):
        now = market.end_time - timedelta(seconds=270)
        trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 0
    assert trader._binance_filtered_skips == 1


def test_other_buckets_not_filtered_by_binance() -> None:
    """Non-T=270s buckets fire regardless of Binance direction."""
    trader = PaperTrader()
    market = _make_market(end_offset_s=60.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.97: 100.0}

    with patch.object(trader._price_feed, "get_direction", return_value=None):
        now = market.end_time - timedelta(seconds=60)
        trader._evaluate_signals(market.up_token_id, now)

    assert len(trader._open_trades) == 1
    assert trader._binance_filtered_skips == 0


# ---------------------------------------------------------------------------
# Realistic execution fields
# ---------------------------------------------------------------------------


def test_paper_trade_has_realistic_fields() -> None:
    """PaperTrade dataclass exposes the four new realistic fields, defaulting to None."""
    market = _make_market(end_offset_s=270.0)
    trade = PaperTrade(
        trade_id="test_id",
        entry_timestamp_utc=datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC),
        market=market,
        side="Up",
        signal_bucket_label="T=270s_0.70-0.85",
        signal_target_time_s=270.0,
        seconds_to_resolution_at_entry=270.0,
        entry_price=0.75,
        simulated_shares=1.0 / 0.75,
        simulated_stake_usdc=1.0,
        fee_usdc=0.005,
    )
    assert trade.realistic_entry_price_1 is None
    assert trade.realistic_entry_price_5 is None
    assert trade.realistic_entry_price_25 is None
    assert trade.realistic_out_of_bucket is None


def test_csv_header_includes_realistic_columns() -> None:
    """_CSV_HEADER contains all four realistic execution columns."""
    assert "realistic_entry_price_1" in _CSV_HEADER
    assert "realistic_entry_price_5" in _CSV_HEADER
    assert "realistic_entry_price_25" in _CSV_HEADER
    assert "realistic_out_of_bucket" in _CSV_HEADER


def test_size_shares_rounded_to_4_decimals_in_live_mode() -> None:
    """Live order size_shares must be ≤ 4 decimal places (Polymarket precision)."""
    stake = 5.0
    best_ask = 0.70
    size_shares = round(stake / best_ask, 4)
    assert size_shares == 7.1429
    assert len(str(size_shares).split(".")[-1]) <= 4
