"""Unit tests for src/paper_trader.py — all offline."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.paper_trader import (
    PaperTrade,
    PaperTrader,
    SIGNAL_RULES,
    STAKE_USDC,
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


def test_signal_rules_cover_three_verified_buckets() -> None:
    assert len(SIGNAL_RULES) == 3
    labels = [r[4] for r in SIGNAL_RULES]
    assert "T=60s_0.90-0.95" in labels
    assert "T=60s_0.95-1.00" in labels
    assert "T=10s_0.95-1.00" in labels


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
    market = _make_market(end_offset_s=10.0)
    trader._tracked[market.condition_id] = market
    trader._token_to_market[market.up_token_id] = (market, "Up")
    trader._book_asks[market.up_token_id] = {0.99: 100.0}

    now = market.end_time - timedelta(seconds=10)
    trader._evaluate_signals(market.up_token_id, now)

    trade = trader._open_trades[0]
    # stake=$1, shares = 1/0.99 ≈ 1.0101
    # fee = shares × 0.072 × 0.99 × 0.01 = 1.0101 × 0.0007128 ≈ 0.00072 USDC
    expected_fee = (1.0 / 0.99) * 0.072 * 0.99 * 0.01
    assert abs(trade.fee_usdc - expected_fee) < 1e-4
    assert abs(trade.simulated_shares - 1.0 / 0.99) < 1e-6


def test_determine_winner_when_up_high_down_low() -> None:
    trader = PaperTrader()
    market = _make_market()
    trader._book_asks[market.up_token_id] = {0.98: 100.0}
    trader._book_asks[market.down_token_id] = {0.02: 100.0}
    assert trader._determine_winner_from_book(market) == "Up"


def test_determine_winner_when_down_high_up_low() -> None:
    trader = PaperTrader()
    market = _make_market()
    trader._book_asks[market.up_token_id] = {0.01: 100.0}
    trader._book_asks[market.down_token_id] = {0.99: 100.0}
    assert trader._determine_winner_from_book(market) == "Down"


def test_determine_winner_unknown_when_mid_price() -> None:
    trader = PaperTrader()
    market = _make_market()
    trader._book_asks[market.up_token_id] = {0.55: 100.0}
    trader._book_asks[market.down_token_id] = {0.45: 100.0}
    assert trader._determine_winner_from_book(market) == "unknown"


def test_resolve_open_trades_marks_win_correctly() -> None:
    trader = PaperTrader()
    market = _make_market(end_offset_s=-60.0)  # end_time = pinned_now - 60s
    # Pass now = end_time + 120s so secs_past_end = 120 (> 30 threshold)
    resolve_now = market.end_time + timedelta(seconds=120)
    trade = PaperTrade(
        trade_id="20260420_000001",
        entry_timestamp_utc=market.end_time - timedelta(seconds=10),
        market=market,
        side="Up",
        signal_bucket_label="T=10s_0.95-1.00",
        signal_target_time_s=10.0,
        seconds_to_resolution_at_entry=10.0,
        entry_price=0.97,
        simulated_shares=1.0 / 0.97,
        simulated_stake_usdc=1.0,
        fee_usdc=0.002,
    )
    trader._open_trades.append(trade)
    # Up wins
    trader._book_asks[market.up_token_id] = {0.99: 100.0}
    trader._book_asks[market.down_token_id] = {0.01: 100.0}
    trader._resolve_open_trades(now=resolve_now)

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
        entry_timestamp_utc=market.end_time - timedelta(seconds=10),
        market=market,
        side="Up",
        signal_bucket_label="T=10s_0.95-1.00",
        signal_target_time_s=10.0,
        seconds_to_resolution_at_entry=10.0,
        entry_price=0.97,
        simulated_shares=1.0 / 0.97,
        simulated_stake_usdc=1.0,
        fee_usdc=0.002,
    )
    trader._open_trades.append(trade)
    # Down wins — Up side loses
    trader._book_asks[market.up_token_id] = {0.01: 100.0}
    trader._book_asks[market.down_token_id] = {0.99: 100.0}
    trader._resolve_open_trades(now=resolve_now)

    assert trader._total_losses == 1
    assert trader._total_wins == 0
    assert trade.winner == "Down"
    assert trade.payout_usdc == 0.0
    assert trade.pnl_usdc is not None and trade.pnl_usdc < 0
