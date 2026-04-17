"""Unit tests for src/scanner.py — all offline, no real API calls."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.scanner import (
    Market,
    is_btc_eth_5min_window,
    parse_market,
    parse_window_times,
)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MARKET = {
    "conditionId": "0xabc123def456",
    "question": "Bitcoin Up or Down - April 17, 3:00PM-3:05PM ET",
    "clobTokenIds": '["111", "222"]',
    "outcomes": '["Up", "Down"]',
    "slug": "btc-up-or-down-april-17-3pm-3-05pm-et",
    "active": True,
    "closed": False,
}

# Reference date pinned so tests don't drift with the calendar.
REF = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# is_btc_eth_5min_window
# ---------------------------------------------------------------------------


def test_is_btc_eth_5min_window_accepts_valid() -> None:
    valid = [
        "Bitcoin Up or Down - April 17, 3:00PM-3:05PM ET",
        "Bitcoin Up or Down - January 1, 11:55PM-12:00AM ET",
        "Ethereum Up or Down - March 3, 9:00AM-9:05AM ET",
        "Ethereum Up or Down - December 31, 4:45PM-4:50PM ET",
    ]
    for q in valid:
        assert is_btc_eth_5min_window(q), f"Expected True for: {q}"


def test_is_btc_eth_5min_window_rejects_invalid() -> None:
    invalid = [
        "Ethereum Up or Down - April 14, 5:15PM-5:30PM ET",   # 15-min
        "Bitcoin Up or Down - April 14, 4:00PM-8:00PM ET",    # 4-hour
        "Bitcoin Up or Down - April 14, 9:00AM-9:10AM ET",    # 10-min
        "Solana Up or Down - April 17, 3:00PM-3:05PM ET",     # wrong asset
        "Bitcoin goes up - April 17, 3:00PM-3:05PM ET",       # wrong format
        "bitcoin up or down - April 17, 3:00PM-3:05PM ET",    # wrong case
        "",                                                     # empty
        "Random market question",                              # unrelated
    ]
    for q in invalid:
        assert not is_btc_eth_5min_window(q), f"Expected False for: {q}"


# ---------------------------------------------------------------------------
# parse_window_times
# ---------------------------------------------------------------------------


def test_parse_window_times_round_trip() -> None:
    question = "Bitcoin Up or Down - April 17, 3:00PM-3:05PM ET"
    result = parse_window_times(question, reference_date=REF)

    assert result is not None
    start_utc, end_utc = result

    # Both must be UTC-aware
    assert start_utc.tzinfo is not None
    assert end_utc.tzinfo is not None

    # Span must be exactly 5 minutes
    assert end_utc - start_utc == timedelta(minutes=5)

    # April 17 3:00 PM ET = 19:00 UTC (EDT = UTC-4)
    assert start_utc.hour == 19
    assert start_utc.minute == 0
    assert end_utc.hour == 19
    assert end_utc.minute == 5

    assert start_utc.year == 2026
    assert start_utc.month == 4
    assert start_utc.day == 17


def test_parse_window_times_midnight_wrap() -> None:
    """11:55PM-12:00AM should produce a 5-minute span crossing midnight."""
    question = "Bitcoin Up or Down - January 1, 11:55PM-12:00AM ET"
    ref = datetime(2026, 1, 1, 4, 0, 0, tzinfo=UTC)
    result = parse_window_times(question, reference_date=ref)

    assert result is not None
    start_utc, end_utc = result
    assert end_utc - start_utc == timedelta(minutes=5)
    assert end_utc > start_utc


def test_parse_window_times_returns_none_on_bad_input() -> None:
    assert parse_window_times("Not a market question") is None
    assert parse_window_times("") is None


# ---------------------------------------------------------------------------
# parse_market
# ---------------------------------------------------------------------------


def test_parse_market_returns_none_on_missing_token_ids() -> None:
    raw = {**SAMPLE_MARKET}
    del raw["clobTokenIds"]
    assert parse_market(raw) is None


def test_parse_market_returns_none_on_non_5min() -> None:
    raw = {
        **SAMPLE_MARKET,
        "question": "Ethereum Up or Down - April 14, 5:15PM-5:30PM ET",
    }
    assert parse_market(raw) is None


def test_parse_market_happy_path() -> None:
    market = parse_market(SAMPLE_MARKET)

    assert market is not None
    assert isinstance(market, Market)

    assert market.condition_id == "0xabc123def456"
    assert market.asset == "BTC"
    assert market.up_token_id == "111"
    assert market.down_token_id == "222"
    assert market.slug == "btc-up-or-down-april-17-3pm-3-05pm-et"

    assert market.end_time - market.start_time == timedelta(minutes=5)
    assert market.start_time.tzinfo is not None
    assert market.end_time.tzinfo is not None

    # raw preserved
    assert market.raw is SAMPLE_MARKET


def test_parse_market_yes_no_outcomes() -> None:
    """Markets that use Yes/No instead of Up/Down should still map correctly."""
    raw = {
        **SAMPLE_MARKET,
        "outcomes": '["Yes", "No"]',
    }
    market = parse_market(raw)
    assert market is not None
    assert market.up_token_id == "111"
    assert market.down_token_id == "222"


def test_parse_market_ethereum() -> None:
    raw = {
        **SAMPLE_MARKET,
        "question": "Ethereum Up or Down - April 17, 3:00PM-3:05PM ET",
        "conditionId": "0xeth999",
    }
    market = parse_market(raw)
    assert market is not None
    assert market.asset == "ETH"
    assert market.condition_id == "0xeth999"
