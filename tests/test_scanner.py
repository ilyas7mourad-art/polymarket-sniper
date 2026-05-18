"""Unit tests for src/scanner.py — all offline, no real API calls."""

from datetime import datetime, timedelta, timezone

from src.scanner import (
    Market,
    SERIES_CONFIG,
    _parse_iso_utc,
    _try_parse_event_market,
    parse_event_market,
)

_5M = timedelta(minutes=5)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Reference date pinned so tests don't drift with the calendar.
REF = datetime(2026, 4, 17, 23, 0, 0, tzinfo=UTC)


def _make_event(
    event_start: str = "2026-04-17T23:05:00Z",
    end_date: str = "2026-04-17T23:10:00Z",
    clob_token_ids: str = '["111", "222"]',
    outcomes: str = '["Up", "Down"]',
    question: str = "Bitcoin Up or Down - April 17, 7:05PM-7:10PM ET",
    condition_id: str = "0xabc123",
    slug: str = "btc-updown-5m-1776467100",
    include_market: bool = True,
) -> dict:
    """Build a realistic event dict mirroring Gamma API structure."""
    event: dict = {
        "slug": slug,
        "eventStartTime": None,   # event-level field is None in real API
        "endDate": end_date,
        "active": True,
        "closed": False,
    }
    if include_market:
        event["markets"] = [{
            "conditionId": condition_id,
            "question": question,
            "clobTokenIds": clob_token_ids,
            "outcomes": outcomes,
            "slug": slug,
            "eventStartTime": event_start,
            "endDate": end_date,
        }]
    else:
        event["markets"] = []
    return event


# ---------------------------------------------------------------------------
# _parse_iso_utc
# ---------------------------------------------------------------------------


def test_parse_iso_utc_handles_z_suffix() -> None:
    dt = _parse_iso_utc("2026-04-17T19:00:00Z")
    assert dt.tzinfo is not None
    assert dt == datetime(2026, 4, 17, 19, 0, tzinfo=UTC)


def test_parse_iso_utc_handles_offset_suffix() -> None:
    assert _parse_iso_utc("2026-04-17T19:00:00+00:00") == _parse_iso_utc("2026-04-17T19:00:00Z")


# ---------------------------------------------------------------------------
# SERIES_CONFIG
# ---------------------------------------------------------------------------


def test_series_config_contains_btc_and_eth() -> None:
    assets = {cfg["asset"] for cfg in SERIES_CONFIG}
    assert "BTC" in assets and "ETH" in assets
    for cfg in SERIES_CONFIG:
        assert isinstance(cfg["series_id"], int)
        assert isinstance(cfg["window"], timedelta)


# ---------------------------------------------------------------------------
# _try_parse_event_market / parse_event_market
# ---------------------------------------------------------------------------


def test_parse_event_market_happy_path_btc() -> None:
    event = _make_event()
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)

    assert reason is None
    assert market is not None
    assert isinstance(market, Market)
    assert market.asset == "BTC"
    assert market.condition_id == "0xabc123"
    assert market.up_token_id == "111"
    assert market.down_token_id == "222"
    assert market.start_time == datetime(2026, 4, 17, 23, 5, tzinfo=UTC)
    assert market.end_time == datetime(2026, 4, 17, 23, 10, tzinfo=UTC)
    assert market.end_time - market.start_time == timedelta(minutes=5)
    assert market.start_time.tzinfo is not None


def test_parse_event_market_happy_path_eth() -> None:
    event = _make_event(
        question="Ethereum Up or Down - April 17, 7:05PM-7:10PM ET",
    )
    market, reason = _try_parse_event_market(event, "ETH", "5m", _5M, reference_date=REF)

    assert reason is None
    assert market is not None
    assert market.asset == "ETH"
    assert market.start_time == datetime(2026, 4, 17, 23, 5, tzinfo=UTC)
    assert market.end_time == datetime(2026, 4, 17, 23, 10, tzinfo=UTC)


def test_parse_event_market_rejects_no_markets() -> None:
    event = _make_event(include_market=False)
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert market is None
    assert reason == "no_markets"


def test_parse_event_market_rejects_missing_dates() -> None:
    event = _make_event(event_start="", end_date="")
    # Both fields are empty strings — parser should return missing_dates
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert market is None
    assert reason == "missing_dates"


def test_parse_event_market_rejects_wrong_window() -> None:
    # 10-minute span instead of 5
    event = _make_event(end_date="2026-04-17T23:15:00Z")
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert market is None
    assert reason == "wrong_window_size"


def test_parse_event_market_rejects_stale() -> None:
    # endDate 2 hours before REF (REF = 23:00, end = 21:05 → stale by >1h)
    event = _make_event(
        event_start="2026-04-17T21:00:00Z",
        end_date="2026-04-17T21:05:00Z",
    )
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert market is None
    assert reason == "stale"


def test_parse_event_market_rejects_too_far_future() -> None:
    # eventStartTime 3 days after REF
    event = _make_event(
        event_start="2026-04-20T23:05:00Z",
        end_date="2026-04-20T23:10:00Z",
    )
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert market is None
    assert reason == "too_far_future"


def test_parse_event_market_rejects_missing_tokens() -> None:
    event = _make_event(clob_token_ids="")
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert market is None
    assert reason == "missing_tokens"


def test_parse_event_market_yes_no_outcomes() -> None:
    """Markets that use Yes/No instead of Up/Down should still map correctly."""
    event = _make_event(outcomes='["Yes", "No"]')
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert reason is None
    assert market is not None
    assert market.up_token_id == "111"
    assert market.down_token_id == "222"


def test_parse_event_market_public_wrapper() -> None:
    """parse_event_market returns a Market (not a tuple)."""
    event = _make_event()
    result = parse_event_market(event, "BTC", "5m", _5M, reference_date=REF)
    assert isinstance(result, Market)
    assert result.asset == "BTC"


def test_parse_event_market_rejects_december_2025_zombie() -> None:
    """Regression test: API returns Dec 2025 events with active=true; must be filtered as stale."""
    ref = datetime(2026, 4, 17, 23, 47, 0, tzinfo=UTC)
    event = _make_event(
        event_start="2025-12-19T16:35:00Z",
        end_date="2025-12-19T16:40:00Z",
    )
    market, reason = _try_parse_event_market(event, "BTC", "5m", _5M, ref)
    assert market is None
    assert reason == "stale"
