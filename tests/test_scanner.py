"""Unit tests for src/scanner.py — all offline, no real API calls."""

from datetime import datetime, timedelta, timezone

from src.scanner import (
    Market,
    _parse_iso_utc,
    is_btc_eth_5min_window,
    parse_market,
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
    "eventStartTime": "2026-04-17T19:00:00Z",
    "endDate": "2026-04-17T19:05:00Z",
    "active": True,
    "closed": False,
}

# 5 minutes before the window opens — market is active and near-term.
REF = datetime(2026, 4, 17, 18, 55, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# _parse_iso_utc
# ---------------------------------------------------------------------------


def test_parse_iso_utc_handles_z_suffix() -> None:
    dt = _parse_iso_utc("2026-04-17T19:00:00Z")
    assert dt.tzinfo is not None
    assert dt.year == 2026
    assert dt.month == 4
    assert dt.day == 17
    assert dt.hour == 19
    assert dt.minute == 0
    assert dt.second == 0


def test_parse_iso_utc_handles_offset_suffix() -> None:
    dt = _parse_iso_utc("2026-04-17T19:00:00+00:00")
    assert dt == _parse_iso_utc("2026-04-17T19:00:00Z")


# ---------------------------------------------------------------------------
# is_btc_eth_5min_window
# ---------------------------------------------------------------------------


def test_is_btc_eth_5min_window_accepts_valid() -> None:
    cases = [
        # BTC, standard 5-min window
        (SAMPLE_MARKET, REF),
        # ETH, different time
        (
            {**SAMPLE_MARKET, "question": "Ethereum Up or Down - April 17, 3:05PM-3:10PM ET",
             "eventStartTime": "2026-04-17T19:05:00Z", "endDate": "2026-04-17T19:10:00Z"},
            REF,
        ),
        # BTC, early morning window
        (
            {**SAMPLE_MARKET, "question": "Bitcoin Up or Down - April 18, 9:00AM-9:05AM ET",
             "eventStartTime": "2026-04-18T13:00:00Z", "endDate": "2026-04-18T13:05:00Z"},
            datetime(2026, 4, 18, 12, 55, 0, tzinfo=UTC),
        ),
        # ETH, within 1h of ref (just started)
        (
            {**SAMPLE_MARKET, "question": "Ethereum Up or Down - April 17, 2:55PM-3:00PM ET",
             "eventStartTime": "2026-04-17T18:55:00Z", "endDate": "2026-04-17T19:00:00Z"},
            REF,
        ),
    ]
    for raw, ref in cases:
        assert is_btc_eth_5min_window(raw, reference_date=ref), (
            f"Expected True for: {raw['question']}"
        )


def test_is_btc_eth_5min_window_rejects_invalid() -> None:
    cases = [
        # Wrong asset
        ({**SAMPLE_MARKET, "question": "Solana Up or Down - April 17, 3:00PM-3:05PM ET"}, REF),
        # Unrecognised format
        ({**SAMPLE_MARKET, "question": "Bitcoin goes up"}, REF),
        # Empty question
        ({**SAMPLE_MARKET, "question": ""}, REF),
        # 10-minute window
        ({**SAMPLE_MARKET, "endDate": "2026-04-17T19:10:00Z"}, REF),
        # 15-minute window
        ({**SAMPLE_MARKET, "endDate": "2026-04-17T19:15:00Z"}, REF),
        # Stale: endDate 2h before REF (REF=18:55, end=16:55)
        (
            {**SAMPLE_MARKET,
             "eventStartTime": "2026-04-17T16:50:00Z",
             "endDate": "2026-04-17T16:55:00Z"},
            REF,
        ),
        # Far future: eventStartTime 3 days after REF
        (
            {**SAMPLE_MARKET,
             "eventStartTime": "2026-04-20T19:00:00Z",
             "endDate": "2026-04-20T19:05:00Z"},
            REF,
        ),
        # Missing eventStartTime
        ({k: v for k, v in SAMPLE_MARKET.items() if k != "eventStartTime"}, REF),
        # Missing endDate
        ({k: v for k, v in SAMPLE_MARKET.items() if k != "endDate"}, REF),
    ]
    for raw, ref in cases:
        assert not is_btc_eth_5min_window(raw, reference_date=ref), (
            f"Expected False for: {raw.get('question')}"
        )


# ---------------------------------------------------------------------------
# parse_market
# ---------------------------------------------------------------------------


def test_parse_market_returns_none_on_missing_token_ids() -> None:
    raw = {k: v for k, v in SAMPLE_MARKET.items() if k != "clobTokenIds"}
    assert parse_market(raw, reference_date=REF) is None


def test_parse_market_returns_none_on_non_5min() -> None:
    raw = {**SAMPLE_MARKET, "endDate": "2026-04-17T19:10:00Z"}
    assert parse_market(raw, reference_date=REF) is None


def test_parse_market_rejects_missing_event_start_time() -> None:
    raw = {k: v for k, v in SAMPLE_MARKET.items() if k != "eventStartTime"}
    assert parse_market(raw, reference_date=REF) is None


def test_parse_market_rejects_stale() -> None:
    raw = {
        **SAMPLE_MARKET,
        "eventStartTime": "2026-04-17T16:50:00Z",
        "endDate": "2026-04-17T16:55:00Z",
    }
    assert parse_market(raw, reference_date=REF) is None


def test_parse_market_rejects_far_future() -> None:
    raw = {
        **SAMPLE_MARKET,
        "eventStartTime": "2026-04-20T19:00:00Z",
        "endDate": "2026-04-20T19:05:00Z",
    }
    assert parse_market(raw, reference_date=REF) is None


def test_parse_market_happy_path() -> None:
    market = parse_market(SAMPLE_MARKET, reference_date=REF)

    assert market is not None
    assert isinstance(market, Market)

    assert market.condition_id == "0xabc123def456"
    assert market.asset == "BTC"
    assert market.up_token_id == "111"
    assert market.down_token_id == "222"
    assert market.slug == "btc-up-or-down-april-17-3pm-3-05pm-et"

    assert market.start_time == datetime(2026, 4, 17, 19, 0, tzinfo=UTC)
    assert market.end_time == datetime(2026, 4, 17, 19, 5, tzinfo=UTC)
    assert market.end_time - market.start_time == timedelta(minutes=5)

    assert market.raw is SAMPLE_MARKET


def test_parse_market_yes_no_outcomes() -> None:
    """Markets that use Yes/No instead of Up/Down should still map correctly."""
    raw = {**SAMPLE_MARKET, "outcomes": '["Yes", "No"]'}
    market = parse_market(raw, reference_date=REF)
    assert market is not None
    assert market.up_token_id == "111"
    assert market.down_token_id == "222"


def test_parse_market_ethereum() -> None:
    raw = {
        **SAMPLE_MARKET,
        "question": "Ethereum Up or Down - April 17, 3:00PM-3:05PM ET",
        "conditionId": "0xeth999",
    }
    market = parse_market(raw, reference_date=REF)
    assert market is not None
    assert market.asset == "ETH"
    assert market.condition_id == "0xeth999"
    assert market.start_time == datetime(2026, 4, 17, 19, 0, tzinfo=UTC)
    assert market.end_time == datetime(2026, 4, 17, 19, 5, tzinfo=UTC)
