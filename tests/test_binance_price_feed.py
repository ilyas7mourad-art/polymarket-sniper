"""Unit tests for src/binance_price_feed.py — all offline."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.binance_price_feed import (
    HISTORY_RETENTION_SECONDS,
    MAX_STALE_SECONDS,
    SIGNAL_LOOKBACK_SECONDS,
    BinancePriceFeed,
)

UTC = timezone.utc


def _ts(offset_seconds: float = 0.0) -> datetime:
    """Return a fixed base time plus offset, always timezone-aware."""
    base = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)
    return base + timedelta(seconds=offset_seconds)


def _make_trade_msg(stream: str, price: float, ts_ms: int) -> str:
    return json.dumps({
        "stream": stream,
        "data": {"s": stream.split("@")[0].upper(), "p": str(price), "T": ts_ms},
    })


def test_constants() -> None:
    assert HISTORY_RETENTION_SECONDS == 60
    assert SIGNAL_LOOKBACK_SECONDS == 30
    assert MAX_STALE_SECONDS == 10


def test_get_direction_returns_none_when_no_data() -> None:
    feed = BinancePriceFeed()
    assert feed.get_direction("BTC") is None
    assert feed.get_direction("ETH") is None


def test_get_direction_returns_none_when_only_one_point() -> None:
    feed = BinancePriceFeed()
    now = _ts()
    feed._record_price("BTC", 65000.0, now)
    with patch("src.binance_price_feed.datetime") as mock_dt:
        mock_dt.now.return_value = now
        result = feed.get_direction("BTC")
    assert result is None


def test_get_direction_returns_up_when_price_rising() -> None:
    feed = BinancePriceFeed()
    base = _ts()
    feed._record_price("BTC", 65000.0, base - timedelta(seconds=20))
    feed._record_price("BTC", 65100.0, base - timedelta(seconds=10))
    feed._record_price("BTC", 65200.0, base)

    with patch("src.binance_price_feed.datetime") as mock_dt:
        mock_dt.now.return_value = base
        result = feed.get_direction("BTC")
    assert result == "Up"


def test_get_direction_returns_down_when_price_falling() -> None:
    feed = BinancePriceFeed()
    base = _ts()
    feed._record_price("ETH", 3000.0, base - timedelta(seconds=20))
    feed._record_price("ETH", 2950.0, base - timedelta(seconds=10))
    feed._record_price("ETH", 2900.0, base)

    with patch("src.binance_price_feed.datetime") as mock_dt:
        mock_dt.now.return_value = base
        result = feed.get_direction("ETH")
    assert result == "Down"


def test_get_direction_returns_none_when_data_outside_lookback() -> None:
    """Prices older than lookback_seconds are excluded from the window."""
    feed = BinancePriceFeed()
    base = _ts()
    # Both points are 60+ seconds old — outside the 30s lookback
    feed._record_price("BTC", 65000.0, base - timedelta(seconds=60))
    feed._record_price("BTC", 65200.0, base - timedelta(seconds=50))

    with patch("src.binance_price_feed.datetime") as mock_dt:
        mock_dt.now.return_value = base
        result = feed.get_direction("BTC")
    assert result is None


def test_get_price_returns_latest_price() -> None:
    feed = BinancePriceFeed()
    base = _ts()
    feed._record_price("BTC", 65000.0, base - timedelta(seconds=5))
    feed._record_price("BTC", 65100.0, base)

    with patch("src.binance_price_feed.datetime") as mock_dt:
        mock_dt.now.return_value = base
        price = feed.get_price("BTC")
    assert price == pytest.approx(65100.0)


def test_get_price_returns_none_when_no_data() -> None:
    feed = BinancePriceFeed()
    assert feed.get_price("BTC") is None


def test_get_price_returns_none_when_stale() -> None:
    feed = BinancePriceFeed()
    base = _ts()
    feed._record_price("BTC", 65000.0, base - timedelta(seconds=MAX_STALE_SECONDS + 5))
    feed._last_update["BTC"] = base - timedelta(seconds=MAX_STALE_SECONDS + 5)

    with patch("src.binance_price_feed.datetime") as mock_dt:
        mock_dt.now.return_value = base
        price = feed.get_price("BTC")
    assert price is None


def test_process_message_records_btc_trade() -> None:
    feed = BinancePriceFeed()
    base = _ts()
    ts_ms = int(base.timestamp() * 1000)
    feed._process_message(_make_trade_msg("btcusdt@trade", 65000.0, ts_ms))

    assert len(feed._history["BTC"]) == 1
    recorded_ts, recorded_price = feed._history["BTC"][0]
    assert recorded_price == pytest.approx(65000.0)
    assert abs((recorded_ts - base).total_seconds()) < 0.01


def test_process_message_records_eth_trade() -> None:
    feed = BinancePriceFeed()
    base = _ts()
    ts_ms = int(base.timestamp() * 1000)
    feed._process_message(_make_trade_msg("ethusdt@trade", 3000.0, ts_ms))

    assert len(feed._history["ETH"]) == 1
    _, recorded_price = feed._history["ETH"][0]
    assert recorded_price == pytest.approx(3000.0)


def test_process_message_ignores_unknown_stream() -> None:
    feed = BinancePriceFeed()
    msg = json.dumps({"stream": "solusdt@trade", "data": {"p": "200", "T": 1000000}})
    feed._process_message(msg)

    assert len(feed._history["BTC"]) == 0
    assert len(feed._history["ETH"]) == 0


def test_record_price_evicts_old_entries() -> None:
    """Prices older than HISTORY_RETENTION_SECONDS are pruned."""
    feed = BinancePriceFeed()
    base = _ts()
    old_ts = base - timedelta(seconds=HISTORY_RETENTION_SECONDS + 10)
    feed._record_price("BTC", 64000.0, old_ts)
    feed._record_price("BTC", 65000.0, base)

    assert len(feed._history["BTC"]) == 1
    assert feed._history["BTC"][0][1] == pytest.approx(65000.0)
