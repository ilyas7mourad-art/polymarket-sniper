"""Unit tests for src/orderbook_observer.py — all offline."""

import asyncio
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import config
from src.orderbook_observer import OrderbookObserver, OrderbookTick
from src.scanner import Market

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market(
    condition_id: str = "0xcond1",
    asset: str = "BTC",
    end_offset_s: int = 300,
    slug: str = "btc-5m-test",
) -> Market:
    now = datetime.now(UTC)
    return Market(
        condition_id=condition_id,
        question=f"{asset} Up or Down - test",
        asset=asset,
        timeframe="5m",
        start_time=now,
        end_time=now + timedelta(seconds=end_offset_s),
        up_token_id=f"up_{condition_id}",
        down_token_id=f"down_{condition_id}",
        slug=slug,
        raw={},
    )


def _make_tick(
    bid: float = 0.70,
    ask: float = 0.72,
    side: str = "Up",
    secs: float = 120.0,
) -> OrderbookTick:
    return OrderbookTick(
        timestamp_utc=datetime.now(UTC),
        market_slug="btc-5m-test",
        asset="BTC",
        condition_id="0xcond1",
        side=side,
        best_bid=bid,
        best_ask=ask,
        seconds_to_resolution=secs,
    )


@pytest.fixture
def observer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OrderbookObserver:
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    return OrderbookObserver(max_markets=5, refresh_interval=60)


# ---------------------------------------------------------------------------
# 1. Mid calculation
# ---------------------------------------------------------------------------


def test_orderbook_tick_mid_calculation() -> None:
    tick = _make_tick(bid=0.70, ask=0.72)
    assert tick.mid == pytest.approx(0.71)


# ---------------------------------------------------------------------------
# 2. Buffer and flush
# ---------------------------------------------------------------------------


def test_observer_buffers_and_flushes(observer: OrderbookObserver, tmp_path: Path) -> None:
    for _ in range(3):
        observer._record_tick(_make_tick())

    assert len(observer._buffer) == 3
    observer._flush()
    assert len(observer._buffer) == 0

    csv_files = list(tmp_path.glob("orderbook_*.csv"))
    assert len(csv_files) == 1

    rows = csv_files[0].read_text().splitlines()
    assert len(rows) == 4  # 1 header + 3 data rows


# ---------------------------------------------------------------------------
# 3. Auto-flush at 50 ticks
# ---------------------------------------------------------------------------


def test_observer_flushes_at_50_ticks(observer: OrderbookObserver, tmp_path: Path) -> None:
    for _ in range(50):
        observer._record_tick(_make_tick())

    # Buffer should have been flushed automatically at 50
    assert len(observer._buffer) == 0

    csv_files = list(tmp_path.glob("orderbook_*.csv"))
    assert len(csv_files) == 1

    rows = csv_files[0].read_text().splitlines()
    # 1 header + 50 data rows
    assert len(rows) == 51


# ---------------------------------------------------------------------------
# 4. CSV schema matches spec
# ---------------------------------------------------------------------------


def test_observer_csv_schema(observer: OrderbookObserver, tmp_path: Path) -> None:
    observer._record_tick(_make_tick())
    observer._flush()

    csv_files = list(tmp_path.glob("orderbook_*.csv"))
    with csv_files[0].open() as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == [
        "timestamp_utc", "market_slug", "asset", "condition_id",
        "side", "best_bid", "best_ask", "mid", "seconds_to_resolution",
    ]


# ---------------------------------------------------------------------------
# 5. Negative seconds_to_resolution writes without error
# ---------------------------------------------------------------------------


def test_observer_handles_negative_seconds_to_resolution(
    observer: OrderbookObserver, tmp_path: Path
) -> None:
    tick = _make_tick(secs=-5.0)
    observer._record_tick(tick)
    observer._flush()  # should not raise

    csv_files = list(tmp_path.glob("orderbook_*.csv"))
    rows = csv_files[0].read_text().splitlines()
    data_row = rows[1]
    assert "-5.0" in data_row


# ---------------------------------------------------------------------------
# 6. Market refresh swaps old for new
# ---------------------------------------------------------------------------


def test_refresh_markets_swaps_old_for_new(
    observer: OrderbookObserver, tmp_path: Path
) -> None:
    market_a = _make_market(condition_id="0xAAAA", slug="btc-a", end_offset_s=300)
    market_b = _make_market(condition_id="0xBBBB", slug="btc-b", end_offset_s=600)
    market_c = _make_market(condition_id="0xCCCC", slug="btc-c", end_offset_s=900)

    with patch("src.orderbook_observer.scan", return_value=[market_a, market_b]):
        asyncio.run(observer._refresh_markets())

    assert "0xAAAA" in observer._tracked
    assert "0xBBBB" in observer._tracked
    assert len(observer._tracked) == 2

    # Second refresh: market_a gone, market_c added
    with patch("src.orderbook_observer.scan", return_value=[market_b, market_c]):
        asyncio.run(observer._refresh_markets())

    assert "0xAAAA" not in observer._tracked
    assert "0xBBBB" in observer._tracked
    assert "0xCCCC" in observer._tracked
    assert len(observer._tracked) == 2

    # Token mappings updated correctly
    assert f"up_0xBBBB" in observer._token_to_market
    assert f"up_0xCCCC" in observer._token_to_market
    assert f"up_0xAAAA" not in observer._token_to_market
