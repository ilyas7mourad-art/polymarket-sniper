"""Unit tests for src/bonereaper_tracker.py — all offline."""

import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bonereaper_tracker import (
    BONEREAPER_PROXY,
    POLL_INTERVAL_SECONDS,
    TRADES_PER_POLL,
    BonereaperTracker,
    _CSV_HEADER,
)

UTC = timezone.utc


def _mock_trade(tx: str, price: float = 0.5, size: float = 10.0,
                side: str = "BUY", outcome: str = "Up",
                slug: str = "btc-updown-5m-test") -> dict:
    """Build a fake API trade response."""
    return {
        "proxyWallet": BONEREAPER_PROXY.lower(),
        "side": side,
        "asset": "12345",
        "conditionId": "0xabc",
        "size": size,
        "price": price,
        "timestamp": 1776700000,
        "title": "Bitcoin Up or Down - test",
        "slug": slug,
        "outcome": outcome,
        "transactionHash": tx,
    }


def test_constants() -> None:
    assert BONEREAPER_PROXY == "0x519e0202046caf341469df75b2e7a7eac4f3d41d"
    assert POLL_INTERVAL_SECONDS == 5
    assert TRADES_PER_POLL == 50


def test_csv_header_has_required_fields() -> None:
    required = {"timestamp_utc", "transaction_hash", "side", "outcome",
                "price", "size", "usdc_size", "condition_id",
                "market_slug", "market_title"}
    assert required.issubset(set(_CSV_HEADER))


def test_fetch_new_trades_filters_dupes() -> None:
    """If we've already seen a tx, it's not returned."""
    tracker = BonereaperTracker()
    tracker._seen_tx.add("0xtx_already_seen")

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        _mock_trade("0xtx_already_seen"),
        _mock_trade("0xtx_new"),
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = asyncio.run(tracker._fetch_new_trades(mock_client))

    assert len(result) == 1
    assert result[0]["transaction_hash"] == "0xtx_new"


def test_fetch_new_trades_computes_usdc_size() -> None:
    """usdc_size = price × size."""
    tracker = BonereaperTracker()
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [_mock_trade("0xt", price=0.97, size=100.0)]
    mock_resp.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = asyncio.run(tracker._fetch_new_trades(mock_client))

    assert len(result) == 1
    assert result[0]["usdc_size"] == pytest.approx(97.0, abs=1e-4)


def test_fetch_new_trades_handles_empty_response() -> None:
    tracker = BonereaperTracker()
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = []
    mock_resp.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = asyncio.run(tracker._fetch_new_trades(mock_client))

    assert result == []


def test_fetch_new_trades_handles_non_list_response() -> None:
    """API returns malformed data — should log warning and return []."""
    tracker = BonereaperTracker()
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"error": "something"}
    mock_resp.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = asyncio.run(tracker._fetch_new_trades(mock_client))
    assert result == []


def test_fetch_new_trades_skips_invalid_price() -> None:
    """Trades with non-numeric price/size are skipped."""
    tracker = BonereaperTracker()
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    bad_trade = _mock_trade("0xbad")
    bad_trade["price"] = "not_a_number"
    mock_resp.json.return_value = [bad_trade]
    mock_resp.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = asyncio.run(tracker._fetch_new_trades(mock_client))
    assert result == []


def test_append_trades_writes_csv_with_header(tmp_path: Path) -> None:
    """First write creates file with header row."""
    tracker = BonereaperTracker()
    trades = [{
        "timestamp_utc": "2026-04-20T12:00:00+00:00",
        "transaction_hash": "0xtx1",
        "side": "BUY",
        "outcome": "Up",
        "price": 0.97,
        "size": 100.0,
        "usdc_size": 97.0,
        "condition_id": "0xabc",
        "asset": "12345",
        "market_slug": "btc-updown-5m-test",
        "market_title": "BTC test",
    }]

    with patch("src.bonereaper_tracker.config") as mock_config:
        mock_config.DATA_DIR = str(tmp_path)
        tracker._append_trades(trades)

    today = datetime.now(UTC).strftime("%Y%m%d")
    csv_path = tmp_path / f"bonereaper_trades_{today}.csv"
    assert csv_path.exists()

    with csv_path.open("r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["transaction_hash"] == "0xtx1"
    assert rows[0]["side"] == "BUY"


def test_append_trades_appends_without_duplicating_header(tmp_path: Path) -> None:
    """Second write appends without re-writing the header."""
    tracker = BonereaperTracker()
    trade = {
        "timestamp_utc": "2026-04-20T12:00:00+00:00",
        "transaction_hash": "0xtx1",
        "side": "BUY", "outcome": "Up", "price": 0.97, "size": 100.0,
        "usdc_size": 97.0, "condition_id": "0xabc", "asset": "12345",
        "market_slug": "test", "market_title": "BTC",
    }

    with patch("src.bonereaper_tracker.config") as mock_config:
        mock_config.DATA_DIR = str(tmp_path)
        tracker._append_trades([trade])
        trade2 = dict(trade, transaction_hash="0xtx2")
        tracker._append_trades([trade2])

    today = datetime.now(UTC).strftime("%Y%m%d")
    csv_path = tmp_path / f"bonereaper_trades_{today}.csv"
    with csv_path.open("r") as f:
        lines = f.readlines()
    # 1 header + 2 data lines
    assert len(lines) == 3


def test_prime_seen_from_csv_loads_existing_tx(tmp_path: Path) -> None:
    """On startup, seen_tx is populated from today's CSV to prevent dupe writes."""
    today = datetime.now(UTC).strftime("%Y%m%d")
    csv_path = tmp_path / f"bonereaper_trades_{today}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        writer.writerow({
            "timestamp_utc": "2026-04-20T12:00:00+00:00",
            "transaction_hash": "0xexisting",
            "side": "BUY", "outcome": "Up", "price": 0.5, "size": 10.0,
            "usdc_size": 5.0, "condition_id": "0xabc", "asset": "12345",
            "market_slug": "test", "market_title": "BTC",
        })

    tracker = BonereaperTracker()
    with patch("src.bonereaper_tracker.config") as mock_config:
        mock_config.DATA_DIR = str(tmp_path)
        tracker._prime_seen_from_csv()

    assert "0xexisting" in tracker._seen_tx


def test_fetch_new_trades_returns_oldest_first() -> None:
    """API returns newest-first; our writer should output oldest-first."""
    tracker = BonereaperTracker()
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        _mock_trade("0x_newest"),
        _mock_trade("0x_middle"),
        _mock_trade("0x_oldest"),
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = asyncio.run(tracker._fetch_new_trades(mock_client))
    assert len(result) == 3
    assert result[0]["transaction_hash"] == "0x_oldest"
    assert result[2]["transaction_hash"] == "0x_newest"


def test_log_identity_check_handles_empty_response() -> None:
    """If the API returns nothing, identity check logs a warning but doesn't crash."""
    tracker = BonereaperTracker()
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = []
    mock_resp.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("src.bonereaper_tracker.httpx.AsyncClient") as mock_client_class:
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)
        # Should not raise
        asyncio.run(tracker._log_identity_check())
