"""Unit tests for src/realistic_executor.py — all offline."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.realistic_executor import (
    SIMULATED_LATENCY_MS,
    SIMULATED_STAKES_USDC,
    RealisticExecutor,
    RealisticFill,
    walk_orderbook,
)


def test_constants() -> None:
    assert SIMULATED_LATENCY_MS == 300
    assert SIMULATED_STAKES_USDC == [1.0, 5.0, 25.0]


def test_walk_orderbook_full_fill_at_one_level() -> None:
    """$1 stake fills entirely from a single level at $0.75 (need 1.33 shares)."""
    asks = [{"price": "0.75", "size": "100"}]
    avg, shares = walk_orderbook(asks, 1.0)
    assert avg == pytest.approx(0.75)
    assert shares == pytest.approx(1.0 / 0.75, abs=0.001)


def test_walk_orderbook_full_fill_with_partial() -> None:
    """$25 stake fills from one level."""
    asks = [{"price": "0.50", "size": "100"}]
    avg, shares = walk_orderbook(asks, 25.0)
    assert avg == pytest.approx(0.50)
    assert shares == pytest.approx(50.0)


def test_walk_orderbook_walks_multiple_levels() -> None:
    """$15 stake walks from $0.50 to $0.55."""
    asks = [
        {"price": "0.50", "size": "10"},   # $5 available at this level
        {"price": "0.55", "size": "100"},  # plenty at next level
    ]
    avg, shares = walk_orderbook(asks, 15.0)
    # First level: 10 shares for $5. Second level: need $10 more → 10/0.55 ≈ 18.18 shares
    expected_shares = 10.0 + (10.0 / 0.55)
    expected_avg = 15.0 / expected_shares
    assert avg == pytest.approx(expected_avg, abs=0.005)
    assert shares == pytest.approx(expected_shares, abs=0.05)


def test_walk_orderbook_empty_book() -> None:
    avg, shares = walk_orderbook([], 1.0)
    assert avg is None
    assert shares == 0.0


def test_walk_orderbook_unsorted_input_handled() -> None:
    """Asks not pre-sorted — function should sort by price ascending."""
    asks = [
        {"price": "0.80", "size": "10"},
        {"price": "0.50", "size": "10"},
        {"price": "0.65", "size": "10"},
    ]
    # $5 stake should fill entirely from the $0.50 level first
    avg, shares = walk_orderbook(asks, 5.0)
    assert avg == pytest.approx(0.50)


def _make_mock_client(asks: list[dict]) -> AsyncMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"asks": asks, "bids": []}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def test_simulate_fill_returns_dict_per_stake_size() -> None:
    """simulate_fill returns a dict keyed by each stake size."""
    executor = RealisticExecutor(latency_ms=0)

    mock_client = _make_mock_client([{"price": "0.75", "size": "1000"}])

    with patch("src.realistic_executor.httpx.AsyncClient", return_value=mock_client):
        fills = asyncio.run(executor.simulate_fill(
            token_id="test_token",
            signal_min_ask=0.70,
            signal_max_ask=0.80,
        ))

    assert set(fills.keys()) == {1.0, 5.0, 25.0}
    assert fills[1.0].weighted_avg_price == pytest.approx(0.75)
    assert fills[1.0].out_of_bucket is False


def test_simulate_fill_detects_out_of_bucket_drift() -> None:
    """Best ask drifted to $0.86, original bucket was [0.70, 0.85) — out of bucket."""
    executor = RealisticExecutor(latency_ms=0)

    mock_client = _make_mock_client([{"price": "0.86", "size": "1000"}])

    with patch("src.realistic_executor.httpx.AsyncClient", return_value=mock_client):
        fills = asyncio.run(executor.simulate_fill(
            token_id="test_token",
            signal_min_ask=0.70,
            signal_max_ask=0.85,
        ))

    assert fills[1.0].out_of_bucket is True


def test_simulate_fill_returns_none_on_api_error() -> None:
    """Network failure → all fills have None price and out_of_bucket=True."""
    executor = RealisticExecutor(latency_ms=0)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("network down"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.realistic_executor.httpx.AsyncClient", return_value=mock_client):
        fills = asyncio.run(executor.simulate_fill(
            token_id="test_token",
            signal_min_ask=0.70,
            signal_max_ask=0.85,
        ))

    assert fills[1.0].weighted_avg_price is None
    assert fills[1.0].out_of_bucket is True
    assert fills[25.0].weighted_avg_price is None


def test_simulate_fill_empty_book_is_out_of_bucket() -> None:
    """An empty book means best_ask is None, which counts as out-of-bucket."""
    executor = RealisticExecutor(latency_ms=0)

    mock_client = _make_mock_client([])  # empty asks

    with patch("src.realistic_executor.httpx.AsyncClient", return_value=mock_client):
        fills = asyncio.run(executor.simulate_fill(
            token_id="test_token",
            signal_min_ask=0.70,
            signal_max_ask=0.85,
        ))

    assert fills[1.0].out_of_bucket is True
    assert fills[1.0].weighted_avg_price is None
