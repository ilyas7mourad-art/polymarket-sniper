"""Unit tests for src/live_executor.py — all offline."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.live_executor import LiveExecutor, LiveOrderResult, compute_clean_order_amounts


def _make_executor() -> LiveExecutor:
    with patch("src.live_executor.ClobClient"):
        executor = LiveExecutor()
    # Skip lazy creds derivation for all tests
    executor._api_creds = MagicMock()
    return executor


def test_place_order_filled() -> None:
    """Full fill: makingAmount covers the full size."""
    executor = _make_executor()
    # 75 USDC at $0.75 → 100 shares filled == size_shares → "filled"
    executor._client.create_order = MagicMock(return_value=MagicMock())
    executor._client.post_order = MagicMock(return_value={
        "success": True,
        "orderID": "order-123",
        "makingAmount": "75",
        "price": "0.75",
    })

    result = asyncio.run(executor.place_order("token-abc", price=0.75, size_shares=100.0))

    assert result.fill_status == "filled"
    assert result.order_id == "order-123"
    assert result.avg_fill_price == pytest.approx(0.75)
    assert result.filled_shares == pytest.approx(100.0, abs=0.01)
    assert result.error_message is None


def test_place_order_partial_fill() -> None:
    """Partial fill: makingAmount covers only part of size_shares."""
    executor = _make_executor()
    # 3.25 USDC at $0.65 → 5 shares filled, size=10 → "partial"
    executor._client.create_order = MagicMock(return_value=MagicMock())
    executor._client.post_order = MagicMock(return_value={
        "success": True,
        "orderID": "order-789",
        "makingAmount": "3.25",
        "price": "0.65",
    })

    result = asyncio.run(executor.place_order("token-abc", price=0.65, size_shares=10.0))

    assert result.fill_status == "partial"
    assert result.filled_shares == pytest.approx(5.0, abs=0.01)


def test_place_order_zero_fill() -> None:
    """Accepted but zero fill → partial with error_message='zero fill'."""
    executor = _make_executor()
    executor._client.create_order = MagicMock(return_value=MagicMock())
    executor._client.post_order = MagicMock(return_value={
        "success": True,
        "orderID": "order-456",
        "makingAmount": "0",
        "takingAmount": "0",
    })

    result = asyncio.run(executor.place_order("token-abc", 0.65, 10.0))

    assert result.fill_status == "partial"
    assert result.error_message == "zero fill"
    assert result.filled_shares == 0.0


def test_place_order_rejected() -> None:
    """Server rejects the order."""
    executor = _make_executor()
    executor._client.create_order = MagicMock(return_value=MagicMock())
    executor._client.post_order = MagicMock(return_value={
        "success": False,
        "errorMsg": "insufficient balance",
    })

    result = asyncio.run(executor.place_order("token-abc", 0.65, 10.0))

    assert result.fill_status == "rejected"
    assert result.avg_fill_price is None
    assert result.error_message == "insufficient balance"


def test_place_order_sdk_exception() -> None:
    """SDK raises during order creation → error result, no crash."""
    executor = _make_executor()
    executor._client.create_order = MagicMock(side_effect=RuntimeError("connection refused"))

    result = asyncio.run(executor.place_order("token-abc", 0.65, 10.0))

    assert result.fill_status == "error"
    assert result.error_message == "connection refused"
    assert result.filled_shares == 0.0
    assert result.order_id is None


def test_place_order_uses_taking_amount_fallback() -> None:
    """makingAmount=0 → fall back to takingAmount for share calculation."""
    executor = _make_executor()
    executor._client.create_order = MagicMock(return_value=MagicMock())
    # 3.25 USDC at $0.65 → 5 shares filled out of 10 requested → "partial"
    executor._client.post_order = MagicMock(return_value={
        "success": True,
        "orderID": "order-999",
        "makingAmount": "0",
        "takingAmount": "3.25",
        "price": "0.65",
    })

    result = asyncio.run(executor.place_order("token-abc", 0.65, 10.0))

    assert result.fill_status == "partial"
    assert result.filled_shares == pytest.approx(5.0, abs=0.01)


@patch("src.live_executor.ClobClient")
def test_get_balance_returns_float(mock_clob_class) -> None:
    """Balance returned as USDC float; BalanceAllowanceParams passed to SDK."""
    mock_client = MagicMock()
    mock_client.get_balance_allowance = MagicMock(return_value={"balance": "10000000"})
    mock_client.derive_api_key = MagicMock(return_value=MagicMock())
    mock_clob_class.return_value = mock_client

    with patch("src.live_executor.config") as mock_config:
        mock_config.WALLET_PRIVATE_KEY = "k"
        mock_config.WALLET_FUNDER = "f"
        mock_config.WALLET_ADDRESS = "a"
        mock_config.CHAIN_ID = 137
        mock_config.CLOB_HOST = "h"

        executor = LiveExecutor()
        balance = asyncio.run(executor.get_balance())

        assert balance == pytest.approx(10.0)
        args, kwargs = mock_client.get_balance_allowance.call_args
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        assert len(args) == 1
        assert isinstance(args[0], BalanceAllowanceParams)
        assert args[0].asset_type == AssetType.COLLATERAL


def test_get_balance_failure() -> None:
    """RPC error → returns 0.0, no crash."""
    executor = _make_executor()
    executor._client.get_balance_allowance = MagicMock(side_effect=RuntimeError("rpc error"))

    balance = asyncio.run(executor.get_balance())

    assert balance == 0.0


@patch("src.live_executor.ClobClient")
def test_get_balance_initializes_creds(mock_clob_class) -> None:
    """get_balance should call _ensure_api_creds before fetching balance."""
    mock_client = MagicMock()
    mock_client.get_balance_allowance = MagicMock(return_value={"balance": "5000000"})
    mock_client.derive_api_key = MagicMock(return_value=MagicMock())
    mock_clob_class.return_value = mock_client

    with patch("src.live_executor.config") as mock_config:
        mock_config.WALLET_PRIVATE_KEY = "k"
        mock_config.WALLET_FUNDER = "f"
        mock_config.WALLET_ADDRESS = "a"
        mock_config.CHAIN_ID = 137
        mock_config.CLOB_HOST = "h"

        executor = LiveExecutor()
        assert executor._api_creds is None

        balance = asyncio.run(executor.get_balance())

        assert executor._api_creds is not None
        assert balance == pytest.approx(5.0)
        mock_client.derive_api_key.assert_called_once()


def test_order_result_dataclass() -> None:
    """LiveOrderResult fields are accessible and optional fields default to None."""
    r = LiveOrderResult(
        order_id=None,
        fill_status="error",
        avg_fill_price=None,
        filled_shares=0.0,
    )
    assert r.error_message is None
    assert r.filled_shares == 0.0


def test_compute_clean_order_amounts_basic() -> None:
    """$5 stake at 0.70: shares ≤ 4 decimals AND shares×price ≤ 2 decimals."""
    shares, notional = compute_clean_order_amounts(5.0, 0.70)
    assert round(shares, 4) == shares
    expected_notional = round(shares * 0.70, 2)
    assert abs(shares * 0.70 - expected_notional) < 1e-9
    assert notional == expected_notional


def test_compute_clean_order_amounts_various_prices() -> None:
    """Both constraints satisfied across all prices in our trading range."""
    for price in [0.70, 0.75, 0.78, 0.85, 0.95, 0.97, 0.99]:
        shares, notional = compute_clean_order_amounts(5.0, price)
        assert round(shares, 4) == shares, f"shares not 4-decimal at price {price}"
        product = shares * price
        assert abs(product - round(product, 2)) < 1e-9, f"notional not 2-decimal at price {price}"
        assert notional <= 5.01, f"notional ${notional:.4f} exceeds stake at price {price}"
        assert notional >= 4.50, f"notional ${notional:.4f} too low at price {price}"


def test_compute_clean_order_amounts_no_decimal_overflow() -> None:
    """Specifically test the failing production case: $5 / 0.70 must not overflow to 3 decimals."""
    shares, notional = compute_clean_order_amounts(5.0, 0.70)
    product = shares * 0.70
    cents = round(product * 100)
    reconstructed = cents / 100.0
    assert abs(product - reconstructed) < 1e-9, f"Product {product} has fractional cents"
