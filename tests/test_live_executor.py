"""Unit tests for src/live_executor.py — all offline (no network calls)."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from src.live_executor import LiveExecutor, LiveOrderResult, compute_clean_order_amounts

# Deterministic dummy private key (valid 32-byte key for eth_account)
_DUMMY_KEY = "0x" + "ab" * 32
_DUMMY_EOA = "0xaB5801a7D398351b8bE11C439e05C5B3259aeC9B"  # derived from dummy key
_DUMMY_FUNDER = "0x1234567890123456789012345678901234567890"


def _make_executor() -> LiveExecutor:
    """Build a LiveExecutor with patched config and pre-seeded API creds."""
    with patch("src.live_executor.config") as mock_cfg:
        mock_cfg.WALLET_PRIVATE_KEY = _DUMMY_KEY
        mock_cfg.WALLET_ADDRESS = _DUMMY_EOA
        mock_cfg.WALLET_FUNDER = _DUMMY_FUNDER
        mock_cfg.CLOB_HOST = "https://clob.polymarket.com"
        executor = LiveExecutor()

    # Skip credential derivation for unit tests
    executor._api_creds = {
        "api_key": "test-api-key",
        "secret": "dGVzdHNlY3JldA==",  # base64("testsecret")
        "passphrase": "test-passphrase",
    }
    return executor


def _mock_post_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.status_code = 200
    return resp


def _mock_get_response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.status_code = status
    return resp


# ── place_order tests ──────────────────────────────────────────────────────────

def test_place_order_filled() -> None:
    """Full fill: makingAmount at price gives shares == size_shares → 'filled'."""
    executor = _make_executor()
    api_resp = {
        "success": True,
        "orderID": "order-123",
        "makingAmount": "75",
        "price": "0.75",
    }
    with patch.object(executor._session, "post", return_value=_mock_post_response(api_resp)):
        result = asyncio.run(executor.place_order("12345", price=0.75, size_shares=100.0))

    assert result.fill_status == "filled"
    assert result.order_id == "order-123"
    assert result.avg_fill_price == pytest.approx(0.75)
    assert result.filled_shares == pytest.approx(100.0, abs=0.01)
    assert result.error_message is None


def test_place_order_partial_fill() -> None:
    """Partial fill: makingAmount / price < size_shares → 'partial'."""
    executor = _make_executor()
    api_resp = {
        "success": True,
        "orderID": "order-789",
        "makingAmount": "3.25",
        "price": "0.65",
    }
    with patch.object(executor._session, "post", return_value=_mock_post_response(api_resp)):
        result = asyncio.run(executor.place_order("12345", price=0.65, size_shares=10.0))

    assert result.fill_status == "partial"
    assert result.filled_shares == pytest.approx(5.0, abs=0.01)


def test_place_order_zero_fill() -> None:
    """Accepted but zero fill → partial with error_message='zero fill'."""
    executor = _make_executor()
    api_resp = {
        "success": True,
        "orderID": "order-456",
        "makingAmount": "0",
        "takingAmount": "0",
    }
    with patch.object(executor._session, "post", return_value=_mock_post_response(api_resp)):
        result = asyncio.run(executor.place_order("12345", 0.65, 10.0))

    assert result.fill_status == "partial"
    assert result.error_message == "zero fill"
    assert result.filled_shares == 0.0


def test_place_order_rejected() -> None:
    """Server rejects the order."""
    executor = _make_executor()
    api_resp = {"success": False, "errorMsg": "insufficient balance"}
    with patch.object(executor._session, "post", return_value=_mock_post_response(api_resp)):
        result = asyncio.run(executor.place_order("12345", 0.65, 10.0))

    assert result.fill_status == "rejected"
    assert result.avg_fill_price is None
    assert result.error_message == "insufficient balance"


def test_place_order_network_exception() -> None:
    """Network error during POST → error result, no crash."""
    executor = _make_executor()
    with patch.object(executor._session, "post", side_effect=RuntimeError("connection refused")):
        result = asyncio.run(executor.place_order("12345", 0.65, 10.0))

    assert result.fill_status == "error"
    assert "connection refused" in result.error_message
    assert result.filled_shares == 0.0
    assert result.order_id is None


def test_place_order_uses_taking_amount_fallback() -> None:
    """makingAmount=0 → fall back to takingAmount for fill calculation."""
    executor = _make_executor()
    api_resp = {
        "success": True,
        "orderID": "order-999",
        "makingAmount": "0",
        "takingAmount": "3.25",
        "price": "0.65",
    }
    with patch.object(executor._session, "post", return_value=_mock_post_response(api_resp)):
        result = asyncio.run(executor.place_order("12345", 0.65, 10.0))

    assert result.fill_status == "partial"
    assert result.filled_shares == pytest.approx(5.0, abs=0.01)


def test_place_order_body_is_valid_json() -> None:
    """Verify the POST body is valid JSON with V2 fields."""
    executor = _make_executor()
    captured_body = {}

    def mock_post(url, headers=None, data=None, **kwargs):
        captured_body.update(json.loads(data))
        return _mock_post_response({"success": True, "orderID": "x", "makingAmount": "5", "price": "0.75"})

    with patch.object(executor._session, "post", side_effect=mock_post):
        asyncio.run(executor.place_order("99999", price=0.75, size_shares=6.6667))

    order = captured_body["order"]
    assert captured_body["orderType"] == "FAK"
    assert captured_body["owner"] == "test-api-key"
    assert order["side"] == "BUY"
    assert order["signatureType"] == 2
    assert order["maker"] == _DUMMY_FUNDER
    assert order["signer"] == _DUMMY_EOA
    assert order["signature"].startswith("0x")
    assert len(order["signature"]) == 132  # 0x + 65 bytes × 2 hex chars
    # V2: taker/nonce/feeRateBps absent; timestamp/metadata/builder present
    assert "taker" not in order
    assert "nonce" not in order
    assert "feeRateBps" not in order
    assert isinstance(order["timestamp"], str) and int(order["timestamp"]) > 0
    assert order["metadata"] == "0x" + "00" * 32
    assert order["builder"] == "0x" + "00" * 32


def test_place_order_neg_risk_uses_different_exchange() -> None:
    """neg_risk=True uses the NegRisk exchange address in EIP-712 domain."""
    executor = _make_executor()
    from src.live_executor import EXCHANGE_ADDRESS, NEG_RISK_EXCHANGE_ADDRESS

    signatures = {}

    def capture_sign(domain_data, message_types, message_data):
        key = domain_data["verifyingContract"]
        signatures[key] = True
        from eth_account import Account
        acct = Account.from_key(_DUMMY_KEY)
        return acct.sign_typed_data(
            domain_data=domain_data,
            message_types=message_types,
            message_data=message_data,
        )

    with patch.object(executor._account, "sign_typed_data", side_effect=capture_sign):
        with patch.object(executor._session, "post", return_value=_mock_post_response(
            {"success": True, "orderID": "x", "makingAmount": "5", "price": "0.75"}
        )):
            asyncio.run(executor.place_order("99999", 0.75, 6.0, neg_risk=False))
            asyncio.run(executor.place_order("99999", 0.75, 6.0, neg_risk=True))

    assert EXCHANGE_ADDRESS in signatures
    assert NEG_RISK_EXCHANGE_ADDRESS in signatures


# ── get_balance tests ──────────────────────────────────────────────────────────

def test_get_balance_returns_float() -> None:
    """Balance raw value 10_000_000 (micro-USDC) → 10.0 USDC."""
    executor = _make_executor()
    api_resp = {"balance": "10000000"}
    with patch.object(executor._session, "get", return_value=_mock_get_response(api_resp)):
        balance = asyncio.run(executor.get_balance())

    assert balance == pytest.approx(10.0)


def test_get_balance_failure() -> None:
    """Network error → returns 0.0, no crash."""
    executor = _make_executor()
    with patch.object(executor._session, "get", side_effect=RuntimeError("rpc error")):
        balance = asyncio.run(executor.get_balance())

    assert balance == 0.0


def test_get_balance_initializes_creds() -> None:
    """get_balance triggers _ensure_api_creds when creds are missing."""
    with patch("src.live_executor.config") as mock_cfg:
        mock_cfg.WALLET_PRIVATE_KEY = _DUMMY_KEY
        mock_cfg.WALLET_ADDRESS = _DUMMY_EOA
        mock_cfg.WALLET_FUNDER = _DUMMY_FUNDER
        mock_cfg.CLOB_HOST = "https://clob.polymarket.com"
        executor = LiveExecutor()

    assert executor._api_creds is None

    derive_resp = _mock_get_response({
        "apiKey": "k",
        "secret": "dGVzdA==",
        "passphrase": "p",
    }, status=200)
    balance_resp = _mock_get_response({"balance": "5000000"})

    call_count = [0]
    def side_effect(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return derive_resp
        return balance_resp

    with patch.object(executor._session, "get", side_effect=side_effect):
        balance = asyncio.run(executor.get_balance())

    assert executor._api_creds is not None
    assert executor._api_creds["api_key"] == "k"
    assert balance == pytest.approx(5.0)


# ── dataclass / utility tests (unchanged) ─────────────────────────────────────

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
    """$5 / 0.70 must not overflow to fractional cents."""
    shares, notional = compute_clean_order_amounts(5.0, 0.70)
    product = shares * 0.70
    cents = round(product * 100)
    reconstructed = cents / 100.0
    assert abs(product - reconstructed) < 1e-9, f"Product {product} has fractional cents"
