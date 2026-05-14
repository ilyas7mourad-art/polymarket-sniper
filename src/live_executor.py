"""Live executor for Polymarket: places real FAK orders via direct REST + EIP-712.

Replaces py-clob-client (broken since Polymarket CLOB V2 migration April 28 2026).
Uses eth_account for EIP-712 signing and requests for HTTP — no external Polymarket SDK.
Implements CLOB V2 order struct (timestamp replaces nonce; taker/nonce/feeRateBps removed).
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import requests
from eth_account import Account

from src.config import config

logger = logging.getLogger(__name__)

# ── Contract addresses (Polygon mainnet, CLOB V2 — active from April 28 2026) ──
EXCHANGE_ADDRESS = "0xE111180000d2663C0091e4f400237545B87B996B"
NEG_RISK_EXCHANGE_ADDRESS = "0xe2222d279d744050d28e00520010520000310F59"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
_ZERO_BYTES32 = b"\x00" * 32  # metadata / builder default

# ── EIP-712 types (CLOB V2) ────────────────────────────────────────────────────
# Domain version bumped to "2" in V2; ClobAuthDomain stays at "1" (unchanged).
_ORDER_DOMAIN_BASE = {
    "name": "Polymarket CTF Exchange",
    "version": "2",
    "chainId": 137,
}

_ORDER_TYPES = {
    "Order": [
        {"name": "salt", "type": "uint256"},
        {"name": "maker", "type": "address"},
        {"name": "signer", "type": "address"},
        {"name": "tokenId", "type": "uint256"},
        {"name": "makerAmount", "type": "uint256"},
        {"name": "takerAmount", "type": "uint256"},
        {"name": "side", "type": "uint8"},
        {"name": "signatureType", "type": "uint8"},
        {"name": "timestamp", "type": "uint256"},
        {"name": "metadata", "type": "bytes32"},
        {"name": "builder", "type": "bytes32"},
    ]
}

_AUTH_DOMAIN = {"name": "ClobAuthDomain", "version": "1", "chainId": 137}

_AUTH_TYPES = {
    "ClobAuth": [
        {"name": "address", "type": "address"},
        {"name": "timestamp", "type": "string"},
        {"name": "nonce", "type": "uint256"},
        {"name": "message", "type": "string"},
    ]
}

_MSG_TO_SIGN = "This message attests that I control the given wallet"

# signatureType: 1 = POLY_PROXY (Polymarket browser wallets)
#                2 = POLY_GNOSIS_SAFE (Gnosis Safe — use for this wallet)
_POLY_GNOSIS_SAFE = 2


def compute_clean_order_amounts(stake_usdc: float, price: float) -> tuple[float, float]:
    """Return (size_shares, actual_notional) satisfying Polymarket's decimal constraints.

    Polymarket requires:
      - shares (taker amount) ≤ 4 decimal places
      - shares × price (maker USDC) ≤ 2 decimal places
    """
    p = round(price * 100)
    stake_cents = round(stake_usdc * 100)

    step = 10000 // math.gcd(p, 10000)
    max_s = (stake_cents * 10000) // p
    s = (max_s // step) * step

    if s <= 0:
        return (round(stake_usdc / price, 4), round(stake_usdc, 2))

    size_shares = s / 10000
    actual_notional = round(size_shares * price, 2)
    return (size_shares, actual_notional)


@dataclass
class LiveOrderResult:
    """Result of attempting to place a live order on Polymarket."""

    order_id: Optional[str]
    fill_status: str  # "filled", "partial", "rejected", "error"
    avg_fill_price: Optional[float]
    filled_shares: float
    error_message: Optional[str] = None


class LiveExecutor:
    """Places real FAK orders on Polymarket via direct REST API calls.

    Authentication:
    - L1 (API key derivation): EIP-712 ClobAuth signature with EOA key
    - L2 (order posting): HMAC-SHA256 with API secret

    Order signing: EIP-712 Order struct, signed by EOA, with Gnosis Safe proxy as maker.
    """

    def __init__(self) -> None:
        self._account = Account.from_key(config.WALLET_PRIVATE_KEY)
        self._eoa = config.WALLET_ADDRESS
        self._funder = config.WALLET_FUNDER  # Gnosis Safe proxy — the onchain maker
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._api_creds: Optional[dict] = None  # {api_key, secret, passphrase}
        self._creds_lock = asyncio.Lock()

    # ── Signing helpers ────────────────────────────────────────────────────────

    def _sign_order(self, order_data: dict, neg_risk: bool = False) -> str:
        """EIP-712 sign an Order struct. Returns hex signature without 0x prefix."""
        exchange = NEG_RISK_EXCHANGE_ADDRESS if neg_risk else EXCHANGE_ADDRESS
        domain = {**_ORDER_DOMAIN_BASE, "verifyingContract": exchange}
        signed = self._account.sign_typed_data(
            domain_data=domain,
            message_types=_ORDER_TYPES,
            message_data=order_data,
        )
        return signed.signature.hex()

    def _sign_clob_auth(self, timestamp: int, nonce: int) -> str:
        """EIP-712 sign a ClobAuth struct for API key derivation. Returns 0x-prefixed hex."""
        value = {
            "address": self._eoa,
            "timestamp": str(timestamp),
            "nonce": nonce,
            "message": _MSG_TO_SIGN,
        }
        signed = self._account.sign_typed_data(
            domain_data=_AUTH_DOMAIN,
            message_types=_AUTH_TYPES,
            message_data=value,
        )
        return "0x" + signed.signature.hex()

    def _l1_headers(self, nonce: int = 0) -> dict:
        ts = int(time.time())
        return {
            "POLY_ADDRESS": self._eoa,
            "POLY_SIGNATURE": self._sign_clob_auth(ts, nonce),
            "POLY_TIMESTAMP": str(ts),
            "POLY_NONCE": str(nonce),
        }

    def _l2_headers(self, method: str, path: str, body: str = "") -> dict:
        ts = int(time.time())
        secret_bytes = base64.urlsafe_b64decode(self._api_creds["secret"])
        message = str(ts) + method + path + body
        sig = base64.urlsafe_b64encode(
            hmac.new(secret_bytes, message.encode(), hashlib.sha256).digest()
        ).decode()
        return {
            "POLY_ADDRESS": self._eoa,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": str(ts),
            "POLY_API_KEY": self._api_creds["api_key"],
            "POLY_PASSPHRASE": self._api_creds["passphrase"],
        }

    # ── API credential management ──────────────────────────────────────────────

    async def _ensure_api_creds(self) -> None:
        if self._api_creds is not None:
            return
        async with self._creds_lock:
            if self._api_creds is not None:
                return
            # Try to derive existing key first (/auth/derive-api-key)
            headers = self._l1_headers()
            resp = await asyncio.to_thread(
                self._session.get,
                f"{config.CLOB_HOST}/auth/derive-api-key",
                headers=headers,
            )
            if resp.status_code != 200:
                # No key yet — create one (/auth/api-key POST)
                headers = self._l1_headers()
                resp = await asyncio.to_thread(
                    self._session.post,
                    f"{config.CLOB_HOST}/auth/api-key",
                    headers=headers,
                )
                resp.raise_for_status()
            data = resp.json()
            self._api_creds = {
                "api_key": data["apiKey"],
                "secret": data["secret"],
                "passphrase": data["passphrase"],
            }
            logger.info("API credentials ready for %s", self._eoa)

    # ── Order placement ────────────────────────────────────────────────────────

    async def place_order(
        self,
        token_id: str,
        price: float,
        size_shares: float,
        side: str = "BUY",
        neg_risk: bool = False,
        order_type: str = "FAK",
    ) -> LiveOrderResult:
        """Place an order.

        Args:
            token_id: Polymarket asset/token ID.
            price: Limit price (0–1).
            size_shares: Number of shares to buy.
            side: "BUY" or "SELL".
            neg_risk: True for multi-outcome (neg-risk) markets.
            order_type: "FAK", "GTC", "GTD", or "FOK".

        Returns:
            LiveOrderResult with fill details.
        """
        await self._ensure_api_creds()

        # Amounts in micro-units (6 decimals): USDC × 1e6
        # BUY: maker gives USDC (makerAmount), receives shares (takerAmount)
        maker_amount = int(round(price * size_shares * 1_000_000))
        taker_amount = int(round(size_shares * 1_000_000))
        # Salt must fit in JS safe-integer range (2^53-1); mirrors TS SDK's
        # Math.round(Math.random() * Date.now()) which tops out at ~1.7e12
        salt = random.randint(1, 9_007_199_254_740_991)
        side_int = 0 if side == "BUY" else 1
        # V2: timestamp (ms) replaces nonce for per-address order uniqueness
        timestamp_ms = int(time.time() * 1000)

        # V2 Order struct: taker/nonce/feeRateBps/expiration removed; timestamp/metadata/builder added
        order_eip712 = {
            "salt": salt,
            "maker": self._funder,
            "signer": self._eoa,
            "tokenId": int(token_id),
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            "side": side_int,
            "signatureType": _POLY_GNOSIS_SAFE,
            "timestamp": timestamp_ms,
            "metadata": _ZERO_BYTES32,
            "builder": _ZERO_BYTES32,
        }

        try:
            signature = await asyncio.to_thread(self._sign_order, order_eip712, neg_risk)
        except Exception as exc:
            logger.warning("Order signing failed: %s", exc)
            return LiveOrderResult(
                order_id=None,
                fill_status="error",
                avg_fill_price=None,
                filled_shares=0.0,
                error_message=f"signing: {exc}",
            )

        body = {
            "deferExec": False,
            "order": {
                "salt": salt,
                "maker": self._funder,
                "signer": self._eoa,
                "tokenId": str(int(token_id)),
                "makerAmount": str(maker_amount),
                "takerAmount": str(taker_amount),
                "side": side,
                "signatureType": _POLY_GNOSIS_SAFE,
                "signature": "0x" + signature,
                # V2 REST API: timestamp and expiration are strings; metadata is "".
                # EIP-712 signs timestamp as uint256 (int) — these are serialisation-only changes.
                "timestamp": str(timestamp_ms),
                "expiration": "0",
                "metadata": "0x" + "00" * 32,
                "builder": "0x" + "00" * 32,
            },
            "owner": self._api_creds["api_key"],
            "orderType": order_type,
        }
        body_str = json.dumps(body, separators=(",", ":"))
        headers = self._l2_headers("POST", "/order", body_str)

        try:
            resp = await asyncio.to_thread(
                self._session.post,
                f"{config.CLOB_HOST}/order",
                headers=headers,
                data=body_str,
            )
            response = resp.json()
            logger.debug("POST /order HTTP %s: %s", resp.status_code, response)
        except Exception as exc:
            logger.warning("Order POST failed for token %s: %s", token_id[:12], exc)
            return LiveOrderResult(
                order_id=None,
                fill_status="error",
                avg_fill_price=None,
                filled_shares=0.0,
                error_message=str(exc),
            )

        success = response.get("success", False)
        order_id = response.get("orderID") or response.get("orderId")

        if not success:
            # Server uses "errorMsg" for app-level rejects, "error" for 4xx payload errors
            err = response.get("errorMsg") or response.get("error") or "unknown error"
            logger.warning("Order rejected by Polymarket: %s", err)
            return LiveOrderResult(
                order_id=order_id,
                fill_status="rejected",
                avg_fill_price=None,
                filled_shares=0.0,
                error_message=err,
            )

        matched_amount = float(response.get("makingAmount", 0) or 0) or float(
            response.get("takingAmount", 0) or 0
        )
        if matched_amount == 0:
            return LiveOrderResult(
                order_id=order_id,
                fill_status="partial",
                avg_fill_price=None,
                filled_shares=0.0,
                error_message="zero fill",
            )

        filled_shares = matched_amount / price if price > 0 else 0.0
        avg_price = price
        actual_price = response.get("price")
        if actual_price is not None:
            try:
                avg_price = float(actual_price)
            except (ValueError, TypeError):
                pass

        status = "filled" if abs(filled_shares - size_shares) < 0.01 else "partial"
        logger.info(
            "ORDER %s: %s %.4f @ %.4f, filled %.2f shares (status=%s, id=%s)",
            side,
            token_id[:12],
            size_shares,
            avg_price,
            filled_shares,
            status,
            order_id,
        )
        return LiveOrderResult(
            order_id=order_id,
            fill_status=status,
            avg_fill_price=avg_price,
            filled_shares=filled_shares,
            error_message=None,
        )

    # ── Balance ────────────────────────────────────────────────────────────────

    async def get_balance(self) -> float:
        """Return USDC balance of the proxy wallet."""
        await self._ensure_api_creds()
        path = "/balance-allowance"
        params = {
            "asset_type": "COLLATERAL",
            "signature_type": 2,  # wallet lookup type: 2 = Gnosis Safe proxy (wallet creation type, not order sig type)
            "proxy_wallet": self._funder,
        }
        headers = self._l2_headers("GET", path)
        full_url = f"{config.CLOB_HOST}{path}?" + "&".join(f"{k}={v}" for k, v in params.items())
        logger.debug("get_balance GET %s", full_url)
        try:
            resp = await asyncio.to_thread(
                self._session.get,
                f"{config.CLOB_HOST}{path}",
                headers=headers,
                params=params,
            )
            data = resp.json()
        except Exception as exc:
            logger.warning("Balance fetch failed: %s", exc)
            return 0.0
        try:
            return float(data.get("balance", "0")) / 1_000_000
        except (ValueError, TypeError):
            return 0.0
