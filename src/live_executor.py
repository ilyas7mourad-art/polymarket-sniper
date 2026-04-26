"""Live executor for Polymarket: places real FAK orders via py-clob-client SDK."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
)

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class LiveOrderResult:
    """Result of attempting to place a live order on Polymarket."""

    order_id: Optional[str]
    fill_status: str  # "filled", "partial", "rejected", "error"
    avg_fill_price: Optional[float]
    filled_shares: float
    error_message: Optional[str] = None


class LiveExecutor:
    """Wraps py-clob-client for FAK order placement.

    Adds retry-friendly structured results and per-order logging on top of the
    raw SDK calls.
    """

    def __init__(self) -> None:
        self._client = ClobClient(
            host=config.CLOB_HOST,
            chain_id=config.CHAIN_ID,
            key=config.WALLET_PRIVATE_KEY,
            signature_type=2,  # POLY_GNOSIS_SAFE
            funder=config.WALLET_FUNDER,
        )
        self._api_creds: Optional[ApiCreds] = None
        self._creds_lock = asyncio.Lock()

    async def _ensure_api_creds(self) -> None:
        """Lazily derive API credentials from the wallet (cached after first call)."""
        if self._api_creds is not None:
            return
        async with self._creds_lock:
            if self._api_creds is not None:
                return
            try:
                creds = await asyncio.to_thread(self._client.derive_api_key)
            except Exception:
                creds = await asyncio.to_thread(self._client.create_or_derive_api_creds)
            self._client.set_api_creds(creds)
            self._api_creds = creds
            logger.info("API credentials initialized for wallet %s", config.WALLET_ADDRESS)

    async def place_order(
        self,
        token_id: str,
        price: float,
        size_shares: float,
        side: str = "BUY",
    ) -> LiveOrderResult:
        """Place a FAK (Fill-and-Kill) order.

        Args:
            token_id: Polymarket asset/token ID.
            price: Limit price.
            size_shares: Number of shares to buy.
            side: "BUY" or "SELL".

        Returns:
            LiveOrderResult with fill details.
        """
        await self._ensure_api_creds()

        order_args = OrderArgs(
            price=price,
            size=size_shares,
            side=side,
            token_id=token_id,
        )

        try:
            signed_order = await asyncio.to_thread(self._client.create_order, order_args)
            response = await asyncio.to_thread(
                self._client.post_order,
                signed_order,
                OrderType.FAK,
            )
        except Exception as exc:
            logger.warning("Order placement failed for token %s: %s", token_id, exc)
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
            err = response.get("errorMsg", "unknown error")
            logger.warning("Order rejected by Polymarket: %s", err)
            return LiveOrderResult(
                order_id=order_id,
                fill_status="rejected",
                avg_fill_price=None,
                filled_shares=0.0,
                error_message=err,
            )

        # FAK responses report fills via makingAmount (shares) or takingAmount (USDC)
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
            "ORDER %s: %s %.4f @ %.4f, filled %.2f shares (status=%s, order_id=%s)",
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

    async def get_balance(self) -> float:
        """Return USDC balance of the wallet as a float."""
        await self._ensure_api_creds()
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        try:
            balance_resp = await asyncio.to_thread(
                self._client.get_balance_allowance, params
            )
        except Exception as exc:
            logger.warning("Balance fetch failed: %s", exc)
            return 0.0
        balance_raw = balance_resp.get("balance", "0")
        try:
            return float(balance_raw) / 1e6
        except (ValueError, TypeError):
            return 0.0
