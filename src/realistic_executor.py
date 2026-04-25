"""Realistic execution simulator: adds latency + orderbook re-fetch to paper trades.

For every entry the paper trader fires, this module sleeps `SIMULATED_LATENCY_MS`,
fetches the live orderbook from Polymarket's REST API, and computes what would
have actually filled at three stake sizes ($1, $5, $25). Records both the paper
prices and realistic prices for later EV comparison.
"""

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Simulated order placement latency (ms). Realistic for HTTP roundtrip + Polymarket order routing.
SIMULATED_LATENCY_MS = 300

# Stake sizes to simulate per trade (USDC). At each size we walk the orderbook
# to compute weighted-average fill price.
SIMULATED_STAKES_USDC = [1.0, 5.0, 25.0]

# Polymarket CLOB orderbook endpoint
_BOOK_URL = "https://clob.polymarket.com/book"


class RealisticFill:
    """Result of simulating a realistic fill at a given stake size."""

    def __init__(
        self,
        stake_usdc: float,
        weighted_avg_price: Optional[float],
        shares_filled: float,
        out_of_bucket: bool,
    ) -> None:
        self.stake_usdc = stake_usdc
        self.weighted_avg_price = weighted_avg_price
        self.shares_filled = shares_filled
        self.out_of_bucket = out_of_bucket


def walk_orderbook(asks: list[dict], stake_usdc: float) -> tuple[Optional[float], float]:
    """Walk the asks (sorted ascending by price) and compute weighted-avg fill price.

    Args:
        asks: List of {"price": str, "size": str} from Polymarket book API.
        stake_usdc: Total USDC we want to deploy.

    Returns:
        (weighted_avg_price, total_shares_filled) — or (None, 0) if book is empty.
    """
    if not asks:
        return (None, 0.0)

    sorted_asks = sorted(asks, key=lambda x: float(x["price"]))
    remaining_usdc = stake_usdc
    total_shares = 0.0
    total_cost = 0.0

    for level in sorted_asks:
        try:
            price = float(level["price"])
            size_avail = float(level["size"])
        except (KeyError, ValueError, TypeError):
            continue

        if remaining_usdc <= 0:
            break
        if size_avail <= 0:
            continue

        cost_at_level = price * size_avail
        if cost_at_level <= remaining_usdc:
            # Take the whole level
            total_shares += size_avail
            total_cost += cost_at_level
            remaining_usdc -= cost_at_level
        else:
            # Take partial — only what fits in remaining_usdc
            partial_shares = remaining_usdc / price
            total_shares += partial_shares
            total_cost += remaining_usdc
            remaining_usdc = 0
            break

    if total_shares == 0:
        return (None, 0.0)

    weighted_avg = total_cost / total_shares
    return (weighted_avg, total_shares)


class RealisticExecutor:
    """Simulates realistic execution: latency, orderbook re-fetch, slippage."""

    def __init__(self, latency_ms: int = SIMULATED_LATENCY_MS) -> None:
        self.latency_ms = latency_ms

    async def simulate_fill(
        self,
        token_id: str,
        signal_min_ask: float,
        signal_max_ask: float,
        stake_sizes: Optional[list[float]] = None,
    ) -> dict[float, RealisticFill]:
        """Sleep for latency_ms, then fetch the live orderbook and compute fills.

        Args:
            token_id: Polymarket asset/token ID.
            signal_min_ask: Lower bound of the signal's price bucket.
            signal_max_ask: Upper bound of the signal's price bucket.
            stake_sizes: List of USDC sizes to simulate. Defaults to SIMULATED_STAKES_USDC.

        Returns:
            Dict mapping stake_size -> RealisticFill.
        """
        if stake_sizes is None:
            stake_sizes = SIMULATED_STAKES_USDC

        # Sleep to simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Fetch live orderbook
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(_BOOK_URL, params={"token_id": token_id})
                resp.raise_for_status()
                book = resp.json()
        except Exception as exc:
            logger.warning("Realistic fill: book fetch failed for %s: %s", token_id, exc)
            return {s: RealisticFill(s, None, 0.0, True) for s in stake_sizes}

        asks = book.get("asks", [])

        # Determine if best ask is still in the bucket
        if asks:
            sorted_asks = sorted(asks, key=lambda x: float(x.get("price", 0)))
            try:
                best_ask = float(sorted_asks[0]["price"])
            except (KeyError, ValueError, TypeError):
                best_ask = None
        else:
            best_ask = None

        out_of_bucket = (
            best_ask is None
            or best_ask < signal_min_ask
            or best_ask >= signal_max_ask
        )

        # Compute fill for each stake size
        results = {}
        for stake in stake_sizes:
            avg_price, shares = walk_orderbook(asks, stake)
            results[stake] = RealisticFill(
                stake_usdc=stake,
                weighted_avg_price=avg_price,
                shares_filled=shares,
                out_of_bucket=out_of_bucket,
            )

        return results
