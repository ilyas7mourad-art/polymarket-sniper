"""Scans Polymarket for eligible BTC/ETH 5-minute up/down markets."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from src.config import config

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Detects asset only — times come from structured API fields, not the question.
_ASSET_RE = re.compile(r"^(Bitcoin|Ethereum) Up or Down")

_ASSET_MAP = {"Bitcoin": "BTC", "Ethereum": "ETH"}


@dataclass(frozen=True)
class Market:
    """A single active BTC/ETH 5-minute up/down Polymarket market.

    Attributes:
        condition_id: Hex string uniquely identifying the market.
        question: Raw question string from the Gamma API.
        asset: "BTC" or "ETH".
        start_time: Window open time (eventStartTime), timezone-aware UTC.
        end_time: Window close time (endDate), timezone-aware UTC.
        up_token_id: CLOB token ID for the Up/Yes outcome.
        down_token_id: CLOB token ID for the Down/No outcome.
        slug: Market URL slug.
        raw: Original Gamma API response dict, for debugging.
    """

    condition_id: str
    question: str
    asset: str
    start_time: datetime
    end_time: datetime
    up_token_id: str
    down_token_id: str
    slug: str
    raw: dict


def _parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp. Accepts trailing 'Z'.

    Args:
        s: ISO-8601 string, e.g. "2026-04-17T19:00:00Z".

    Returns:
        Timezone-aware UTC datetime.
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch_active_markets(limit: int = 500) -> list[dict]:
    """Fetch active, non-closed markets from Gamma API.

    Args:
        limit: Maximum number of markets to request.

    Returns:
        List of raw market dicts. Empty list on any error.
    """
    url = (
        f"{config.GAMMA_URL}/markets"
        f"?limit={limit}&active=true&closed=false&order=endDate&ascending=true"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning("Gamma API returned %s: %s", resp.status_code, resp.text[:200])
            return []
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch markets: %s", exc)
        return []


def is_btc_eth_5min_window(
    raw: dict,
    reference_date: datetime | None = None,
) -> bool:
    """Return True iff the market dict represents a BTC/ETH 5-minute up/down window.

    Args:
        raw: A single market dict from the Gamma API response.
        reference_date: UTC datetime used as "now" for staleness checks. Defaults to now.

    Returns:
        True iff all of these hold:
        1. Question matches Bitcoin/Ethereum "Up or Down" pattern.
        2. Both eventStartTime and endDate are present and valid ISO-8601.
        3. endDate - eventStartTime == exactly 5 minutes.
        4. endDate is not more than 1 hour in the past.
        5. eventStartTime is not more than 2 days in the future.
    """
    question = raw.get("question", "")
    if not _ASSET_RE.match(question):
        return False

    try:
        start = _parse_iso_utc(raw["eventStartTime"])
        end = _parse_iso_utc(raw["endDate"])
    except (KeyError, ValueError):
        return False

    if end - start != timedelta(minutes=5):
        return False

    ref = reference_date or datetime.now(UTC)
    if (end - ref).total_seconds() < -3600:
        return False
    if (start - ref).total_seconds() > 2 * 86400:
        return False

    return True


def _resolve_token_ids(raw: dict) -> tuple[str, str] | None:
    """Extract (up_token_id, down_token_id) from a raw market dict.

    Returns None if the token IDs are missing, malformed, or ambiguous.
    """
    clob_raw = raw.get("clobTokenIds")
    outcomes_raw = raw.get("outcomes")

    if not clob_raw or not outcomes_raw:
        return None

    try:
        token_ids: list[str] = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
        outcomes: list[str] = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    except (json.JSONDecodeError, TypeError):
        return None

    if len(token_ids) < 2 or len(outcomes) < 2:
        return None

    up_idx: int | None = None
    down_idx: int | None = None
    for i, outcome in enumerate(outcomes):
        o = outcome.strip().lower()
        if o in ("up", "yes"):
            up_idx = i
        elif o in ("down", "no"):
            down_idx = i

    if up_idx is None or down_idx is None:
        logger.warning("Cannot determine Up/Down index from outcomes: %s", outcomes)
        return None

    return token_ids[up_idx], token_ids[down_idx]


def _try_parse_market(
    raw: dict,
    reference_date: datetime | None = None,
) -> tuple["Market | None", "str | None"]:
    """Parse a Gamma API market dict, returning a reason string on failure.

    Args:
        raw: A single market dict from the Gamma API response.
        reference_date: UTC datetime used as "now". Defaults to now.

    Returns:
        (Market, None) on success, or (None, reason) on failure where reason is
        one of: "wrong_asset", "missing_dates", "wrong_window_size", "stale",
        "too_far_future", "missing_tokens".
    """
    question = raw.get("question", "")
    logger.debug("Parsing: %s", question)

    asset_match = _ASSET_RE.match(question)
    if not asset_match:
        return None, "wrong_asset"

    try:
        start = _parse_iso_utc(raw["eventStartTime"])
        end = _parse_iso_utc(raw["endDate"])
    except (KeyError, ValueError):
        return None, "missing_dates"

    if end - start != timedelta(minutes=5):
        return None, "wrong_window_size"

    ref = reference_date or datetime.now(UTC)
    if (end - ref).total_seconds() < -3600:
        return None, "stale"
    if (start - ref).total_seconds() > 2 * 86400:
        return None, "too_far_future"

    token_pair = _resolve_token_ids(raw)
    if token_pair is None:
        logger.warning("Missing/bad token IDs for: %s", question)
        return None, "missing_tokens"
    up_token_id, down_token_id = token_pair

    asset = _ASSET_MAP[asset_match.group(1)]

    return Market(
        condition_id=raw.get("conditionId", ""),
        question=question,
        asset=asset,
        start_time=start,
        end_time=end,
        up_token_id=up_token_id,
        down_token_id=down_token_id,
        slug=raw.get("slug", ""),
        raw=raw,
    ), None


def parse_market(
    raw: dict,
    reference_date: datetime | None = None,
) -> "Market | None":
    """Parse a Gamma API market dict into a Market.

    Args:
        raw: A single market dict from the Gamma API response.
        reference_date: UTC datetime used as "now". Defaults to now.

    Returns:
        A Market dataclass, or None if the market is not a valid BTC/ETH 5-min
        up/down market or required fields are missing/malformed.
    """
    market, _ = _try_parse_market(raw, reference_date)
    return market


def scan(reference_date: datetime | None = None) -> list[Market]:
    """Fetch and parse all active BTC/ETH 5-minute markets.

    Args:
        reference_date: UTC datetime used as "now" for staleness checks. Defaults to now.

    Returns:
        List of parsed Market objects, sorted by end_time ascending.
    """
    raw_markets = fetch_active_markets()
    markets: list[Market] = []
    rejected: dict[str, int] = {
        "wrong_asset": 0,
        "wrong_window_size": 0,
        "stale": 0,
        "too_far_future": 0,
        "missing_dates": 0,
        "missing_tokens": 0,
    }

    for raw in raw_markets:
        market, reason = _try_parse_market(raw, reference_date)
        if market is not None:
            markets.append(market)
        elif reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1

    logger.info(
        "Scanner: %d total, %d kept, rejected: %s",
        len(raw_markets),
        len(markets),
        ", ".join(f"{k}={v}" for k, v in rejected.items() if v > 0),
    )
    return sorted(markets, key=lambda m: m.end_time)


if __name__ == "__main__":
    markets = scan()
    for m in markets:
        print(f"{m.asset} {m.start_time:%m-%d %H:%M}-{m.end_time:%H:%M} UTC  {m.condition_id[:10]}...")
    print(f"\nTotal: {len(markets)} active BTC/ETH 5-min markets")
