"""Scans Polymarket for eligible BTC/ETH 5-minute up/down markets."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from src.config import config

logger = logging.getLogger(__name__)

UTC = timezone.utc
ET = ZoneInfo("America/New_York")

# Matches: "Bitcoin Up or Down - April 17, 3:00PM-3:05PM ET"
_QUESTION_RE = re.compile(
    r"^(Bitcoin|Ethereum) Up or Down - (\w+ \d+), (\d+:\d+)(AM|PM)-(\d+:\d+)(AM|PM) ET$"
)

_ASSET_MAP = {"Bitcoin": "BTC", "Ethereum": "ETH"}


@dataclass(frozen=True)
class Market:
    """A single active BTC/ETH 5-minute up/down Polymarket market.

    Attributes:
        condition_id: Hex string uniquely identifying the market.
        question: Raw question string from the Gamma API.
        asset: "BTC" or "ETH".
        start_time: Window open time, timezone-aware UTC.
        end_time: Window close time, timezone-aware UTC.
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
    question: str,
    reference_date: datetime | None = None,
) -> bool:
    """Return True iff the question matches a BTC/ETH 5-minute up/down pattern.

    Args:
        question: The market question string from the Gamma API.
        reference_date: UTC datetime used to infer the year. Defaults to now.

    Returns:
        True if the question is a valid 5-minute BTC/ETH window.
    """
    result = parse_window_times(question, reference_date=reference_date)
    if result is None:
        return False
    start, end = result
    return (end - start) == timedelta(minutes=5)


def parse_window_times(
    question: str,
    reference_date: datetime | None = None,
) -> tuple[datetime, datetime] | None:
    """Parse start and end times from a market question string.

    Args:
        question: The market question string, e.g.
            "Bitcoin Up or Down - April 17, 3:00PM-3:05PM ET".
        reference_date: UTC datetime used to infer the year. Defaults to now.

    Returns:
        (start_utc, end_utc) as timezone-aware UTC datetimes, or None if the
        question doesn't match the expected format.
    """
    m = _QUESTION_RE.match(question)
    if not m:
        return None

    _, date_str, start_time_str, start_ampm, end_time_str, end_ampm = m.groups()

    ref = reference_date or datetime.now(UTC)

    def _parse_dt(date_part: str, time_part: str, ampm: str, year: int) -> datetime:
        raw = f"{date_part} {year} {time_part}{ampm}"
        naive = datetime.strptime(raw, "%B %d %Y %I:%M%p")
        return naive.replace(tzinfo=ET).astimezone(UTC)

    # Try current year; if end_time is implausible, try adjacent years.
    for year_offset in (0, 1, -1):
        year = ref.year + year_offset
        try:
            start_utc = _parse_dt(date_str, start_time_str, start_ampm, year)
            end_utc = _parse_dt(date_str, end_time_str, end_ampm, year)
        except ValueError:
            continue

        # Handle midnight wrap (e.g. 11:55PM-12:00AM)
        if end_utc <= start_utc:
            end_utc += timedelta(days=1)

        # Active markets are always near-term. Accept -1h past to +2 days future.
        delta = (end_utc - ref).total_seconds()
        if -3600 <= delta <= 2 * 86400:
            return start_utc, end_utc

    return None


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


def parse_market(
    raw: dict,
    reference_date: datetime | None = None,
) -> "Market | None":
    """Parse a Gamma API market dict into a Market.

    Args:
        raw: A single market dict from the Gamma API response.
        reference_date: UTC datetime used to infer the year. Defaults to now.

    Returns:
        A Market dataclass, or None if the market is not a BTC/ETH 5-min
        up/down market or if required fields are missing/malformed.
    """
    question = raw.get("question", "")
    logger.debug("Parsing: %s", question)

    if not is_btc_eth_5min_window(question, reference_date=reference_date):
        return None

    times = parse_window_times(question, reference_date=reference_date)
    if times is None:
        return None
    start_utc, end_utc = times

    token_pair = _resolve_token_ids(raw)
    if token_pair is None:
        logger.warning("Missing/bad token IDs for: %s", question)
        return None
    up_token_id, down_token_id = token_pair

    m = _QUESTION_RE.match(question)
    asset = _ASSET_MAP[m.group(1)]  # type: ignore[index]

    return Market(
        condition_id=raw.get("conditionId", ""),
        question=question,
        asset=asset,
        start_time=start_utc,
        end_time=end_utc,
        up_token_id=up_token_id,
        down_token_id=down_token_id,
        slug=raw.get("slug", ""),
        raw=raw,
    )


def scan() -> list[Market]:
    """Fetch and parse all active BTC/ETH 5-minute markets.

    Returns:
        List of parsed Market objects, sorted by end_time ascending.
    """
    raw_markets = fetch_active_markets()
    markets: list[Market] = []

    for raw in raw_markets:
        market = parse_market(raw)
        if market is not None:
            markets.append(market)

    logger.info("Fetched %d markets, kept %d after filtering", len(raw_markets), len(markets))
    return sorted(markets, key=lambda m: m.end_time)


if __name__ == "__main__":
    markets = scan()
    for m in markets:
        print(f"{m.asset} {m.start_time:%m-%d %H:%M}-{m.end_time:%H:%M} UTC  {m.condition_id[:10]}...")
    print(f"\nTotal: {len(markets)} active BTC/ETH 5-min markets")
