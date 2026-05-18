"""Scans Polymarket for active BTC/ETH up/down markets via series_id (5m, 15m, 4h)."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from src.config import config

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Series configs for all BTC/ETH up/down markets we trade.
# To rediscover: GET /events?slug=<asset>-updown-<tf>-<timestamp> → event.series[0].id
SERIES_CONFIG: list[dict] = [
    {"asset": "BTC", "timeframe": "5m",  "series_id": 10684, "window": timedelta(minutes=5)},
    {"asset": "ETH", "timeframe": "5m",  "series_id": 10683, "window": timedelta(minutes=5)},
    {"asset": "BTC", "timeframe": "15m", "series_id": 10192, "window": timedelta(minutes=15)},
    {"asset": "ETH", "timeframe": "15m", "series_id": 10191, "window": timedelta(minutes=15)},
    {"asset": "BTC", "timeframe": "4h",  "series_id": 10331, "window": timedelta(hours=4)},
]

# Time window acceptance bounds.
MAX_PAST = timedelta(hours=1)    # reject if endDate is more than this in the past
MAX_FUTURE = timedelta(days=2)   # reject if eventStartTime is more than this in the future


@dataclass(frozen=True)
class Market:
    """A single active BTC/ETH up/down Polymarket market.

    Attributes:
        condition_id: Hex string uniquely identifying the market.
        question: Raw question string from the Gamma API.
        asset: "BTC" or "ETH".
        timeframe: "5m", "15m", or "4h".
        start_time: Window open time (eventStartTime), timezone-aware UTC.
        end_time: Window close time (endDate), timezone-aware UTC.
        up_token_id: CLOB token ID for the Up/Yes outcome.
        down_token_id: CLOB token ID for the Down/No outcome.
        slug: Market URL slug.
        raw: Original Gamma API market dict, for debugging.
    """

    condition_id: str
    question: str
    asset: str
    timeframe: str
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


def fetch_events_for_series(series_id: int, limit: int = 500) -> list[dict]:
    """Fetch non-closed events for a given series_id from the Gamma API.

    Returns events unsorted; the caller is expected to filter by time window
    (see MAX_PAST / MAX_FUTURE). The API includes zombie markets with
    active=true but endDate months in the past — these are caught by the
    stale rejection in _try_parse_event_market.

    Args:
        series_id: The Polymarket series ID (e.g. 10684 for BTC 5m).
        limit: Max events to return.

    Returns:
        List of raw event dicts. Empty list on any error.
    """
    url = (
        f"{config.GAMMA_URL}/events"
        f"?series_id={series_id}&closed=false&limit={limit}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning(
                "Gamma API (series_id=%d) returned %s: %s",
                series_id, resp.status_code, resp.text[:200],
            )
            return []
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch events for series_id=%d: %s", series_id, exc)
        return []


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


def _try_parse_event_market(
    event: dict,
    asset: str,
    timeframe: str,
    window: timedelta,
    reference_date: datetime | None = None,
) -> tuple["Market | None", "str | None"]:
    """Try to extract a valid Market from an event dict.

    Args:
        event: Raw event dict from /events?series_id=...
        asset: "BTC" or "ETH".
        timeframe: "5m", "15m", or "4h".
        window: Expected candle duration (used to validate the market).
        reference_date: UTC datetime for freshness checks. Defaults to now.

    Returns:
        (Market, None) on success, or (None, reason) on failure.
    """
    ref = reference_date or datetime.now(UTC)

    markets = event.get("markets") or []
    if not markets:
        return None, "no_markets"

    market_raw = markets[0]

    event_start = market_raw.get("eventStartTime") or event.get("eventStartTime")
    end_date = market_raw.get("endDate") or event.get("endDate")

    if not event_start or not end_date:
        return None, "missing_dates"

    try:
        start_utc = _parse_iso_utc(event_start)
        end_utc = _parse_iso_utc(end_date)
    except (ValueError, TypeError):
        return None, "missing_dates"

    if (end_utc - start_utc) != window:
        return None, "wrong_window_size"

    if (ref - end_utc) > MAX_PAST:
        return None, "stale"

    if (start_utc - ref) > MAX_FUTURE:
        return None, "too_far_future"

    token_pair = _resolve_token_ids(market_raw)
    if token_pair is None:
        logger.warning("Missing/bad token IDs for market in event: %s", event.get("slug", "?"))
        return None, "missing_tokens"
    up_token_id, down_token_id = token_pair

    return Market(
        condition_id=market_raw.get("conditionId", ""),
        question=market_raw.get("question", ""),
        asset=asset,
        timeframe=timeframe,
        start_time=start_utc,
        end_time=end_utc,
        up_token_id=up_token_id,
        down_token_id=down_token_id,
        slug=market_raw.get("slug") or event.get("slug", ""),
        raw=market_raw,
    ), None


def parse_event_market(
    event: dict,
    asset: str,
    timeframe: str,
    window: timedelta,
    reference_date: datetime | None = None,
) -> "Market | None":
    """Public wrapper around _try_parse_event_market — returns Market or None."""
    market, _ = _try_parse_event_market(event, asset, timeframe, window, reference_date)
    return market


def scan(reference_date: datetime | None = None) -> list[Market]:
    """Fetch and parse all active BTC/ETH markets across all configured series.

    Args:
        reference_date: UTC datetime used as "now" for staleness checks. Defaults to now.

    Returns:
        List of Market objects sorted by end_time ascending.
    """
    all_markets: list[Market] = []
    rejected: dict[str, int] = {}
    total_events = 0

    for cfg in SERIES_CONFIG:
        events = fetch_events_for_series(cfg["series_id"])
        total_events += len(events)
        for event in events:
            market, reason = _try_parse_event_market(
                event, cfg["asset"], cfg["timeframe"], cfg["window"], reference_date
            )
            if market is not None:
                all_markets.append(market)
            elif reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1

    rejected_summary = ", ".join(f"{k}={v}" for k, v in rejected.items() if v > 0)
    logger.info(
        "Scanner: %d events across %d series, %d markets kept, rejected: %s",
        total_events, len(SERIES_CONFIG), len(all_markets), rejected_summary or "none",
    )
    return sorted(all_markets, key=lambda m: m.end_time)


if __name__ == "__main__":
    markets = scan()
    for m in markets:
        print(f"{m.asset:3} {m.timeframe:3}  {m.start_time:%m-%d %H:%M}-{m.end_time:%H:%M} UTC  {m.condition_id[:10]}...")
    print(f"\nTotal: {len(markets)} active markets")
