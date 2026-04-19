"""Unit tests for src/analysis.py — all offline, no real CSV data."""

import io
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import (
    BucketStats,
    EntrySnapshot,
    MarketOutcome,
    compute_win_rates_by_bucket,
    determine_winners,
    extract_entry_snapshots,
    load_ticks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = "timestamp_utc,market_slug,asset,condition_id,side,best_bid,best_ask,mid,seconds_to_resolution\n"


def _make_csv_rows(*rows: tuple) -> str:
    """Build CSV content from (ts, slug, asset, cid, side, bid, ask, mid, secs) tuples."""
    lines = [_CSV_HEADER]
    for r in rows:
        lines.append(",".join(str(v) for v in r) + "\n")
    return "".join(lines)


def _make_df(*rows: tuple) -> pd.DataFrame:
    """Build a DataFrame directly from row tuples (same schema as CSV)."""
    cols = ["timestamp_utc", "market_slug", "asset", "condition_id", "side",
            "best_bid", "best_ask", "mid", "seconds_to_resolution"]
    data = [dict(zip(cols, r)) for r in rows]
    df = pd.DataFrame(data)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    for col in ("best_bid", "best_ask", "mid", "seconds_to_resolution"):
        df[col] = df[col].astype("float64")
    for col in ("market_slug", "asset", "condition_id", "side"):
        df[col] = df[col].astype("string")
    return df


_TS = "2026-04-19T10:00:00+00:00"
_CID = "0xabc"
_SLUG = "btc-updown-5m-111"


# ---------------------------------------------------------------------------
# 1. load_ticks — dtype assertions
# ---------------------------------------------------------------------------


def test_load_ticks_parses_types(tmp_path: Path) -> None:
    content = _make_csv_rows(
        (_TS, _SLUG, "BTC", _CID, "Up", 0.49, 0.51, 0.50, 120.0),
        (_TS, _SLUG, "BTC", _CID, "Down", 0.49, 0.51, 0.50, 120.0),
    )
    csv_file = tmp_path / "orderbook_test.csv"
    csv_file.write_text(content)

    df = load_ticks([csv_file])

    assert hasattr(df["timestamp_utc"].dtype, "tz") and str(df["timestamp_utc"].dtype.tz) == "UTC"
    assert df["best_bid"].dtype == "float64"
    assert df["best_ask"].dtype == "float64"
    assert df["mid"].dtype == "float64"
    assert df["seconds_to_resolution"].dtype == "float64"
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 2. determine_winners — Up resolution
# ---------------------------------------------------------------------------


def test_determine_winners_obvious_up_resolution() -> None:
    df = _make_df(
        (_TS, _SLUG, "BTC", _CID, "Up", 0.98, 0.99, 0.98, 2.0),
        (_TS, _SLUG, "BTC", _CID, "Up", 0.97, 0.98, 0.97, 5.0),
        (_TS, _SLUG, "BTC", _CID, "Down", 0.01, 0.02, 0.02, 2.0),
        (_TS, _SLUG, "BTC", _CID, "Down", 0.01, 0.03, 0.03, 5.0),
    )
    winners = determine_winners(df)
    assert winners[_CID].winner == "Up"


# ---------------------------------------------------------------------------
# 3. determine_winners — Down resolution
# ---------------------------------------------------------------------------


def test_determine_winners_obvious_down_resolution() -> None:
    df = _make_df(
        (_TS, _SLUG, "BTC", _CID, "Up", 0.01, 0.02, 0.02, 2.0),
        (_TS, _SLUG, "BTC", _CID, "Up", 0.01, 0.03, 0.03, 5.0),
        (_TS, _SLUG, "BTC", _CID, "Down", 0.97, 0.98, 0.97, 2.0),
        (_TS, _SLUG, "BTC", _CID, "Down", 0.96, 0.97, 0.96, 5.0),
    )
    winners = determine_winners(df)
    assert winners[_CID].winner == "Down"


# ---------------------------------------------------------------------------
# 4. determine_winners — ambiguous
# ---------------------------------------------------------------------------


def test_determine_winners_ambiguous() -> None:
    df = _make_df(
        (_TS, _SLUG, "BTC", _CID, "Up", 0.59, 0.61, 0.60, 2.0),
        (_TS, _SLUG, "BTC", _CID, "Down", 0.39, 0.41, 0.40, 2.0),
    )
    winners = determine_winners(df)
    assert winners[_CID].winner == "unknown"


# ---------------------------------------------------------------------------
# 5. extract_entry_snapshots — picks the closest tick
# ---------------------------------------------------------------------------


def test_extract_entry_snapshots_finds_closest_tick() -> None:
    # Ticks at seconds_to_resolution = 65, 62, 58, 55
    # target=60, tolerance=2 → window [58, 62]
    # Both 62 (diff=2) and 58 (diff=2) are in window; 62 is first by sort, but
    # idxmin picks whichever has smaller diff — both equal, so either is valid.
    # The spec says "closest to target", so diff=2 for both 62 and 58.
    # We just verify exactly ONE snapshot is returned and it's within the window.
    cid = "0xtest"
    slug = "btc-updown-5m-999"
    df = _make_df(
        (_TS, slug, "BTC", cid, "Up", 0.49, 0.51, 0.50, 65.0),
        (_TS, slug, "BTC", cid, "Up", 0.49, 0.51, 0.50, 62.0),
        (_TS, slug, "BTC", cid, "Up", 0.49, 0.51, 0.50, 58.0),
        (_TS, slug, "BTC", cid, "Up", 0.49, 0.51, 0.50, 55.0),
        (_TS, slug, "BTC", cid, "Down", 0.49, 0.51, 0.50, 62.0),
    )
    winners = {
        cid: MarketOutcome(
            condition_id=cid,
            asset="BTC",
            market_slug=slug,
            winner="Up",
            resolved_at=pd.Timestamp(_TS),
        )
    }
    snaps = extract_entry_snapshots(df, winners, target_seconds=60.0, tolerance_seconds=2.0)

    # Should get one Up snap and one Down snap (both have ticks at 62)
    up_snaps = [s for s in snaps if s.side == "Up"]
    assert len(up_snaps) == 1
    assert abs(up_snaps[0].seconds_to_resolution - 60.0) <= 2.0

    down_snaps = [s for s in snaps if s.side == "Down"]
    assert len(down_snaps) == 1
    assert down_snaps[0].seconds_to_resolution == 62.0


# ---------------------------------------------------------------------------
# 6. extract_entry_snapshots — skips unknown winners
# ---------------------------------------------------------------------------


def test_extract_entry_snapshots_skips_unknown_winners() -> None:
    cid = "0xunknown"
    slug = "eth-updown-5m-999"
    df = _make_df(
        (_TS, slug, "ETH", cid, "Up", 0.49, 0.51, 0.50, 60.0),
        (_TS, slug, "ETH", cid, "Down", 0.49, 0.51, 0.50, 60.0),
    )
    winners = {
        cid: MarketOutcome(
            condition_id=cid,
            asset="ETH",
            market_slug=slug,
            winner="unknown",
            resolved_at=pd.Timestamp(_TS),
        )
    }
    snaps = extract_entry_snapshots(df, winners, target_seconds=60.0, tolerance_seconds=2.0)
    assert snaps == []


# ---------------------------------------------------------------------------
# 7. compute_win_rates_by_bucket — known outcomes
# ---------------------------------------------------------------------------


def _make_snapshot(side: str, ask: float, winner: str, cid: str = "0xtest") -> EntrySnapshot:
    return EntrySnapshot(
        condition_id=cid,
        asset="BTC",
        side=side,
        best_ask=ask,
        mid=ask - 0.005,
        seconds_to_resolution=60.0,
        eventual_winner=winner,
    )


def test_compute_win_rates_by_bucket_simple() -> None:
    # 4 Up snapshots in [0.60-0.65], 3 win → win_rate = 3/4 = 0.75
    # 6 Up snapshots in [0.70-0.75], 2 win → win_rate = 2/6 ≈ 0.333
    snaps = (
        [_make_snapshot("Up", 0.62, "Up")] * 3
        + [_make_snapshot("Up", 0.61, "Down")]
        + [_make_snapshot("Up", 0.72, "Up")] * 2
        + [_make_snapshot("Up", 0.71, "Down")] * 4
    )
    buckets = [(0.60, 0.65), (0.70, 0.75)]
    stats = compute_win_rates_by_bucket(snaps, buckets)

    assert len(stats) == 2
    s60 = stats[0]
    assert s60.lower == 0.60
    assert s60.n_samples == 4
    assert s60.n_wins == 3
    assert abs(s60.win_rate - 0.75) < 1e-9

    s70 = stats[1]
    assert s70.lower == 0.70
    assert s70.n_samples == 6
    assert s70.n_wins == 2
    assert abs(s70.win_rate - 2 / 6) < 1e-9


# ---------------------------------------------------------------------------
# 8. compute_win_rates_by_bucket — empty bucket doesn't crash
# ---------------------------------------------------------------------------


def test_compute_win_rates_by_bucket_empty_bucket_omitted_or_zero() -> None:
    snaps = [_make_snapshot("Up", 0.62, "Up")]
    buckets = [(0.60, 0.65), (0.80, 0.85)]  # second bucket has no data

    stats = compute_win_rates_by_bucket(snaps, buckets)

    assert len(stats) == 2
    populated = next(s for s in stats if s.lower == 0.60)
    empty = next(s for s in stats if s.lower == 0.80)

    assert populated.n_samples == 1
    assert empty.n_samples == 0
    assert empty.n_wins == 0
    assert empty.win_rate == 0.0
