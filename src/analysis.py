"""Analysis of orderbook observer data: win rates by entry price and time-to-resolution."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MarketOutcome:
    """Final resolution of a single market."""

    condition_id: str
    asset: str
    market_slug: str
    winner: str  # "Up" or "Down" or "unknown" if we can't determine
    resolved_at: pd.Timestamp


@dataclass(frozen=True)
class EntrySnapshot:
    """One side's state at a specific time-to-resolution snapshot."""

    condition_id: str
    asset: str
    side: str           # "Up" or "Down"
    best_ask: float     # what you'd pay to enter
    mid: float
    seconds_to_resolution: float
    eventual_winner: str  # filled in after we know the outcome


def load_ticks(csv_paths: list[Path]) -> pd.DataFrame:
    """Load one or more orderbook CSVs into a single DataFrame.

    Returns DataFrame with proper dtypes:
    - timestamp_utc as datetime64[ns, UTC]
    - numeric columns as float64
    - string columns as string
    """
    dtype_map = {
        "market_slug": "string",
        "asset": "string",
        "condition_id": "string",
        "side": "string",
        "best_bid": "float64",
        "best_ask": "float64",
        "mid": "float64",
        "seconds_to_resolution": "float64",
    }
    frames = []
    for path in csv_paths:
        df = pd.read_csv(
            path,
            parse_dates=["timestamp_utc"],
            dtype=dtype_map,
        )
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    return combined


def determine_winners(df: pd.DataFrame) -> dict[str, MarketOutcome]:
    """For each condition_id, determine which side won.

    We take the post-resolution tick (smallest / most-negative seconds_to_resolution)
    for each side. The observer logs a few seconds past resolution, so the last
    tick we see has the settled state: one side pinned at ~1.00, the other at ~0.00.

    Heuristic: a market is "resolved Up" if the post-resolution Up mid >= 0.95
    AND the post-resolution Down mid <= 0.05. Vice versa for Down.
    If neither condition holds, winner is "unknown".

    Returns dict keyed by condition_id.
    """
    # "Last tick" = the tick with the smallest seconds_to_resolution (closest to / past resolution).
    # Sort ascending so the minimum seconds_to_resolution row is first per group, then take head(1).
    sorted_df = df.sort_values(["condition_id", "side", "seconds_to_resolution"])
    last_ticks = sorted_df.groupby(["condition_id", "side"]).head(1)

    # Pivot so we have one row per (condition_id, Up_mid, Down_mid)
    up_last = last_ticks[last_ticks["side"] == "Up"].set_index("condition_id")[["mid", "asset", "market_slug", "timestamp_utc"]]
    down_last = last_ticks[last_ticks["side"] == "Down"].set_index("condition_id")[["mid"]]

    outcomes: dict[str, MarketOutcome] = {}

    all_condition_ids = df["condition_id"].unique()
    for cid in all_condition_ids:
        has_up = cid in up_last.index
        has_down = cid in down_last.index

        if not has_up or not has_down:
            # Can't determine — only one side observed
            row = df[df["condition_id"] == cid].iloc[0]
            outcomes[cid] = MarketOutcome(
                condition_id=cid,
                asset=str(row["asset"]),
                market_slug=str(row["market_slug"]),
                winner="unknown",
                resolved_at=row["timestamp_utc"],
            )
            continue

        up_mid = float(up_last.loc[cid, "mid"])
        down_mid = float(down_last.loc[cid, "mid"])
        asset = str(up_last.loc[cid, "asset"])
        slug = str(up_last.loc[cid, "market_slug"])
        resolved_at = up_last.loc[cid, "timestamp_utc"]

        if up_mid >= 0.95 and down_mid <= 0.05:
            winner = "Up"
        elif down_mid >= 0.95 and up_mid <= 0.05:
            winner = "Down"
        else:
            winner = "unknown"

        outcomes[cid] = MarketOutcome(
            condition_id=cid,
            asset=asset,
            market_slug=slug,
            winner=winner,
            resolved_at=resolved_at,
        )

    return outcomes


def extract_entry_snapshots(
    df: pd.DataFrame,
    winners: dict[str, MarketOutcome],
    target_seconds: float,
    tolerance_seconds: float = 2.0,
) -> list[EntrySnapshot]:
    """For each resolved market, find the tick closest to `target_seconds` before resolution.

    For each (condition_id, side), finds the tick where seconds_to_resolution is within
    [target - tolerance, target + tolerance]. If multiple, picks the one closest to target.

    Skips markets with winner="unknown".
    Returns list of EntrySnapshot with eventual_winner filled in.
    """
    # Only keep resolved markets
    resolved_ids = {cid for cid, outcome in winners.items() if outcome.winner != "unknown"}
    if not resolved_ids:
        return []

    sub = df[df["condition_id"].isin(resolved_ids)].copy()
    sub["_diff"] = (sub["seconds_to_resolution"] - target_seconds).abs()

    in_window = sub[sub["_diff"] <= tolerance_seconds].copy()
    if in_window.empty:
        return []

    # Pick the row with min diff per (condition_id, side)
    idx = in_window.groupby(["condition_id", "side"])["_diff"].idxmin()
    best = in_window.loc[idx]

    snapshots: list[EntrySnapshot] = []
    for _, row in best.iterrows():
        cid = str(row["condition_id"])
        outcome = winners[cid]
        snapshots.append(
            EntrySnapshot(
                condition_id=cid,
                asset=str(row["asset"]),
                side=str(row["side"]),
                best_ask=float(row["best_ask"]),
                mid=float(row["mid"]),
                seconds_to_resolution=float(row["seconds_to_resolution"]),
                eventual_winner=outcome.winner,
            )
        )

    return snapshots


@dataclass
class BucketStats:
    """Win-rate stats for one entry-price bucket."""

    bucket_label: str       # e.g. "0.70-0.75"
    lower: float
    upper: float
    n_samples: int
    n_wins: int
    win_rate: float         # wins / samples
    avg_entry_ask: float    # mean best_ask within this bucket


def compute_win_rates_by_bucket(
    snapshots: list[EntrySnapshot],
    buckets: list[tuple[float, float]],
) -> list[BucketStats]:
    """Bucket snapshots by best_ask price and compute win rate per bucket.

    A "win" = the snapshot's side is the eventual_winner.
    Only considers snapshots where best_ask falls in a defined bucket range.
    Skips snapshots with eventual_winner not in {"Up","Down"}.

    Returns stats sorted by bucket lower-bound ascending.
    """
    results: list[BucketStats] = []
    valid = [s for s in snapshots if s.eventual_winner in ("Up", "Down")]

    for lower, upper in sorted(buckets, key=lambda b: b[0]):
        label = f"{lower:.2f}-{upper:.2f}"
        in_bucket = [s for s in valid if lower <= s.best_ask < upper]
        n = len(in_bucket)
        if n == 0:
            results.append(BucketStats(
                bucket_label=label,
                lower=lower,
                upper=upper,
                n_samples=0,
                n_wins=0,
                win_rate=0.0,
                avg_entry_ask=0.0,
            ))
            continue

        wins = sum(1 for s in in_bucket if s.side == s.eventual_winner)
        avg_ask = sum(s.best_ask for s in in_bucket) / n
        results.append(BucketStats(
            bucket_label=label,
            lower=lower,
            upper=upper,
            n_samples=n,
            n_wins=wins,
            win_rate=wins / n,
            avg_entry_ask=avg_ask,
        ))

    return results
