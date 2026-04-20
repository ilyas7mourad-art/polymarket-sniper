"""Run win-rate analysis on collected orderbook data.

Loads all orderbook CSVs, determines market winners, and reports win rates
by entry price bucket at several time-to-resolution snapshots.
Output goes to stdout and data/analysis_report_<timestamp>.md.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow importing from src/ when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    DEFAULT_FEE_RATE,
    BucketStats,
    EntrySnapshot,
    compute_win_rates_by_bucket,
    determine_winners,
    extract_entry_snapshots,
    load_ticks,
)

BUCKETS: list[tuple[float, float]] = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.00),
]

TARGET_SECONDS: list[float] = [120.0, 60.0, 30.0, 10.0]


def _bucket_table(stats: list[BucketStats]) -> str:
    header = "| bucket | N | wins | win rate | avg ask | fee/share | EV (naive) | EV (real) |"
    sep = "|--------|---|------|----------|---------|-----------|------------|-----------|"
    rows = [header, sep]
    for s in stats:
        if s.n_samples == 0:
            rows.append(
                f"| {s.bucket_label} | 0 | 0 | — | {s.avg_entry_ask:.4f} | — | — | — |"
            )
        else:
            rows.append(
                f"| {s.bucket_label} | {s.n_samples} | {s.n_wins} | {s.win_rate:.1%} | {s.avg_entry_ask:.4f} | {s.fee_per_share:.5f} | {s.naive_ev:+.4f} | {s.fee_adjusted_ev:+.4f} |"
            )
    return "\n".join(rows)


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    csv_paths = sorted(data_dir.glob("orderbook_*.csv"))

    if not csv_paths:
        print("No orderbook CSV files found in data/", file=sys.stderr)
        sys.exit(1)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# Polymarket Orderbook Analysis")
    emit(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    emit()

    emit("## Data files loaded")
    for p in csv_paths:
        emit(f"- `{p.name}`")
    emit()

    emit("Loading ticks...")
    df = load_ticks(csv_paths)
    emit(f"Total rows loaded: {len(df):,}")
    emit()

    emit("Determining market winners...")
    winners = determine_winners(df)

    total = len(winners)
    resolved_up = sum(1 for o in winners.values() if o.winner == "Up")
    resolved_down = sum(1 for o in winners.values() if o.winner == "Down")
    resolved = resolved_up + resolved_down
    unknown = total - resolved

    emit(f"## Market resolution summary")
    emit(f"- Total markets observed: **{total}**")
    emit(f"- Resolved Up: **{resolved_up}**")
    emit(f"- Resolved Down: **{resolved_down}**")
    emit(f"- Resolved total: **{resolved}**")
    emit(f"- Unknown (cut off / mid-trade): **{unknown}**")
    emit()

    if resolved == 0:
        emit("WARNING: No markets resolved. Observation window may be too short.")
        emit("Cannot compute win rates — exiting.")
        _write_report(data_dir, lines)
        return

    # Collect best stats across all target times for the summary
    best_wr = 0.0
    best_label = ""
    best_t = 0.0
    total_high_ask_samples = 0

    # Store per-target results for final EV summary
    per_target: dict[float, list[BucketStats]] = {}

    for t in TARGET_SECONDS:
        emit(f"---")
        emit(f"## T = {int(t)}s before resolution")
        emit()

        snapshots = extract_entry_snapshots(df, winners, target_seconds=t, tolerance_seconds=2.0)

        n_snap = len(snapshots)
        if n_snap == 0:
            emit(f"No snapshots found within ±2s of T={int(t)}s.")
            emit()
            per_target[t] = []
            continue

        # Count by side
        sides = {}
        for s in snapshots:
            sides[s.side] = sides.get(s.side, 0) + 1
        emit(f"Snapshots: {n_snap} ({', '.join(f'{k}={v}' for k,v in sorted(sides.items()))})")
        emit()

        stats = compute_win_rates_by_bucket(snapshots, BUCKETS)
        per_target[t] = stats

        # Overall win rate at this target time (all snapshots, regardless of bucket)
        valid_snaps = [s for s in snapshots if s.eventual_winner in ("Up", "Down")]
        overall_wins = sum(1 for s in valid_snaps if s.side == s.eventual_winner)
        overall_wr = overall_wins / len(valid_snaps) if valid_snaps else 0.0

        emit(_bucket_table(stats))
        emit()
        emit(f"**Overall win rate at T={int(t)}s: {overall_wr:.1%}** ({overall_wins}/{len(valid_snaps)} valid snapshots)")
        emit()

        # Track best bucket for final summary
        for s in stats:
            if s.n_samples >= 5 and s.win_rate > best_wr:
                best_wr = s.win_rate
                best_label = s.bucket_label
                best_t = t

        # High-ask count
        total_high_ask_samples += sum(s.n_samples for s in stats if s.lower >= 0.90)

    emit("---")
    emit("## Data-driven summary")
    emit()

    if best_label:
        emit(f"1. **Strongest edge found** at T={int(best_t)}s, price bucket {best_label}, win rate {best_wr:.1%}")
    else:
        emit("1. **No bucket with ≥5 samples had a notable edge** — sample sizes may be too small.")

    emit(f"2. **Samples with best_ask ≥ 0.90:** {total_high_ask_samples} (across all target times)")
    emit()

    emit("3. **Fee-adjusted EV per trade by bucket** (real Polymarket formula, crypto rate 0.072):")
    emit()
    for t in TARGET_SECONDS:
        stats = per_target.get(t, [])
        if not stats:
            continue
        emit(f"   **T={int(t)}s:**")
        for s in stats:
            if s.n_samples == 0:
                continue
            emit(f"   - {s.bucket_label}: fee_adj_EV = {s.fee_adjusted_ev:+.4f}, naive_EV = {s.naive_ev:+.4f}, fee/share = {s.fee_per_share:.5f} (win_rate={s.win_rate:.1%}, avg_ask={s.avg_entry_ask:.4f}, N={s.n_samples})")
        emit()

    emit("---")
    emit("## Fee assumption")
    emit()
    emit("Polymarket taker fee formula: fee_usdc = shares × fee_rate × price × (1 - price).")
    emit(f"Crypto category fee_rate = {DEFAULT_FEE_RATE}. Fee peaks at price=0.50, approaches zero at 0.01 and 0.99.")
    emit("Only takers pay; makers receive rebates (not modeled).")
    emit()

    _write_report(data_dir, lines)


def _write_report(data_dir: Path, lines: list[str]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = data_dir / f"analysis_report_{ts}.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
