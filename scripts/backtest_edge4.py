"""
Offline backtest — Edge 4: Momentum within epoch.

Signal rule: first orderbook snapshot per (condition_id, side) where
    55 <= seconds_to_resolution <= 185  (targeting the 60-180s window)
    mid >= 0.70

Entry price: best_ask at that snapshot.
Exit: resolution — payout $1/share on win, $0 on loss.
Fee: 2% of notional (Polymarket taker).
PnL = payout - stake - fee.

Outcome ground truth: paper_trades_*.csv (condition_id → winner).
"""

import glob
import io
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA = "/home/mma/polymarket-sniper/data"
STAKE_USDC = 1.0
FEE_RATE = 0.02
MID_THRESHOLD = 0.70
TTL_MIN = 55
TTL_MAX = 185


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def awk_filter_ttl(files: list[str]) -> pd.DataFrame:
    """Load orderbook rows where seconds_to_resolution is in [TTL_MIN, TTL_MAX].

    Uses awk on each file for speed — col 9 is seconds_to_resolution per header:
    timestamp_utc,market_slug,asset,condition_id,side,best_bid,best_ask,mid,seconds_to_resolution
    """
    frames = []
    for f in sorted(files):
        cmd = (
            f"awk -F',' 'NR==1 || ($9 >= {TTL_MIN} && $9 <= {TTL_MAX})' {f}"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            try:
                frames.append(pd.read_csv(io.StringIO(result.stdout)))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def extract_window(slug: str) -> str:
    for w in ["5m", "15m", "1h"]:
        if w in str(slug):
            return w
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Outcome map from paper trades
# ─────────────────────────────────────────────────────────────────────────────
print("=== Loading outcome map from paper trades ===")

pt_files = glob.glob(f"{DATA}/paper_trades_*.csv")
pt_frames = []
for f in sorted(pt_files):
    try:
        pt_frames.append(pd.read_csv(f, usecols=["condition_id", "winner"]))
    except Exception:
        pass

if not pt_frames:
    raise SystemExit("No paper_trades_*.csv files found — cannot determine outcomes.")

outcomes_df = pd.concat(pt_frames, ignore_index=True)
outcomes_df = outcomes_df[
    outcomes_df["winner"].notna() & (outcomes_df["winner"] != "unknown")
]
outcome_map: dict[str, str] = (
    outcomes_df.drop_duplicates("condition_id")
    .set_index("condition_id")["winner"]
    .to_dict()
)
print(f"Known outcomes: {len(outcome_map):,} condition_ids")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load orderbook filtered to signal window
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Loading orderbook (TTL {TTL_MIN}-{TTL_MAX}s via awk) ===")

ob_files = glob.glob(f"{DATA}/orderbook_*.csv")
if not ob_files:
    raise SystemExit("No orderbook_*.csv files found.")

ob = awk_filter_ttl(ob_files)
ob["timestamp_utc"] = pd.to_datetime(ob["timestamp_utc"], format="ISO8601", utc=True)
ob["asset"] = ob["asset"].str.upper().str.strip()
ob = ob.sort_values("timestamp_utc").reset_index(drop=True)
print(f"Rows in signal window: {len(ob):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Apply signal and pick entries
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Applying signal: mid >= {MID_THRESHOLD:.0%} ===")

signal_rows = ob[ob["mid"] >= MID_THRESHOLD].copy()
print(f"Rows matching threshold: {len(signal_rows):,}")

# One entry per (condition_id, side): first qualifying snapshot
entries = (
    signal_rows
    .sort_values("timestamp_utc")
    .drop_duplicates(subset=["condition_id", "side"], keep="first")
    .copy()
)
print(f"Unique (condition_id, side) entries: {len(entries):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Join outcomes
# ─────────────────────────────────────────────────────────────────────────────
entries["winner"] = entries["condition_id"].map(outcome_map)
known = entries[entries["winner"].notna()].copy()
n_no_outcome = len(entries) - len(known)
print(f"Entries with known outcome: {len(known):,}  (dropped {n_no_outcome} without outcome data)")

if len(known) < 10:
    raise SystemExit("Too few trades with known outcomes to produce meaningful results.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compute PnL
# ─────────────────────────────────────────────────────────────────────────────
known["won"] = (known["side"] == known["winner"]).astype(int)
known["shares"] = STAKE_USDC / known["best_ask"]
known["fee_usdc"] = FEE_RATE * STAKE_USDC
known["payout"] = known["won"] * known["shares"]
known["pnl"] = known["payout"] - STAKE_USDC - known["fee_usdc"]
known["window"] = known["market_slug"].apply(extract_window)
known = known.sort_values("timestamp_utc").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────
n = len(known)
wr = known["won"].mean()
total_pnl = known["pnl"].sum()
mean_pnl = known["pnl"].mean()
sharpe = known["pnl"].mean() / (known["pnl"].std() + 1e-9)
cum = known["pnl"].cumsum()
max_dd = (cum - cum.cummax()).min()

print(f"\n{'='*60}")
print(f"  BACKTEST RESULTS — Edge 4 Momentum (mid≥{MID_THRESHOLD:.0%}, TTL {TTL_MIN}-{TTL_MAX}s)")
print(f"{'='*60}")
print(f"  Trades       : {n}")
print(f"  Win rate     : {wr:.1%}")
print(f"  Total PnL    : ${total_pnl:.4f}")
print(f"  Mean PnL     : ${mean_pnl:.4f} per trade")
print(f"  Sharpe       : {sharpe:.4f} (per-trade, not annualized)")
print(f"  Max drawdown : ${max_dd:.4f}")
ci_lo, ci_hi = bootstrap_ci(known["won"].values)
print(f"  Win rate 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sensitivity: win rate by mid threshold
# ─────────────────────────────────────────────────────────────────────────────
print("--- Win rate & EV by mid threshold ---")
threshold_rows = []
for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    sub = known[known["mid"] >= thresh]
    if len(sub) < 5:
        continue
    wr_s = sub["won"].mean()
    avg_ask = sub["best_ask"].mean()
    ev = wr_s * 1.0 - avg_ask - FEE_RATE * avg_ask
    ci_l, ci_h = bootstrap_ci(sub["won"].values)
    threshold_rows.append({
        "mid_threshold": f">={thresh:.0%}",
        "n": len(sub),
        "win_rate": round(wr_s, 4),
        "ci_95": f"[{ci_l:.3f},{ci_h:.3f}]",
        "avg_ask": round(avg_ask, 4),
        "EV_per_share": round(ev, 5),
    })

df_thresh = pd.DataFrame(threshold_rows)
if not df_thresh.empty:
    print(df_thresh.to_string(index=False))
print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Breakdown by asset, market window, TTL bucket, UTC hour
# ─────────────────────────────────────────────────────────────────────────────
print("--- By asset ---")
for asset, g in known.groupby("asset"):
    if len(g) >= 5:
        ci_l, ci_h = bootstrap_ci(g["won"].values)
        print(f"  {asset}: n={len(g):4d}  wr={g['won'].mean():.3f}  "
              f"CI=[{ci_l:.3f},{ci_h:.3f}]  total_pnl=${g['pnl'].sum():.4f}")
print()

print("--- By market window ---")
for window, g in known.groupby("window"):
    if len(g) >= 5:
        print(f"  {window}: n={len(g):4d}  wr={g['won'].mean():.3f}  "
              f"mean_pnl=${g['pnl'].mean():.5f}  total_pnl=${g['pnl'].sum():.4f}")
print()

print("--- By TTL at entry ---")
known["ttl_bin"] = pd.cut(
    known["seconds_to_resolution"],
    bins=[TTL_MIN, 70, 90, 110, 130, 150, TTL_MAX],
    labels=["55-70s", "70-90s", "90-110s", "110-130s", "130-150s", "150-185s"],
)
ttl_g = known.groupby("ttl_bin", observed=True).agg(
    n=("pnl", "count"),
    win_rate=("won", "mean"),
    mean_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
).round(5)
print(ttl_g.to_string())
print()

print("--- By UTC hour (all, sorted by win rate) ---")
known["hour"] = known["timestamp_utc"].dt.hour
tod = known.groupby("hour").agg(
    n=("pnl", "count"),
    win_rate=("won", "mean"),
    mean_pnl=("pnl", "mean"),
).sort_values("win_rate", ascending=False)
print(tod.to_string())
print()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Equity curve checkpoints
# ─────────────────────────────────────────────────────────────────────────────
print("--- Equity curve (10 checkpoints) ---")
cum_series = known["pnl"].cumsum()
step = max(1, n // 10)
for i in range(0, n, step):
    print(f"  trade {i+1:4d}: cumPnL=${cum_series.iloc[i]:.4f}")
print(f"  trade {n:4d}: cumPnL=${cum_series.iloc[-1]:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Stat test: is win rate > entry price?
# ─────────────────────────────────────────────────────────────────────────────
print("--- Statistical test: win_rate vs entry_price ---")
r, p = stats.pearsonr(known["mid"], known["won"])
print(f"  Pearson(mid, won): r={r:.4f}  p={p:.5e}")

# One-sided binomial test: WR > implied by avg ask
avg_ask_all = known["best_ask"].mean()
binom = stats.binomtest(int(known["won"].sum()), n, avg_ask_all, alternative="greater")
print(f"  Binomial test WR > avg_ask ({avg_ask_all:.3f}): p={binom.pvalue:.5e}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Save report
# ─────────────────────────────────────────────────────────────────────────────
best_thresh_row = df_thresh.loc[df_thresh["EV_per_share"].idxmax()] if not df_thresh.empty else None

report_lines = [
    "# Backtest Report — Edge 4: Momentum within Epoch",
    f"**Signal**: mid ≥ {MID_THRESHOLD:.0%}, TTL {TTL_MIN}–{TTL_MAX}s | "
    f"**Fee**: {FEE_RATE*100:.0f}% | **Stake**: ${STAKE_USDC}/trade | "
    f"**Data**: Apr 18–May 7 2026",
    "",
    "## Summary",
    "| Metric | Value |",
    "|--------|-------|",
    f"| Trades | {n} |",
    f"| Win rate | {wr:.1%} |",
    f"| Win rate 95% CI | [{ci_lo:.3f}, {ci_hi:.3f}] |",
    f"| Total PnL | ${total_pnl:.4f} |",
    f"| Mean PnL/trade | ${mean_pnl:.5f} |",
    f"| Sharpe (per trade) | {sharpe:.4f} |",
    f"| Max drawdown | ${max_dd:.4f} |",
    "",
    "## Threshold Sensitivity",
    df_thresh.to_markdown(index=False) if not df_thresh.empty else "_no data_",
    "",
]

if best_thresh_row is not None:
    report_lines += [
        f"**Best EV threshold**: mid {best_thresh_row['mid_threshold']} "
        f"→ EV={best_thresh_row['EV_per_share']:.5f}/share, "
        f"n={best_thresh_row['n']}, wr={best_thresh_row['win_rate']:.3f}",
        "",
    ]

report_lines += [
    "## Breakdown by Asset",
    known.groupby("asset").agg(n=("pnl","count"), win_rate=("won","mean"),
                                mean_pnl=("pnl","mean"), total_pnl=("pnl","sum"))
         .round(4).to_markdown(),
    "",
    "## Breakdown by Market Window",
    known.groupby("window").agg(n=("pnl","count"), win_rate=("won","mean"),
                                 mean_pnl=("pnl","mean"), total_pnl=("pnl","sum"))
         .round(4).to_markdown(),
    "",
    "## Breakdown by TTL at Entry",
    ttl_g.to_markdown(),
    "",
    "## Breakdown by UTC Hour",
    tod.to_markdown(),
    "",
    "## Statistical Significance",
    f"- Pearson(mid, won): r={r:.4f}, p={p:.5e}",
    f"- Binomial test (WR > avg_ask={avg_ask_all:.3f}): p={binom.pvalue:.5e}",
    "",
    "## Notes",
    "- Outcome ground truth from paper_trades CSVs; markets without a known winner are excluded.",
    "- Entry at best_ask (taker). Fee = 2% × stake. One entry per (condition_id, side).",
    "- Max drawdown computed on per-$1-stake basis; scale linearly for larger position sizes.",
]

report_text = "\n".join(report_lines) + "\n"

report_path = f"{DATA}/backtest_report.md"
with open(report_path, "w") as fh:
    fh.write(report_text)
print(f"Report saved → {report_path}")
