"""
Stress test — Edge 4: Momentum within epoch.

Four tests:
  1. Walk-forward (out-of-sample): Apr 18-30 in-sample, May 1-7 out-of-sample.
  2. Temporal stability: rolling win rate + weekly + daily breakdown.
  3. Parameter sensitivity: grid over TTL window, mid threshold, skip-hours, assets.
  4. Monte Carlo: 5 000 bootstrap paths → drawdown/ruin at real stake sizes.
  5. TTL sub-bucket deep dive (100-105s was net-negative — consistent or noise?).

Streams files one at a time — peak RAM ≈ 1 file (~600 MB), not the entire dataset.
Saves a Markdown report to data/stress_test_report.md.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = "/home/mma/polymarket-sniper/data"
FEE_RATE = 0.07  # Polymarket crypto taker rate; effective fee = rate*(1-ask)*stake

BASELINE = dict(ttl_min=90, ttl_max=110, mid_threshold=0.70,
                assets={"BTC"}, skip_hours={7, 9, 14})

TRAIN_END  = "20260430"
TEST_START = "20260501"

OB_COLS  = ["timestamp_utc", "market_slug", "asset", "condition_id",
            "side", "best_bid", "best_ask", "mid", "seconds_to_resolution"]
OB_DTYPE = {"best_bid": "float32", "best_ask": "float32", "mid": "float32",
            "seconds_to_resolution": "float32"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_outcomes(files: list[str]) -> dict[str, str]:
    frames = []
    for f in sorted(files):
        try:
            frames.append(pd.read_csv(f, usecols=["condition_id", "winner"]))
        except Exception:
            pass
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df[df["winner"].notna() & (df["winner"] != "unknown")]
    return df.drop_duplicates("condition_id").set_index("condition_id")["winner"].to_dict()


def _apply_one(ob: pd.DataFrame, outcomes: dict, p: dict) -> pd.DataFrame:
    mask = (
        (ob["seconds_to_resolution"] >= p["ttl_min"]) &
        (ob["seconds_to_resolution"] <= p["ttl_max"]) &
        (ob["mid"] >= p["mid_threshold"]) &
        (ob["asset"].isin(p["assets"])) &
        (~ob["hour"].isin(p["skip_hours"]))
    )
    signal = ob[mask]
    if signal.empty:
        return pd.DataFrame()
    entries = signal.drop_duplicates(subset=["condition_id", "side"], keep="first").copy()
    entries["winner"] = entries["condition_id"].map(outcomes)
    known = entries[entries["winner"].notna()].copy()
    if known.empty:
        return pd.DataFrame()
    known["won"] = (known["side"] == known["winner"]).astype(int)
    known["fee_usdc"] = FEE_RATE * (1.0 - known["best_ask"])
    known["payout"] = known["won"] * (1.0 / known["best_ask"])
    known["pnl"] = known["payout"] - 1.0 - known["fee_usdc"]
    return known


def stream_filter(files: list[str], outcomes: dict, param_grid: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """
    Stream files one at a time and apply every param set per file.
    Peak RAM ≈ one file in memory + tiny accumulated trades.
    """
    accum: dict[str, list] = {k: [] for k in param_grid}
    n = len(files)
    for i, f in enumerate(sorted(files)):
        try:
            ob = pd.read_csv(f, usecols=OB_COLS, dtype=OB_DTYPE)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}", flush=True)
            continue
        ob["timestamp_utc"] = pd.to_datetime(ob["timestamp_utc"], format="ISO8601", utc=True)
        ob["asset"] = ob["asset"].str.upper().str.strip()
        ob["hour"] = ob["timestamp_utc"].dt.hour
        print(f"  [{i+1}/{n}] {os.path.basename(f)}: {len(ob):,} rows", flush=True)
        for key, p in param_grid.items():
            res = _apply_one(ob, outcomes, p)
            if not res.empty:
                accum[key].append(res)
        del ob

    out: dict[str, pd.DataFrame] = {}
    for k, frames in accum.items():
        if not frames:
            out[k] = pd.DataFrame()
        else:
            df = pd.concat(frames, ignore_index=True)
            df = (df.drop_duplicates(subset=["condition_id", "side"])
                    .sort_values("timestamp_utc")
                    .reset_index(drop=True))
            out[k] = df
    return out


def _combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Concat two trade frames and dedup."""
    parts = [x for x in [a, b] if not x.empty]
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
              .drop_duplicates(subset=["condition_id", "side"])
              .sort_values("timestamp_utc")
              .reset_index(drop=True))


def metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 5:
        return {"n": 0}
    wr = df["won"].mean()
    pnl_arr = df["pnl"].values.astype(float)
    sharpe = pnl_arr.mean() / (pnl_arr.std() + 1e-9)
    cum = np.cumsum(pnl_arr)
    max_dd = float((cum - np.maximum.accumulate(cum)).min())
    rng = np.random.default_rng(42)
    boot = rng.choice(df["won"].values, size=(2000, len(df)), replace=True).mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {
        "n": len(df),
        "win_rate": round(float(wr), 4),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "total_pnl": round(float(pnl_arr.sum()), 4),
        "mean_pnl": round(float(pnl_arr.mean()), 5),
        "sharpe": round(float(sharpe), 4),
        "max_dd": round(max_dd, 4),
    }


def fmt(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "insufficient data"
    return (
        f"n={m['n']}  WR={m['win_rate']:.1%}  CI=[{m['ci_lo']:.3f},{m['ci_hi']:.3f}]  "
        f"PnL=${m['total_pnl']:.2f}  Sharpe={m['sharpe']:.4f}  DD=${m['max_dd']:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# File lists
# ─────────────────────────────────────────────────────────────────────────────

all_ob_files = sorted(glob.glob(f"{DATA}/orderbook_*.csv"))
all_pt_files = sorted(glob.glob(f"{DATA}/paper_trades_*.csv"))

def _ob_date(p):  return p.split("orderbook_")[1].replace(".csv", "")
def _pt_date(p):  return p.split("paper_trades_")[1].replace(".csv", "")

train_ob_files = [f for f in all_ob_files if _ob_date(f) <= TRAIN_END]
test_ob_files  = [f for f in all_ob_files if _ob_date(f) >= TEST_START]
train_pt_files = [f for f in all_pt_files if _pt_date(f) <= TRAIN_END]
test_pt_files  = [f for f in all_pt_files if _pt_date(f) >= TEST_START]

print(f"Train: {len(train_ob_files)} ob files, {len(train_pt_files)} pt files  (Apr 18-30)")
print(f"Test:  {len(test_ob_files)} ob files, {len(test_pt_files)} pt files   (May 1-7)")


# ─────────────────────────────────────────────────────────────────────────────
# Build full parameter grid — streamed in two passes (train + test)
# ─────────────────────────────────────────────────────────────────────────────

ttl_windows    = [(80,100),(85,105),(90,110),(95,115),(100,120),(105,110)]
mid_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
skip_variants  = {
    "none":                    set(),
    "baseline {7,9,14}":       {7, 9, 14},
    "extended {2,7,9,14,18}":  {2, 7, 9, 14, 18},
}
asset_variants = [
    ({"BTC"},       "BTC only"),
    ({"ETH"},       "ETH only"),
    ({"BTC","ETH"}, "BTC+ETH"),
]
sub_buckets = [(90,95),(95,100),(100,105),(105,110)]

weeks = [
    ("Week 1 (Apr 18-24)", "20260418", "20260424"),
    ("Week 2 (Apr 25-May 1)", "20260425", "20260501"),
    ("Week 3 (May 2-7)",   "20260502", "20260507"),
]

param_grid: dict[str, dict] = {"baseline": dict(BASELINE)}
for tmin, tmax in ttl_windows:
    param_grid[f"ttl_{tmin}_{tmax}"] = dict(BASELINE, ttl_min=tmin, ttl_max=tmax)
for thresh in mid_thresholds:
    param_grid[f"mid_{thresh}"] = dict(BASELINE, mid_threshold=thresh)
for label, skip_set in skip_variants.items():
    param_grid[f"skip_{label}"] = dict(BASELINE, skip_hours=skip_set)
for asset_set, label in asset_variants:
    param_grid[f"asset_{label}"] = dict(BASELINE, assets=frozenset(asset_set))
for tmin, tmax in sub_buckets:
    param_grid[f"sub_{tmin}_{tmax}_train"] = dict(BASELINE, ttl_min=tmin, ttl_max=tmax)
    param_grid[f"sub_{tmin}_{tmax}_test"]  = dict(BASELINE, ttl_min=tmin, ttl_max=tmax)
param_grid["sub_90_100"]  = dict(BASELINE, ttl_min=90,  ttl_max=100)
param_grid["sub_105_110"] = dict(BASELINE, ttl_min=105, ttl_max=110)

# Joint combos for final param selection: mid × skip_hours (BTC, TTL 90-110s)
joint_combos = [
    ("jt_m70_sB", dict(BASELINE, mid_threshold=0.70, skip_hours={7, 9, 14})),
    ("jt_m70_sE", dict(BASELINE, mid_threshold=0.70, skip_hours={2, 7, 9, 14, 18})),
    ("jt_m75_sB", dict(BASELINE, mid_threshold=0.75, skip_hours={7, 9, 14})),
    ("jt_m75_sE", dict(BASELINE, mid_threshold=0.75, skip_hours={2, 7, 9, 14, 18})),
    ("jt_m80_sE", dict(BASELINE, mid_threshold=0.80, skip_hours={2, 7, 9, 14, 18})),
]
for key, p in joint_combos:
    param_grid[key] = p


# ─────────────────────────────────────────────────────────────────────────────
# Stream train files
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Loading outcomes ===")
outcomes_train = load_outcomes(train_pt_files)
outcomes_test  = load_outcomes(test_pt_files)
outcomes_full  = {**outcomes_train, **outcomes_test}
print(f"  train={len(outcomes_train):,}  test={len(outcomes_test):,}  total={len(outcomes_full):,}", flush=True)

# Train-only params (sub-buckets IS split) + shared params
train_param_grid = {k: v for k, v in param_grid.items() if not k.endswith("_test")}
print("\n=== Streaming TRAIN files ===", flush=True)
train_results = stream_filter(train_ob_files, outcomes_train, train_param_grid)

# Test-only params
test_param_grid = {k: v for k, v in param_grid.items() if not k.endswith("_train")}
print("\n=== Streaming TEST files ===", flush=True)
test_results = stream_filter(test_ob_files, outcomes_test, test_param_grid)

# ─────────────────────────────────────────────────────────────────────────────
# Assemble named result DataFrames
# ─────────────────────────────────────────────────────────────────────────────

train_trades = train_results["baseline"]
test_trades  = test_results["baseline"]
full_trades  = _combine(train_trades, test_trades)

print(f"\nBaseline trades — full={len(full_trades)}, train={len(train_trades)}, test={len(test_trades)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Walk-forward
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1 — Walk-forward (out-of-sample)")
print("="*60)

train_m = metrics(train_trades)
test_m  = metrics(test_trades)

print(f"  IN-SAMPLE  (Apr 18-30): {fmt(train_m)}")
print(f"  OUT-OF-SAMPLE (May 1-7): {fmt(test_m)}")

if train_m.get("n", 0) > 0 and test_m.get("n", 0) > 0:
    delta = test_m["win_rate"] - train_m["win_rate"]
    print(f"  WR delta (OOS - IS): {delta:+.1%}")
    if test_m["ci_lo"] > 0.50:
        print("  VERDICT: Edge holds OOS (CI lower bound > 50%).")
    else:
        print("  VERDICT: WARNING — OOS CI lower bound <= 50%.")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Temporal stability
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2 — Temporal stability")
print("="*60)

print("\n--- Weekly ---")
week_results = []
for label, start, end in weeks:
    # Re-use train/test results filtered by date
    wk_frames = []
    for df in [train_trades, test_trades]:
        if df.empty:
            continue
        ts = df["timestamp_utc"]
        s = pd.Timestamp(f"{start[:4]}-{start[4:6]}-{start[6:8]}", tz="UTC")
        e = pd.Timestamp(f"{end[:4]}-{end[4:6]}-{end[6:8]}", tz="UTC") + pd.Timedelta(days=1)
        chunk = df[(ts >= s) & (ts < e)]
        if not chunk.empty:
            wk_frames.append(chunk)
    wk_df = pd.concat(wk_frames, ignore_index=True) if wk_frames else pd.DataFrame()
    wk_m = metrics(wk_df)
    week_results.append((label, wk_m))
    print(f"  {label}: {fmt(wk_m)}")

print("\n--- Daily win rate ---")
if not full_trades.empty:
    full_trades["date"] = full_trades["timestamp_utc"].dt.date
    daily = (full_trades.groupby("date")
             .agg(n=("won", "count"), win_rate=("won", "mean"), pnl=("pnl", "sum"))
             .reset_index())
    for _, row in daily.iterrows():
        bar = "#" * int(row["win_rate"] * 30)
        print(f"  {row['date']}  n={row['n']:4d}  WR={row['win_rate']:.1%}  {bar}")

print("\n--- Rolling 300-trade win rate ---")
if not full_trades.empty:
    roll = full_trades["won"].rolling(300, min_periods=50).mean()
    for i in range(299, len(full_trades), 300):
        ts = full_trades["timestamp_utc"].iloc[i].strftime("%b %d")
        print(f"  trade {i+1:4d} ({ts}): rolling WR={roll.iloc[i]:.1%}")
    last = len(full_trades) - 1
    print(f"  trade {last+1:4d} ({full_trades['timestamp_utc'].iloc[last].strftime('%b %d')}): "
          f"rolling WR={roll.iloc[last]:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Parameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3 — Parameter sensitivity (all in-memory from streamed results)")
print("="*60)

def _full_for(key: str) -> pd.DataFrame:
    """Combine train+test results for a given param key."""
    tr = train_results.get(key, pd.DataFrame())
    te = test_results.get(key, pd.DataFrame())
    return _combine(tr, te)

print("\n--- TTL window (mid≥70%, BTC, skip={7,9,14}) ---")
ttl_rows = []
for tmin, tmax in ttl_windows:
    m = metrics(_full_for(f"ttl_{tmin}_{tmax}"))
    ttl_rows.append({"TTL": f"{tmin}-{tmax}s", **m})
    print(f"  {tmin}-{tmax}s: {fmt(m)}")

print("\n--- Mid threshold (TTL=90-110s, BTC, skip={7,9,14}) ---")
mid_rows = []
for thresh in mid_thresholds:
    m = metrics(_full_for(f"mid_{thresh}"))
    mid_rows.append({"mid": f">={thresh:.0%}", **m})
    print(f"  mid>={thresh:.0%}: {fmt(m)}")

print("\n--- Skip-hours variant (TTL=90-110s, mid≥70%, BTC) ---")
hour_rows = []
for label, _ in skip_variants.items():
    m = metrics(_full_for(f"skip_{label}"))
    hour_rows.append({"skip_hours": label, **m})
    print(f"  {label}: {fmt(m)}")

print("\n--- Asset (TTL=90-110s, mid≥70%, skip={7,9,14}) ---")
asset_rows = []
for _, label in asset_variants:
    m = metrics(_full_for(f"asset_{label}"))
    asset_rows.append({"assets": label, **m})
    print(f"  {label}: {fmt(m)}")


# ─────────────────────────────────────────────────────────────────────────────
# PARAM SELECTION — joint mid × skip-hours grid
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PARAM SELECTION — joint mid × skip-hours (BTC, TTL 90-110s)")
print("="*60)

selection_rows = []
for key, p in joint_combos:
    m = metrics(_full_for(key))
    label = f"mid≥{p['mid_threshold']:.0%}, skip={sorted(p['skip_hours'])}"
    selection_rows.append({"label": label, "mid": p["mid_threshold"],
                           "skip": sorted(p["skip_hours"]), **m})
    print(f"  {label}: {fmt(m)}")

valid_sel = [r for r in selection_rows if r.get("n", 0) > 0]
best_sel = max(valid_sel, key=lambda r: r.get("sharpe", -999)) if valid_sel else None
if best_sel:
    print(f"\n  *** RECOMMENDED: {best_sel['label']} ***")
    print(f"      Sharpe={best_sel['sharpe']:.4f}  WR={best_sel['win_rate']:.1%}"
          f"  n={best_sel['n']}  PnL=${best_sel['total_pnl']:.2f}"
          f"  CI=[{best_sel['ci_lo']:.3f},{best_sel['ci_hi']:.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 4 — Monte Carlo (5,000 paths)")
print("="*60)

N_SIM = 5_000
BATCH = 500
STAKE_SIZES = [1, 5, 10, 20, 50]

if not full_trades.empty:
    pnl_unit = full_trades["pnl"].values.astype(float)
    n_trades = len(pnl_unit)

    rng = np.random.default_rng(42)
    sim_final  = np.empty(N_SIM)
    sim_max_dd = np.empty(N_SIM)
    for i in range(0, N_SIM, BATCH):
        b = min(BATCH, N_SIM - i)
        batch_pnl = rng.choice(pnl_unit, size=(b, n_trades), replace=True)
        batch_cum = np.cumsum(batch_pnl, axis=1)
        run_max   = np.maximum.accumulate(batch_cum, axis=1)
        sim_max_dd[i:i+b] = (batch_cum - run_max).min(axis=1)
        sim_final[i:i+b]  = batch_cum[:, -1]
        del batch_pnl, batch_cum, run_max

    print(f"\n  Trades per simulation: {n_trades}")
    print("\n--- Final equity percentiles (per $1 stake) ---")
    for pct in [5, 25, 50, 75, 95]:
        print(f"  P{pct:2d}: ${np.percentile(sim_final, pct):.2f}")

    print("\n--- Max drawdown distribution (per $1 stake) ---")
    for pct in [50, 75, 90, 95, 99]:
        print(f"  P{pct:2d} worst drawdown: ${np.percentile(sim_max_dd, pct):.2f}")

    print(f"\n--- Projected drawdown & profit at real stake sizes ---")
    p95_dd_unit  = float(np.percentile(sim_max_dd, 95))
    p50_fin_unit = float(np.percentile(sim_final, 50))
    for stake in STAKE_SIZES:
        print(f"  ${stake:3d}/trade → P95 max drawdown: ${p95_dd_unit*stake:.0f}  |  "
              f"median profit over {n_trades} trades: ${p50_fin_unit*stake:.0f}")

    print("\n--- Ruin probability (drawdown > 30% of 20x-stake bankroll) ---")
    for stake in STAKE_SIZES:
        bankroll = 20 * stake
        threshold_unit = -(0.30 * bankroll / stake)
        ruin_p = (sim_max_dd < threshold_unit).mean()
        print(f"  ${stake:3d}/trade  bankroll=${bankroll}  "
              f"ruin_threshold=${-threshold_unit*stake:.0f}  ruin_prob={ruin_p:.2%}")

    avg_ask = float(full_trades["best_ask"].mean())
    wr_full = float(full_trades["won"].mean())
    b_val = (1.0 / avg_ask) - 1.0
    kelly     = wr_full - (1 - wr_full) / b_val
    half_kelly = kelly / 2
    print(f"\n--- Kelly criterion ---")
    print(f"  avg ask={avg_ask:.4f}  WR={wr_full:.4f}  b={b_val:.4f}")
    print(f"  Full Kelly: {kelly:.4f} ({kelly*100:.2f}% of bankroll per trade)")
    print(f"  Half Kelly: {half_kelly:.4f} ({half_kelly*100:.2f}% of bankroll per trade)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — TTL sub-bucket deep dive
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 5 — TTL sub-bucket deep dive (100-105s was losing)")
print("="*60)

sub_results = {}
for tmin, tmax in sub_buckets:
    tr_m = metrics(train_results.get(f"sub_{tmin}_{tmax}_train", pd.DataFrame()))
    te_m = metrics(test_results.get(f"sub_{tmin}_{tmax}_test",  pd.DataFrame()))
    sub_results[(tmin, tmax)] = (tr_m, te_m)
    print(f"  TTL {tmin}-{tmax}s  IS : {fmt(tr_m)}")
    print(f"             OOS: {fmt(te_m)}")

print("\n--- Edge 4 with 100-105s bucket dropped (TTL 90-100s + 105-110s combined) ---")
t1 = _full_for("sub_90_100")
t2 = _full_for("sub_105_110")
if not t1.empty or not t2.empty:
    t_combined = _combine(t1, t2)
    m_drop = metrics(t_combined)
    print(f"  Without 100-105s: {fmt(m_drop)}")
    print(f"  With    100-105s: {fmt(metrics(full_trades))}")


# ─────────────────────────────────────────────────────────────────────────────
# Save Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def m_row(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "| — | — | — | — | — | — | — |"
    return (
        f"| {m['n']} | {m['win_rate']:.1%} "
        f"| [{m['ci_lo']:.3f},{m['ci_hi']:.3f}] "
        f"| ${m['total_pnl']:.2f} | {m['mean_pnl']:.5f} "
        f"| {m['sharpe']:.4f} | ${m['max_dd']:.2f} |"
    )

HDR = "| Split | n | WR | CI 95% | Total PnL | Mean PnL | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|"

report_lines = [
    "# Stress Test Report — Edge 4: Momentum within Epoch",
    "**Baseline**: mid≥70%, TTL 90-110s, BTC only, skip UTC hours {7,9,14}, fee=0.07×(1-ask)×stake (Polymarket crypto taker)",
    "**Data**: Apr 18 – May 7 2026 | **In-sample**: Apr 18-30 | **Out-of-sample**: May 1-7",
    "",
    "---",
    "## 1. Walk-forward (Out-of-Sample)",
    HDR,
    f"| In-sample (Apr 18-30) {m_row(train_m)}",
    f"| Out-of-sample (May 1-7) {m_row(test_m)}",
    "",
]
if train_m.get("n") and test_m.get("n"):
    delta = test_m["win_rate"] - train_m["win_rate"]
    verdict = ("Edge holds OOS — CI lower bound > 50%."
               if test_m.get("ci_lo", 0) > 0.50
               else "WARNING: OOS CI lower bound <= 50%.")
    report_lines += [f"**WR delta (OOS − IS)**: {delta:+.1%}  |  **Verdict**: {verdict}", ""]

report_lines += [
    "---",
    "## 2. Temporal Stability — Weekly",
    HDR,
]
for label, wk_m in week_results:
    report_lines.append(f"| {label} {m_row(wk_m)}")

if not full_trades.empty:
    report_lines += [
        "",
        "### Daily win rate",
        daily.assign(
            win_rate=daily["win_rate"].map("{:.1%}".format),
            pnl=daily["pnl"].map("${:.2f}".format)
        ).to_markdown(index=False),
    ]

report_lines += [
    "",
    "---",
    "## 3. Parameter Sensitivity",
    "",
    "### TTL window",
    "| TTL | n | WR | CI 95% | Total PnL | Mean PnL | Sharpe | Max DD |",
    "|---|---|---|---|---|---|---|---|",
]
for row in ttl_rows:
    report_lines.append(f"| {row['TTL']} {m_row(row)}")

report_lines += [
    "",
    "### Mid threshold",
    "| Mid | n | WR | CI 95% | Total PnL | Mean PnL | Sharpe | Max DD |",
    "|---|---|---|---|---|---|---|---|",
]
for row in mid_rows:
    report_lines.append(f"| {row['mid']} {m_row(row)}")

report_lines += [
    "",
    "### Skip-hours variant",
    "| Skip hours | n | WR | CI 95% | Total PnL | Mean PnL | Sharpe | Max DD |",
    "|---|---|---|---|---|---|---|---|",
]
for row in hour_rows:
    report_lines.append(f"| {row['skip_hours']} {m_row(row)}")

report_lines += [
    "",
    "### Asset filter",
    "| Assets | n | WR | CI 95% | Total PnL | Mean PnL | Sharpe | Max DD |",
    "|---|---|---|---|---|---|---|---|",
]
for row in asset_rows:
    report_lines.append(f"| {row['assets']} {m_row(row)}")

report_lines += [
    "",
    "### Joint param selection (mid × skip-hours, BTC, TTL 90-110s)",
    "| Params | n | WR | CI 95% | Total PnL | Mean PnL | Sharpe | Max DD |",
    "|---|---|---|---|---|---|---|---|",
]
for row in selection_rows:
    report_lines.append(f"| {row['label']} {m_row(row)}")
if best_sel:
    report_lines += [
        "",
        f"**RECOMMENDED LIVE PARAMS**: {best_sel['label']}",
        f"Sharpe={best_sel['sharpe']:.4f} | WR={best_sel['win_rate']:.1%} | "
        f"n={best_sel['n']} | PnL=${best_sel['total_pnl']:.2f} | "
        f"CI=[{best_sel['ci_lo']:.3f},{best_sel['ci_hi']:.3f}]",
    ]

if not full_trades.empty:
    report_lines += [
        "",
        "---",
        "## 4. Monte Carlo (5,000 paths, bootstrap with replacement)",
        f"Trades per simulation: {n_trades}",
        "",
        "### Final equity percentiles (per $1 stake)",
        "| Pct | Final PnL |",
        "|---|---|",
    ]
    for pct in [5, 25, 50, 75, 95]:
        report_lines.append(f"| P{pct} | ${np.percentile(sim_final, pct):.2f} |")

    report_lines += [
        "",
        "### Drawdown & profit at real stake sizes",
        "| Stake | P95 max drawdown | Median profit |",
        "|---|---|---|",
    ]
    for stake in STAKE_SIZES:
        report_lines.append(
            f"| ${stake}/trade | ${p95_dd_unit*stake:.0f} | ${p50_fin_unit*stake:.0f} |"
        )

    report_lines += [
        "",
        "### Ruin probability (drawdown > 30% of 20x-stake bankroll)",
        "| Stake | Bankroll | Ruin threshold | Ruin prob |",
        "|---|---|---|---|",
    ]
    for stake in STAKE_SIZES:
        bankroll = 20 * stake
        rt_unit = -(0.30 * bankroll / stake)
        rp = (sim_max_dd < rt_unit).mean()
        report_lines.append(
            f"| ${stake}/trade | ${bankroll} | -${-rt_unit*stake:.0f} | {rp:.2%} |"
        )

    report_lines += [
        "",
        f"**Full Kelly**: {kelly:.4f} ({kelly*100:.2f}% of bankroll/trade)",
        f"**Half Kelly**: {half_kelly:.4f} ({half_kelly*100:.2f}% of bankroll/trade)",
    ]

report_lines += [
    "",
    "---",
    "## 5. TTL Sub-bucket Walk-forward",
    "| TTL | Split | n | WR | CI 95% | Total PnL | Sharpe | Max DD |",
    "|---|---|---|---|---|---|---|---|",
]
for (tmin, tmax), (tr_m2, te_m2) in sub_results.items():
    report_lines.append(f"| {tmin}-{tmax}s | In-sample {m_row(tr_m2)}")
    report_lines.append(f"| {tmin}-{tmax}s | Out-of-sample {m_row(te_m2)}")

if not t1.empty or not t2.empty:
    report_lines += [
        "",
        f"**Without 100-105s bucket**: {fmt(m_drop)}",
        f"**With 100-105s bucket**:    {fmt(metrics(full_trades))}",
    ]

report_lines += ["", "---", ""]

report_path = f"{DATA}/stress_test_report.md"
with open(report_path, "w") as fh:
    fh.write("\n".join(report_lines) + "\n")
print(f"\nReport saved → {report_path}")
