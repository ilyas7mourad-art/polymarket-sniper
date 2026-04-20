#!/usr/bin/env python3
"""One-time backfill: re-query Polymarket CLOB API for any 'unknown' trades in the given CSV.

Usage:
    python scripts/backfill_unknowns.py path/to/paper_trades_YYYYMMDD.csv

Rewrites the CSV in place. The original is backed up to <name>.bak before any changes.
"""

import asyncio
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

UTC = timezone.utc
_CLOB_MARKETS_URL = "https://clob.polymarket.com/markets"


async def _fetch_winner(client: httpx.AsyncClient, condition_id: str) -> str | None:
    """Returns 'Up', 'Down', or None."""
    try:
        resp = await client.get(f"{_CLOB_MARKETS_URL}/{condition_id}", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  API error for {condition_id}: {exc}", file=sys.stderr)
        return None

    if not data.get("closed", False):
        return None
    for token in data.get("tokens", []):
        if token.get("winner") is True:
            outcome = token.get("outcome")
            if outcome in ("Up", "Down"):
                return outcome
    return None


async def main(csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows or not fieldnames:
        print("CSV is empty.", file=sys.stderr)
        return 1

    unknowns = [r for r in rows if r["winner"] == "unknown"]
    print(f"Found {len(unknowns)} 'unknown' rows out of {len(rows)} total.")

    if not unknowns:
        print("Nothing to do.")
        return 0

    backup_path = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup_path)
    print(f"Backed up original to: {backup_path}")

    fixed = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i, row in enumerate(unknowns, start=1):
            print(f"[{i}/{len(unknowns)}] {row['market_slug']}...", end=" ", flush=True)
            winner = await _fetch_winner(client, row["condition_id"])
            if winner is None:
                print("still unresolved or error")
                failed += 1
                continue

            side = row["side"]
            shares = float(row["simulated_shares"])
            stake = float(row["simulated_stake_usdc"])
            fee = float(row["fee_usdc"])

            if side == winner:
                payout = shares * 1.0
                pnl = payout - stake - fee
            else:
                payout = 0.0
                pnl = -stake - fee

            row["winner"] = winner
            row["payout_usdc"] = f"{payout:.4f}"
            row["pnl_usdc"] = f"{pnl:.4f}"
            row["resolution_timestamp_utc"] = datetime.now(UTC).isoformat(timespec="milliseconds")
            fixed += 1
            print(f"resolved as {winner}, pnl={pnl:+.4f}")

    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)

    print()
    print(f"Done. Fixed: {fixed}, still unresolved: {failed}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/backfill_unknowns.py path/to/paper_trades_YYYYMMDD.csv", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(main(Path(sys.argv[1]))))
