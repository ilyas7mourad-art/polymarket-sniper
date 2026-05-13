#!/bin/bash
# Quick health check for the observer

set -e
cd ~/polymarket-sniper

echo "===== TMUX SESSION ====="
if tmux has-session -t observer 2>/dev/null; then
    echo "✓ observer session is running"
else
    echo "✗ observer session NOT running"
    exit 1
fi

echo ""
echo "===== LATEST 5 LOG LINES ====="
tmux capture-pane -t observer -p | tail -5

echo ""
echo "===== CSV STATS ====="
LATEST_CSV=$(ls -t data/orderbook_*.csv 2>/dev/null | head -1)
if [ -z "$LATEST_CSV" ]; then
    echo "No CSV files found yet"
    exit 0
fi
TOTAL_ROWS=$(wc -l < "$LATEST_CSV")
SIZE=$(du -h "$LATEST_CSV" | cut -f1)
LAST_TIMESTAMP=$(tail -1 "$LATEST_CSV" | cut -d',' -f1)
echo "Latest CSV: $LATEST_CSV"
echo "  Size: $SIZE"
echo "  Total rows: $TOTAL_ROWS"
echo "  Last row timestamp: $LAST_TIMESTAMP"

echo ""
echo "===== ROWS LOGGED IN LAST HOUR ====="
ONE_HOUR_AGO=$(date -u -d '1 hour ago' +%Y-%m-%dT%H)
RECENT=$(awk -F',' -v cutoff="$ONE_HOUR_AGO" 'NR>1 && $1 > cutoff' "$LATEST_CSV" | wc -l)
echo "Rows: $RECENT"

echo ""
echo "===== DISK SPACE ====="
df -h ~ | tail -1
