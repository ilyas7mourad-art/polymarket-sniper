#!/bin/bash
# Cleanly stop the observer (sends SIGTERM so the buffer flushes)

if tmux has-session -t observer 2>/dev/null; then
    echo "Sending Ctrl-C to observer (will trigger graceful flush)..."
    tmux send-keys -t observer C-c
    sleep 3
    tmux kill-session -t observer
    echo "Observer stopped."
else
    echo "No observer session running."
fi

# Show final CSV stats
LATEST_CSV=$(ls -t ~/polymarket-sniper/data/orderbook_*.csv 2>/dev/null | head -1)
if [ -n "$LATEST_CSV" ]; then
    echo ""
    echo "Final CSV: $LATEST_CSV"
    echo "  Size: $(du -h "$LATEST_CSV" | cut -f1)"
    echo "  Rows: $(wc -l < "$LATEST_CSV")"
fi
