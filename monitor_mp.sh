#!/usr/bin/env bash
# monitor_mp.sh — live monitor for mp dataset generation
# Usage: bash monitor_mp.sh [interval_seconds]

INTERVAL=${1:-30}
LOG=/share/NFS/u/wendler/code/doom-dashboard/mp_gen.log
SHARD_DIR=/share/NFS/u/wendler/code/doom-dashboard/mp_dataset

while true; do
    clear
    echo "══════════════════════════════════════════════════════"
    echo " MP Dataset Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════════════"

    # Check if generation is still running
    if pgrep -f "generate-mp-dataset" > /dev/null 2>&1; then
        echo " STATUS: ✓ Running (PID $(pgrep -f 'generate-mp-dataset' | head -1))"
    else
        echo " STATUS: ✗ Not running — check log for errors"
    fi

    echo ""
    echo "── Shards ──────────────────────────────────────────"
    TOTAL_SHARDS=$(ls "$SHARD_DIR"/mp-shard-*.tar 2>/dev/null | wc -l)
    NONEMPTY=$(ls -s "$SHARD_DIR"/mp-shard-*.tar 2>/dev/null | awk '$1 > 0 {count++} END {print count+0}')
    TOTAL_SIZE=$(du -sh "$SHARD_DIR" 2>/dev/null | cut -f1)
    echo "  Shards created:      $TOTAL_SHARDS"
    echo "  Non-empty shards:    $NONEMPTY"
    echo "  Total size on disk:  ${TOTAL_SIZE:-0}"

    # Count episodes from log
    EPISODES=$(grep -c '"done"' "$LOG" 2>/dev/null || grep -c 'game_tics' "$LOG" 2>/dev/null || echo "?")
    # Hours collected from last log line mentioning hours
    HOURS_LINE=$(grep -oE '[0-9]+\.[0-9]+ h/s|[0-9]+\.[0-9]+ h ' "$LOG" 2>/dev/null | tail -1)
    echo "  Episodes in log:     $(grep -c '("done"' "$LOG" 2>/dev/null || echo 0)"
    
    # Parse progress bar line from tqdm
    PROGRESS=$(grep -oE 'MP Dataset:.*\[.*\]' "$LOG" 2>/dev/null | tail -1)
    [ -n "$PROGRESS" ] && echo "  Progress:            $PROGRESS"

    echo ""
    echo "── Recent log ──────────────────────────────────────"
    if [ -f "$LOG" ]; then
        grep -vE "^Contacting|^Got connect|^Waiting for|^Total players|^Console player|^Received|^Found P|^Exchanging|^Sending all|^BindToPort|^Network game" "$LOG" | tail -12
    else
        echo "  (log not found)"
    fi

    echo ""
    echo "── GPU usage ───────────────────────────────────────"
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | wc -l)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU processes:  $GPU_PROCS"
    echo "  GPU memory:     $GPU_MEM"

    echo ""
    echo "  [Refreshing every ${INTERVAL}s — Ctrl+C to exit]"
    sleep "$INTERVAL"
done
