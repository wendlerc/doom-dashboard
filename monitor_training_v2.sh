#!/bin/bash
# Monitor training progress and auto-generate submission when models improve
set -e

BC_LOG="/tmp/bc_finetune_v2_highent.log"
SCRATCH_LOG="/tmp/scratch_v2_highent.log"
SUBMIT_DIR="results_latest/competition_submission_v3"
mkdir -p "$SUBMIT_DIR"

echo "=== Training Monitor v2 ==="
echo "BC+PPO log: $BC_LOG"
echo "Scratch log: $SCRATCH_LOG"

while true; do
    echo ""
    echo "=== $(date +%H:%M) ==="

    # BC+PPO progress
    BC_STEPS=$(grep "total_timesteps" "$BC_LOG" 2>/dev/null | tail -1 | grep -oP '\d+' | tail -1)
    BC_ENTROPY=$(grep "entropy_loss" "$BC_LOG" 2>/dev/null | tail -1 | grep -oP '[\d.-]+' | tail -1)
    BC_REWARD=$(grep "ep_rew_mean" "$BC_LOG" 2>/dev/null | tail -1 | grep -oP '[\d.-]+' | tail -1)
    echo "BC+PPO: ${BC_STEPS:-0} steps  entropy=${BC_ENTROPY:-?}  reward=${BC_REWARD:-?}"

    # Scratch progress
    SC_STEPS=$(grep "total_timesteps" "$SCRATCH_LOG" 2>/dev/null | tail -1 | grep -oP '\d+' | tail -1)
    SC_ENTROPY=$(grep "entropy_loss" "$SCRATCH_LOG" 2>/dev/null | tail -1 | grep -oP '[\d.-]+' | tail -1)
    SC_REWARD=$(grep "ep_rew_mean" "$SCRATCH_LOG" 2>/dev/null | tail -1 | grep -oP '[\d.-]+' | tail -1)
    echo "Scratch: ${SC_STEPS:-0} steps  entropy=${SC_ENTROPY:-?}  reward=${SC_REWARD:-?}"

    # Check for new best models
    BC_BEST="trained_policies/bc_finetune_v2_highent_best/best_model.zip"
    SC_BEST="trained_policies/scratch_v2_highent_best/best_model.zip"

    if [ -f "$BC_BEST" ]; then
        BC_MTIME=$(stat -c %Y "$BC_BEST" 2>/dev/null || echo 0)
        BC_SUBMIT="$SUBMIT_DIR/bc_v2_best.zip"
        BC_SUBMIT_MTIME=$(stat -c %Y "$BC_SUBMIT" 2>/dev/null || echo 0)
        if [ "$BC_MTIME" -gt "$BC_SUBMIT_MTIME" ]; then
            cp "$BC_BEST" "$BC_SUBMIT"
            cp "trained_policies/bc_finetune_v2_highent.meta.json" "$SUBMIT_DIR/bc_v2_best.meta.json" 2>/dev/null || true
            echo "  -> Updated $BC_SUBMIT"
        fi
    fi

    if [ -f "$SC_BEST" ]; then
        SC_MTIME=$(stat -c %Y "$SC_BEST" 2>/dev/null || echo 0)
        SC_SUBMIT="$SUBMIT_DIR/scratch_v2_best.zip"
        SC_SUBMIT_MTIME=$(stat -c %Y "$SC_SUBMIT" 2>/dev/null || echo 0)
        if [ "$SC_MTIME" -gt "$SC_SUBMIT_MTIME" ]; then
            cp "$SC_BEST" "$SC_SUBMIT"
            cp "trained_policies/scratch_v2_highent.meta.json" "$SUBMIT_DIR/scratch_v2_best.meta.json" 2>/dev/null || true
            echo "  -> Updated $SC_SUBMIT"
        fi
    fi

    sleep 300  # Check every 5 minutes
done
