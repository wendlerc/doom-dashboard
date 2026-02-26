#!/bin/bash
# selfplay_overnight.sh — League-style self-play using train_overnight_dm.py
#
# Strategy:
#   Round 1: Train vs bots (init from best compact/fullaction model)
#   Round 2+: Train vs bots + evaluate vs previous round's model
#   Each round uses the previous round's best model as init
#
# Usage: nohup bash selfplay_overnight.sh [scenario] [init_model] &
#   scenario: deathmatch_compact (default) or deathmatch_fullaction
#   init_model: path to initial model (optional, will use latest best)
set -e
cd /share/NFS/u/wendler/code/doom-dashboard
export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/share/NFS/u/wendler/.local/lib:${LD_LIBRARY_PATH:-}"

SCENARIO="${1:-deathmatch_compact}"
INIT="${2:-}"
ROUNDS=4
STEPS_PER_ROUND=2000000
LOG="selfplay.log"

# Determine cfg path and policy type
if [[ "$SCENARIO" == *fullaction* ]]; then
    CFG="doom_dashboard/scenarios/deathmatch_fullaction.cfg"
    ENT_COEF="0.05"
else
    CFG="doom_dashboard/scenarios/deathmatch_compact.cfg"
    ENT_COEF="0.03"
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Find initial model if not specified
if [ -z "$INIT" ]; then
    for d in trained_policies/v12a_fullaction_lstm_best trained_policies/v11a_lstm_best; do
        if [ -f "$d/best_model.zip" ]; then
            INIT="$d/best_model.zip"
            break
        fi
    done
fi

log "=== Self-Play Training Started ==="
log "Scenario: $SCENARIO"
log "Config: $CFG"
log "Init model: ${INIT:-none (from scratch)}"
log "Rounds: $ROUNDS x $STEPS_PER_ROUND steps"

CURRENT_MODEL="$INIT"
POOL=()

for round in $(seq 1 $ROUNDS); do
    NAME="sp_r${round}_${SCENARIO}"
    INIT_ARG=""
    [ -n "$CURRENT_MODEL" ] && INIT_ARG="--init-model $CURRENT_MODEL"

    log "--- Round $round/$ROUNDS: Training $NAME ---"

    # Train with bots (+ higher bot count in later rounds)
    BOTS=$((1 + round / 2))
    [ "$BOTS" -gt 3 ] && BOTS=3

    uv run python train_overnight_dm.py \
        --name "$NAME" \
        --cfg "$CFG" \
        --maps map01 --timesteps "$STEPS_PER_ROUND" --envs 2 \
        --bots "$BOTS" --bots-eval "$BOTS" \
        --timelimit-minutes 2.5 --frame-skip 4 \
        --obs-height 120 --obs-width 160 \
        --n-steps 1024 --batch-size 1024 --n-epochs 4 \
        --learning-rate 3e-4 --ent-coef "$ENT_COEF" --target-kl 0.05 \
        --policy-hidden-size 512 --policy-hidden-layers 2 \
        --cnn-type impala --cnn-features-dim 256 \
        --policy-type lstm --lstm-hidden-size 256 --lstm-num-layers 1 \
        --video-freq 500000 \
        --bench-episodes 8 --bench-timelimit 2.0 \
        --device cuda --wandb \
        --frag-bonus 200 --hit-bonus 12.0 --damage-bonus 0.2 \
        --death-penalty 30 --attack-bonus 0.05 --move-bonus 0.2 \
        --noop-penalty 0.1 --reward-scale 0.1 \
        $INIT_ARG \
        2>&1 | tee -a "$LOG"

    # Find best model from this round
    BEST_DIR="trained_policies/${NAME}_best"
    CKPT_DIR="trained_policies/${NAME}_ckpts"
    if [ -f "$BEST_DIR/best_model.zip" ]; then
        ROUND_MODEL="$BEST_DIR/best_model.zip"
    else
        ROUND_MODEL=$(ls -t "$CKPT_DIR"/ppo_*_steps.zip 2>/dev/null | head -1)
    fi

    if [ -n "$ROUND_MODEL" ]; then
        log "Round $round best model: $ROUND_MODEL"
        POOL+=("$ROUND_MODEL")

        # Evaluate vs all previous round models using multiplayer bench
        if [ ${#POOL[@]} -gt 1 ]; then
            log "Evaluating round $round model vs pool (${#POOL[@]} opponents)..."
            for opp in "${POOL[@]}"; do
                [ "$opp" == "$ROUND_MODEL" ] && continue
                OPP_NAME=$(basename "$(dirname "$opp")")
                log "  vs $OPP_NAME..."
                uv run python bench_model.py "$ROUND_MODEL" \
                    --cfg "$CFG" --episodes 8 --timelimit 2.0 \
                    --out "trained_policies/${NAME}_vs_${OPP_NAME}.json" 2>&1 | tail -5 | tee -a "$LOG" || true
            done
        fi

        CURRENT_MODEL="$ROUND_MODEL"
    else
        log "Warning: no model found for round $round"
    fi

    log "Round $round complete"
done

# Copy final model to results
mkdir -p results_latest/selfplay
if [ -n "$CURRENT_MODEL" ]; then
    cp "$CURRENT_MODEL" results_latest/selfplay/final_selfplay.zip 2>/dev/null
    log "Final self-play model: results_latest/selfplay/final_selfplay.zip"
fi

log "=== Self-Play Training Complete ==="
