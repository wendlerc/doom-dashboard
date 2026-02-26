#!/bin/bash
# FIGHT v11 — LSTM recurrent policy + IMPALA CNN
# Architecture: IMPALA CNN (256) + LSTM (256 hidden) + 2x512 MLP
# Key: position-based move reward, recurrent policy for temporal reasoning
export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/share/NFS/u/wendler/.local/lib:${LD_LIBRARY_PATH:-}"
cd /share/NFS/u/wendler/code/doom-dashboard

LOG="fight_v11.log"
MAX=20
n=0
NAME="overnight_fight_v11"
TRAIN_PID=""

cleanup() {
    echo "[L $(date +%H:%M:%S)] cleanup signal received" >> "$LOG"
    [ -n "$TRAIN_PID" ] && kill -9 "$TRAIN_PID" 2>/dev/null
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    for pid in $pids; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    done
    sleep 2
    exit 0
}
trap cleanup TERM INT HUP

cleanup_gpu() {
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    for pid in $pids; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    done
    sleep 3
}

# ─── Training with crash recovery + GPU cleanup ───
while [ $n -lt $MAX ]; do
    ia=""
    d="trained_policies/${NAME}_ckpts"
    if [ -d "$d" ]; then
        f=$(ls -t "$d"/ppo_*_steps.zip 2>/dev/null | head -1)
        [ -n "$f" ] && ia="--init-model $f" && echo "[L $(date +%H:%M:%S)] resume $(basename "$f")" >> "$LOG"
    fi

    echo "[L $(date +%H:%M:%S)] go n=$n v11_lstm ent=0.03 move=0.2 4envs impala+lstm256+2x512" >> "$LOG"

    uv run python train_overnight_dm.py \
        --name "$NAME" \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --maps map01 --timesteps 4000000 --envs 4 --bots 1 --bots-eval 1 \
        --timelimit-minutes 2.5 --frame-skip 4 \
        --obs-height 120 --obs-width 160 \
        --n-steps 2048 --batch-size 2048 --n-epochs 4 \
        --learning-rate 3e-4 --ent-coef 0.03 --target-kl 0.05 \
        --policy-hidden-size 512 --policy-hidden-layers 2 \
        --cnn-type impala --cnn-features-dim 256 \
        --policy-type lstm --lstm-hidden-size 256 --lstm-num-layers 1 \
        --video-freq 200000 \
        --bench-episodes 16 --bench-timelimit 2.0 \
        --device cuda --wandb \
        --frag-bonus 200 --hit-bonus 12.0 --damage-bonus 0.2 \
        --death-penalty 30 --attack-bonus 0.05 --move-bonus 0.2 \
        --noop-penalty 0.1 --reward-scale 0.1 \
        $ia >> "$LOG" 2>&1 &
    TRAIN_PID=$!
    wait $TRAIN_PID
    rc=$?
    TRAIN_PID=""

    [ $rc -eq 0 ] && echo "[L $(date +%H:%M:%S)] train done ok" >> "$LOG" && break

    n=$((n + 1))
    echo "[L $(date +%H:%M:%S)] crash $n rc=$rc — cleaning up GPU" >> "$LOG"
    cleanup_gpu
    sleep 5
done

echo "[L $(date +%H:%M:%S)] TRAINING COMPLETE" >> "$LOG"
