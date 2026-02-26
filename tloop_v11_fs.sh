#!/bin/bash
# FIGHT v11b — MLP + 4-frame stacking (2 envs to share GPU)
export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/share/NFS/u/wendler/.local/lib:${LD_LIBRARY_PATH:-}"
cd /share/NFS/u/wendler/code/doom-dashboard

LOG="fight_v11b_fs.log"
MAX=20
n=0
NAME="v11b_framestack"
TRAIN_PID=""

cleanup() {
    echo "[L $(date +%H:%M:%S)] cleanup" >> "$LOG"
    [ -n "$TRAIN_PID" ] && kill -9 "$TRAIN_PID" 2>/dev/null
    exit 0
}
trap cleanup TERM INT HUP

while [ $n -lt $MAX ]; do
    ia=""
    d="trained_policies/${NAME}_ckpts"
    if [ -d "$d" ]; then
        f=$(ls -t "$d"/ppo_*_steps.zip 2>/dev/null | head -1)
        [ -n "$f" ] && ia="--init-model $f" && echo "[L $(date +%H:%M:%S)] resume $(basename "$f")" >> "$LOG"
    fi

    echo "[L $(date +%H:%M:%S)] go n=$n v11b_framestack 2envs fs=4" >> "$LOG"

    uv run python train_overnight_dm.py \
        --name "$NAME" \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --maps map01 --timesteps 4000000 --envs 2 --bots 1 --bots-eval 1 \
        --timelimit-minutes 2.5 --frame-skip 4 \
        --obs-height 120 --obs-width 160 \
        --n-steps 1024 --batch-size 2048 --n-epochs 4 \
        --learning-rate 3e-4 --ent-coef 0.03 --target-kl 0.05 \
        --policy-hidden-size 512 --policy-hidden-layers 2 \
        --cnn-type impala --cnn-features-dim 256 \
        --policy-type mlp --frame-stack 4 \
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
    [ $rc -eq 0 ] && echo "[L $(date +%H:%M:%S)] done ok" >> "$LOG" && break
    n=$((n + 1))
    echo "[L $(date +%H:%M:%S)] crash $n rc=$rc" >> "$LOG"
    sleep 5
done
echo "[L $(date +%H:%M:%S)] COMPLETE" >> "$LOG"
