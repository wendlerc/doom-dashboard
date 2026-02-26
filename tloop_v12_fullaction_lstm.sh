#!/bin/bash
# tloop_v12_fullaction_lstm.sh — LSTM agent on fullaction deathmatch (with monsters + weapon select)
# Fullaction has 18 buttons → 38 macro actions
set -e
cd /share/NFS/u/wendler/code/doom-dashboard
export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/share/NFS/u/wendler/.local/lib:${LD_LIBRARY_PATH:-}"

INIT_MODEL_ARG=""
if [ -f "trained_policies/bc_fullaction_lstm.zip" ]; then
    INIT_MODEL_ARG="--init-model trained_policies/bc_fullaction_lstm.zip"
    echo "Using BC-pretrained init model"
fi

uv run python train_overnight_dm.py \
    --name "v12a_fullaction_lstm" \
    --cfg doom_dashboard/scenarios/deathmatch_fullaction.cfg \
    --maps map01 --timesteps 4000000 --envs 2 --bots 1 --bots-eval 1 \
    --timelimit-minutes 2.5 --frame-skip 4 \
    --obs-height 120 --obs-width 160 \
    --n-steps 1024 --batch-size 1024 --n-epochs 4 \
    --learning-rate 3e-4 --ent-coef 0.05 --target-kl 0.05 \
    --policy-hidden-size 512 --policy-hidden-layers 2 \
    --cnn-type impala --cnn-features-dim 256 \
    --policy-type lstm --lstm-hidden-size 256 --lstm-num-layers 1 \
    --video-freq 200000 \
    --bench-episodes 16 --bench-timelimit 2.0 \
    --device cuda --wandb \
    --frag-bonus 200 --hit-bonus 12.0 --damage-bonus 0.2 \
    --death-penalty 30 --attack-bonus 0.05 --move-bonus 0.2 \
    --noop-penalty 0.1 --reward-scale 0.1 \
    $INIT_MODEL_ARG \
    >> fight_v12a_fullaction_lstm.log 2>&1

echo "[tloop] v12a_fullaction_lstm finished"
