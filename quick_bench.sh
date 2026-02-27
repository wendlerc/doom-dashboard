#!/bin/bash
# Quick 50k-step experiments to compare configs
set -e
cd /share/NFS/u/wendler/code/doom-dashboard
export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/share/NFS/u/wendler/.local/lib:${LD_LIBRARY_PATH:-}"

COMMON="--cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
    --maps map01 --timesteps 50000 --envs 2 --bots 1 --bots-eval 1 \
    --timelimit-minutes 2.5 --frame-skip 4 \
    --obs-height 120 --obs-width 160 --n-epochs 4 \
    --learning-rate 3e-4 --target-kl 0.05 \
    --cnn-type impala --cnn-features-dim 256 \
    --video-freq 0 --bench-episodes 8 --bench-timelimit 1.0 \
    --device cuda \
    --frag-bonus 200 --hit-bonus 12.0 --damage-bonus 0.2 \
    --death-penalty 30 --attack-bonus 0.05 --move-bonus 0.2 \
    --noop-penalty 0.1 --reward-scale 0.1"

run() {
    local name=$1; shift
    echo -n "[$name] "
    local t0=$(date +%s)
    uv run python train_overnight_dm.py --name "qb_${name}" $COMMON "$@" 2>&1 | tail -1
    local t1=$(date +%s)
    local bench="trained_policies/qb_${name}_bench_mp.json"
    if [ -f "$bench" ]; then
        python3 -c "
import json
with open('$bench') as f: d=json.load(f)
h=d.get('model_hit_mean',0); dm=d.get('model_damage_mean',0)
ds=d.get('model_distance_mean',0); p=d.get('pass',False)
print(f'  hits={h:.1f} dmg={dm:.0f} dist={ds:.0f} pass={p} ({int($t1-$t0)}s)')
"
    else
        echo "  no_bench (${t1-t0}s)"
    fi
    sleep 2
}

echo "=== Quick Bench Suite (50k steps each) ==="

run "mlp" --policy-type mlp --policy-hidden-size 512 --policy-hidden-layers 2 \
    --n-steps 1024 --batch-size 2048 --ent-coef 0.03

run "lstm256" --policy-type lstm --lstm-hidden-size 256 --lstm-num-layers 1 \
    --policy-hidden-size 512 --policy-hidden-layers 2 \
    --n-steps 1024 --batch-size 1024 --ent-coef 0.03

run "mlp_fs4" --policy-type mlp --policy-hidden-size 512 --policy-hidden-layers 2 \
    --frame-stack 4 --n-steps 1024 --batch-size 2048 --ent-coef 0.03

run "lstm256_fs4" --policy-type lstm --lstm-hidden-size 256 --lstm-num-layers 1 \
    --policy-hidden-size 512 --policy-hidden-layers 2 \
    --frame-stack 4 --n-steps 1024 --batch-size 1024 --ent-coef 0.03

echo ""
echo "=== Done ==="
