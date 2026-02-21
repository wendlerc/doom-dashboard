#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# post_training_launch.sh
#
# Run after train_policies.py completes:
#   1. Generate annotated dashboard sample videos (1 per scenario×policy)
#   2. Launch the 200h parallelized 1v1 multiplayer WebDataset generator
#
# Usage:
#   chmod +x post_training_launch.sh
#   xvfb-run -a bash post_training_launch.sh
#
# Or to run in background detached:
#   nohup xvfb-run -a bash post_training_launch.sh > post_training.log 2>&1 &
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config.yaml"
HOURS="${MP_HOURS:-200}"          # override with: MP_HOURS=50 bash ...
WORKERS="${MP_WORKERS:-}"         # blank = auto
RANDOM_RATIO="${RANDOM_RATIO:-0.05}"
RESOLUTION="${RESOLUTION:-RES_320X240}"
TIMELIMIT="${TIMELIMIT:-5.0}"

echo "============================================================"
echo " Doom Policy — Post-Training Launch"
echo " Target multiplayer hours : $HOURS"
echo " Random policy ratio      : $RANDOM_RATIO"
echo " Resolution               : $RESOLUTION"
echo " Timelimit per game (min) : $TIMELIMIT"
echo "============================================================"

# 1. Verify checkpoints exist
EXPECTED=(basic defend_the_center deadly_corridor health_gathering my_way_home)
MISSING=()
for ckpt in "${EXPECTED[@]}"; do
    if [[ ! -f "trained_policies/${ckpt}.zip" ]]; then
        MISSING+=("$ckpt")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "⚠  Missing checkpoints: ${MISSING[*]}"
    echo "   Run: xvfb-run -a uv run python train_policies.py"
    echo "   Proceeding with whatever policies exist + Random."
fi

# 2. Generate dashboard sample videos
echo ""
echo "── Step 1: Generating dashboard sample videos ──────────────"
uv run python -m doom_dashboard generate-samples \
    --config "$CONFIG" \
    --output-dir samples
echo "✓ Samples generated → samples/"

# 3. Generate the multiplayer dataset
echo ""
echo "── Step 2: Generating ${HOURS}h of 1v1 multiplayer gameplay ─"

WORKERS_ARG=""
if [[ -n "$WORKERS" ]]; then
    WORKERS_ARG="--workers $WORKERS"
fi

uv run python -m doom_dashboard generate-mp-dataset \
    --config "$CONFIG" \
    --output-dir mp_dataset \
    --hours "$HOURS" \
    --random-ratio "$RANDOM_RATIO" \
    --resolution "$RESOLUTION" \
    --timelimit "$TIMELIMIT" \
    --frame-skip 4 \
    --shard-mb 512 \
    ${WORKERS_ARG}

echo ""
echo "✓ All done. Dataset is in mp_dataset/"
echo "  To start the dashboard: uv run python -m doom_dashboard serve --config config.yaml"
