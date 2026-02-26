#!/bin/bash
# validate_and_showcase.sh — Run bench, generate videos, validate movement, update results
# Usage: bash validate_and_showcase.sh <model_path> <name> [scenario]
set -e
cd /share/NFS/u/wendler/code/doom-dashboard
export DISPLAY="${DISPLAY:-:0}"

MODEL="${1:?Usage: validate_and_showcase.sh <model_path> <name> [scenario]}"
NAME="${2:?Provide a model name}"
SCENARIO="${3:-deathmatch_compact}"  # or deathmatch_nomonsters

OUT_DIR="results_latest/${NAME}"
mkdir -p "$OUT_DIR"

echo "=== Validating ${NAME} ==="
echo "  Model: ${MODEL}"
echo "  Scenario: ${SCENARIO}"
echo "  Output: ${OUT_DIR}"

# 1. Run bench
echo ""
echo "--- Step 1: Benchmark ---"
uv run python bench_model.py --model "$MODEL" \
    --cfg "doom_dashboard/scenarios/${SCENARIO}.cfg" \
    --episodes 16 --bots 4 --timelimit 2.0 \
    --output "${OUT_DIR}/bench.json" 2>&1 || true

if [ -f "${OUT_DIR}/bench.json" ]; then
    echo "Bench results:"
    uv run python -c "
import json
with open('${OUT_DIR}/bench.json') as f:
    d = json.load(f)
print(f'  hits={d.get(\"model_hit_mean\",0):.1f}, damage={d.get(\"model_damage_mean\",0):.1f}, ratio={d.get(\"damage_ratio\",0):.1f}')
print(f'  distance={d.get(\"model_distance_mean\",\"N/A\")}')
print(f'  pass={d.get(\"pass\",False)}')
" 2>/dev/null
fi

# 2. Generate videos
echo ""
echo "--- Step 2: Generate Videos ---"
cat > "${OUT_DIR}/config.yaml" << YAML
scenarios:
  - name: "Scenario"
    cfg: "doom_dashboard/scenarios/${SCENARIO}.cfg"
    frame_skip: 4
    render_resolution: "RES_640X480"
    render_hud: false

policies:
  - name: "${NAME}"
    type: "sb3"
    path: "${MODEL}"
    algo: "PPO"
    device: "auto"

  - name: "Random"
    type: "random"

dashboard:
  samples_per_map: 1
  render_resolution: "RES_640X480"
  fps: 30
  output_dir: "${OUT_DIR}/videos"
YAML

uv run python -m doom_dashboard generate-elim-videos \
    --config "${OUT_DIR}/config.yaml" \
    --output-dir "${OUT_DIR}/videos" \
    --scenario "$SCENARIO" \
    --games 4 --timelimit 2.0 \
    --frame-skip 4 --resolution RES_640X480 \
    --policy "$NAME" --policy "Random" \
    --allow-mismatch 2>&1 || true

# 3. Validate movement
echo ""
echo "--- Step 3: Validate Movement ---"
uv run python << 'PYEOF'
import cv2, numpy as np, glob, json, sys, os

video_dir = "${OUT_DIR}/videos"
mp4s = glob.glob(f"{video_dir}/*.mp4")
results = []
for mp4 in sorted(mp4s):
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 10:
        cap.release()
        continue
    indices = np.linspace(0, n-2, min(25, n//2), dtype=int)
    diffs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if ok1 and ok2:
            w = f1.shape[1]
            mid = w // 2
            diffs.append(np.mean(np.abs(f1[:,:mid].astype(float) - f2[:,:mid].astype(float))))
    cap.release()
    mean_diff = float(np.mean(diffs)) if diffs else 0
    moving = mean_diff >= 5.0
    results.append({"file": os.path.basename(mp4), "mean_diff": round(mean_diff, 1), "moving": moving})
    status = "MOVING" if moving else "STATIC"
    print(f"  [{status}] {os.path.basename(mp4)}: diff={mean_diff:.1f}")

with open(f"{video_dir}/movement_check.json", "w") as f:
    json.dump(results, f, indent=2)

all_moving = all(r["moving"] for r in results) if results else False
print(f"\n  Overall: {'ALL MOVING' if all_moving else 'SOME STATIC'}")
PYEOF

echo ""
echo "=== Done: results in ${OUT_DIR} ==="
