#!/bin/bash
# overnight_monitor.sh — Monitor parallel training runs, launch follow-ups, generate results
# Run with: nohup bash overnight_monitor.sh &>/dev/null &
set -e
cd /share/NFS/u/wendler/code/doom-dashboard
export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/share/NFS/u/wendler/.local/lib:${LD_LIBRARY_PATH:-}"

LOG="overnight_monitor.log"
RESULTS="results_latest"
mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_for_process() {
    local name=$1 pid=$2
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "$name (PID=$pid) finished"
}

# Find running training PIDs
LSTM_PID=$(pgrep -f "train_overnight_dm.*v11a_lstm" 2>/dev/null | head -1)
FS_PID=$(pgrep -f "train_overnight_dm.*v11b_framestack" 2>/dev/null | head -1)
LSTM_LOOP=$(pgrep -f "tloop_v11_lstm" 2>/dev/null | head -1)
FS_LOOP=$(pgrep -f "tloop_v11_fs" 2>/dev/null | head -1)

log "=== Overnight Monitor Started ==="
log "LSTM training PID: ${LSTM_PID:-not found}, loop: ${LSTM_LOOP:-not found}"
log "FrameStack training PID: ${FS_PID:-not found}, loop: ${FS_LOOP:-not found}"

# ─── Phase 1: Wait for FrameStack to finish (faster run) ───
if [ -n "$FS_LOOP" ]; then
    log "Waiting for FrameStack run to complete..."
    wait_for_process "FrameStack loop" "$FS_LOOP"
fi

# ─── Phase 2: Benchmark FrameStack model ───
FS_MODEL=$(ls -t trained_policies/v11b_framestack_ckpts/ppo_*_steps.zip 2>/dev/null | head -1)
FS_BEST="trained_policies/v11b_framestack_best/best_model.zip"
[ -f "$FS_BEST" ] && FS_MODEL="$FS_BEST"

if [ -n "$FS_MODEL" ]; then
    log "Benchmarking FrameStack model: $FS_MODEL"
    mkdir -p "$RESULTS/v11b_framestack"
    cp "$FS_MODEL" "$RESULTS/v11b_framestack/model.zip" 2>/dev/null

    # Bench on compact
    uv run python bench_model.py --model "$FS_MODEL" \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --episodes 16 --bots 4 --timelimit 2.0 \
        --output "$RESULTS/v11b_framestack/bench_compact.json" 2>&1 | tee -a "$LOG" || true

    # Bench on nomonsters
    uv run python bench_model.py --model "$FS_MODEL" \
        --cfg doom_dashboard/scenarios/deathmatch_nomonsters.cfg \
        --episodes 16 --bots 4 --timelimit 2.0 \
        --output "$RESULTS/v11b_framestack/bench_nomonsters.json" 2>&1 | tee -a "$LOG" || true

    log "FrameStack benchmark done"
fi

# ─── Phase 3: Launch 3rd run with freed GPU (LSTM+FS combo or bigger LSTM) ───
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
log "GPU free after FrameStack: ${GPU_FREE}MB"

if [ "${GPU_FREE:-0}" -gt 15000 ]; then
    log "Launching Phase 3: Larger LSTM (512 hidden) with freed GPU"

    uv run python train_overnight_dm.py \
        --name "v11c_lstm512" \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --maps map01 --timesteps 4000000 --envs 2 --bots 1 --bots-eval 1 \
        --timelimit-minutes 2.5 --frame-skip 4 \
        --obs-height 120 --obs-width 160 \
        --n-steps 1024 --batch-size 1024 --n-epochs 4 \
        --learning-rate 3e-4 --ent-coef 0.03 --target-kl 0.05 \
        --policy-hidden-size 512 --policy-hidden-layers 2 \
        --cnn-type impala --cnn-features-dim 256 \
        --policy-type lstm --lstm-hidden-size 512 --lstm-num-layers 1 \
        --video-freq 200000 \
        --bench-episodes 16 --bench-timelimit 2.0 \
        --device cuda --wandb \
        --frag-bonus 200 --hit-bonus 12.0 --damage-bonus 0.2 \
        --death-penalty 30 --attack-bonus 0.05 --move-bonus 0.2 \
        --noop-penalty 0.1 --reward-scale 0.1 \
        >> "$LOG" 2>&1 &
    P3_PID=$!
    log "Phase 3 launched: v11c_lstm512 PID=$P3_PID"
fi

# ─── Phase 4: Wait for LSTM run to finish ───
if [ -n "$LSTM_LOOP" ]; then
    log "Waiting for LSTM run to complete..."
    wait_for_process "LSTM loop" "$LSTM_LOOP"
fi

# ─── Phase 5: Benchmark LSTM model ───
LSTM_MODEL=$(ls -t trained_policies/v11a_lstm_ckpts/ppo_*_steps.zip 2>/dev/null | head -1)
LSTM_BEST="trained_policies/v11a_lstm_best/best_model.zip"
[ -f "$LSTM_BEST" ] && LSTM_MODEL="$LSTM_BEST"

if [ -n "$LSTM_MODEL" ]; then
    log "Benchmarking LSTM model: $LSTM_MODEL"
    mkdir -p "$RESULTS/v11a_lstm"
    cp "$LSTM_MODEL" "$RESULTS/v11a_lstm/model.zip" 2>/dev/null

    uv run python bench_model.py --model "$LSTM_MODEL" \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --episodes 16 --bots 4 --timelimit 2.0 \
        --output "$RESULTS/v11a_lstm/bench_compact.json" 2>&1 | tee -a "$LOG" || true

    uv run python bench_model.py --model "$LSTM_MODEL" \
        --cfg doom_dashboard/scenarios/deathmatch_nomonsters.cfg \
        --episodes 16 --bots 4 --timelimit 2.0 \
        --output "$RESULTS/v11a_lstm/bench_nomonsters.json" 2>&1 | tee -a "$LOG" || true

    log "LSTM benchmark done"
fi

# ─── Phase 6: Wait for Phase 3 if running ───
if [ -n "$P3_PID" ]; then
    log "Waiting for Phase 3 (lstm512)..."
    wait $P3_PID 2>/dev/null || true

    LSTM512_MODEL=$(ls -t trained_policies/v11c_lstm512_ckpts/ppo_*_steps.zip 2>/dev/null | head -1)
    LSTM512_BEST="trained_policies/v11c_lstm512_best/best_model.zip"
    [ -f "$LSTM512_BEST" ] && LSTM512_MODEL="$LSTM512_BEST"

    if [ -n "$LSTM512_MODEL" ]; then
        log "Benchmarking LSTM-512 model: $LSTM512_MODEL"
        mkdir -p "$RESULTS/v11c_lstm512"
        cp "$LSTM512_MODEL" "$RESULTS/v11c_lstm512/model.zip" 2>/dev/null

        uv run python bench_model.py --model "$LSTM512_MODEL" \
            --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
            --episodes 16 --bots 4 --timelimit 2.0 \
            --output "$RESULTS/v11c_lstm512/bench_compact.json" 2>&1 | tee -a "$LOG" || true

        uv run python bench_model.py --model "$LSTM512_MODEL" \
            --cfg doom_dashboard/scenarios/deathmatch_nomonsters.cfg \
            --episodes 16 --bots 4 --timelimit 2.0 \
            --output "$RESULTS/v11c_lstm512/bench_nomonsters.json" 2>&1 | tee -a "$LOG" || true
    fi
fi

# ─── Phase 7: Generate showcase videos for all models ───
log "Generating showcase videos..."

generate_video() {
    local model=$1 name=$2 scenario=$3 outdir=$4
    local cfg="doom_dashboard/scenarios/${scenario}.cfg"

    uv run python -c "
import vizdoom as vzd, numpy as np, cv2, os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from train_overnight_dm import DeathmatchMacroEnv, ImpalaCnnExtractor, materialize_cfg, parse_available_buttons, build_macro_actions, ShapingCfg
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env

# Try loading as RecurrentPPO first, then PPO
try:
    model = RecurrentPPO.load('$model', device='cuda')
    is_lstm = True
except:
    model = PPO.load('$model', device='cuda')
    is_lstm = False

cfg_runtime = materialize_cfg('$cfg')
shaping = ShapingCfg()

def make_fn():
    return DeathmatchMacroEnv(cfg_path=cfg_runtime, frame_skip=4, obs_shape=(120,160), maps=['map01'], bots=2, timelimit_minutes=2.0, shaping=shaping, spawn_farthest=False, no_autoaim=False)

env = make_vec_env(make_fn, n_envs=1)
env = VecTransposeImage(env)

obs = env.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)
frames = []

game = env.envs[0].unwrapped if hasattr(env.envs[0], 'unwrapped') else env.envs[0]

for step in range(2000):
    if is_lstm:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    else:
        action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_starts = done

    # Get frame from env
    screen = env.envs[0].render() if hasattr(env.envs[0], 'render') else None
    if screen is None:
        # Get from obs
        img = obs[0].transpose(1,2,0) if obs[0].shape[0] == 3 else obs[0]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        frames.append(cv2.resize(img[:,:,:3], (640, 480)))
    else:
        frames.append(cv2.resize(screen, (640, 480)))

env.close()

# Save video
out_path = '$outdir/${name}_${scenario}.mp4'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, 30, (640, 480))
for f in frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
print(f'Video: {out_path} ({len(frames)} frames)')
" 2>&1 | tee -a "$LOG" || true
}

# Generate videos for each model that has a benchmark
for model_dir in "$RESULTS"/v11*; do
    [ -d "$model_dir" ] || continue
    model_file="$model_dir/model.zip"
    [ -f "$model_file" ] || continue
    name=$(basename "$model_dir")
    log "Video for $name..."
    generate_video "$model_file" "$name" "deathmatch_compact" "$model_dir/videos"
    generate_video "$model_file" "$name" "deathmatch_nomonsters" "$model_dir/videos"
done

# ─── Phase 8: Pick winner, update README ───
log "=== Selecting best model ==="

uv run python << 'PYEOF'
import json, os, glob

results_dir = "results_latest"
models = {}
for d in glob.glob(f"{results_dir}/v11*/bench_compact.json"):
    name = os.path.basename(os.path.dirname(d))
    try:
        with open(d) as f:
            bench = json.load(f)
        models[name] = {
            "compact": bench,
            "hits": bench.get("model_hit_mean", 0),
            "damage": bench.get("model_damage_mean", 0),
            "distance": bench.get("model_distance_mean", 0),
            "pass": bench.get("pass", False),
        }
    except:
        pass

if not models:
    print("No benchmark results found yet")
else:
    # Score = hits * 2 + damage / 10 + distance / 1000
    for name, m in models.items():
        m["score"] = m["hits"] * 2 + m["damage"] / 10 + m["distance"] / 1000
        print(f"  {name}: hits={m['hits']:.1f} dmg={m['damage']:.0f} dist={m['distance']:.0f} "
              f"pass={m['pass']} score={m['score']:.1f}")

    winner = max(models, key=lambda k: models[k]["score"])
    print(f"\n  WINNER: {winner} (score={models[winner]['score']:.1f})")
PYEOF

log "=== Overnight Monitor Complete ==="
