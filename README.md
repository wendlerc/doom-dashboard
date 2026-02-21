# Doom Dashboard

A ViZDoom-based dataset generation and visualization toolkit. Trains Stable-Baselines3 PPO agents on five ViZDoom scenarios, records both single-player and 1v1 multiplayer gameplay as WebDataset shards, and serves a Flask dashboard to browse sample videos.

---

## What Was Built

| Component | Description |
|---|---|
| `train_policies.py` | Trains 5 PPO agents in parallel (Basic, DefendTheCenter, HealthGathering, MyWayHome, DeadlyCorridor) |
| `doom_dashboard/rollout.py` | Single-player episode rollout — headless, frameskip, structured output |
| `doom_dashboard/multiplayer_rollout.py` | 1v1 multiplayer rollout via `subprocess.Popen` host+join pair |
| `doom_dashboard/dataset_gen.py` | Parallelized WebDataset generator for single-player gameplay |
| `doom_dashboard/mp_dataset_gen.py` | Parallelized WebDataset generator for 1v1 multiplayer gameplay |
| `doom_dashboard/annotate.py` | Frame annotation (action labels, reward bar) + MP4 encoding |
| `doom_dashboard/server.py` | Flask dashboard with sample video browser and dataset generation UI |
| `doom_dashboard/cli.py` | Click CLI: `train`, `generate-dataset`, `generate-mp-dataset`, `generate-samples`, `serve` |
| `monitor_mp.sh` | Live terminal monitor for multiplayer generation progress |
| `post_training_launch.sh` | One-shot automation: verify checkpoints → samples → mp dataset |

### Dataset Schema

Each WebDataset shard (`.tar`) contains episodes. Per episode:

**Single-player** (`dataset/`):
```
frames.npy       (T, H, W, 3)  uint8
actions.npy      (T, n_buttons)
rewards.npy      (T,)           float32
meta.json        {scenario, map, policy, steps, total_reward, ...}
```

**Multiplayer** (`mp_dataset/`):
```
frames_p1.npy    (T, H, W, 3)  uint8     ← player 1 perspective
actions_p1.npy / rewards_p1.npy
frames_p2.npy    (T, H, W, 3)  uint8     ← player 2 perspective
actions_p2.npy / rewards_p2.npy
meta.json        {scenario, map, policy_p1, policy_p2, ...}
```

---

## Setup

**Requirements:** Python ≥ 3.10, `uv`, `xvfb-run` (headless rendering)

```bash
git clone git@github.com:wendlerc/doom-dashboard.git
cd doom-dashboard
uv sync
```

> **Note:** ViZDoom requires system libraries. On Ubuntu/Debian:
> ```bash
> sudo apt install libsdl2-dev libboost-all-dev cmake git xvfb
> ```

---

## Usage

### 1. Train Policies

Trains 5 PPO agents in parallel (~1.5h each on CPU). Saves checkpoints to `./trained_policies/`.

```bash
xvfb-run -a uv run python train_policies.py
```

Resulting checkpoints:
```
trained_policies/basic.zip
trained_policies/defend_the_center.zip
trained_policies/health_gathering.zip
trained_policies/my_way_home.zip
trained_policies/deadly_corridor.zip
```

### 2. Generate Sample Videos (for Dashboard)

```bash
xvfb-run -a uv run python -m doom_dashboard generate-samples --config config.yaml
```

### 3. Generate Single-Player Dataset

```bash
xvfb-run -a uv run python -m doom_dashboard generate-dataset \
    --config config.yaml --output-dir dataset --hours 10
```

### 4. Generate Multiplayer Dataset (1v1)

Run in a `screen` session so it persists:

```bash
screen -dmS doom_mp bash -c '
  cd /path/to/doom-dashboard
  xvfb-run -a uv run python -m doom_dashboard generate-mp-dataset \
    --config config.yaml \
    --output-dir mp_dataset \
    --hours 200 \
    --workers 8 \
    --random-ratio 0.05 \
    --timelimit 5.0 2>&1 | tee mp_gen.log
'
# Attach to monitor: screen -r doom_mp
# Or use the monitoring script:
bash monitor_mp.sh 30
```

**Worker architecture:** each worker process spawns a host+join subprocess pair (fresh Python interpreters) for each episode — avoids CUDA/fork conflicts. Results are returned via temp pickle files.

**Policy sampling:** 95% trained policies (uniform across the 5), 5% random.

### 5. Launch Dashboard

```bash
uv run python -m doom_dashboard serve --config config.yaml --port 5000
```

Then open `http://localhost:5000` in a browser.

---

## Configuration

`config.yaml` controls scenarios, policies, and generation parameters:

```yaml
policies:
  - name: PPO-Basic
    type: sb3
    path: ./trained_policies/basic.zip
    algo: PPO
    device: auto
  # ... more policies

dataset:
  scenarios:
    - name: basic
      ratio: 0.2
  # ...

multiplayer:
  timelimit: 5.0       # minutes per episode
  scenarios: [deathmatch, cig, multi_duel]
```

---

## Monitoring Generation

```bash
# Live monitor (refreshes every 30s):
bash monitor_mp.sh

# Quick check:
tail -f mp_gen.log | grep -v "^Contacting\|^Got connect\|^Waiting"

# Screen session:
screen -r doom_mp
```

---

## Post-Training Automation

```bash
# Generates sample videos + launches 200h multiplayer dataset in one shot:
bash post_training_launch.sh

# With overrides:
MP_HOURS=50 MP_WORKERS=4 bash post_training_launch.sh
```

---

## Project Structure

```
doom-dashboard/
├── config.yaml                      # Main configuration
├── train_policies.py                # Policy training script
├── post_training_launch.sh          # Post-training automation
├── monitor_mp.sh                    # Generation progress monitor
├── doom_dashboard/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                       # Click CLI entrypoint
│   ├── config.py                    # Config dataclasses + YAML loading
│   ├── policies.py                  # RandomPolicy, SB3Policy, TorchPolicy
│   ├── _archs.py                    # DuelQNet architecture
│   ├── rollout.py                   # Single-player episode rollout
│   ├── multiplayer_rollout.py       # 1v1 multiplayer rollout
│   ├── dataset_gen.py               # Single-player dataset generator
│   ├── mp_dataset_gen.py            # Multiplayer dataset generator
│   ├── annotate.py                  # Frame annotation + MP4 encoding
│   ├── server.py                    # Flask dashboard server
│   └── static/
│       └── index.html               # Dashboard frontend
├── trained_policies/                # PPO checkpoints (gitignored)
├── dataset/                         # Single-player shards (gitignored)
├── mp_dataset/                      # Multiplayer shards (gitignored)
└── samples/                         # Sample videos (gitignored)
```
