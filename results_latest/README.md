# Doom Deathmatch Agent — Latest Results

## Competition Submission (v3) — CURRENT

**Folder**: `results_latest/competition_submission_v3/`
**WandB**: https://wandb.ai/chrisxx/doom-overnight

### Two Best Models

#### 1. BC Only (LSTM) — Best K/D, Human-Like Behavior
| Metric | Value |
|--------|-------|
| **Kills/episode** | 48.2 (5-ep avg, 1 bot) |
| **Deaths/episode** | 3.6 |
| **K/D ratio** | **13.4** |
| **Best episode** | 65 kills, 1 death |
| **Actions used** | 15-18/19 |
| **Inference mode** | `deterministic=False` (stochastic) |
| **Model file** | `bc_only_lstm.zip` |

```python
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load("bc_only_lstm.zip", device="cuda")
# Use LSTM state tracking:
action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
```

#### 2. PPO (BC+RL) — Most Kills, More Aggressive
| Metric | Value |
|--------|-------|
| **Kills/episode** | 48.4 (5-ep avg, 1 bot) |
| **Deaths/episode** | 7.6 |
| **K/D ratio** | 6.4 |
| **Best episode** | 52 kills, 2 deaths |
| **Actions used** | 19/19 (all macro actions) |
| **Inference mode** | `deterministic=False` (stochastic) |
| **Model file** | `bc_v2_highent_best.zip` |

```python
from stable_baselines3 import PPO
model = PPO.load("bc_v2_highent_best.zip", device="cuda")
action, _ = model.predict(obs, deterministic=False)  # MUST be False!
```

### Key Findings
- **BC-only model has 2x better K/D** (13.4 vs 6.4) with same kill rate — it dies much less
- **BC model looks more human-like**: navigates naturally, learned from human demos
- **PPO model is more aggressive**: uses all 19 actions, fires more often, but dies more
- Both models use **stochastic inference** (`deterministic=False`) — this is essential

### Showcase Videos
```
videos/
  ppo_vs_random.mp4    # PPO model left, random agent right (3 episodes, 180s)
  bc_only_vs_random.mp4  # BC model left, random agent right (3 episodes, 180s)
  ppo_vs_bc.mp4        # PPO left, BC right — direct comparison (3 episodes, 180s)
```

### Critical Fix from v2
v2 models had **ent_coef=0.0** (bug: BC model didn't set it, and `.load()` preserved it).
This caused complete policy collapse — all models repeated a single action.
v3 uses **ent_coef=0.1** + stochastic inference, resulting in diverse behavior.

---

## Architecture
- **CNN**: IMPALA ResNet (3 blocks: 16, 32, 32 channels, features_dim=256)
- **Policy**: PPO MLP (2x512) or RecurrentPPO LSTM (256 hidden)
- **Actions**: 19 macro-discrete (8 buttons: ATTACK, SPEED, MOVE_R/L/F/B, TURN_R/L)
- **Obs**: 120x160 RGB + 8 gamevars, frame_skip=4

## Training Pipeline
1. **Human behavioral cloning**: 10,460 frames, 100 epochs, 47.6% val accuracy → `bc_only_lstm.zip`
2. **PPO fine-tuning**: 775k best checkpoint (3M run), ent_coef=0.1, no attack_bonus, noop_penalty=0.1 → `bc_v2_highent_best.zip`

## Key Scripts
- `train_overnight_dm.py` — main PPO training
- `pretrain_bc.py` — behavioral cloning from human demos
- `make_showcase.py` — generate side-by-side comparison videos (model vs random, or model vs model)
- `test_stochastic_video.py` — generate test videos
- `diagnose_policy.py` — diagnose action diversity / policy collapse
- `record_vs_ai.py` — play against trained AI

## Files
```
results_latest/
  competition_submission_v3/       # <-- USE THIS
    bc_only_lstm.zip               # BC model (RecurrentPPO, best K/D)
    bc_only_lstm.meta.json         # BC model metadata
    bc_v2_highent_best.zip         # PPO model (most aggressive)
    bc_v2_highent_best.meta.json   # PPO model metadata
    videos/                        # Showcase comparison videos
    SUBMISSION.md                  # Full details and loading instructions
  competition_submission_v2/       # OLD (broken — spinning models)
```
