# Doom Deathmatch Agent — Latest Results

## Competition Submission (v3) — CURRENT

**Folder**: `results_latest/competition_submission_v3/`
**WandB**: https://wandb.ai/chrisxx/doom-overnight

### Best Model: BC Only (LSTM) — K/D 18.7

The behavior cloning model trained purely from human demonstrations is our best agent.
PPO fine-tuning actually *degrades* survivability — the RL reward doesn't penalize deaths enough.

| Metric | BC Only (Best) | PPO 2M (Best PPO) | PPO 775k (Old Best) |
|--------|---------------|-------------------|-------------------|
| **Kills/episode** | **54.2** | 49.9 | 49.6 |
| **Deaths/episode** | **2.9** | 3.9 | 6.8 |
| **K/D ratio** | **18.7** | 12.8 | 7.3 |
| **Actions used** | 16/19 | 19/19 | 19/19 |
| **Architecture** | RecurrentPPO (LSTM) | PPO (MLP) | PPO (MLP) |
| **Inference** | stochastic | stochastic | stochastic |

*All stats: 10+ episode averages, stochastic inference, vs 1 bot on deathmatch_compact map01.*

### Model Files

| File | Description |
|------|-------------|
| `bc_only_lstm.zip` | **BEST** — BC model (RecurrentPPO LSTM), K/D=18.7 |
| `bc_v2_highent_best.zip` | PPO model (775k steps), more aggressive |
| `ppo_2M.zip` | PPO 2M steps — best PPO checkpoint by K/D |

### How to Use

```python
# Best model (BC Only, LSTM)
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load("bc_only_lstm.zip", device="cuda")
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)
action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
episode_starts = np.zeros((1,), dtype=bool)  # after first step

# PPO model (no LSTM state needed)
from stable_baselines3 import PPO
model = PPO.load("bc_v2_highent_best.zip", device="cuda")
action, _ = model.predict(obs, deterministic=False)  # MUST be False!
```

### Key Findings from WandB Analysis

1. **BC-only beats PPO**: Human imitation learned better survival skills than RL rewards could teach
2. **PPO fine-tuning is noisy**: Neighboring checkpoints (750k vs 800k) have K/D ranging from 6.7 to 15.8
3. **High entropy essential**: ent_coef=0.1 prevents policy collapse, ent_coef=0.01 is borderline
4. **Stochastic inference required**: deterministic=True collapses all models to single actions
5. **Old overnight models were overrated**: The "100% win rate" overnight_185028 model actually gets K/D=7.4 in stochastic eval (vs 18.7 for BC)

### Showcase Videos
```
videos/
  ppo_vs_random.mp4       # PPO (BC+RL) left, random right (3 episodes, 180s)
  bc_only_vs_random.mp4   # BC model left, random right (3 episodes, 180s)
  ppo_vs_bc.mp4           # PPO left, BC right — direct comparison (3 episodes, 180s)
```

---

## Architecture
- **CNN**: IMPALA ResNet (3 blocks: 16, 32, 32 channels, features_dim=256)
- **BC model**: RecurrentPPO LSTM (256 hidden), 16-18/19 actions
- **PPO model**: PPO MLP (2x512 hidden), 19/19 actions
- **Actions**: 19 macro-discrete (8 buttons: ATTACK, SPEED, MOVE_R/L/F/B, TURN_R/L)
- **Obs**: 120x160 RGB + 8 gamevars, frame_skip=4

## Training Pipeline
1. **Human behavioral cloning**: 10,460 frames, 100 epochs, 47.6% val acc → `bc_only_lstm.zip` (BEST)
2. **PPO fine-tuning**: 3M steps, ent_coef=0.1, best at 2M → `ppo_2M.zip` (optional, more aggressive)

## Key Scripts
- `train_overnight_dm.py` — main PPO training
- `pretrain_bc.py` — behavioral cloning from human demos
- `make_showcase.py` — side-by-side comparison videos (model vs random, or model vs model)
- `test_stochastic_video.py` — generate test videos
- `diagnose_policy.py` — diagnose action diversity / policy collapse
- `record_vs_ai.py` — play against trained AI
