# Doom Deathmatch Competition Submission (v2)

## Primary Model: BC+PPO Fine-tuned LSTM (`bc_finetune_best.zip`)

**Recommended for submission.**

### Performance
- **Best showcase episode**: 62 kills, 4 deaths (K-D = 58)
- **Average (3 episodes)**: 54 kills, 7.3 deaths
- **Eval reward**: 148.0 (peak at 950k PPO steps, still training)
- **Config**: deathmatch_compact.cfg, 1 bot, map01

### Architecture
- **CNN**: IMPALA ResNet (3 blocks: 16, 32, 32 channels, features_dim=256)
- **Policy**: RecurrentPPO (LSTM hidden=256, 1 layer, shared=False, enable_critic_lstm=True)
- **MLP**: 2x512 hidden layers
- **Actions**: 19 macro-discrete (8 buttons: ATTACK, SPEED, MOVE_R/L/F/B, TURN_R/L)
- **Obs**: 120x160 RGB + 8 gamevars, frame_skip=4

### Training Pipeline
1. **Human behavioral cloning**: 10,460 frames, 100 epochs, 47.6% validation accuracy
2. **PPO fine-tuning**: 950k+ steps with reward shaping (kills, hits, damage, movement)

## Backup Models

| Model | File | Steps | Eval | Notes |
|-------|------|-------|------|-------|
| **BC+PPO** | `bc_finetune_best.zip` | 950k PPO | **148.0** | Human-initialized (primary) |
| **v11a continued** | `v11a_continued_best.zip` | 4M+350k | 120.9 | Extra training with 2 bots |
| **v11a original** | `v11a_lstm_best.zip` | 4M | 219* | Pure RL from scratch |

*v11a eval was with 0 bots (solo); multiplayer scores are lower.

## How to Load

```python
from sb3_contrib import RecurrentPPO
import numpy as np

model = RecurrentPPO.load("bc_finetune_best.zip", device="cuda")

obs = env.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

while True:
    action, lstm_states = model.predict(
        obs, state=lstm_states,
        episode_start=episode_starts,
        deterministic=True
    )
    obs, reward, done, info = env.step(action)
    episode_starts = done
```

## Demo Videos
- `bc_agent_showcase.mp4` — Best episode: 62 kills, 4 deaths (29s)
- `v11a_lstm_showcase.mp4` — v11a pure RL comparison (57s)

## WandB
- BC pretraining: https://wandb.ai/chrisxx/doom-overnight/runs/abdstxt8
- PPO fine-tuning: https://wandb.ai/chrisxx/doom-overnight (run: bc_finetune_lstm_v1)

## Meta
See `.meta.json` files for button names, action map, obs shape.
