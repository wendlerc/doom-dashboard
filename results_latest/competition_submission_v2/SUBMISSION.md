# Doom Deathmatch Competition Submission (v2)

## Primary Model: BC+PPO Fine-tuned LSTM (`bc_finetune_best.zip`)

**Recommended for submission.**

### Performance (2M PPO steps, final model)
- **Best showcase**: 53 kills, 4 deaths (K-D = 49)
- **Average (3 episodes)**: 49.3 kills, 5.0 deaths (K-D = 44.3)
- **Eval reward**: 168.7 (peak at 2M steps)
- **Config**: deathmatch_compact.cfg, 1 bot, map01, 2 min timelimit

### Architecture
- **CNN**: IMPALA ResNet (3 blocks: 16, 32, 32 channels, features_dim=256)
- **Policy**: RecurrentPPO (LSTM hidden=256, 1 layer, shared=False, enable_critic_lstm=True)
- **MLP**: 2x512 hidden layers
- **Actions**: 19 macro-discrete (8 buttons: ATTACK, SPEED, MOVE_R/L/F/B, TURN_R/L)
- **Obs**: 120x160 RGB + 8 gamevars, frame_skip=4

### Training Pipeline
1. **Human behavioral cloning**: 10,460 frames, 100 epochs, 47.6% val accuracy
2. **PPO fine-tuning**: 2M steps with reward shaping (kills, hits, damage, movement)

## Backup Models

| Model | File | Eval | Kills/Ep | Deaths/Ep |
|-------|------|------|----------|-----------|
| **BC+PPO 2M** | `bc_finetune_best.zip` | **168.7** | 49.3 | 5.0 |
| v11a continued | `v11a_continued_best.zip` | 120.9 | ~50 | ~12 |
| v11a original | `v11a_lstm_best.zip` | 219* | ~53 | ~12 |

*v11a original eval was solo (0 bots); multiplayer scores differ.

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
- `bc_agent_showcase.mp4` — Best episode: 53 kills, 4 deaths (29s)
- `v11a_lstm_showcase.mp4` — v11a pure RL comparison (57s)

## WandB
- BC pretraining: https://wandb.ai/chrisxx/doom-overnight/runs/abdstxt8
- PPO fine-tuning: https://wandb.ai/chrisxx/doom-overnight/runs/0nydidq1
