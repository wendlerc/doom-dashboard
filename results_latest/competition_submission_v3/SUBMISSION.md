# Doom Deathmatch Competition Submission (v3)

## Primary Model: BC+PPO High-Entropy (`bc_v2_highent_best.zip`)

**Use stochastic inference: `deterministic=False`**

### Performance Summary (10-episode average, 2 bots)
- **Kills per episode**: 51.1
- **Deaths per episode**: 16.2
- **K/D ratio**: 3.2
- **Best episode**: 65 kills, 3 deaths
- **Actions used**: 19/19 (all macro actions, diverse distribution)

### Performance vs 1 bot (3 episodes)
- **Kills**: 169, Deaths: 28, K/D = 6.0
- **56.3 kills/ep**, navigates, aims, fights, retreats

### Architecture
- **CNN**: IMPALA ResNet (3 blocks: 16, 32, 32 channels, features_dim=256)
- **Policy**: PPO (MLP, 2x512 hidden layers)
- **Actions**: 19 macro-discrete (8 buttons: ATTACK, SPEED, MOVE_R/L/F/B, TURN_R/L)
- **Obs**: 120x160 RGB + 8 gamevars, frame_skip=4

### Training Pipeline
1. **Human behavioral cloning**: 10,460 frames from human gameplay, 100 epochs, 47.6% val accuracy
2. **PPO fine-tuning**: 775k steps, ent_coef=0.1, no attack_bonus, noop_penalty=0.1

### Key Design Decisions
- **ent_coef = 0.1**: Prevents policy collapse to single action
- **Stochastic inference**: Samples from learned distribution for diverse behavior
- **attack_bonus = 0**: No reward for constant firing (prevents degenerate spinning)
- **noop_penalty = 0.1**: Encourages active gameplay
- **Position-based move reward**: Rewards actual displacement, not button presses

## How to Load

```python
from stable_baselines3 import PPO
import numpy as np

model = PPO.load("bc_v2_highent_best.zip", device="cuda")

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=False)  # IMPORTANT: Must be False!
    obs, reward, done, info = env.step(action)
```

## Files
- `bc_v2_highent_best.zip` — Primary model (40MB)
- `bc_v2_highent_best.meta.json` — Model metadata (action names, button map)
- `bc_v2_highent_775k.zip` — Same model (backup copy)
- `bc_v2_showcase_5ep.mp4` — 5-episode demo vs 2 bots: 262 kills, 45 deaths
- `bc_v2_showcase.mp4` — 3-episode demo vs 1 bot: 169 kills, 28 deaths

## WandB
- BC pretraining: https://wandb.ai/chrisxx/doom-overnight/runs/abdstxt8
- PPO fine-tuning: https://wandb.ai/chrisxx/doom-overnight/runs/in2baom2

## Config: deathmatch_compact.cfg
- Map: map01, 2 min timelimit
- 8 buttons: ATTACK, SPEED, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT
- 19 macro actions (all combinations of simultaneous button presses)
