# Doom Deathmatch Competition Submission (v2)

## Models

### Primary: BC+PPO Fine-tuned LSTM (`bc_finetune_best.zip`)
- **Architecture**: IMPALA CNN + RecurrentPPO (LSTM-256, 1 layer)
- **Training**: Human behavioral cloning (47.6% val acc, 100 epochs) + PPO fine-tuning (475k+ steps)
- **Human data**: 10,460 frames of human gameplay, mapped to 19 macro-discrete actions
- **Performance**: ~52 kills, ~12 deaths per 2-min episode (2 bots, map01)
- **Eval reward**: 133.4 (peak at 475k steps, still training)

### Backup: v11a LSTM (`v11a_lstm_best.zip`)
- **Architecture**: Same (IMPALA CNN + RecurrentPPO LSTM-256)
- **Training**: 4M PPO steps from scratch (no human data)
- **Performance**: ~53 kills, ~12 deaths per 2-min episode (2 bots, map01)
- **Eval reward**: 219 (peak at ~1.2M steps)

## Architecture Details
- **CNN**: IMPALA ResNet (3 blocks: 16, 32, 32 channels, features_dim=256)
- **Policy**: RecurrentPPO (LSTM hidden=256, 1 layer, shared=False, enable_critic_lstm=True)
- **MLP**: 2x512 hidden layers
- **Actions**: 19 macro-discrete (8 buttons: move, turn, attack, speed)
- **Obs**: 120x160 RGB + 8 gamevars, frame_skip=4
- **Config**: deathmatch_compact.cfg, map01

## How to Load

```python
from sb3_contrib import RecurrentPPO
import numpy as np

# Load model
model = RecurrentPPO.load("bc_finetune_best.zip", device="cuda")

# Inference with LSTM state
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
- `bc_agent_showcase.mp4` — BC+PPO agent playing deathmatch (57s, 43 kills)
- `v11a_lstm_showcase.mp4` — v11a pure RL agent comparison (57s, 71 kills)

## Training Pipeline
1. **Human demo recording** via `record_gameplay.sh`
2. **Behavioral cloning** via `pretrain_bc.py` (LSTM, 100 epochs, 47.6% accuracy)
3. **PPO fine-tuning** via `train_overnight_dm.py --init-model bc_model.zip`
4. **Evaluation** via `bench_model.py`

## WandB
- BC pretraining: https://wandb.ai/chrisxx/doom-overnight/runs/abdstxt8
- PPO fine-tuning: https://wandb.ai/chrisxx/doom-overnight (run: bc_finetune_lstm_v1)

## Meta
See `.meta.json` files for button names, action map, obs shape.
