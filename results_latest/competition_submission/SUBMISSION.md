# Doom Deathmatch Competition Submission

## Best Model: v11a_lstm (RecurrentPPO + IMPALA CNN)

**Recommended model for submission: `v11a_lstm_best.zip`**

### Models (pick one)

| File | Steps | Eval Reward | Notes |
|------|-------|-------------|-------|
| `v11a_lstm_best.zip` | ~1.2M | **219** (peak) | Best eval callback model |
| `v11a_lstm_2850k.zip` | 2.85M | 210 | Strong late-training model |
| `v11a_lstm_3400k.zip` | 3.4M | 188 | Latest checkpoint |
| `v11a_lstm_latest.zip` | 3.4M | 188 | Same as above |
| `v10_mlp_4000k.zip` | 4M | ~140 | Fallback (no LSTM, simpler) |

### Architecture
- **CNN**: IMPALA ResNet (3 blocks: 16,32,32 channels, features_dim=256)
- **Policy**: RecurrentPPO (LSTM hidden=256, 1 layer, shared=False, enable_critic_lstm=True)
- **MLP**: 2x512 hidden layers
- **Actions**: 19 macro-discrete (8 buttons: move, turn, attack, speed)
- **Obs**: 120x160 RGB, frame_skip=4
- **Training**: deathmatch_compact.cfg, 1 bot, map01

### How to load
```python
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load("v11a_lstm_best.zip", device="cuda")

# For inference with LSTM state:
obs = env.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_starts = done
```

### Meta info
See `v11a_lstm_best.meta.json` for button names, action map, obs shape.

### WandB
https://wandb.ai/chrisxx/doom-overnight
