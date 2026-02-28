# Doom Deathmatch Agent — Latest Results

## Sample Factory Models — CURRENT BEST

**WandB**: https://wandb.ai/chrisxx/doom-deathmatch
**Framework**: [Sample Factory](https://github.com/wendlerc/sample-factory/tree/doom-arena) (APPO)
**Fork**: https://github.com/wendlerc/sample-factory (branch: `doom-arena`)

### Trained From Scratch (sf_dm_train_v1) — BEST MODEL

Trained from scratch to 500M frames (~2.5h on A6000). **3x better than pretrained models!**

| Metric | Value |
|--------|-------|
| **Avg Reward** | **75.3 +/- 6.0** |
| **Avg Frags/ep** | **63.3** |
| Deaths/ep | ~20 |
| Damage/ep | ~6500 |
| K/D Ratio | ~3.2 |
| Frames Trained | 500M |

Reward progression:
```
 10M: -5.7 → 20M: 0.0 → 30M: 0.9 → 47M: 2.7 → 72M: 4.8 → 100M: 9.15 → 155M: 29.8 → 500M: 75.3
```

### Pretrained Models (HuggingFace)

| Model | Source | Avg Reward | Frags/ep |
|-------|--------|-----------|----------|
| SF Seed 0 | andrewzhang505/doom_deathmatch_bots | 28.0 | 25.2 |
| SF Seed 2222 | edbeeching/doom_deathmatch_bots_2222 | 24.0 | ~24 |
| SF Seed 3333 | edbeeching/doom_deathmatch_bots_3333 | 25.1 | ~25 |

### Architecture
- **Algorithm**: APPO (Async PPO) with GAE (lambda=0.95)
- **Encoder**: ConvNet Simple -> 512 MLP -> LSTM 512
- **Input**: 128x72 RGB, CHW format, frameskip=4
- **Actions**: 39 discrete (full Doom action space)
- **Training**: 16 workers x 8 envs, batch_size=2048
- **Map**: dwango5.wad (deathmatch, 7 bots)

### How to Run (using the fork)

```bash
# Install the fork
pip install -e "/path/to/sample-factory[vizdoom]"

# Download pretrained models
python -m sf_examples.vizdoom.doom_arena.download_models

# Run best trained agent
python -m sf_examples.vizdoom.doom_arena.run_agent \
    --experiment sf_dm_train_v1 --episodes 5 --output gameplay.mp4

# Train from scratch with wandb
python -m sf_examples.vizdoom.doom_arena.train_deathmatch \
    --experiment=my_run --with_wandb=True --train_for_env_steps=500000000

# Evaluate a model
python -m sf_examples.vizdoom.doom_arena.eval_detailed --experiment sf_dm_train_v1 --episodes 10

# Sample frames for visual inspection
python -m sf_examples.vizdoom.doom_arena.sample_frames
```

### Showcase Videos
```
sf_trained_500M.mp4             # Best model (500M), 5 episodes, avg reward 75.3
sf_trained_155M_showcase.mp4    # 155M checkpoint, avg reward 29.8
sf_trained_100M.mp4             # 100M checkpoint, avg reward 7.7
sf_best_showcase.mp4            # Best pretrained (Seed 0), avg reward 25.6
```

---

## Previous Results (SB3-based, deprecated)

BC Only LSTM (K/D=18.7 vs 1 bot) and PPO models. User found these visually lacking.
See git history for details.
