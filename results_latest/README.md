# Doom Deathmatch Agent — Latest Results

## Sample Factory Models — CURRENT BEST

**WandB**: https://wandb.ai/chrisxx/doom-deathmatch
**Framework**: [Sample Factory](https://github.com/alex-petrenko/sample-factory) (APPO)

### Pretrained Models (HuggingFace) — Best Available

Three pretrained deathmatch agents trained with APPO + LSTM (~2B frames each).
These agents **win 1st place** against 7 bots in most matches.

| Model | Source | Avg Reward | Frags/ep | Wins (out of 10) |
|-------|--------|-----------|----------|-------------------|
| **SF Seed 0** | andrewzhang505/doom_deathmatch_bots | **28.0** | **25.2** | **9/10 (1st place)** |
| SF Seed 2222 | edbeeching/doom_deathmatch_bots_2222 | 24.0 | ~24 | - |
| SF Seed 3333 | edbeeching/doom_deathmatch_bots_3333 | 25.1 | ~25 | - |

*10-episode averages, 8-player FFA (1 agent + 7 bots).*

### Trained From Scratch (sf_dm_train_v1) — In Progress

Training from scratch at ~28k FPS on A6000. Currently resuming to 500M frames.

| Checkpoint | Frames | Avg Reward | Frags/ep |
|-----------|--------|-----------|----------|
| 100M (completed) | 100M | 7.8 | 7.2 |
| 500M (training...) | 500M | TBD | TBD |

Reward progression (100M run):
```
 10M: -5.7 → 20M: 0.0 → 30M: 0.9 → 47M: 2.7 → 72M: 4.8 → 100M: 9.15
```

### Architecture
- **Algorithm**: APPO (Async PPO) with GAE (lambda=0.95)
- **Encoder**: ConvNet Simple → 512 MLP → LSTM 512
- **Input**: 128x72 RGB, CHW format, frameskip=4
- **Actions**: 39 discrete (full Doom action space)
- **Training**: 16 workers x 8 envs, batch_size=2048

### How to Run

```bash
# Run pretrained model
uv run python run_sf_agent.py --experiment 00_bots_128_fs2_narrow_see_0 --episodes 5 --output gameplay.mp4

# Train from scratch with wandb
uv run python train_sf_deathmatch.py --experiment=sf_dm_v2 --with_wandb=True

# Resume training
uv run python train_sf_deathmatch.py --experiment=sf_dm_train_v1 --restart_behavior=resume --with_wandb=True

# Evaluate a model
uv run python eval_sf_detailed.py --experiment sf_dm_train_v1 --episodes 10

# Sample frames for visual inspection
uv run python sample_sf_frames.py

# Monitor training progress (logs videos to wandb)
uv run python monitor_sf_training.py --experiment sf_dm_train_v1 --interval 600
```

### Showcase Videos
```
sf_best_showcase.mp4        # Best pretrained (Seed 0), 10 episodes, avg reward 25.6
sf_pretrained.mp4           # Seed 0, 5 episodes
sf_model_2222.mp4           # Seed 2222, 3 episodes
sf_model_3333.mp4           # Seed 3333, 5 episodes
sf_trained_100M.mp4         # Trained from scratch, 100M frames, avg reward 7.7
sf_training_progress_47M.mp4 # Training checkpoint at 47M
```

### Frame Samples
See `results_latest/sf_frame_samples/` for sampled gameplay frames from all 3 pretrained models.

---

## Previous Results (SB3-based, deprecated)

BC Only LSTM (K/D=18.7 vs 1 bot) and PPO models. User found these visually lacking.
See git history for details.

### Key Scripts
- `run_sf_agent.py` — Run SF pretrained models, generate videos
- `train_sf_deathmatch.py` — Train SF deathmatch from scratch with wandb
- `eval_sf_detailed.py` — Detailed evaluation (frags, deaths, damage)
- `sample_sf_frames.py` — Sample frames for visual inspection
- `monitor_sf_training.py` — Monitor training, log videos to wandb
- `log_sf_results_wandb.py` — Log evaluation results to wandb
- `make_showcase.py` — Side-by-side comparison videos
