# Doom Deathmatch Agent — Latest Results

## Sample Factory Pretrained Models (v4) — CURRENT BEST

**Folder**: `results_latest/competition_submission_v3/videos/`
**WandB**: https://wandb.ai/chrisxx/doom-deathmatch
**Framework**: [Sample Factory](https://github.com/alex-petrenko/sample-factory) (APPO)

### Pretrained Models from HuggingFace

Three pretrained deathmatch models trained with APPO (Async PPO) + LSTM, downloaded from HuggingFace.
These are *significantly* better than the SB3-based models and show diverse, aggressive gameplay.

| Model | Source | Avg Reward | Frags/ep | Resolution |
|-------|--------|-----------|----------|------------|
| **SF Seed 0** | andrewzhang505/doom_deathmatch_bots | **25.0** | ~25 | 128x72 |
| **SF Seed 2222** | edbeeching/doom_deathmatch_bots_2222 | 24.0 | ~24 | 128x72 |
| **SF Seed 3333** | edbeeching/doom_deathmatch_bots_3333 | **25.1** | ~25-29 | 128x72 |

*5-episode averages, vs multiple bots on doom_deathmatch_bots map.*

### SF Architecture
- **Algorithm**: APPO (Async PPO) with V-trace off
- **Encoder**: ConvNet Simple → 512 MLP → LSTM 512
- **Input**: 128x72 RGB, CHW format, frameskip=4
- **Actions**: 39 discrete (full Doom action space)
- **Training**: ~2B+ environment steps, 20 workers x 12 envs

### How to Run Pretrained Models

```bash
# Generate video from pretrained model
uv run python run_sf_agent.py --experiment 00_bots_128_fs2_narrow_see_0 --episodes 5 --output sf_gameplay.mp4

# Other models
uv run python run_sf_agent.py --experiment doom_deathmatch_bots_2222 --episodes 3 --output sf_2222.mp4
uv run python run_sf_agent.py --experiment doom_deathmatch_bots_3333 --episodes 5 --output sf_3333.mp4
```

### Training From Scratch

```bash
# Start training with wandb logging (requires GPU)
uv run python train_sf_deathmatch.py --experiment=sf_dm_train_v1 --with_wandb=True

# Resume training
uv run python train_sf_deathmatch.py --experiment=sf_dm_train_v1 --restart_behavior=resume --with_wandb=True
```

Training runs at ~27k FPS on A6000 with 16 workers x 8 envs. Currently training `sf_dm_train_v1` from scratch.

### Showcase Videos
```
videos/
  sf_pretrained.mp4     # SF Seed 0 — 5 episodes (300s @ 35fps)
  sf_model_2222.mp4     # SF Seed 2222 — 3 episodes
  sf_model_3333.mp4     # SF Seed 3333 — 5 episodes
  ppo_vs_random.mp4     # Old SB3 PPO vs random (for comparison)
  bc_only_vs_random.mp4 # Old SB3 BC vs random (for comparison)
  ppo_vs_bc.mp4         # Old SB3 PPO vs BC
```

### Frame Samples
See `results_latest/sf_frame_samples/` for sampled gameplay frames from all 3 models.

---

## Previous Results (SB3-based, v3)

### BC Only (LSTM) — K/D 18.7 (Previous Best)

| Metric | BC Only | PPO 2M | PPO 775k |
|--------|---------|--------|----------|
| Kills/ep | 54.2 | 49.9 | 49.6 |
| Deaths/ep | 2.9 | 3.9 | 6.8 |
| K/D ratio | 18.7 | 12.8 | 7.3 |
| Architecture | RecurrentPPO LSTM | PPO MLP | PPO MLP |

*Note: User evaluated these visually and found them lacking — the agent doesn't look natural.*

### Key Scripts
- `run_sf_agent.py` — Run Sample Factory pretrained models, generate videos
- `train_sf_deathmatch.py` — Train SF deathmatch agent from scratch
- `sample_sf_frames.py` — Sample frames for visual inspection
- `log_sf_results_wandb.py` — Log SF results to wandb
- `make_showcase.py` — Side-by-side comparison videos
- `pretrain_bc.py` — Behavioral cloning from human demos
- `train_overnight_dm.py` — SB3 PPO training
