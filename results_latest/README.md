# Doom Deathmatch Agent — Latest Results

## Competition Submission (v3) — CURRENT

**Folder**: `results_latest/competition_submission_v3/`
**WandB**: https://wandb.ai/chrisxx/doom-overnight

### Best Model: BC+PPO High-Entropy

| Metric | Value |
|--------|-------|
| **Kills/episode** | 51.1 (10-ep avg, 2 bots) |
| **Deaths/episode** | 16.2 |
| **K/D ratio** | 3.2 (10-ep avg), up to 15 in best runs |
| **Best episode** | 65 kills, 3 deaths |
| **Actions used** | 19/19 (all macro actions) |
| **Inference mode** | `deterministic=False` (stochastic) |

### Critical Fix from v2
v2 models had **ent_coef=0.0** (bug: BC model didn't set it, and `.load()` preserved it).
This caused complete policy collapse — all models repeated a single action.

v3 uses **ent_coef=0.1** + stochastic inference, resulting in diverse, human-like behavior.

### How to Use
```python
from stable_baselines3 import PPO
model = PPO.load("bc_v2_highent_best.zip", device="cuda")
action, _ = model.predict(obs, deterministic=False)  # MUST be False!
```

### Training Pipeline
1. **Human behavioral cloning**: 10,460 frames, 100 epochs, 47.6% val accuracy
2. **PPO fine-tuning**: 775k best checkpoint (3M run), ent_coef=0.1, no attack_bonus, noop_penalty=0.1

---

## Architecture
- **CNN**: IMPALA ResNet (3 blocks: 16, 32, 32 channels, features_dim=256)
- **Policy**: PPO MLP (2x512 hidden layers)
- **Actions**: 19 macro-discrete (8 buttons: ATTACK, SPEED, MOVE_R/L/F/B, TURN_R/L)
- **Obs**: 120x160 RGB + 8 gamevars, frame_skip=4

## Key Scripts
- `train_overnight_dm.py` — main PPO training
- `pretrain_bc.py` — behavioral cloning from human demos
- `test_stochastic_video.py` — generate test videos
- `diagnose_policy.py` — diagnose action diversity / policy collapse

## Files
```
results_latest/
  competition_submission_v3/       # <-- USE THIS
    bc_v2_highent_best.zip         # Best model (PPO, stochastic inference)
    bc_v2_highent_best.meta.json   # Model metadata
    bc_v2_highent_775k.zip         # Backup copy of best model
    bc_v2_showcase_5ep.mp4         # 5-episode demo vs 2 bots: 262 kills, 45 deaths
    bc_v2_showcase.mp4             # 3-episode demo vs 1 bot: 169 kills, 28 deaths
    SUBMISSION.md                  # Full details and loading instructions
  competition_submission_v2/       # OLD (broken — spinning models)
```
