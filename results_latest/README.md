# Doom Deathmatch Agent — Latest Results

## Best Model: v10 (4M steps) — PASSES BOTH SCENARIOS

**WandB**: https://wandb.ai/chrisxx/doom-overnight
**Best Model**: `results_latest/overnight_fight_v10/model_4000k.zip`
**Videos**: `results_latest/overnight_fight_v10/videos/` (also served on port 8766)

### Final Benchmark (16 episodes, 2 bots, 2 min timelimit)

| Model | Scenario | Pass | Hits | Damage | Distance | vs Random |
|-------|----------|------|------|--------|----------|-----------|
| **v10 4M** | **Compact (monsters)** | **YES** | **13.0** | **191** | **13,600** | **2.4x hits, 2.9x dmg** |
| **v10 4M** | **No Monsters** | **YES** | **15.5** | **192** | **11,537** | **4.4x hits, 3.1x dmg** |

### Training Progression

| Checkpoint | Compact Hits | Compact Dmg | NoMonsters Hits | NoMonsters Dmg | Pass? |
|------------|-------------|-------------|-----------------|----------------|-------|
| 1.2M | 14.2 | 216 | 14.6 | 167 | Both |
| 2.0M | 30.0 | 476 | 13.6 | 177 | Both |
| 3.0M | 21.4 | 290 | 15.2 | 247 | Both |
| 4.0M | 13.0 | 191 | 15.5 | 192 | Both |

Note: High variance between runs due to stochastic ViZDoom deathmatch. All checkpoints pass.

### Movement Validation (frame-diff analysis)
All training videos from 400k+ show 100% moving frames (mean diff > 20).
Previous v9 model was a stationary turret (mean diff 3.2).

### Key Breakthrough: Position-Based Movement Reward
Previous models (v9) were **stationary turrets** — they learned to stand still and shoot.
The v10 model uses **position-based movement reward** (POSITION_X/Y displacement per step)
instead of action-based reward, preventing the agent from gaming the reward by spinning in place.

### Architecture
- IMPALA ResNet CNN (features_dim=256) + 2x512 MLP
- PPO with macro-discrete actions (19 actions from 8 buttons)
- Trained on `deathmatch_compact.cfg` (with monsters), transfers to nomonsters
- 4 envs, 1 bot/env, frame_skip=4, obs=120x160
- Entropy coef=0.03 (prevents policy collapse to 1-2 actions)

### Reward Shaping
| Component | Value | Type |
|-----------|-------|------|
| frag_bonus | 200 | Per player kill |
| hit_bonus | 12.0 | Per hit landed |
| damage_bonus | 0.2 | Per damage dealt |
| death_penalty | 30 | Per death |
| attack_bonus | 0.05 | Per attack action |
| **move_bonus** | **0.2** | **Position displacement (min 1.0 units, cap 1.5x)** |
| noop_penalty | 0.1 | Per idle frame |
| reward_scale | 0.1 | Global multiplier |

### Benchmark Gate
```
hit_mean >= 8               # land hits consistently
damage_mean >= 100          # deal meaningful damage
hit_mean > random_hit_mean  # outperform random on hits
distance > random * 0.5     # must actually move (no turrets)
kills > random OR damage > random  # outperform random overall
```

## Files
```
results_latest/overnight_fight_v10/
  model_4000k.zip                          # Best model (4M steps)
  model_3000k.zip                          # 3M checkpoint
  model_2000k.zip                          # 2M checkpoint
  model_1200k.zip                          # 1.2M checkpoint (first passing)
  bench_compact_4000k.json                 # Final compact benchmark
  bench_nomonsters_4000k.json              # Final nomonsters benchmark
  videos/
    showcase_compact_4000k.mp4             # Latest compact gameplay
    showcase_nomonsters_4000k.mp4          # Latest nomonsters gameplay
    showcase_compact_2000k.mp4             # 2M compact gameplay
    showcase_nomonsters_2000k.mp4          # 2M nomonsters gameplay
    showcase_compact_1200k.mp4             # First passing model
    showcase_nomonsters_1200k.mp4
```

## Previous Models

| Model | Scenario | Moves? | Fights? | Pass? | Notes |
|-------|----------|--------|---------|-------|-------|
| **v10 (4M)** | **both** | **YES** | **YES** | **YES** | Position-based reward |
| v9 (2.15M) | nomonsters | NO | YES | old gate | Turret |
| exp_fighter_v2 | compact | YES | Monsters only | NO | Kills monsters not players |

## External Resources Investigated
- **Sample Factory APPO** (HuggingFace) — different framework, not SB3 compatible
- **P-H-B-D-a16z/ViZDoom-Deathmatch-PPO** — 5k episodes, 18 actions, no quality labels
- **thavlik/doom-gameplay-dataset** — 170h video, no actions
- **Conclusion**: No drop-in pretrained models for SB3; training from scratch is best

## Key Scripts
- `tloop_v10.sh` — training loop with crash recovery
- `train_overnight_dm.py` — training + position-based reward + bench
- `validate_and_showcase.sh` — end-to-end validation
