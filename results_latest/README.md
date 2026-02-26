# Doom Deathmatch Agent — Latest Results

## Training In Progress (v11 + v12 pipeline)

**WandB**: https://wandb.ai/chrisxx/doom-overnight
**Monitoring**: `overnight_monitor.sh` (auto-benchmarks and generates videos)

### Active Training Runs (GPU: A6000 48GB, 2 parallel runs)

| Run | Config | Architecture | Steps | Reward | FPS | GPU |
|-----|--------|-------------|-------|--------|-----|-----|
| **v11a_lstm** | compact (8 btn, 19 macros) | IMPALA CNN + LSTM-256 | ~500k/4M | ~97 | 126 | 24.7GB |
| **v11b_framestack** | compact (8 btn, 19 macros) | IMPALA CNN + MLP + FS4 | ~700k/4M | ~99 | 174 | 23.7GB |

### Planned Pipeline (overnight_monitor.sh)
1. When FrameStack finishes (~5h) → benchmark it → launch **v12a_fullaction_lstm**
2. When LSTM finishes (~7h) → benchmark it
3. Fullaction LSTM trains on `deathmatch_fullaction.cfg` (18 buttons, 38 macros, weapon select)
4. Generate showcase videos for all models
5. Pick winner by composite score (hits, damage, distance)

### After Current Runs: Self-Play
- `selfplay_overnight.sh` — league-style self-play with opponent pool
- Trains against bots + evaluates against previous round's model
- Progressively harder opponents (1→3 bots across rounds)

---

## Previous Best: v10 (4M steps) — PASSES BOTH SCENARIOS

**Best Model**: `results_latest/overnight_fight_v10/model_4000k.zip`
**Videos**: `results_latest/overnight_fight_v10/videos/`

### v10 Benchmark (16 episodes, 2 bots, 2 min timelimit)

| Model | Scenario | Pass | Hits | Damage | Distance | vs Random |
|-------|----------|------|------|--------|----------|-----------|
| **v10 4M** | **Compact (monsters)** | **YES** | **13.0** | **191** | **13,600** | **2.4x hits, 2.9x dmg** |
| **v10 4M** | **No Monsters** | **YES** | **15.5** | **192** | **11,537** | **4.4x hits, 3.1x dmg** |

---

## Architecture Evolution

### v10 (baseline): Feedforward MLP + IMPALA CNN
- PPO with 19 macro-discrete actions (8 buttons → 19 combos)
- No temporal context — single frame input
- Good performance but limited strategic behavior

### v11: Temporal Context (LSTM / Frame Stacking)
- **v11a**: RecurrentPPO LSTM-256 — full hidden state across frames
- **v11b**: PPO + VecFrameStack(4) — cheap temporal context via stacked frames
- Both show improved eval reward vs v10 (178 eval reward at 400k steps for LSTM!)

### v12: Fullaction Deathmatch (weapon selection)
- 18 buttons including SELECT_WEAPON1-6, NEXT/PREV WEAPON
- 38 macro actions (vs 19 for compact)
- LSTM architecture for weapon-switching strategy
- Higher entropy coef (0.05) to encourage exploring the larger action space

### Future: Self-Play
- League-style training via `selfplay_overnight.sh` / `train_self_play.py`
- Opponent pool with progressive difficulty
- True 1v1 multiplayer evaluation (subprocess isolation)

---

## Human Gameplay Data

**Recording tool**: `bash record_gameplay.sh [scenario] [map] [minutes] [bots]`
**Demo location**: `human_demos/`
**BC pretraining**: `uv run python pretrain_bc.py --demo-dir human_demos/ --cfg ...`

### Current Status
- `human_deathmatch_fullaction_map01` — 865MB, **truncated** during git transfer
  - 3950/10470 frames recovered at 720p, NO action labels
  - Recommend re-recording at `--resolution RES_320X240` to avoid Zip64 issues

### BC Pipeline (ready when demos available)
1. `pretrain_bc.py` — supervised learning from human gameplay
2. Supports MLP and LSTM policies
3. Maps human button presses to macro actions (cross-format: fullaction→compact)
4. Fine-tune with PPO via `train_overnight_dm.py --init-model bc_model.zip`

---

## Reward Shaping
| Component | Value | Type |
|-----------|-------|------|
| frag_bonus | 200 | Per player kill |
| hit_bonus | 12.0 | Per hit landed |
| damage_bonus | 0.2 | Per damage dealt |
| death_penalty | 30 | Per death |
| attack_bonus | 0.05 | Per attack action |
| **move_bonus** | **0.2** | **Position displacement** |
| noop_penalty | 0.1 | Per idle frame |
| reward_scale | 0.1 | Global multiplier |

## Key Scripts
- `tloop_v11_lstm.sh` — LSTM training loop
- `tloop_v11_fs.sh` — FrameStack training loop
- `tloop_v12_fullaction_lstm.sh` — Fullaction LSTM training
- `overnight_monitor.sh` — automated pipeline (bench, launch, video, pick winner)
- `selfplay_overnight.sh` — league-style self-play
- `train_overnight_dm.py` — main training script (supports LSTM/MLP/FrameStack)
- `pretrain_bc.py` — behavioral cloning from human demos
- `bench_model.py` — standalone model benchmark
- `record_gameplay.sh` — human gameplay recorder

## Files
```
results_latest/
  overnight_fight_v10/
    model_4000k.zip                          # v10 best model
    videos/                                  # v10 showcase videos
  README.md                                  # this file

trained_policies/
  v11a_lstm_ckpts/ppo_*_steps.zip           # v11 LSTM checkpoints
  v11a_lstm_best/best_model.zip             # v11 LSTM best
  v11b_framestack_ckpts/ppo_*_steps.zip     # v11 FrameStack checkpoints
  v11b_framestack_best/best_model.zip       # v11 FrameStack best
  (v12a_fullaction_lstm_* coming soon)

human_demos/
  extracted_frames.npy                       # 3950 frames from human demo
  *.meta.json                                # demo metadata
```
