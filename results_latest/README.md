# Doom Deathmatch Agent — Latest Results

## Competition Submission (v2)

**Folder**: `results_latest/competition_submission_v2/`
**WandB**: https://wandb.ai/chrisxx/doom-overnight

### Best Models

| Model | Architecture | Steps | Eval Reward | Kills/2min | Deaths/2min | Notes |
|-------|-------------|-------|-------------|------------|-------------|-------|
| **bc_finetune_best** | IMPALA+LSTM-256 | 475k PPO | 133.4 | ~52 | ~12 | BC pretrained on human data |
| **v11a_lstm_best** | IMPALA+LSTM-256 | 4M PPO | 219 | ~53 | ~12 | Pure RL from scratch |

### Training Pipeline
1. **Human gameplay recording** (10,460 frames, fullaction config)
2. **Behavioral cloning** (47.6% val accuracy, 100 epochs) -> `bc_lstm_human_v2.zip`
3. **PPO fine-tuning** from BC model -> `bc_finetune_best.zip`
4. **Pure RL** (v11a_lstm: 4M steps from scratch) -> `v11a_lstm_best.zip`

### Key Features
- **IMPALA ResNet CNN** (3 blocks: 16, 32, 32 channels)
- **RecurrentPPO** with LSTM-256 for temporal context
- **19 macro-discrete actions** (compact: 8 buttons -> 19 combos)
- **Pretrained encoder support**: ResNet-18/34, EfficientNet-B0 (ImageNet weights)
- **Human behavioral cloning** pipeline with cross-config action mapping

---

## Architecture Evolution

### v10 (baseline): Feedforward MLP + IMPALA CNN
- PPO with 19 macro-discrete actions
- No temporal context, single frame input

### v11: Temporal Context
- **v11a**: RecurrentPPO LSTM-256 (4M steps, eval=219)
- **v11b**: PPO + VecFrameStack(4) (4M steps, poor multiplayer)

### v12: Human-initialized (BC + PPO)
- Behavioral cloning from 10k human gameplay frames
- PPO fine-tuning with same architecture as v11a
- Reaches v11a-level performance in ~475k steps (8x fewer)

---

## Reward Shaping
| Component | Value | Type |
|-----------|-------|------|
| frag_bonus | 200 | Per player kill |
| hit_bonus | 12.0 | Per hit landed |
| damage_bonus | 0.2 | Per damage dealt |
| death_penalty | 30 | Per death |
| attack_bonus | 0.05 | Per attack action |
| move_bonus | 0.2 | Position displacement |
| noop_penalty | 0.1 | Per idle frame |
| reward_scale | 0.1 | Global multiplier |

## Key Scripts
- `train_overnight_dm.py` — main training (supports LSTM/MLP/FrameStack, IMPALA/ResNet/EfficientNet)
- `pretrain_bc.py` — behavioral cloning from human demos (with wandb logging)
- `bench_model.py` — standalone model benchmark
- `record_gameplay.sh` — human gameplay recorder

## Files
```
results_latest/
  competition_submission_v2/
    bc_finetune_best.zip              # BC+PPO best model (primary submission)
    v11a_lstm_best.zip                # Pure RL model (backup)
    *.meta.json                       # Model metadata
    bc_agent_showcase.mp4             # BC model video demo
    v11a_lstm_showcase.mp4            # v11a model video demo
    SUBMISSION.md                     # Loading instructions
  README.md                          # This file
```
