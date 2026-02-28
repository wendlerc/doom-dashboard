#!/usr/bin/env python3
"""
Train Sample Factory deathmatch agent with wandb logging.

Usage:
    uv run python train_sf_deathmatch.py --experiment=sf_dm_v2 --with_wandb=True
    uv run python train_sf_deathmatch.py --experiment=sf_dm_v2 --with_wandb=True --restart_behavior=resume
"""
import sys
import os

# Patch torch.load for PyTorch 2.6+
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components, parse_vizdoom_cfg
from sample_factory.train import run_rl


def main():
    register_vizdoom_components()

    # Default args - can be overridden from command line
    default_args = [
        "--algo=APPO",
        "--env=doom_deathmatch_bots",
        "--train_dir=./sf_train_dir",
        "--experiment=sf_dm_v2",
        "--num_workers=16",
        "--num_envs_per_worker=8",
        "--batch_size=2048",
        "--num_epochs=1",
        "--rollout=32",
        "--recurrence=32",
        "--gamma=0.99",
        "--use_rnn=True",
        "--rnn_type=lstm",
        "--nonlinearity=relu",
        "--learning_rate=0.0001",
        "--max_grad_norm=0.0",
        "--exploration_loss_coeff=0.001",
        "--decorrelate_experience_max_seconds=1",
        "--heartbeat_reporting_interval=300",
        "--train_for_env_steps=100000000",
        "--save_every_sec=120",
        "--save_best_every_sec=30",
        "--wandb_project=doom-deathmatch",
        "--wandb_group=sf_training",
    ]

    # Merge: command line args override defaults
    cli_args = sys.argv[1:]
    cli_keys = {a.split("=")[0] for a in cli_args if a.startswith("--")}
    argv = [a for a in default_args if a.split("=")[0] not in cli_keys] + cli_args

    cfg = parse_vizdoom_cfg(argv=argv)
    cfg.skip_measurements_head = True

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
