#!/usr/bin/env python3
"""
Sample frames from Sample Factory pretrained models for visual comparison.
Creates a grid of sample frames from each model.

Usage:
    uv run python sample_sf_frames.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

# Patch torch.load for PyTorch 2.6+
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def run_model_and_sample(experiment, train_dir="./sf_train_dir", num_samples=8, device="cpu"):
    """Run a model and sample frames at regular intervals."""
    from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
    register_vizdoom_components()

    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict

    # Load config
    cfg_path = Path(train_dir) / experiment / "cfg.json"
    if not cfg_path.exists():
        cfg_path = Path(train_dir) / experiment / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg = AttrDict(cfg)
    cfg.train_dir = train_dir
    cfg.experiment = experiment
    cfg.no_render = True
    cfg.skip_measurements_head = True

    # Create environment
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)
    env = make_env_func_batched(cfg, env_config=env_config, render_mode="rgb_array")

    # Create and load model
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    dev = torch.device(device)
    actor_critic = actor_critic.to(dev)

    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, 0), "best_*"
    )
    checkpoint_dict = Learner.load_checkpoint(checkpoints, dev)
    actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
    actor_critic.eval()

    # Run one episode and sample frames
    rnn_size = get_rnn_size(cfg)
    num_agents = env.num_agents
    obs, info = env.reset()
    rnn_states = torch.zeros(num_agents, rnn_size, device=dev)

    frames = []
    step = 0
    while True:
        obs_torch = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                obs_torch[key] = val.float().to(dev)
            else:
                t = torch.from_numpy(np.array(val)).float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                obs_torch[key] = t.to(dev)

        with torch.no_grad():
            normalized = actor_critic.normalize_obs(obs_torch)
            result = actor_critic(normalized, rnn_states)
            rnn_states = result["new_rnn_states"]
            actions = result["actions"]

        action = actions.cpu().numpy()
        obs, rew, terminated, truncated, infos = env.step(action)

        if "obs" in obs:
            raw_frame = obs["obs"]
            if isinstance(raw_frame, torch.Tensor):
                raw_frame = raw_frame[0].cpu().numpy()
                raw_frame = np.transpose(raw_frame, (1, 2, 0))
            frames.append(raw_frame.copy())

        step += 1
        if isinstance(terminated, (list, np.ndarray)):
            done = bool(terminated[0]) or bool(truncated[0])
        else:
            done = bool(terminated) or bool(truncated)
        if done:
            break

    env.close()

    # Sample frames at regular intervals
    if len(frames) < num_samples:
        sampled = frames
    else:
        indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
        sampled = [frames[i] for i in indices]

    return sampled


def create_frame_grid(frames_dict, output_path, scale=3):
    """Create a grid of sampled frames from multiple models."""
    num_models = len(frames_dict)
    num_samples = max(len(f) for f in frames_dict.values())

    # Get frame dimensions
    sample_frame = list(frames_dict.values())[0][0]
    h, w = sample_frame.shape[:2]
    sh, sw = h * scale, w * scale

    # Create grid with labels
    label_h = 30
    grid = np.zeros((num_models * (sh + label_h) + label_h, num_samples * sw, 3), dtype=np.uint8)

    for i, (model_name, frames) in enumerate(frames_dict.items()):
        y_offset = i * (sh + label_h) + label_h

        # Add model label
        cv2.putText(grid, model_name, (10, y_offset - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for j, frame in enumerate(frames):
            x_offset = j * sw
            # Scale up frame
            scaled = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
            if scaled.shape[2] == 3:
                scaled = cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)
            grid[y_offset:y_offset + sh, x_offset:x_offset + sw] = scaled

    cv2.imwrite(output_path, grid)
    return output_path


def main():
    output_dir = Path("results_latest/sf_frame_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "SF Seed 0": "00_bots_128_fs2_narrow_see_0",
        "SF Seed 2222": "doom_deathmatch_bots_2222",
        "SF Seed 3333": "doom_deathmatch_bots_3333",
    }

    all_frames = {}
    for label, experiment in models.items():
        print(f"Sampling frames from {label} ({experiment})...")
        frames = run_model_and_sample(experiment, num_samples=8)
        all_frames[label] = frames
        print(f"  Got {len(frames)} sample frames")

        # Save individual frames
        for i, f in enumerate(frames):
            fname = f"{experiment}_sample_{i:02d}.png"
            cv2.imwrite(
                str(output_dir / fname),
                cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            )

    # Create comparison grid
    grid_path = str(output_dir / "sf_models_comparison_grid.png")
    create_frame_grid(all_frames, grid_path)
    print(f"\nSaved comparison grid: {grid_path}")

    # Also create individual model grids
    for label, frames in all_frames.items():
        h, w = frames[0].shape[:2]
        cols = 4
        rows = (len(frames) + cols - 1) // cols
        scale = 3
        sh, sw = h * scale, w * scale
        grid = np.zeros((rows * sh, cols * sw, 3), dtype=np.uint8)
        for i, f in enumerate(frames):
            r, c = i // cols, i % cols
            scaled = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_NEAREST)
            if scaled.shape[2] == 3:
                scaled = cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)
            grid[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = scaled
        safe_label = label.replace(" ", "_").lower()
        cv2.imwrite(str(output_dir / f"{safe_label}_grid.png"), grid)

    print(f"\nAll frames saved to {output_dir}")


if __name__ == "__main__":
    main()
