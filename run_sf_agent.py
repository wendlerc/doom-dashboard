#!/usr/bin/env python3
"""
Run Sample Factory pretrained deathmatch agent and record frames/video.

Usage:
    uv run python run_sf_agent.py --episodes 5 --output results_latest/sf_showcase.mp4
    uv run python run_sf_agent.py --episodes 10 --save-frames sf_dataset/  # save frame data
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
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


def main():
    ap = argparse.ArgumentParser(description="Run Sample Factory pretrained deathmatch agent")
    ap.add_argument("--train-dir", default="./sf_train_dir")
    ap.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0")
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", "-o", default=None, help="Save video to this path")
    ap.add_argument("--save-frames", default=None, help="Save frame data to this directory")
    ap.add_argument("--fps", type=int, default=35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-render", action="store_true", default=True)
    args = ap.parse_args()

    # Register vizdoom components
    from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
    register_vizdoom_components()

    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict

    # Load config
    cfg_path = Path(args.train_dir) / args.experiment / "cfg.json"
    if not cfg_path.exists():
        cfg_path = Path(args.train_dir) / args.experiment / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg = AttrDict(cfg)
    cfg.train_dir = args.train_dir
    cfg.experiment = args.experiment
    cfg.no_render = True
    cfg.skip_measurements_head = True  # For pretrained model compatibility

    # Create environment
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)
    env = make_env_func_batched(cfg, env_config=env_config, render_mode="rgb_array")

    # Create and load model
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    device = torch.device(args.device)
    actor_critic = actor_critic.to(device)

    # Load checkpoint
    name_prefix = "best" if args.checkpoint == "best" else "checkpoint"
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, 0), f"{name_prefix}_*"
    )
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
    actor_critic.eval()

    print(f"Loaded model from {args.experiment}")
    print(f"  Checkpoint: {checkpoints[-1] if checkpoints else 'none'}")
    print(f"  Env: {cfg.env}")

    # Run episodes
    all_frames = []
    all_stats = []
    rnn_size = get_rnn_size(cfg)
    num_agents = env.num_agents

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep+1}/{args.episodes} ---")

        obs, info = env.reset()
        rnn_states = torch.zeros(num_agents, rnn_size, device=device)
        ep_reward = 0
        ep_frames = []
        step = 0

        while True:
            # Prepare observations - obs is already a dict of tensors
            obs_torch = {}
            for key, val in obs.items():
                if isinstance(val, torch.Tensor):
                    obs_torch[key] = val.float().to(device)
                else:
                    t = torch.from_numpy(np.array(val)).float()
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    obs_torch[key] = t.to(device)

            # Get action
            with torch.no_grad():
                normalized = actor_critic.normalize_obs(obs_torch)
                result = actor_critic(normalized, rnn_states)
                rnn_states = result["new_rnn_states"]
                actions = result["actions"]

            # Step environment
            action = actions.cpu().numpy()
            obs, rew, terminated, truncated, infos = env.step(action)
            if isinstance(rew, (list, np.ndarray)):
                ep_reward += float(rew[0]) if len(rew) > 0 else float(rew)
            else:
                ep_reward += float(rew)

            # Capture frame from the obs (raw game screen)
            if "obs" in obs:
                raw_frame = obs["obs"]
                if isinstance(raw_frame, torch.Tensor):
                    raw_frame = raw_frame[0].cpu().numpy()  # [C, H, W]
                    raw_frame = np.transpose(raw_frame, (1, 2, 0))  # [H, W, C]
                ep_frames.append(raw_frame.copy())

            step += 1
            if isinstance(terminated, (list, np.ndarray)):
                done = bool(terminated[0]) or bool(truncated[0])
            else:
                done = bool(terminated) or bool(truncated)
            if done:
                break

        # Extract stats from info
        info_str = ""
        if infos and len(infos) > 0:
            info_str = str(infos[0])

        print(f"  Steps: {step}, Reward: {ep_reward:.1f}, Frames: {len(ep_frames)}")
        all_frames.extend(ep_frames)
        all_stats.append({"episode": ep, "reward": float(ep_reward), "steps": step})

    env.close()

    # Save video
    if args.output and all_frames:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for f in all_frames:
            if f.shape[2] == 3:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            else:
                writer.write(f)
        writer.release()
        duration = len(all_frames) / args.fps
        print(f"\nSaved video: {args.output}")
        print(f"  {len(all_frames)} frames, {duration:.1f}s @ {args.fps}fps, {w}x{h}")

    # Save frame data
    if args.save_frames and all_frames:
        out_dir = Path(args.save_frames)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(all_frames):
            cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        print(f"\nSaved {len(all_frames)} frames to {args.save_frames}")

    # Print summary
    print(f"\n=== Summary ({args.episodes} episodes) ===")
    rewards = [s["reward"] for s in all_stats]
    print(f"  Avg reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Total frames: {len(all_frames)}")


if __name__ == "__main__":
    main()
