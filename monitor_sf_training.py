#!/usr/bin/env python3
"""
Monitor SF training: periodically evaluate the latest checkpoint and log videos to wandb.

Usage:
    uv run python monitor_sf_training.py --experiment sf_dm_train_v1 --interval 300
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb

# Patch torch.load for PyTorch 2.6+
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def evaluate_checkpoint(experiment, train_dir="./sf_train_dir", episodes=3, device="cpu"):
    """Run a few episodes with the latest checkpoint and return stats + frames."""
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
    if not cfg_path.exists():
        return None
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

    # Try best checkpoint first, then latest
    for prefix in ["best_*", "checkpoint_*"]:
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(cfg, 0), prefix
        )
        if checkpoints:
            break

    if not checkpoints:
        env.close()
        return None

    checkpoint_dict = Learner.load_checkpoint(checkpoints, dev)
    actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
    actor_critic.eval()

    checkpoint_name = os.path.basename(checkpoints[-1])

    # Run episodes
    rnn_size = get_rnn_size(cfg)
    num_agents = env.num_agents
    all_rewards = []
    all_frames = []

    for ep in range(episodes):
        obs, info = env.reset()
        rnn_states = torch.zeros(num_agents, rnn_size, device=dev)
        ep_reward = 0
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
            if isinstance(rew, (list, np.ndarray)):
                ep_reward += float(rew[0]) if len(rew) > 0 else float(rew)
            else:
                ep_reward += float(rew)

            if "obs" in obs:
                raw_frame = obs["obs"]
                if isinstance(raw_frame, torch.Tensor):
                    raw_frame = raw_frame[0].cpu().numpy()
                    raw_frame = np.transpose(raw_frame, (1, 2, 0))
                all_frames.append(raw_frame.copy())

            step += 1
            if isinstance(terminated, (list, np.ndarray)):
                done = bool(terminated[0]) or bool(truncated[0])
            else:
                done = bool(terminated) or bool(truncated)
            if done:
                break

        all_rewards.append(ep_reward)

    env.close()

    return {
        "checkpoint": checkpoint_name,
        "rewards": all_rewards,
        "avg_reward": float(np.mean(all_rewards)),
        "frames": all_frames,
    }


def save_video(frames, path, fps=35):
    """Save frames as video."""
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        if f.shape[2] == 3:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        else:
            writer.write(f)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="sf_dm_train_v1")
    ap.add_argument("--train-dir", default="./sf_train_dir")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between evaluations")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-evals", type=int, default=50)
    args = ap.parse_args()

    run = wandb.init(
        project="doom-deathmatch",
        name=f"monitor_{args.experiment}",
        group="sf_monitoring",
        tags=["monitoring", "sample_factory"],
        config={"experiment": args.experiment, "eval_episodes": args.episodes},
    )

    eval_count = 0
    last_checkpoint = None

    while eval_count < args.max_evals:
        print(f"\n--- Evaluation {eval_count + 1} ---")
        result = evaluate_checkpoint(
            args.experiment, args.train_dir, args.episodes
        )

        if result is None:
            print("No checkpoint available yet, waiting...")
            time.sleep(args.interval)
            continue

        if result["checkpoint"] == last_checkpoint:
            print(f"Same checkpoint ({result['checkpoint']}), waiting...")
            time.sleep(args.interval)
            continue

        last_checkpoint = result["checkpoint"]
        eval_count += 1

        print(f"  Checkpoint: {result['checkpoint']}")
        print(f"  Avg reward: {result['avg_reward']:.1f}")
        print(f"  Rewards: {result['rewards']}")

        # Save and log video
        video_path = f"results_latest/sf_training_progress/{args.experiment}_eval_{eval_count:03d}.mp4"
        save_video(result["frames"], video_path)

        log_data = {
            "eval/avg_reward": result["avg_reward"],
            "eval/min_reward": min(result["rewards"]),
            "eval/max_reward": max(result["rewards"]),
            "eval/checkpoint": result["checkpoint"],
            "eval/eval_count": eval_count,
        }

        if os.path.exists(video_path):
            log_data["eval/gameplay_video"] = wandb.Video(
                video_path, fps=35, format="mp4",
                caption=f"Eval {eval_count}: reward={result['avg_reward']:.1f}"
            )

        wandb.log(log_data)
        print(f"  Logged to wandb (eval {eval_count})")

        if eval_count < args.max_evals:
            time.sleep(args.interval)

    wandb.finish()
    print(f"\nMonitoring complete ({eval_count} evaluations)")


if __name__ == "__main__":
    main()
