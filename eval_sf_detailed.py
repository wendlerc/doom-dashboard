#!/usr/bin/env python3
"""
Detailed evaluation of SF models: extract kills, deaths, damage from game variables.

Usage:
    uv run python eval_sf_detailed.py --experiment 00_bots_128_fs2_narrow_see_0 --episodes 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Patch torch.load for PyTorch 2.6+
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0")
    ap.add_argument("--train-dir", default="./sf_train_dir")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

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
    cfg.skip_measurements_head = True

    # Create environment
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)
    env = make_env_func_batched(cfg, env_config=env_config, render_mode="rgb_array")

    # Create and load model
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    dev = torch.device(args.device)
    actor_critic = actor_critic.to(dev)

    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, 0), "best_*"
    )
    checkpoint_dict = Learner.load_checkpoint(checkpoints, dev)
    actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
    actor_critic.eval()

    print(f"Model: {args.experiment}")
    print(f"Checkpoint: {checkpoints[-1]}")

    # Run episodes
    rnn_size = get_rnn_size(cfg)
    num_agents = env.num_agents
    episode_stats = []

    for ep in range(args.episodes):
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

            step += 1
            if isinstance(terminated, (list, np.ndarray)):
                done = bool(terminated[0]) or bool(truncated[0])
            else:
                done = bool(terminated) or bool(truncated)
            if done:
                break

        # Extract stats from infos
        info_dict = {}
        if infos and len(infos) > 0:
            inf = infos[0] if isinstance(infos, list) else infos
            if isinstance(inf, dict):
                info_dict = inf

        stat = {
            "episode": ep,
            "reward": ep_reward,
            "steps": step,
        }

        # Try to extract game stats from reward structure in the log
        # The reward is a composite - look at info for FRAGCOUNT/DEATHCOUNT
        for key in ["FRAGCOUNT", "DEATHCOUNT", "HITCOUNT", "DAMAGECOUNT", "HEALTH", "ARMOR"]:
            if key in info_dict:
                stat[key.lower()] = float(info_dict[key])

        episode_stats.append(stat)
        print(f"  Ep {ep+1}: reward={ep_reward:.1f}, steps={step}, info={info_dict}")

    env.close()

    # Print summary
    rewards = [s["reward"] for s in episode_stats]
    print(f"\n=== Summary ({args.episodes} episodes) ===")
    print(f"  Experiment: {args.experiment}")
    print(f"  Avg reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Min/Max reward: {np.min(rewards):.1f} / {np.max(rewards):.1f}")

    # Print game stats if available
    for key in ["fragcount", "deathcount", "hitcount", "damagecount"]:
        vals = [s.get(key) for s in episode_stats if key in s]
        if vals:
            print(f"  Avg {key}: {np.mean(vals):.1f}")

    # Save results
    results_path = Path("results_latest") / f"eval_{args.experiment}.json"
    with open(results_path, "w") as f:
        json.dump({
            "experiment": args.experiment,
            "checkpoint": str(checkpoints[-1]),
            "episodes": args.episodes,
            "stats": episode_stats,
            "summary": {
                "avg_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
            }
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
