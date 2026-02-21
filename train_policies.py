#!/usr/bin/env python3
"""
Train SB3 PPO policies on key VizDoom scenarios in parallel.

One subprocess per scenario — each trains its own PPO agent and saves
a .zip checkpoint to ./trained_policies/<scenario>.zip

Usage:
    uv run python train_policies.py [--timesteps 500000] [--envs 8]

Requirements: stable-baselines3, opencv-python-headless, vizdoom
(all in pyproject.toml)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import cv2
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

import vizdoom  # noqa – registers envs
import vizdoom.gymnasium_wrapper  # noqa – registers Gymnasium envs


# ─── scenario → gymnasium env name ───────────────────────────────
SCENARIO_ENV_MAP = {
    "basic":               "VizdoomBasic-v1",
    "defend_the_center":   "VizdoomDefendCenter-v1",
    "deadly_corridor":     "VizdoomDeadlyCorridor-v1",
    "health_gathering":    "VizdoomHealthGathering-v1",
    "my_way_home":         "VizdoomMyWayHome-v1",
}

IMAGE_SHAPE = (60, 80)
REWARD_SCALE = 0.01
FRAME_SKIP = 4
N_STEPS = 128


# ─── observation wrapper (same as the SB3 example) ────────────────
IMAGE_BUFFER_KEYS = ("screen", "depth", "labels", "automap")


class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.image_buffer_keys = [
            k for k in env.observation_space.spaces if k in IMAGE_BUFFER_KEYS
        ]
        spaces = {}
        for key, space in env.observation_space.spaces.items():
            if key in IMAGE_BUFFER_KEYS:
                nc = space.shape[-1] if len(space.shape) >= 3 else 1
                spaces[key] = gymnasium.spaces.Box(0, 255, shape=(shape[0], shape[1], nc), dtype=np.uint8)
            else:
                spaces[key] = space
        self.observation_space = gymnasium.spaces.Dict(spaces)

    def observation(self, obs):
        out = {}
        for k, v in obs.items():
            if k in self.image_buffer_keys:
                r = cv2.resize(v, self.image_shape_reverse)
                if r.ndim == 2:
                    r = r[..., None]
                out[k] = r
            elif k == "gamevariables":
                out[k] = np.asarray(v, dtype=np.float32)
            else:
                out[k] = v
        return out


# ─── training function (runs in a subprocess) ────────────────────

def train_one(scenario_name: str, env_id: str, out_dir: str,
              total_timesteps: int, n_envs: int):
    """Train PPO on one scenario and save the checkpoint."""
    print(f"[{scenario_name}] Starting training on {env_id} "
          f"({total_timesteps:,} steps, {n_envs} envs)")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scenario_name}.zip")

    def make_env():
        env = gymnasium.make(env_id, frame_skip=FRAME_SKIP)
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * REWARD_SCALE)
        return env

    vec_env = make_vec_env(make_env, n_envs=n_envs)

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1000),
        save_path=os.path.join(out_dir, f"{scenario_name}_ckpts"),
        name_prefix="ppo",
        verbose=0,
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        n_steps=N_STEPS,
        learning_rate=1e-3,
        verbose=1,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=ckpt_cb, progress_bar=True)
    except ImportError:
        model.learn(total_timesteps=total_timesteps, callback=ckpt_cb)

    model.save(out_path)
    print(f"[{scenario_name}] ✓ Saved to {out_path}")
    vec_env.close()


# ─── entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train VizDoom PPO policies")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps per scenario (default: 500k)")
    parser.add_argument("--envs", type=int, default=8,
                        help="Parallel envs per scenario during training (default: 8)")
    parser.add_argument("--out", type=str, default="./trained_policies",
                        help="Output directory for checkpoints")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_ENV_MAP.keys()),
                        choices=list(SCENARIO_ENV_MAP.keys()),
                        help="Scenarios to train on (default: all)")
    args = parser.parse_args()

    selected = {k: SCENARIO_ENV_MAP[k] for k in args.scenarios}
    print(f"Training {len(selected)} policies: {list(selected.keys())}")
    print(f"  timesteps/scenario={args.timesteps:,}  envs/scenario={args.envs}")
    print(f"  output: {args.out}\n")

    # Use 'spawn' context to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    procs = []
    for sc_name, env_id in selected.items():
        p = ctx.Process(
            target=train_one,
            args=(sc_name, env_id, args.out, args.timesteps, args.envs),
            name=f"train-{sc_name}",
        )
        p.start()
        procs.append((sc_name, p))

    failed = []
    for sc_name, p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"[{sc_name}] ✗ Training failed (exit code {p.exitcode})")
            failed.append(sc_name)
        else:
            print(f"[{sc_name}] ✓ Done")

    if failed:
        print(f"\n{len(failed)} scenario(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\n✓ All policies saved to: {args.out}/")
        print("\nNow add them to config.yaml under 'policies:', e.g.:")
        for sc in selected:
            print(f"  - name: 'PPO-{sc}'")
            print(f"    type: sb3")
            print(f"    path: '{args.out}/{sc}.zip'")
            print(f"    algo: PPO")


if __name__ == "__main__":
    main()
