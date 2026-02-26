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
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

import cv2
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage

import vizdoom  # noqa – registers envs
import vizdoom.gymnasium_wrapper  # noqa – registers Gymnasium envs


# ─── scenario → gymnasium env name ───────────────────────────────
SCENARIO_ENV_MAP = {
    "basic":               "VizdoomBasic-v1",
    "defend_the_center":   "VizdoomDefendCenter-v1",
    "deadly_corridor":     "VizdoomDeadlyCorridor-v1",
    "health_gathering":    "VizdoomHealthGathering-v1",
    "my_way_home":         "VizdoomMyWayHome-v1",
    # Compact deathmatch action set for more stable combat learning.
    "deathmatch":          "CFG::deathmatch_compact",
}

IMAGE_SHAPE = (60, 80)
REWARD_SCALE = 0.01
FRAME_SKIP = 4
N_STEPS = 128
PPO_CFG = {
    "n_steps": 256,
    "batch_size": 512,
    "n_epochs": 4,
    "learning_rate": 2.5e-4,
    "clip_range": 0.1,
    "ent_coef": 0.001,
    "target_kl": 0.03,
}


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


class BinaryOnlyActionWrapper(gymnasium.ActionWrapper):
    """Expose only the `binary` branch when env action space is Dict(binary, continuous)."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.action_space, gymnasium.spaces.Dict) or "binary" not in env.action_space.spaces:
            raise TypeError("BinaryOnlyActionWrapper requires Dict action space with key 'binary'.")
        self._bin = env.action_space.spaces["binary"]
        self._cont = env.action_space.spaces.get("continuous")
        self.action_space = self._bin

    def action(self, act):
        if isinstance(self._bin, gymnasium.spaces.Discrete):
            out_bin = int(act)
        else:
            out_bin = np.asarray(act, dtype=np.int8)
        out = {"binary": out_bin}
        if self._cont is not None:
            out["continuous"] = np.zeros(self._cont.shape, dtype=np.float32)
        return out


@dataclass
class DeathmatchShapingConfig:
    frag_bonus: float = 5.0
    kill_bonus: float = 2.0
    hit_bonus: float = 0.25
    damage_bonus: float = 0.01
    hit_taken_penalty: float = 0.15
    damage_taken_penalty: float = 0.01
    death_penalty: float = 3.0
    living_penalty: float = 0.001


class DeathmatchRewardShapingWrapper(gymnasium.Wrapper):
    """Dense reward shaping for sparse deathmatch feedback.

    Uses positive deltas of combat counters when available and applies small
    living penalty so idle behavior is disfavored.
    """

    def __init__(self, env, cfg: DeathmatchShapingConfig):
        super().__init__(env)
        self.cfg = cfg
        self._prev_gv = None
        self._var_to_idx = {}
        try:
            vars_ = env.unwrapped.game.get_available_game_variables()
            self._var_to_idx = {str(v).split(".")[-1]: i for i, v in enumerate(vars_)}
        except Exception:
            self._var_to_idx = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        gv = obs.get("gamevariables") if isinstance(obs, dict) else None
        self._prev_gv = np.asarray(gv, dtype=np.float32).copy() if gv is not None else None
        return obs, info

    def _delta(self, prev: np.ndarray, curr: np.ndarray, key: str) -> float:
        idx = self._var_to_idx.get(key)
        if idx is None or idx >= prev.shape[0] or idx >= curr.shape[0]:
            return 0.0
        return float(curr[idx] - prev[idx])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        gv = obs.get("gamevariables") if isinstance(obs, dict) else None
        curr = np.asarray(gv, dtype=np.float32).reshape(-1) if gv is not None else None
        prev = self._prev_gv
        shaped = float(reward) - self.cfg.living_penalty
        if prev is not None and curr is not None and prev.shape[0] > 0 and curr.shape[0] > 0:
            # Positive progress
            shaped += max(0.0, self._delta(prev, curr, "FRAGCOUNT")) * self.cfg.frag_bonus
            shaped += max(0.0, self._delta(prev, curr, "KILLCOUNT")) * self.cfg.kill_bonus
            shaped += max(0.0, self._delta(prev, curr, "HITCOUNT")) * self.cfg.hit_bonus
            shaped += max(0.0, self._delta(prev, curr, "DAMAGECOUNT")) * self.cfg.damage_bonus
            # Negative events
            shaped -= max(0.0, self._delta(prev, curr, "HITS_TAKEN")) * self.cfg.hit_taken_penalty
            shaped -= max(0.0, self._delta(prev, curr, "DAMAGE_TAKEN")) * self.cfg.damage_taken_penalty
            shaped -= max(0.0, self._delta(prev, curr, "DEATHCOUNT")) * self.cfg.death_penalty

        self._prev_gv = curr.copy() if curr is not None else None
        return obs, shaped, terminated, truncated, info


class DeathmatchBotsWrapper(gymnasium.Wrapper):
    """Inject Doom bots into deathmatch episodes.

    Uses VizDoom console commands:
      - `removebots`
      - repeated `addbot`
    """

    def __init__(self, env, n_bots: int):
        super().__init__(env)
        self.n_bots = max(0, int(n_bots))

    def reset(self, **kwargs):
        # Remove leftover bots from previous episode before new_episode() is called.
        try:
            g0 = self.env.unwrapped.game
            g0.send_game_command("removebots")
        except Exception:
            pass

        obs, info = self.env.reset(**kwargs)
        if self.n_bots <= 0:
            return obs, info
        try:
            g = self.env.unwrapped.game
            g.send_game_command("removebots")
            for _ in range(self.n_bots):
                g.send_game_command("addbot")
        except Exception:
            # Keep training robust even if bot commands fail in some env builds.
            pass
        return obs, info


# ─── training function (runs in a subprocess) ────────────────────

def train_one(scenario_name: str, env_id: str, out_dir: str,
              total_timesteps: int, n_envs: int, ppo_cfg: dict, frame_skip: int, reward_scale: float,
              name_suffix: str = "", init_model: str | None = None,
              dm_shaping: DeathmatchShapingConfig | None = None,
              dm_max_buttons_pressed: int = 2,
              dm_bots: int = 0,
              dm_bots_eval: int = 0,
              obs_shape: tuple[int, int] = IMAGE_SHAPE,
              policy_hidden_size: int = 256,
              policy_hidden_layers: int = 2):
    """Train PPO on one scenario and save the checkpoint."""
    print(f"[{scenario_name}] Starting training on {env_id} "
          f"({total_timesteps:,} steps, {n_envs} envs)")
    os.makedirs(out_dir, exist_ok=True)
    model_stem = f"{scenario_name}{name_suffix}"
    out_path = os.path.join(out_dir, f"{model_stem}.zip")
    meta_path = os.path.join(out_dir, f"{model_stem}.meta.json")

    def make_raw_env():
        if env_id.startswith("CFG::"):
            from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv

            cfg_name = env_id.split("::", 1)[1]
            if cfg_name == "deathmatch_compact":
                template_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "doom_dashboard",
                    "scenarios",
                    "deathmatch_compact.cfg",
                )
                runtime_dir = os.path.join(tempfile.gettempdir(), "doom_compact_runtime")
                os.makedirs(runtime_dir, exist_ok=True)
                cfg_path = os.path.join(runtime_dir, "deathmatch_compact.runtime.cfg")
                wad_src = os.path.join(vizdoom.scenarios_path, "deathmatch.wad")
                wad_dst = os.path.join(runtime_dir, "deathmatch.wad")
                if not os.path.exists(wad_dst):
                    shutil.copy2(wad_src, wad_dst)
                shutil.copy2(template_path, cfg_path)
            else:
                raise ValueError(f"Unknown custom cfg alias: {cfg_name}")
            return VizdoomEnv(
                config_file=cfg_path,
                frame_skip=frame_skip,
                max_buttons_pressed=int(dm_max_buttons_pressed),
            )
        make_kwargs = {"frame_skip": frame_skip}
        if scenario_name == "deathmatch":
            make_kwargs["max_buttons_pressed"] = int(dm_max_buttons_pressed)
        return gymnasium.make(env_id, **make_kwargs)

    def make_env():
        env = make_raw_env()
        if isinstance(env.action_space, gymnasium.spaces.Dict) and "binary" in env.action_space.spaces:
            env = BinaryOnlyActionWrapper(env)
        if scenario_name == "deathmatch" and dm_bots > 0:
            env = DeathmatchBotsWrapper(env, n_bots=dm_bots)
        if scenario_name == "deathmatch" and dm_shaping is not None:
            env = DeathmatchRewardShapingWrapper(env, dm_shaping)
        env = ObservationWrapper(env, shape=obs_shape)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * reward_scale)
        return env

    def make_eval_env():
        env = make_raw_env()
        if isinstance(env.action_space, gymnasium.spaces.Dict) and "binary" in env.action_space.spaces:
            env = BinaryOnlyActionWrapper(env)
        if scenario_name == "deathmatch" and dm_bots_eval > 0:
            env = DeathmatchBotsWrapper(env, n_bots=dm_bots_eval)
        if scenario_name == "deathmatch" and dm_shaping is not None:
            env = DeathmatchRewardShapingWrapper(env, dm_shaping)
        env = ObservationWrapper(env, shape=obs_shape)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * reward_scale)
        return env

    # Probe raw env once to persist action mapping metadata for robust inference.
    probe = make_raw_env()
    try:
        btn_names = [str(b) for b in probe.unwrapped.game.get_available_buttons()]
        btn_map = getattr(probe.unwrapped, "button_map", None)
        action_button_map = [np.asarray(x, dtype=np.int8).tolist() for x in btn_map] if btn_map else None
        action_space_n = None
        if hasattr(probe.action_space, "spaces") and "binary" in probe.action_space.spaces:
            bspace = probe.action_space.spaces["binary"]
            action_space_n = int(getattr(bspace, "n", 0)) if hasattr(bspace, "n") else None
    finally:
        probe.close()

    vec_env = make_vec_env(make_env, n_envs=n_envs)
    vec_env = VecTransposeImage(vec_env)
    eval_env = make_vec_env(make_eval_env, n_envs=1)
    eval_env = VecTransposeImage(eval_env)

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1000),
        save_path=os.path.join(out_dir, f"{model_stem}_ckpts"),
        name_prefix="ppo",
        verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(out_dir, f"{model_stem}_best"),
        log_path=os.path.join(out_dir, f"{model_stem}_eval"),
        eval_freq=max(25_000 // n_envs, 1000),
        deterministic=True,
        render=False,
        n_eval_episodes=8,
        verbose=1,
    )
    callbacks = CallbackList([ckpt_cb, eval_cb])

    if init_model:
        print(f"[{scenario_name}] Loading init model: {init_model}")
        try:
            model = PPO.load(init_model, env=vec_env, device="auto")
        except Exception as e:
            # Keep long-running sweeps robust when env/action-space changes between runs.
            print(f"[{scenario_name}] Warning: init model incompatible ({e}); training from scratch.")
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                n_steps=ppo_cfg["n_steps"],
                batch_size=ppo_cfg["batch_size"],
                n_epochs=ppo_cfg["n_epochs"],
                learning_rate=ppo_cfg["learning_rate"],
                clip_range=ppo_cfg["clip_range"],
                ent_coef=ppo_cfg["ent_coef"],
                target_kl=ppo_cfg["target_kl"],
                policy_kwargs={
                    "net_arch": {
                        "pi": [int(policy_hidden_size)] * int(policy_hidden_layers),
                        "vf": [int(policy_hidden_size)] * int(policy_hidden_layers),
                    }
                },
                verbose=1,
            )
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            learning_rate=ppo_cfg["learning_rate"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            target_kl=ppo_cfg["target_kl"],
            policy_kwargs={
                "net_arch": {
                    "pi": [int(policy_hidden_size)] * int(policy_hidden_layers),
                    "vf": [int(policy_hidden_size)] * int(policy_hidden_layers),
                }
            },
            verbose=1,
        )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    except ImportError:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(out_path)
    meta = {
        "scenario": scenario_name,
        "env_id": env_id,
        "frame_skip": int(frame_skip),
        "obs_shape": [int(obs_shape[0]), int(obs_shape[1])],
        "max_buttons_pressed": int(dm_max_buttons_pressed),
        "button_names": btn_names,
        "action_button_map": action_button_map,
        "action_space_n": action_space_n,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{scenario_name}] ✓ Saved to {out_path}")
    print(f"[{scenario_name}] ✓ Saved metadata to {meta_path}")
    vec_env.close()
    eval_env.close()


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
    parser.add_argument("--name-suffix", type=str, default="",
                        help="Optional suffix appended to saved model names, e.g. '_elim_v1'")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP,
                        help=f"VizDoom frame_skip (default: {FRAME_SKIP})")
    parser.add_argument("--reward-scale", type=float, default=REWARD_SCALE,
                        help=f"Reward scale multiplier (default: {REWARD_SCALE})")
    parser.add_argument("--n-steps", type=int, default=PPO_CFG["n_steps"])
    parser.add_argument("--batch-size", type=int, default=PPO_CFG["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=PPO_CFG["n_epochs"])
    parser.add_argument("--learning-rate", type=float, default=PPO_CFG["learning_rate"])
    parser.add_argument("--clip-range", type=float, default=PPO_CFG["clip_range"])
    parser.add_argument("--ent-coef", type=float, default=PPO_CFG["ent_coef"])
    parser.add_argument("--target-kl", type=float, default=PPO_CFG["target_kl"])
    parser.add_argument("--init-model", type=str, default=None,
                        help="Optional SB3 .zip to continue training from")
    parser.add_argument("--dm-frag-bonus", type=float, default=5.0)
    parser.add_argument("--dm-kill-bonus", type=float, default=2.0)
    parser.add_argument("--dm-hit-bonus", type=float, default=0.25)
    parser.add_argument("--dm-damage-bonus", type=float, default=0.01)
    parser.add_argument("--dm-hit-taken-penalty", type=float, default=0.15)
    parser.add_argument("--dm-damage-taken-penalty", type=float, default=0.01)
    parser.add_argument("--dm-death-penalty", type=float, default=3.0)
    parser.add_argument("--dm-living-penalty", type=float, default=0.001)
    parser.add_argument("--dm-max-buttons-pressed", type=int, default=2,
                        help="Deathmatch only: max simultaneous binary buttons (default: 2)")
    parser.add_argument("--dm-bots", type=int, default=0,
                        help="Deathmatch only: number of NPC bots to add each episode (default: 0)")
    parser.add_argument("--dm-bots-eval", type=int, default=0,
                        help="Deathmatch only: number of NPC bots in eval env (default: 0 for stability)")
    parser.add_argument("--obs-height", type=int, default=IMAGE_SHAPE[0],
                        help=f"Resized observation height before policy (default: {IMAGE_SHAPE[0]})")
    parser.add_argument("--obs-width", type=int, default=IMAGE_SHAPE[1],
                        help=f"Resized observation width before policy (default: {IMAGE_SHAPE[1]})")
    parser.add_argument("--policy-hidden-size", type=int, default=256,
                        help="PPO policy/value MLP hidden size after CNN features")
    parser.add_argument("--policy-hidden-layers", type=int, default=2,
                        help="Number of hidden layers in policy/value heads")
    args = parser.parse_args()

    selected = {k: SCENARIO_ENV_MAP[k] for k in args.scenarios}
    if args.init_model and len(selected) != 1:
        raise ValueError("--init-model currently supports exactly one scenario per run.")
    ppo_cfg = {
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl,
    }
    if ppo_cfg["batch_size"] % max(1, ppo_cfg["n_steps"]) != 0:
        print(
            f"Warning: batch_size ({ppo_cfg['batch_size']}) is not a multiple of n_steps ({ppo_cfg['n_steps']})."
        )
    print(f"Training {len(selected)} policies: {list(selected.keys())}")
    print(f"  timesteps/scenario={args.timesteps:,}  envs/scenario={args.envs}")
    print(f"  output: {args.out}\n")
    print(
        f"  ppo: n_steps={ppo_cfg['n_steps']} batch={ppo_cfg['batch_size']} epochs={ppo_cfg['n_epochs']} "
        f"lr={ppo_cfg['learning_rate']} clip={ppo_cfg['clip_range']} ent={ppo_cfg['ent_coef']} kl={ppo_cfg['target_kl']}"
    )
    print(
        f"  frame_skip={args.frame_skip} reward_scale={args.reward_scale}"
        f" obs={args.obs_height}x{args.obs_width}"
        f" policy_head={args.policy_hidden_layers}x{args.policy_hidden_size}"
        f" suffix='{args.name_suffix}'\n"
    )
    dm_shaping = DeathmatchShapingConfig(
        frag_bonus=args.dm_frag_bonus,
        kill_bonus=args.dm_kill_bonus,
        hit_bonus=args.dm_hit_bonus,
        damage_bonus=args.dm_damage_bonus,
        hit_taken_penalty=args.dm_hit_taken_penalty,
        damage_taken_penalty=args.dm_damage_taken_penalty,
        death_penalty=args.dm_death_penalty,
        living_penalty=args.dm_living_penalty,
    )
    if "deathmatch" in selected:
        print(
            "  deathmatch shaping:"
            f" frag={dm_shaping.frag_bonus} kill={dm_shaping.kill_bonus}"
            f" hit={dm_shaping.hit_bonus} dmg={dm_shaping.damage_bonus}"
            f" hit_taken={dm_shaping.hit_taken_penalty}"
            f" dmg_taken={dm_shaping.damage_taken_penalty}"
            f" death={dm_shaping.death_penalty}"
            f" living={dm_shaping.living_penalty}"
            f" max_buttons_pressed={args.dm_max_buttons_pressed}"
            f" bots_train={args.dm_bots}"
            f" bots_eval={args.dm_bots_eval}\n"
        )

    # Use 'spawn' context to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    procs = []
    for sc_name, env_id in selected.items():
        p = ctx.Process(
            target=train_one,
            args=(
                sc_name, env_id, args.out, args.timesteps, args.envs, ppo_cfg,
                args.frame_skip, args.reward_scale, args.name_suffix, args.init_model, dm_shaping,
                args.dm_max_buttons_pressed, args.dm_bots, args.dm_bots_eval,
                (args.obs_height, args.obs_width),
                args.policy_hidden_size, args.policy_hidden_layers,
            ),
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
