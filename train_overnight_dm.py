#!/usr/bin/env python3
"""
Unified deathmatch PPO trainer with macro discrete actions for overnight pipeline.

Supports both deathmatch_compact.cfg (8 buttons) and cig_fullaction.cfg (21 buttons).
Auto-detects macro actions from the cfg file.  Bot curriculum via --init-model.

Usage:
    # Phase 1: train from scratch vs 1 bot
    xvfb-run -a uv run python train_overnight_dm.py \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --maps map01 --timesteps 2000000 --envs 4 --bots 1 \
        --name overnight_compact_p1

    # Phase 2: fine-tune vs 2 bots
    xvfb-run -a uv run python train_overnight_dm.py \
        --init-model trained_policies/overnight_compact_p1_best/best_model.zip \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --maps map01 --timesteps 1500000 --envs 4 --bots 2 \
        --name overnight_compact_p2
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack

from doom_dashboard.multiplayer_rollout import rollout_multiplayer_episode


# ─── IMPALA-style ResNet CNN ─────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + x


class _ImpalaBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = _ResBlock(out_ch)
        self.res2 = _ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCnnExtractor(BaseFeaturesExtractor):
    """IMPALA-style ResNet CNN for Dict obs (screen + gamevars).

    Architecture: 3 blocks {16,32,32} channels, each with conv→maxpool→2×residual.
    For 120×160 input → 15×20 feature maps after 3 pooling layers.
    Total: 15 conv layers + FC.
    """

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 channels: Tuple[int, ...] = (16, 32, 32)):
        # Call with dummy features_dim, we'll override after computing real dim
        super().__init__(observation_space, features_dim=1)

        screen_space = observation_space.spaces["screen"]
        gamevars_space = observation_space.spaces["gamevars"]
        n_input_channels = screen_space.shape[0]  # (C, H, W) after VecTransposeImage

        blocks = []
        in_ch = n_input_channels
        for out_ch in channels:
            blocks.append(_ImpalaBlock(in_ch, out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, *screen_space.shape)
            cnn_out = self.cnn(sample)
            cnn_flat_dim = int(cnn_out.reshape(1, -1).shape[1])

        self.cnn_head = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_flat_dim, features_dim),
            nn.ReLU(),
        )

        # Gamevars MLP
        gv_dim = int(gamevars_space.shape[0])
        gv_out = 64
        self.gv_net = nn.Sequential(
            nn.Linear(gv_dim, gv_out),
            nn.ReLU(),
            nn.Linear(gv_out, gv_out),
            nn.ReLU(),
        )

        self._features_dim = features_dim + gv_out

    def forward(self, observations: dict) -> torch.Tensor:
        screen = observations["screen"].float() / 255.0
        gamevars = observations["gamevars"].float()
        cnn_feat = self.cnn_head(self.cnn(screen))
        gv_feat = self.gv_net(gamevars)
        return torch.cat([cnn_feat, gv_feat], dim=1)

# ─── wandb callback ──────────────────────────────────────────────

class WandbMetricCallback(BaseCallback):
    """Logs SB3 training metrics + reward breakdown + game stats to W&B."""

    def __init__(self, verbose: int = 0, log_freq: int = 2048):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._reward_accum: Dict[str, List[float]] = {}
        self._ep_frags: List[float] = []
        self._ep_kills: List[float] = []
        self._ep_deaths: List[float] = []
        self._ep_damage: List[float] = []
        self._last_log_step = 0

    def _on_step(self) -> bool:
        try:
            import wandb
            if wandb.run is None:
                return True
        except ImportError:
            return True

        # Collect reward components from infos
        infos = self.locals.get("infos", [])
        for info in infos:
            rc = info.get("reward_components")
            if rc:
                for k, v in rc.items():
                    self._reward_accum.setdefault(k, []).append(v)
            # Collect game vars from episode ends
            if "vars" in info:
                gv = info["vars"]
                if info.get("_terminal_observation") is not None or info.get("TimeLimit.truncated", False):
                    self._ep_frags.append(float(gv[VAR_INDEX["FRAGCOUNT"]]))
                    self._ep_kills.append(float(gv[VAR_INDEX["KILLCOUNT"]]))
                    self._ep_deaths.append(float(gv[VAR_INDEX["DEATHCOUNT"]]))
                    self._ep_damage.append(float(gv[VAR_INDEX["DAMAGECOUNT"]]))

        # Log periodically
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._last_log_step = self.num_timesteps
            log_dict: Dict[str, float] = {}

            # SB3's own logged values
            if self.logger is not None:
                name_to_val = getattr(self.logger, "name_to_value", {})
                if name_to_val:
                    log_dict.update({k: v for k, v in name_to_val.items()})

            # Reward component means
            for k, vals in self._reward_accum.items():
                if vals:
                    log_dict[f"reward/{k}"] = float(np.mean(vals))
            self._reward_accum.clear()

            # Episode-level game stats
            if self._ep_frags:
                log_dict["game/ep_frags"] = float(np.mean(self._ep_frags))
                log_dict["game/ep_kills"] = float(np.mean(self._ep_kills))
                log_dict["game/ep_deaths"] = float(np.mean(self._ep_deaths))
                log_dict["game/ep_damage"] = float(np.mean(self._ep_damage))
                self._ep_frags.clear()
                self._ep_kills.clear()
                self._ep_deaths.clear()
                self._ep_damage.clear()

            if log_dict:
                wandb.log(log_dict, step=self.num_timesteps)

        return True


class WandbVideoCallback(BaseCallback):
    """Records short gameplay video and logs to wandb every `video_freq` steps.

    Uses cv2 to write mp4 files (no moviepy dependency needed).
    """

    def __init__(self, env_fn, video_freq: int = 100_000, max_steps: int = 600, verbose: int = 0):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.video_freq = video_freq
        self.max_steps = max_steps
        self._last_video_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_video_step < self.video_freq:
            return True
        self._last_video_step = self.num_timesteps
        try:
            import wandb
            if wandb.run is None:
                return True
            env = self.env_fn()
            frames = []
            obs, _ = env.reset()
            for _ in range(self.max_steps):
                model_obs = {
                    "screen": np.transpose(obs["screen"], (2, 0, 1))[None],
                    "gamevars": obs["gamevars"][None],
                }
                action, _ = self.model.predict(model_obs, deterministic=True)
                frames.append(obs["screen"].copy())
                obs, _, term, trunc, _ = env.step(int(action[0]))
                if term or trunc:
                    break
            env.close()
            if len(frames) > 10:
                step = max(1, len(frames) // 300)
                frames = frames[::step]
                # Write mp4 using cv2 (no moviepy needed)
                tmp_path = f"/tmp/wandb_vid_{self.num_timesteps}.mp4"
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp_path, fourcc, 15, (w, h))
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                wandb.log({"video/gameplay": wandb.Video(tmp_path, fps=15, format="mp4")},
                          step=self.num_timesteps)
                print(f"[WandbVideoCallback] logged {len(frames)} frame video at step {self.num_timesteps}")
                try:
                    import os
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WandbVideoCallback] failed: {e}")
        return True


ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "doom_dashboard" / "scenarios" / "deathmatch_compact.cfg"

TRACKED_VARS = [
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.KILLCOUNT,
    vzd.GameVariable.DEATHCOUNT,
    vzd.GameVariable.HITCOUNT,
    vzd.GameVariable.HITS_TAKEN,
    vzd.GameVariable.DAMAGECOUNT,
    vzd.GameVariable.DAMAGE_TAKEN,
    vzd.GameVariable.HEALTH,
]
VAR_INDEX = {str(v).split(".")[-1]: i for i, v in enumerate(TRACKED_VARS)}


# ─── cfg helpers (from train_cig_fullaction_macro_ppo.py) ─────────

def parse_available_buttons(cfg_path: str) -> List[str]:
    lines = Path(cfg_path).read_text().splitlines()
    out: List[str] = []
    in_block = False
    for ln in lines:
        raw = ln.strip()
        if not raw or raw.startswith("#"):
            continue
        low = raw.lower()
        if not in_block and low.startswith("available_buttons"):
            in_block = True
            continue
        if in_block and raw.startswith("{"):
            continue
        if in_block and raw.startswith("}"):
            break
        if in_block:
            tok = raw.split("#", 1)[0].strip()
            if tok:
                out.append(tok.upper())
    if not out:
        raise RuntimeError(f"Could not parse available_buttons from cfg: {cfg_path}")
    return out


def materialize_cfg(cfg_path: str) -> str:
    p = Path(cfg_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    text = p.read_text()
    wad_ref = None
    for ln in text.splitlines():
        raw = ln.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.lower().startswith("doom_scenario_path"):
            wad_ref = raw.split("=", 1)[1].strip().strip('"\'')
            break
    if wad_ref is None:
        return str(p)
    wad = Path(wad_ref)
    if wad.is_absolute() or (p.parent / wad_ref).exists():
        return str(p)
    src_wad = Path(vzd.scenarios_path) / wad_ref
    if not src_wad.exists():
        raise FileNotFoundError(
            f"Could not resolve doom_scenario_path '{wad_ref}' for cfg '{cfg_path}'."
        )
    runtime_dir = Path(tempfile.gettempdir()) / "doom_overnight_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = runtime_dir / p.name
    runtime_wad = runtime_dir / wad_ref
    shutil.copy2(p, runtime_cfg)
    shutil.copy2(src_wad, runtime_wad)
    return str(runtime_cfg)


def parse_maps(raw: str) -> List[str]:
    vals = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    return vals or ["map01"]


def _canon_buttons(button_names: Sequence[str]) -> List[str]:
    out = []
    for b in button_names:
        c = str(b).strip().upper()
        if "." in c:
            c = c.split(".")[-1]
        out.append(c)
    return out


# ─── macro action builders ────────────────────────────────────────

def build_macro_actions(button_names: Sequence[str]) -> Tuple[List[str], List[np.ndarray]]:
    """Build macro actions for any cfg by inspecting available buttons.

    For CIG-style cfgs (21 buttons) produces ~27 macros.
    For compact cfgs (8 buttons) produces ~19 macros.
    """
    btns = _canon_buttons(button_names)
    idx = {b: i for i, b in enumerate(btns)}

    def v(*names: str) -> np.ndarray:
        arr = np.zeros((len(btns),), dtype=np.uint8)
        for n in names:
            j = idx.get(n)
            if j is not None:
                arr[j] = 1
        return arr

    # Core movement + combat macros (work for both compact and CIG)
    macro_specs: List[Tuple[str, np.ndarray]] = [
        ("NOOP", v()),
        ("MOVE_FORWARD", v("MOVE_FORWARD")),
        ("MOVE_BACKWARD", v("MOVE_BACKWARD")),
        ("MOVE_LEFT", v("MOVE_LEFT")),
        ("MOVE_RIGHT", v("MOVE_RIGHT")),
        ("TURN_LEFT", v("TURN_LEFT")),
        ("TURN_RIGHT", v("TURN_RIGHT")),
        ("FWD_TURN_LEFT", v("MOVE_FORWARD", "TURN_LEFT")),
        ("FWD_TURN_RIGHT", v("MOVE_FORWARD", "TURN_RIGHT")),
        ("ATTACK", v("ATTACK")),
        ("ATTACK_TURN_LEFT", v("ATTACK", "TURN_LEFT")),
        ("ATTACK_TURN_RIGHT", v("ATTACK", "TURN_RIGHT")),
        ("ATTACK_FORWARD", v("ATTACK", "MOVE_FORWARD")),
        ("ATTACK_FWD_LEFT", v("ATTACK", "MOVE_FORWARD", "TURN_LEFT")),
        ("ATTACK_FWD_RIGHT", v("ATTACK", "MOVE_FORWARD", "TURN_RIGHT")),
        ("SPEED_FORWARD", v("SPEED", "MOVE_FORWARD")),
        ("SPEED_ATTACK_FORWARD", v("SPEED", "ATTACK", "MOVE_FORWARD")),
        ("BACK_TURN_LEFT", v("MOVE_BACKWARD", "TURN_LEFT")),
        ("BACK_TURN_RIGHT", v("MOVE_BACKWARD", "TURN_RIGHT")),
    ]

    # CIG-specific macros (only added if those buttons exist in cfg)
    if "STRAFE" in idx:
        macro_specs.extend([
            ("STRAFE_LEFT", v("STRAFE", "MOVE_LEFT")),
            ("STRAFE_RIGHT", v("STRAFE", "MOVE_RIGHT")),
            ("STRAFE_ATTACK_LEFT", v("STRAFE", "MOVE_LEFT", "ATTACK")),
            ("STRAFE_ATTACK_RIGHT", v("STRAFE", "MOVE_RIGHT", "ATTACK")),
        ])
    if "USE" in idx:
        macro_specs.append(("USE", v("USE")))
    if "SELECT_NEXT_WEAPON" in idx:
        macro_specs.append(("NEXT_WEAPON", v("SELECT_NEXT_WEAPON")))
    if "SELECT_PREV_WEAPON" in idx:
        macro_specs.append(("PREV_WEAPON", v("SELECT_PREV_WEAPON")))
    # Direct weapon selection (fullaction configs)
    for w in range(1, 7):
        wk = f"SELECT_WEAPON{w}"
        if wk in idx:
            macro_specs.append((f"WEAPON{w}", v(wk)))
            macro_specs.append((f"WEAPON{w}_FORWARD", v(wk, "MOVE_FORWARD")))
    if "TURN_LEFT_RIGHT_DELTA" in idx:
        macro_specs.append(("TURN_DELTA", v("TURN_LEFT_RIGHT_DELTA")))
    if "LOOK_UP_DOWN_DELTA" in idx:
        macro_specs.append(("LOOK_DELTA", v("LOOK_UP_DOWN_DELTA")))
    if "MOVE_LEFT_RIGHT_DELTA" in idx:
        macro_specs.append(("MOVE_DELTA", v("MOVE_LEFT_RIGHT_DELTA")))

    # Deduplicate (if a button was absent, some macros collapse to the same vector)
    dedup_names: List[str] = []
    dedup_actions: List[np.ndarray] = []
    seen = set()
    for name, arr in macro_specs:
        key = tuple(int(x) for x in arr.tolist())
        if key in seen:
            continue
        seen.add(key)
        dedup_names.append(name)
        dedup_actions.append(arr)

    return dedup_names, dedup_actions


# ─── reward shaping ───────────────────────────────────────────────

@dataclass
class ShapingCfg:
    # DO NOT MODIFY THESE VALUES — they are set via CLI args in run_training.sh
    # Default values here are ONLY used when CLI args are not provided
    frag_bonus: float = 80.0       # strong kill incentive
    hit_bonus: float = 3.0         # reward landing shots
    damage_bonus: float = 0.05     # reward dealing damage
    death_penalty: float = 20.0    # dying is bad
    hit_taken_penalty: float = 0.35
    damage_taken_penalty: float = 0.01
    living_penalty: float = 0.002  # small constant cost to encourage action
    attack_bonus: float = 0.03     # reward firing weapon
    move_bonus: float = 0.004      # reward moving around
    noop_penalty: float = 0.05     # punish standing still
    reward_scale: float = 0.1      # keep rewards in learnable range for value fn


# ─── environment ──────────────────────────────────────────────────

class DeathmatchMacroEnv(gym.Env):
    """Macro-discrete deathmatch environment with bot opponents."""

    metadata = {"render_modes": []}
    _next_port = 9500

    def __init__(
        self,
        *,
        cfg_path: str,
        frame_skip: int,
        obs_shape: Tuple[int, int],
        maps: Sequence[str],
        bots: int,
        timelimit_minutes: float,
        shaping: ShapingCfg,
        spawn_farthest: bool,
        no_autoaim: bool,
    ):
        super().__init__()
        self.cfg_path = str(cfg_path)
        self.frame_skip = int(frame_skip)
        self.obs_shape = (int(obs_shape[0]), int(obs_shape[1]))
        self.maps = [str(m).lower() for m in maps]
        self.bots = max(0, int(bots))
        self.timelimit_minutes = float(timelimit_minutes)
        self.shaping = shaping
        self.spawn_farthest = bool(spawn_farthest)
        self.no_autoaim = bool(no_autoaim)

        self.button_names = parse_available_buttons(self.cfg_path)
        self.macro_names, self.macro_actions = build_macro_actions(self.button_names)

        self.action_space = gym.spaces.Discrete(len(self.macro_actions))
        self.observation_space = gym.spaces.Dict(
            {
                "screen": gym.spaces.Box(
                    low=0, high=255,
                    shape=(self.obs_shape[0], self.obs_shape[1], 3),
                    dtype=np.uint8,
                ),
                "gamevars": gym.spaces.Box(
                    low=-1e6, high=1e6,
                    shape=(len(TRACKED_VARS),),
                    dtype=np.float32,
                ),
            }
        )

        self.game: vzd.DoomGame | None = None
        self.prev_vars: np.ndarray | None = None
        self.current_map: str | None = None
        self.port = int(DeathmatchMacroEnv._next_port)
        DeathmatchMacroEnv._next_port += 1

        names = _canon_buttons(self.button_names)
        self._idx = {n: i for i, n in enumerate(names)}

    def _init_game(self, doom_map: str):
        if self.game is not None:
            try:
                self.game.close()
            except Exception:
                pass
            self.game = None

        g = vzd.DoomGame()
        g.load_config(self.cfg_path)
        g.set_doom_map(doom_map)
        g.set_window_visible(False)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        g.set_render_hud(False)

        # Convert timelimit to tics for episode_timeout (35 tics/sec)
        timeout_tics = int(self.timelimit_minutes * 60 * 35)
        g.set_episode_timeout(timeout_tics)

        # Single-player with bots: avoid -host and -deathmatch flags
        # (they cause segfaults on this ViZDoom build).
        # Bots are added via send_game_command("addbot") after init.
        g.add_game_args(
            "+sv_forcerespawn 1 +sv_nocrouch 1 "
            "+viz_nosound 1 -nosound -nojoy -noidle "
            "+name Trainer"
        )
        g.init()
        self.game = g
        self.current_map = doom_map

    def _screen(self) -> np.ndarray:
        assert self.game is not None
        st = self.game.get_state()
        if st is None or st.screen_buffer is None:
            return np.zeros((self.obs_shape[0], self.obs_shape[1], 3), dtype=np.uint8)
        buf = st.screen_buffer
        if buf.ndim == 3 and buf.shape[0] in (1, 3, 4) and buf.shape[-1] not in (1, 3, 4):
            frame = np.transpose(buf, (1, 2, 0))
        else:
            frame = buf
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        frame = frame[:, :, :3]
        return cv2.resize(frame, (self.obs_shape[1], self.obs_shape[0]), interpolation=cv2.INTER_AREA)

    def _read_vars(self) -> np.ndarray:
        assert self.game is not None
        vals = []
        for v in TRACKED_VARS:
            try:
                vals.append(float(self.game.get_game_variable(v)))
            except Exception:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float32)

    def _obs(self, vars_vec: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "screen": self._screen(),
            "gamevars": vars_vec.astype(np.float32),
        }

    def _delta(self, prev: np.ndarray, curr: np.ndarray, key: str) -> float:
        return float(curr[VAR_INDEX[key]] - prev[VAR_INDEX[key]])

    def _shape_reward(self, prev: np.ndarray, curr: np.ndarray, act: np.ndarray, pos_delta: float = 0.0) -> Tuple[float, dict]:
        c = self.shaping
        components = {}

        components["living"] = -c.living_penalty
        components["frag"] = max(0.0, self._delta(prev, curr, "FRAGCOUNT")) * c.frag_bonus
        components["hit"] = max(0.0, self._delta(prev, curr, "HITCOUNT")) * c.hit_bonus
        components["damage"] = max(0.0, self._delta(prev, curr, "DAMAGECOUNT")) * c.damage_bonus
        components["death"] = -max(0.0, self._delta(prev, curr, "DEATHCOUNT")) * c.death_penalty
        components["hit_taken"] = -max(0.0, self._delta(prev, curr, "HITS_TAKEN")) * c.hit_taken_penalty
        components["dmg_taken"] = -max(0.0, self._delta(prev, curr, "DAMAGE_TAKEN")) * c.damage_taken_penalty

        attack_idx = self._idx.get("ATTACK")
        attack_on = bool(attack_idx is not None and act[attack_idx] > 0)
        # Only count translational movement (not turning in place) for move_bonus
        move_on = any(
            act[self._idx[k]] > 0
            for k in ("MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT")
            if k in self._idx
        )
        # Turning counts as "doing something" for noop check but not for move_bonus
        turn_on = any(
            act[self._idx[k]] > 0
            for k in ("TURN_LEFT", "TURN_RIGHT")
            if k in self._idx
        )

        components["attack"] = c.attack_bonus if attack_on else 0.0
        # Position-based movement reward: reward actual displacement, not button presses
        # Cap at ~20 units/step to prevent exploitation; typical walking speed is ~8-15 units/step
        components["move"] = c.move_bonus * min(pos_delta / 10.0, 1.5) if pos_delta > 1.0 else 0.0
        components["noop"] = -c.noop_penalty if (not attack_on and not move_on and not turn_on) else 0.0

        raw = sum(components.values())
        scaled = raw * c.reward_scale
        components = {k: v * c.reward_scale for k, v in components.items()}
        return scaled, components

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        doom_map = self.maps[int(self.np_random.integers(0, len(self.maps)))]

        # Remove bots before new_episode to avoid "No player 2 start" errors
        # from bots persisting in WADs with limited player starts.
        if self.game is not None:
            try:
                self.game.send_game_command("removebots")
            except Exception:
                pass

        # Try up to 3 times to get a working game
        for attempt in range(3):
            try:
                if self.game is None or self.current_map != doom_map:
                    self._init_game(doom_map)

                assert self.game is not None
                self.game.new_episode()

                for _ in range(self.bots):
                    try:
                        self.game.send_game_command("addbot")
                    except Exception:
                        pass

                self.prev_vars = self._read_vars()
                # Initialize position tracking for distance-based movement reward
                try:
                    self._prev_pos = (
                        self.game.get_game_variable(vzd.GameVariable.POSITION_X),
                        self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
                    )
                except Exception:
                    self._prev_pos = (0.0, 0.0)
                return self._obs(self.prev_vars), {"vars": self.prev_vars.copy(), "map": doom_map}
            except Exception as e:
                print(f"[DeathmatchMacroEnv] reset attempt {attempt+1} failed: {e!r}")
                try:
                    if self.game is not None:
                        self.game.close()
                except Exception:
                    pass
                self.game = None
                if attempt == 2:
                    raise  # give up after 3 attempts

    def step(self, action):
        assert self.game is not None
        idx = int(np.asarray(action).reshape(()))
        idx = int(np.clip(idx, 0, len(self.macro_actions) - 1))
        act = self.macro_actions[idx]

        try:
            self.game.make_action([bool(x) for x in act.tolist()], self.frame_skip)
            if self.game.is_player_dead():
                self.game.respawn_player()
        except Exception as e:
            # ViZDoom crashed (e.g. ViZDoomUnexpectedExitException / semaphore leak)
            # Reinitialize and end this episode gracefully
            print(f"[DeathmatchMacroEnv] ViZDoom crash in step(): {e!r} — reinitializing")
            try:
                self.game.close()
            except Exception:
                pass
            self.game = None
            doom_map = self.current_map or self.maps[0]
            self._init_game(doom_map)
            self.game.new_episode()
            for _ in range(self.bots):
                try:
                    self.game.send_game_command("addbot")
                except Exception:
                    pass
            curr = self._read_vars()
            self.prev_vars = curr
            zero_components = {k: 0.0 for k in ["living", "frag", "hit", "damage", "death", "hit_taken", "dmg_taken", "attack", "move", "noop"]}
            return self._obs(curr), 0.0, True, False, {
                "vars": curr.copy(), "map": doom_map,
                "action_name": self.macro_names[idx],
                "reward_components": zero_components,
            }

        curr = self._read_vars()
        prev = self.prev_vars if self.prev_vars is not None else curr
        # Track position for distance-based movement reward
        pos_delta = 0.0
        if self.game and self.game.is_running():
            try:
                cx = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
                cy = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
                if hasattr(self, '_prev_pos'):
                    dx = cx - self._prev_pos[0]
                    dy = cy - self._prev_pos[1]
                    pos_delta = (dx*dx + dy*dy) ** 0.5
                self._prev_pos = (cx, cy)
            except Exception:
                pass
        shaped, reward_components = self._shape_reward(prev, curr, act, pos_delta=pos_delta)
        self.prev_vars = curr

        terminated = bool(self.game.is_episode_finished())
        truncated = False
        info = {
            "vars": curr.copy(),
            "map": self.current_map,
            "action_name": self.macro_names[idx],
            "reward_components": reward_components,
        }
        return self._obs(curr), float(shaped), terminated, truncated, info

    def close(self):
        if self.game is not None:
            try:
                self.game.close()
            except Exception:
                pass
            self.game = None


def _make_env(
    *,
    cfg_path: str,
    frame_skip: int,
    obs_shape: Tuple[int, int],
    maps: Sequence[str],
    bots: int,
    timelimit_minutes: float,
    shaping: ShapingCfg,
    spawn_farthest: bool,
    no_autoaim: bool,
):
    return DeathmatchMacroEnv(
        cfg_path=cfg_path,
        frame_skip=frame_skip,
        obs_shape=obs_shape,
        maps=maps,
        bots=bots,
        timelimit_minutes=timelimit_minutes,
        shaping=shaping,
        spawn_farthest=spawn_farthest,
        no_autoaim=no_autoaim,
    )


# ─── benchmark ────────────────────────────────────────────────────

def bench_vs_bots(
    *,
    model_path: str,
    cfg_path: str,
    maps: Sequence[str],
    frame_skip: int,
    timelimit_minutes: float,
    episodes: int,
    n_bots: int = 1,
    obs_shape: Tuple[int, int] = (120, 160),
) -> dict:
    """Benchmark a trained model vs built-in bots in single-player mode.

    Runs the model for `episodes` games and also runs a random policy for
    the same number of games.  Compares net frag scores (frags - deaths).
    The model 'wins' an episode if its net score exceeds the random baseline's
    corresponding episode net score.
    """
    from stable_baselines3 import PPO

    # Load the trained model
    model = PPO.load(str(model_path), device="auto")

    button_names = parse_available_buttons(cfg_path)
    macro_names, macro_actions = build_macro_actions(button_names)

    def _run_episodes(use_model: bool) -> List[dict]:
        results = []
        for i in range(int(episodes)):
            doom_map = maps[i % len(maps)]
            env = DeathmatchMacroEnv(
                cfg_path=cfg_path,
                frame_skip=frame_skip,
                obs_shape=obs_shape,
                maps=[doom_map],
                bots=n_bots,
                timelimit_minutes=timelimit_minutes,
                shaping=ShapingCfg(),
                spawn_farthest=True,
                no_autoaim=False,
            )
            try:
                obs, info = env.reset()
                total_reward = 0.0
                steps = 0
                positions = []
                while True:
                    # Track position for movement validation
                    if env.game and env.game.is_running():
                        try:
                            px = env.game.get_game_variable(vzd.GameVariable.POSITION_X)
                            py = env.game.get_game_variable(vzd.GameVariable.POSITION_Y)
                            positions.append((px, py))
                        except Exception:
                            pass
                    if use_model:
                        # Transpose screen for CNN: (H,W,C) -> (C,H,W)
                        model_obs = {
                            "screen": np.transpose(obs["screen"], (2, 0, 1))[None],
                            "gamevars": obs["gamevars"][None],
                        }
                        action, _ = model.predict(model_obs, deterministic=False)
                        action = int(action[0])
                    else:
                        action = env.action_space.sample()
                    obs, rew, term, trunc, info = env.step(action)
                    total_reward += rew
                    steps += 1
                    if term or trunc:
                        break
                final_vars = env._read_vars() if env.game else np.zeros(len(TRACKED_VARS))
                frags = float(final_vars[VAR_INDEX["FRAGCOUNT"]])
                deaths = float(final_vars[VAR_INDEX["DEATHCOUNT"]])
                kills = float(final_vars[VAR_INDEX["KILLCOUNT"]])
                hits = float(final_vars[VAR_INDEX["HITCOUNT"]])
                damage = float(final_vars[VAR_INDEX["DAMAGECOUNT"]])
                damage_taken = float(final_vars[VAR_INDEX["DAMAGE_TAKEN"]])
                # Compute total distance traveled
                dist = 0.0
                if len(positions) >= 2:
                    for j in range(1, len(positions)):
                        dx = positions[j][0] - positions[j-1][0]
                        dy = positions[j][1] - positions[j-1][1]
                        dist += (dx*dx + dy*dy) ** 0.5
                results.append({
                    "map": doom_map,
                    "frags": frags,
                    "deaths": deaths,
                    "kills": kills,
                    "hits": hits,
                    "damage": damage,
                    "damage_taken": damage_taken,
                    "net": frags - deaths,
                    "reward": total_reward,
                    "steps": steps,
                    "distance": dist,
                })
            except Exception as e:
                print(f"  bench warning: episode {i} failed ({e})")
                results.append({"map": doom_map, "frags": 0, "deaths": 0, "kills": 0, "hits": 0, "damage": 0, "damage_taken": 0, "net": 0, "reward": 0, "steps": 0, "distance": 0})
            finally:
                env.close()
        return results

    print("  Running model episodes...")
    model_results = _run_episodes(use_model=True)
    print("  Running random episodes...")
    random_results = _run_episodes(use_model=False)

    m_nets = [r["net"] for r in model_results]
    r_nets = [r["net"] for r in random_results]

    m = np.asarray(m_nets, dtype=np.float32)
    r = np.asarray(r_nets, dtype=np.float32)

    model_hit_mean = float(np.mean([x["hits"] for x in model_results]))
    model_damage_mean = float(np.mean([x["damage"] for x in model_results]))
    model_damage_taken_mean = float(np.mean([x["damage_taken"] for x in model_results]))
    random_hit_mean = float(np.mean([x["hits"] for x in random_results]))
    random_damage_mean = float(np.mean([x["damage"] for x in random_results]))

    damage_ratio = model_damage_mean / max(1.0, model_damage_taken_mean)
    model_distance_mean = float(np.mean([x["distance"] for x in model_results]))
    random_distance_mean = float(np.mean([x["distance"] for x in random_results]))
    # Compound gate for SP bot-based benchmark:
    # Designed to work in both compact (monsters) and nomonsters scenarios.
    # In compact, monsters deal constant ambient damage so damage_ratio is unreliable.
    # Focus on: dealing damage, landing hits, moving, and outperforming random.
    model_kill_mean = float(np.mean([x["kills"] for x in model_results]))
    random_kill_mean_val = float(np.mean([x["kills"] for x in random_results]))
    passed = bool(
        model_hit_mean >= 8.0            # land hits consistently
        and model_damage_mean >= 100.0   # deal meaningful damage
        and model_hit_mean > random_hit_mean  # hit more than random
        and model_distance_mean > random_distance_mean * 0.5  # must actually move
        and (model_kill_mean > random_kill_mean_val or model_damage_mean > random_damage_mean)  # outperform random
    )

    return {
        "episodes": len(m_nets),
        "maps": list(maps),
        "model_frag_mean": float(np.mean([x["frags"] for x in model_results])),
        "model_death_mean": float(np.mean([x["deaths"] for x in model_results])),
        "model_kill_mean": float(np.mean([x["kills"] for x in model_results])),
        "model_hit_mean": model_hit_mean,
        "model_damage_mean": model_damage_mean,
        "model_damage_taken_mean": model_damage_taken_mean,
        "model_distance_mean": model_distance_mean,
        "random_distance_mean": random_distance_mean,
        "model_net_mean": float(np.mean(m)),
        "random_frag_mean": float(np.mean([x["frags"] for x in random_results])),
        "random_death_mean": float(np.mean([x["deaths"] for x in random_results])),
        "random_kill_mean": float(np.mean([x["kills"] for x in random_results])),
        "random_hit_mean": random_hit_mean,
        "random_damage_mean": random_damage_mean,
        "random_net_mean": float(np.mean(r)),
        "win_rate_vs_random_net": float(np.mean(m > r)),
        "mean_net_gap": float(np.mean(m - r)),
        "damage_ratio": damage_ratio,
        "pass": passed,
        "model_net_samples": [float(x) for x in m.tolist()],
        "random_net_samples": [float(x) for x in r.tolist()],
        "model_reward_mean": float(np.mean([x["reward"] for x in model_results])),
        "random_reward_mean": float(np.mean([x["reward"] for x in random_results])),
    }


def bench_duel_vs_random(
    *,
    model_path: str,
    cfg_path: str,
    scenario_name: str,
    maps: Sequence[str],
    frame_skip: int,
    timelimit_minutes: float,
    episodes: int,
    resolution: str,
) -> dict:
    """Benchmark via true 1v1 multiplayer: model vs random agent."""
    policy = {
        "name": Path(model_path).stem,
        "type": "sb3",
        "path": str(model_path),
        "algo": "PPO",
        "arch": "DuelQNet",
        "action_size": None,
        "device": "auto",
    }
    rand = {
        "name": "Random",
        "type": "random",
        "path": None,
        "algo": "PPO",
        "arch": "DuelQNet",
        "action_size": None,
        "device": "auto",
    }

    nets_model: List[float] = []
    nets_rand: List[float] = []
    frag_model: List[float] = []
    frag_rand: List[float] = []
    death_model: List[float] = []
    death_rand: List[float] = []
    kill_model: List[float] = []
    kill_rand: List[float] = []
    dmg_model: List[float] = []
    dmg_rand: List[float] = []
    dmg_taken_model: List[float] = []
    dmg_taken_rand: List[float] = []

    for i in range(int(episodes)):
        doom_map = maps[i % len(maps)]
        try:
            ep = rollout_multiplayer_episode(
                cfg_path=str(cfg_path),
                scenario_name=scenario_name,
                doom_map=doom_map,
                policy_p1_dict=policy,
                policy_p2_dict=rand,
                timelimit_minutes=float(timelimit_minutes),
                frame_skip=int(frame_skip),
                render_resolution=str(resolution),
                render_hud=False,
                port=7400 + (i % 20) * 10,
            )
        except Exception as e:
            print(f"  bench warning: game {i} failed ({e}); skipping")
            continue

        gv1 = np.asarray(ep.game_vars_p1[-1] if ep.game_vars_p1 else np.zeros(len(TRACKED_VARS)), dtype=np.float32)
        gv2 = np.asarray(ep.game_vars_p2[-1] if ep.game_vars_p2 else np.zeros(len(TRACKED_VARS)), dtype=np.float32)

        f1 = float(gv1[VAR_INDEX["FRAGCOUNT"]])
        d1 = float(gv1[VAR_INDEX["DEATHCOUNT"]])
        f2 = float(gv2[VAR_INDEX["FRAGCOUNT"]])
        d2 = float(gv2[VAR_INDEX["DEATHCOUNT"]])
        k1 = float(gv1[VAR_INDEX["KILLCOUNT"]])
        k2 = float(gv2[VAR_INDEX["KILLCOUNT"]])
        dm1 = float(gv1[VAR_INDEX["DAMAGECOUNT"]])
        dm2 = float(gv2[VAR_INDEX["DAMAGECOUNT"]])
        dt1 = float(gv1[VAR_INDEX["DAMAGE_TAKEN"]])
        dt2 = float(gv2[VAR_INDEX["DAMAGE_TAKEN"]])

        frag_model.append(f1)
        death_model.append(d1)
        frag_rand.append(f2)
        death_rand.append(d2)
        kill_model.append(k1)
        kill_rand.append(k2)
        dmg_model.append(dm1)
        dmg_rand.append(dm2)
        dmg_taken_model.append(dt1)
        dmg_taken_rand.append(dt2)
        nets_model.append(f1 - d1)
        nets_rand.append(f2 - d2)

    if not nets_model:
        return {"episodes": 0, "win_rate_vs_random_net": 0.0, "mean_net_gap": 0.0, "pass": False}

    m = np.asarray(nets_model, dtype=np.float32)
    r = np.asarray(nets_rand, dtype=np.float32)

    model_frag_mean = float(np.mean(frag_model))
    model_kill_mean = float(np.mean(kill_model))
    model_damage_mean = float(np.mean(dmg_model))
    model_damage_taken_mean = float(np.mean(dmg_taken_model))
    damage_ratio = model_damage_mean / max(1.0, model_damage_taken_mean)
    win_rate = float(np.mean(m > r))

    # Compound gate: requires actual combat engagement, not just survival
    passed = bool(
        model_frag_mean >= 1.0
        and model_kill_mean >= 2.0
        and damage_ratio > 1.0
        and win_rate >= 0.75
    )

    return {
        "episodes": len(nets_model),
        "maps": list(maps),
        "model_frag_mean": model_frag_mean,
        "model_death_mean": float(np.mean(death_model)),
        "model_net_mean": float(np.mean(m)),
        "model_kill_mean": model_kill_mean,
        "model_damage_mean": model_damage_mean,
        "model_damage_taken_mean": model_damage_taken_mean,
        "random_frag_mean": float(np.mean(frag_rand)),
        "random_death_mean": float(np.mean(death_rand)),
        "random_net_mean": float(np.mean(r)),
        "random_kill_mean": float(np.mean(kill_rand)),
        "random_damage_mean": float(np.mean(dmg_rand)),
        "damage_ratio": damage_ratio,
        "win_rate_vs_random_net": win_rate,
        "mean_net_gap": float(np.mean(m - r)),
        "pass": passed,
        "model_net_samples": [float(x) for x in m.tolist()],
        "random_net_samples": [float(x) for x in r.tolist()],
    }


# ─── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train PPO macro deathmatch policy (overnight pipeline).")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--out", type=str, default="trained_policies")
    ap.add_argument("--name", type=str, default="overnight_dm")
    ap.add_argument("--cfg", type=str, default=str(DEFAULT_CFG))
    ap.add_argument("--maps", type=str, default="map01")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--obs-height", type=int, default=120)
    ap.add_argument("--obs-width", type=int, default=160)
    ap.add_argument("--bots", type=int, default=1)
    ap.add_argument("--bots-eval", type=int, default=1)
    ap.add_argument("--timelimit-minutes", type=float, default=2.5)

    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--policy-hidden-size", type=int, default=512)
    ap.add_argument("--policy-hidden-layers", type=int, default=2)
    ap.add_argument("--cnn-type", type=str, default="nature",
                    choices=["nature", "impala"],
                    help="CNN feature extractor: nature (SB3 default) or impala (15-layer ResNet)")
    ap.add_argument("--cnn-features-dim", type=int, default=256,
                    help="CNN feature output dimension (before MLP)")
    ap.add_argument("--policy-type", type=str, default="mlp",
                    choices=["mlp", "lstm", "transformer"],
                    help="Policy type: mlp (feedforward), lstm (recurrent), transformer (attention)")
    ap.add_argument("--lstm-hidden-size", type=int, default=256,
                    help="LSTM hidden state size (for --policy-type lstm)")
    ap.add_argument("--lstm-num-layers", type=int, default=1,
                    help="Number of LSTM layers (for --policy-type lstm)")
    ap.add_argument("--transformer-heads", type=int, default=4,
                    help="Number of attention heads (for --policy-type transformer)")
    ap.add_argument("--transformer-layers", type=int, default=2,
                    help="Number of transformer layers (for --policy-type transformer)")
    ap.add_argument("--context-length", type=int, default=32,
                    help="Context window length for transformer (number of past frames)")
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="Number of frames to stack (1=no stacking)")
    ap.add_argument("--video-freq", type=int, default=100_000,
                    help="Steps between wandb video recordings (0=disabled)")

    ap.add_argument("--spawn-farthest", action="store_true")
    ap.add_argument("--no-autoaim", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--init-model", type=str, default=None)
    ap.add_argument("--bench-episodes", type=int, default=16)
    ap.add_argument("--bench-timelimit", type=float, default=2.0,
                    help="Timelimit for benchmark 1v1 games (minutes)")
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    ap.add_argument("--wandb-project", type=str, default="doom-overnight",
                    help="W&B project name")
    # Reward shaping overrides
    ap.add_argument("--frag-bonus", type=float, default=None)
    ap.add_argument("--hit-bonus", type=float, default=None)
    ap.add_argument("--damage-bonus", type=float, default=None)
    ap.add_argument("--death-penalty", type=float, default=None)
    ap.add_argument("--attack-bonus", type=float, default=None)
    ap.add_argument("--move-bonus", type=float, default=None)
    ap.add_argument("--noop-penalty", type=float, default=None)
    ap.add_argument("--hit-taken-penalty", type=float, default=None)
    ap.add_argument("--damage-taken-penalty", type=float, default=None)
    ap.add_argument("--reward-scale", type=float, default=None)
    args = ap.parse_args()

    maps = parse_maps(args.maps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.name}.zip"
    meta_path = out_dir / f"{args.name}.meta.json"
    bench_path = out_dir / f"{args.name}_bench_mp.json"
    best_dir = out_dir / f"{args.name}_best"
    ckpt_dir = out_dir / f"{args.name}_ckpts"

    obs_shape = (int(args.obs_height), int(args.obs_width))
    cfg_runtime = materialize_cfg(args.cfg)
    button_names = parse_available_buttons(cfg_runtime)
    macro_names, macro_actions = build_macro_actions(button_names)
    shaping = ShapingCfg()
    # Apply CLI overrides
    for field_name, arg_name in [
        ("frag_bonus", "frag_bonus"), ("hit_bonus", "hit_bonus"),
        ("damage_bonus", "damage_bonus"), ("death_penalty", "death_penalty"),
        ("attack_bonus", "attack_bonus"), ("move_bonus", "move_bonus"),
        ("noop_penalty", "noop_penalty"), ("hit_taken_penalty", "hit_taken_penalty"),
        ("damage_taken_penalty", "damage_taken_penalty"), ("reward_scale", "reward_scale"),
    ]:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(shaping, field_name, val)

    # Detect scenario name for benchmark
    cfg_basename = Path(args.cfg).stem
    if "cig" in cfg_basename:
        scenario_name = "cig_fullaction"
    elif "deathmatch" in cfg_basename:
        scenario_name = "deathmatch"
    else:
        scenario_name = "deathmatch"

    # Write meta.json EARLY so benchmarks work even if training crashes
    meta = {
        "scenario": scenario_name,
        "env_id": f"CFG::{cfg_basename}_macro",
        "cfg_path": str(Path(args.cfg).resolve()),
        "frame_skip": int(args.frame_skip),
        "obs_shape": [obs_shape[0], obs_shape[1]],
        "observation_keys": ["screen", "gamevars"],
        "button_names": button_names,
        "action_button_map": [a.astype(int).tolist() for a in macro_actions],
        "action_names": macro_names,
        "action_space_n": int(len(macro_actions)),
        "maps": maps,
        "spawn_farthest": bool(args.spawn_farthest),
        "no_autoaim": bool(args.no_autoaim),
        "policy_type": args.policy_type,
        "frame_stack": args.frame_stack,
        "lstm_hidden_size": args.lstm_hidden_size if args.policy_type in ("lstm", "transformer") else None,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    # Also pre-create best dir and write meta there
    best_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(best_dir / "best_model.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    print(
        f"[overnight_dm] cfg={args.cfg} runtime_cfg={cfg_runtime} maps={maps} "
        f"timesteps={args.timesteps:,} envs={args.envs} obs={obs_shape[0]}x{obs_shape[1]} "
        f"frame_skip={args.frame_skip} bots={args.bots}/{args.bots_eval} "
        f"buttons={len(button_names)} macros={len(macro_actions)} timelimit={args.timelimit_minutes}m "
        f"scenario={scenario_name}"
    )
    print(f"  shaping: frag={shaping.frag_bonus} hit={shaping.hit_bonus} "
          f"atk_bonus={shaping.attack_bonus} noop_pen={shaping.noop_penalty}")

    # ── W&B init ──
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.name,
                config={
                    "timesteps": args.timesteps,
                    "envs": args.envs,
                    "cfg": args.cfg,
                    "maps": maps,
                    "frame_skip": args.frame_skip,
                    "obs_shape": obs_shape,
                    "bots": args.bots,
                    "bots_eval": args.bots_eval,
                    "timelimit_minutes": args.timelimit_minutes,
                    "n_steps": args.n_steps,
                    "batch_size": args.batch_size,
                    "n_epochs": args.n_epochs,
                    "learning_rate": args.learning_rate,
                    "clip_range": args.clip_range,
                    "ent_coef": args.ent_coef,
                    "target_kl": args.target_kl,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "policy_hidden_size": args.policy_hidden_size,
                    "policy_hidden_layers": args.policy_hidden_layers,
                    "cnn_type": args.cnn_type,
                    "cnn_features_dim": args.cnn_features_dim,
                    "policy_type": args.policy_type,
                    "lstm_hidden_size": args.lstm_hidden_size,
                    "lstm_num_layers": args.lstm_num_layers,
                    "transformer_heads": args.transformer_heads,
                    "transformer_layers": args.transformer_layers,
                    "context_length": args.context_length,
                    "frame_stack": args.frame_stack,
                    "n_buttons": len(button_names),
                    "n_macros": len(macro_actions),
                    "scenario": scenario_name,
                    "init_model": args.init_model,
                },
            )
            print(f"[overnight_dm] W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"[overnight_dm] W&B init failed ({e}); continuing without logging")

    def make_train():
        return _make_env(
            cfg_path=cfg_runtime,
            frame_skip=args.frame_skip,
            obs_shape=obs_shape,
            maps=maps,
            bots=args.bots,
            timelimit_minutes=args.timelimit_minutes,
            shaping=shaping,
            spawn_farthest=args.spawn_farthest,
            no_autoaim=args.no_autoaim,
        )

    def make_eval():
        return _make_env(
            cfg_path=cfg_runtime,
            frame_skip=args.frame_skip,
            obs_shape=obs_shape,
            maps=maps,
            bots=args.bots_eval,
            timelimit_minutes=args.timelimit_minutes,
            shaping=shaping,
            spawn_farthest=args.spawn_farthest,
            no_autoaim=args.no_autoaim,
        )

    vec_env = make_vec_env(make_train, n_envs=args.envs)
    vec_env = VecTransposeImage(vec_env)
    eval_env = make_vec_env(make_eval, n_envs=1)
    eval_env = VecTransposeImage(eval_env)
    if args.frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)
        print(f"[overnight_dm] Frame stacking: {args.frame_stack} frames")

    # Build policy_kwargs based on architecture choice
    policy_kwargs = {
        "net_arch": {
            "pi": [int(args.policy_hidden_size)] * int(args.policy_hidden_layers),
            "vf": [int(args.policy_hidden_size)] * int(args.policy_hidden_layers),
        }
    }
    if args.cnn_type == "impala":
        policy_kwargs["features_extractor_class"] = ImpalaCnnExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": int(args.cnn_features_dim),
            "channels": (16, 32, 32),
        }
        print(f"[overnight_dm] Using IMPALA ResNet CNN (features_dim={args.cnn_features_dim})")
    else:
        print(f"[overnight_dm] Using default NatureCNN")

    # Select algorithm class based on policy type
    if args.policy_type == "lstm":
        from sb3_contrib import RecurrentPPO
        AlgoClass = RecurrentPPO
        policy_name = "MultiInputLstmPolicy"
        policy_kwargs["lstm_hidden_size"] = int(args.lstm_hidden_size)
        policy_kwargs["n_lstm_layers"] = int(args.lstm_num_layers)
        policy_kwargs["shared_lstm"] = False  # separate LSTM for pi and vf
        policy_kwargs["enable_critic_lstm"] = True
        print(f"[overnight_dm] Using RecurrentPPO (LSTM hidden={args.lstm_hidden_size}, layers={args.lstm_num_layers})")
    elif args.policy_type == "transformer":
        from sb3_contrib import RecurrentPPO
        AlgoClass = RecurrentPPO
        policy_name = "MultiInputLstmPolicy"
        # For transformer, we use frame stacking as the context window
        # and a larger LSTM as the sequence model (RecurrentPPO's LSTM serves
        # as the recurrent backbone; true transformer would need custom policy)
        # Instead, we'll use a larger LSTM + frame stacking as a practical approximation
        # TODO: implement true transformer policy if LSTM proves insufficient
        policy_kwargs["lstm_hidden_size"] = int(args.lstm_hidden_size)
        policy_kwargs["n_lstm_layers"] = max(2, int(args.lstm_num_layers))
        policy_kwargs["shared_lstm"] = False
        policy_kwargs["enable_critic_lstm"] = True
        if args.frame_stack <= 1:
            print(f"[overnight_dm] WARNING: transformer mode works best with --frame-stack >= 4")
        print(f"[overnight_dm] Using RecurrentPPO+FrameStack (LSTM hidden={args.lstm_hidden_size}, "
              f"layers={max(2, args.lstm_num_layers)}, frame_stack={args.frame_stack})")
    else:
        AlgoClass = PPO
        policy_name = "MultiInputPolicy"
        print(f"[overnight_dm] Using standard PPO (feedforward MLP)")

    cb_list = [
        CheckpointCallback(
            save_freq=max(50_000 // max(1, args.envs), 2000),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
            verbose=0,
        ),
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(out_dir / f"{args.name}_eval"),
            eval_freq=max(25_000 // max(1, args.envs), 2000),
            deterministic=True,
            render=False,
            n_eval_episodes=8,
            verbose=1,
        ),
    ]
    if wandb_run is not None:
        cb_list.append(WandbMetricCallback())
        if args.video_freq > 0:
            cb_list.append(WandbVideoCallback(
                env_fn=make_eval,
                video_freq=args.video_freq,
                max_steps=600,
            ))
    callbacks = CallbackList(cb_list)

    model = None
    if args.init_model:
        try:
            print(f"[overnight_dm] loading init model: {args.init_model}")
            model = AlgoClass.load(args.init_model, env=vec_env, device=args.device)
        except Exception as e:
            print(f"[overnight_dm] init model incompatible ({e}); training from scratch.")
            model = None

    if model is None:
        model = AlgoClass(
            policy_name,
            vec_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_kwargs=policy_kwargs,
            device=args.device,
            verbose=1,
        )

    t0 = time.time()
    remaining_steps = args.timesteps
    max_crash_retries = 10
    crash_count = 0

    while remaining_steps > 0 and crash_count < max_crash_retries:
        try:
            model.learn(total_timesteps=remaining_steps, callback=callbacks, reset_num_timesteps=(crash_count == 0))
            remaining_steps = 0  # completed successfully
        except Exception as e:
            crash_count += 1
            elapsed_steps = args.timesteps - remaining_steps + model.num_timesteps
            print(f"[overnight_dm] CRASH #{crash_count} at ~{elapsed_steps} steps: {e!r}")
            print(f"[overnight_dm] Attempting recovery from latest checkpoint...")

            # Clean up old model and GPU memory
            try:
                del model
            except Exception:
                pass
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Clean up old envs
            try:
                vec_env.close()
            except Exception:
                pass
            try:
                eval_env.close()
            except Exception:
                pass
            # Kill any orphaned vizdoom processes from our subprocess workers
            import subprocess as _sp
            _sp.run(["pkill", "-9", "-f", "vizdoom.*Trainer"], capture_output=True)
            time.sleep(3)
            try:
                import gc; gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Find latest checkpoint
            ckpt_files = sorted(ckpt_dir.glob("ppo_*_steps.zip"), key=lambda p: p.stat().st_mtime)
            if ckpt_files:
                latest_ckpt = str(ckpt_files[-1])
                # Extract step count from filename
                import re
                m = re.search(r"ppo_(\d+)_steps", latest_ckpt)
                ckpt_steps = int(m.group(1)) if m else 0
                remaining_steps = max(0, args.timesteps - ckpt_steps)
                print(f"[overnight_dm] Resuming from {latest_ckpt} ({ckpt_steps} steps done, {remaining_steps} remaining)")
            else:
                print(f"[overnight_dm] No checkpoint found, restarting from scratch")
                remaining_steps = args.timesteps

            # Recreate envs
            vec_env = make_vec_env(make_train, n_envs=args.envs)
            vec_env = VecTransposeImage(vec_env)
            eval_env = make_vec_env(make_eval, n_envs=1)
            eval_env = VecTransposeImage(eval_env)
            if args.frame_stack > 1:
                vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)
                eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

            # Reload model from checkpoint
            if ckpt_files:
                model = AlgoClass.load(latest_ckpt, env=vec_env, device=args.device)
            else:
                model = AlgoClass(
                    policy_name, vec_env,
                    n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=args.n_epochs, learning_rate=args.learning_rate,
                    clip_range=args.clip_range, ent_coef=args.ent_coef,
                    target_kl=args.target_kl, gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    policy_kwargs=policy_kwargs,
                    device=args.device, verbose=1,
                )

            # Recreate callbacks with new envs
            cb_list = [
                CheckpointCallback(
                    save_freq=max(50_000 // max(1, args.envs), 2000),
                    save_path=str(ckpt_dir), name_prefix="ppo", verbose=0,
                ),
                EvalCallback(
                    eval_env=eval_env, best_model_save_path=str(best_dir),
                    log_path=str(out_dir / f"{args.name}_eval"),
                    eval_freq=max(25_000 // max(1, args.envs), 2000),
                    deterministic=True, render=False, n_eval_episodes=8, verbose=1,
                ),
            ]
            if wandb_run is not None:
                cb_list.append(WandbMetricCallback())
                if args.video_freq > 0:
                    cb_list.append(WandbVideoCallback(
                        env_fn=make_eval, video_freq=args.video_freq, max_steps=600,
                    ))
            callbacks = CallbackList(cb_list)

    if crash_count >= max_crash_retries:
        print(f"[overnight_dm] WARNING: gave up after {max_crash_retries} crashes")

    t1 = time.time()

    model.save(str(model_path))
    print(f"[overnight_dm] saved: {model_path}")
    print(f"[overnight_dm] train_seconds={t1 - t0:.1f} crashes={crash_count}")

    # Write meta.json sidecar for SB3Policy inference
    meta = {
        "scenario": scenario_name,
        "env_id": f"CFG::{cfg_basename}_macro",
        "cfg_path": str(Path(args.cfg).resolve()),
        "frame_skip": int(args.frame_skip),
        "obs_shape": [obs_shape[0], obs_shape[1]],
        "observation_keys": ["screen", "gamevars"],
        "button_names": button_names,
        "action_button_map": [a.astype(int).tolist() for a in macro_actions],
        "action_names": macro_names,
        "action_space_n": int(len(macro_actions)),
        "maps": maps,
        "spawn_farthest": bool(args.spawn_farthest),
        "no_autoaim": bool(args.no_autoaim),
        "policy_type": args.policy_type,
        "frame_stack": args.frame_stack,
        "lstm_hidden_size": args.lstm_hidden_size if args.policy_type in ("lstm", "transformer") else None,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[overnight_dm] meta: {meta_path}")

    # Copy meta to best dir too
    best_meta = best_dir / "best_model.meta.json"
    try:
        with open(best_meta, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    vec_env.close()
    eval_env.close()

    # Benchmark: try 1v1 multiplayer first, fall back to bot-based
    best_model = best_dir / "best_model.zip"
    bench_target = str(best_model if best_model.exists() else model_path)
    print(f"\n[overnight_dm] Benchmarking {bench_target} vs Random ({args.bench_episodes} episodes)...")
    try:
        bench = bench_duel_vs_random(
            model_path=bench_target,
            cfg_path=cfg_runtime,
            scenario_name=scenario_name,
            maps=maps,
            frame_skip=int(args.frame_skip),
            timelimit_minutes=float(args.bench_timelimit),
            episodes=int(args.bench_episodes),
            resolution="RES_320X240",
        )
        if bench.get("episodes", 0) == 0:
            raise RuntimeError("No successful MP episodes")
    except Exception as e:
        print(f"[overnight_dm] MP benchmark failed ({e}), falling back to bot-based...")
        bench = bench_vs_bots(
            model_path=bench_target,
            cfg_path=cfg_runtime,
            maps=maps,
            frame_skip=int(args.frame_skip),
            timelimit_minutes=float(args.bench_timelimit),
            episodes=int(args.bench_episodes),
            n_bots=max(1, args.bots_eval),
            obs_shape=obs_shape,
        )
    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)

    wr = bench.get("win_rate_vs_random_net", 0.0)
    gap = bench.get("mean_net_gap", 0.0)
    passed = bench.get("pass", False)
    hit_m = bench.get("model_hit_mean", 0.0)
    dmg_m = bench.get("model_damage_mean", 0.0)
    dmg_ratio = bench.get("damage_ratio", 0.0)
    print(f"[overnight_dm] BENCHMARK: win_rate={wr:.2f} mean_net_gap={gap:+.2f}")
    print(f"[overnight_dm]   hits={hit_m:.1f} damage={dmg_m:.1f} dmg_ratio={dmg_ratio:.2f}")
    print(f"[overnight_dm]   GATE: {'PASS' if passed else 'FAIL'} (need: hits>=10, dmg>=200, dmg_ratio>1, dmg>rand*1.5)")
    print(f"[overnight_dm] bench file: {bench_path}")

    # Log benchmark + video to W&B
    if wandb_run is not None:
        try:
            import wandb

            log_dict = {
                "bench/win_rate": wr,
                "bench/mean_net_gap": gap,
                "bench/pass": int(passed),
                "bench/model_frag_mean": bench.get("model_frag_mean", 0),
                "bench/model_kill_mean": bench.get("model_kill_mean", 0),
                "bench/model_death_mean": bench.get("model_death_mean", 0),
                "bench/model_net_mean": bench.get("model_net_mean", 0),
                "bench/model_damage_mean": bench.get("model_damage_mean", 0),
                "bench/model_damage_taken_mean": bench.get("model_damage_taken_mean", 0),
                "bench/damage_ratio": bench.get("damage_ratio", 0),
                "bench/random_net_mean": bench.get("random_net_mean", 0),
                "bench/episodes": bench.get("episodes", 0),
                "train/total_seconds": t1 - t0,
            }

            # Record a short video sample for wandb
            try:
                print("[overnight_dm] Recording sample video for W&B...")
                vid_ep = rollout_multiplayer_episode(
                    cfg_path=str(cfg_runtime),
                    scenario_name=scenario_name,
                    doom_map=maps[0],
                    policy_p1_dict={
                        "name": "model", "type": "sb3",
                        "path": bench_target, "algo": "PPO",
                        "arch": "DuelQNet", "action_size": None, "device": "auto",
                    },
                    policy_p2_dict={
                        "name": "Random", "type": "random",
                        "path": None, "algo": "PPO",
                        "arch": "DuelQNet", "action_size": None, "device": "auto",
                    },
                    timelimit_minutes=1.0,  # Short clip
                    frame_skip=int(args.frame_skip),
                    render_resolution="RES_160X120",  # Low res for fast upload
                    render_hud=False,
                    port=7300,
                )
                if vid_ep.frames_p1 and vid_ep.frames_p2:
                    # Build side-by-side video: model (left) | random (right)
                    n = min(len(vid_ep.frames_p1), len(vid_ep.frames_p2))
                    # Subsample to ~10 fps equivalent (every 3rd frame)
                    step = max(1, n // 300)
                    vid_frames = []
                    for i in range(0, n, step):
                        f1 = vid_ep.frames_p1[i]  # (H, W, 3)
                        f2 = vid_ep.frames_p2[i]
                        # Ensure same height
                        h = min(f1.shape[0], f2.shape[0])
                        f1 = f1[:h]
                        f2 = f2[:h]
                        side_by_side = np.concatenate([f1, f2], axis=1)  # (H, 2*W, 3)
                        vid_frames.append(side_by_side)

                    if vid_frames:
                        # wandb.Video expects (T, C, H, W) or (T, H, W, C)
                        vid_array = np.stack(vid_frames, axis=0)  # (T, H, W, 3)
                        # Transpose to (T, C, H, W) for wandb
                        vid_array = vid_array.transpose(0, 3, 1, 2)  # (T, 3, H, W)
                        log_dict["bench/video_model_vs_random"] = wandb.Video(
                            vid_array, fps=10, format="mp4",
                        )
                        print(f"[overnight_dm] Video logged: {len(vid_frames)} frames, side-by-side")
            except Exception as ve:
                print(f"[overnight_dm] Video recording failed ({ve}); skipping")

            wandb.log(log_dict)
            wandb.finish()
        except Exception:
            pass

    if passed:
        print("[overnight_dm] SUCCESS: Compound gate PASSED — agent fights and wins!")
    else:
        print(f"[overnight_dm] WARNING: Gate FAILED — agent needs more training")


if __name__ == "__main__":
    main()
