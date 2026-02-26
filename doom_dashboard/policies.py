"""
Policy loaders: Random, Stable-Baselines3, custom PyTorch.

Usage
-----
from doom_dashboard.policies import load_policy
from doom_dashboard.config import PolicyConfig

policy = load_policy(PolicyConfig(name="rand", type="random"))
action = policy.predict(obs, available_buttons)   # obs: np.ndarray H×W×C uint8
"""
from __future__ import annotations

import importlib
import json
import os
import random
import re
from typing import List, Optional, Any

import numpy as np
import torch


# ──────────────────────────── base ───────────────────────────────

class BasePolicy:
    """Minimal interface all policies must implement."""
    name: str = "base"

    def predict(
        self,
        obs: np.ndarray,
        available_buttons: Optional[List[str]] = None,
        game_variables: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a binary button array of shape (n_buttons,)."""
        raise NotImplementedError

    def close(self):
        pass


# ──────────────────────────── random ─────────────────────────────

class RandomPolicy(BasePolicy):
    """Uniformly samples a random action each step."""

    def __init__(self, name: str = "Random"):
        self.name = name
        self._n = None  # inferred from first call

    def predict(
        self,
        obs: np.ndarray,
        available_buttons: Optional[List[str]] = None,
        game_variables: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(available_buttons) if available_buttons else (self._n or 3)
        self._n = n
        # Pick exactly one button to press (or none) with equal probability
        action = np.zeros(n, dtype=np.float32)
        idx = random.randint(-1, n - 1)
        if idx >= 0:
            action[idx] = 1.0
        return action


# ──────────────────────────── SB3 ────────────────────────────────

class SB3Policy(BasePolicy):
    """Wraps a Stable-Baselines3 saved model (.zip)."""

    def __init__(self, path: str, algo: str = "PPO", device: str = "auto", name: str = "SB3"):
        self.name = name
        self.path = path
        self.algo = algo
        self.device = device
        self._model = None
        self._button_map: Optional[List[np.ndarray]] = None
        self._train_button_names: Optional[List[str]] = None
        self._action_button_map: Optional[List[np.ndarray]] = None
        self._obs_shape: tuple[int, int] = (60, 80)

    def _load_sidecar_meta(self):
        stem, _ = os.path.splitext(self.path)
        meta_path = stem + ".meta.json"
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path) as f:
                m = json.load(f)
            names = m.get("button_names")
            amap = m.get("action_button_map")
            oshape = m.get("obs_shape")
            if isinstance(names, list):
                self._train_button_names = [str(x) for x in names]
            if isinstance(amap, list):
                self._action_button_map = [np.asarray(x, dtype=np.float32) for x in amap]
                # Prefer explicit mapping from metadata over env reconstruction.
                self._button_map = self._action_button_map
            if (
                isinstance(oshape, list)
                and len(oshape) == 2
                and int(oshape[0]) > 0
                and int(oshape[1]) > 0
            ):
                self._obs_shape = (int(oshape[0]), int(oshape[1]))
        except Exception:
            return

    @staticmethod
    def _canon_button_name(name: str) -> str:
        """Normalize button names across ViZDoom enum/string variants."""
        raw = str(name).strip().upper()
        if "." in raw:
            raw = raw.split(".")[-1]
        return raw

    @staticmethod
    def _infer_scenario_key(name: str, path: str) -> Optional[str]:
        canon = {
            "basic": "basic",
            "defendthecenter": "defend_the_center",
            "deadlycorridor": "deadly_corridor",
            "healthgathering": "health_gathering",
            "mywayhome": "my_way_home",
            "deathmatch": "deathmatch",
            "multiduel": "multi_duel",
            "eliminationarena": "multi_duel",
        }

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())

        candidates = [name, os.path.splitext(os.path.basename(path))[0]]
        if path:
            p = os.path.abspath(path)
            parent = os.path.basename(os.path.dirname(p))
            grand = os.path.basename(os.path.dirname(os.path.dirname(p)))
            candidates.extend([parent, grand, p])
        for raw in candidates:
            c = _norm(raw)
            if c.startswith("ppo"):
                c = c[3:]
            if c in canon:
                return canon[c]
            for k, v in canon.items():
                if c.startswith(k) or k in c:
                    return v
        return None

    def _load_button_map(self):
        if self._button_map is not None:
            return
        self._button_map = []
        sc = self._infer_scenario_key(self.name, self.path)
        if sc is None:
            return
        if sc == "multi_duel":
            action_n = None
            try:
                if self._model is not None and hasattr(self._model, "action_space"):
                    sp = self._model.action_space
                    if hasattr(sp, "n"):
                        action_n = int(sp.n)
            except Exception:
                action_n = None
            # Fallback for in-progress checkpoints before sidecar metadata is available.
            # Button order in multi_duel.cfg: MOVE_LEFT, MOVE_RIGHT, ATTACK
            if action_n == 6:
                self._button_map = [
                    np.asarray([0, 0, 0], dtype=np.float32),
                    np.asarray([1, 0, 0], dtype=np.float32),
                    np.asarray([0, 1, 0], dtype=np.float32),
                    np.asarray([0, 0, 1], dtype=np.float32),
                    np.asarray([1, 0, 1], dtype=np.float32),
                    np.asarray([0, 1, 1], dtype=np.float32),
                ]
            return
        env_id = {
            "basic": "VizdoomBasic-v1",
            "defend_the_center": "VizdoomDefendCenter-v1",
            "deadly_corridor": "VizdoomDeadlyCorridor-v1",
            "health_gathering": "VizdoomHealthGathering-v1",
            "my_way_home": "VizdoomMyWayHome-v1",
            # Match current training setup for deathmatch (discrete actions).
            "deathmatch": "VizdoomDeathmatch-v1",
        }.get(sc)
        if env_id is None:
            return
        try:
            import gymnasium
            import vizdoom  # noqa: F401
            import vizdoom.gymnasium_wrapper  # noqa: F401

            action_n = None
            try:
                if self._model is not None and hasattr(self._model, "action_space"):
                    sp = self._model.action_space
                    if hasattr(sp, "n"):
                        action_n = int(sp.n)
            except Exception:
                action_n = None

            candidates = [{}]
            if sc == "deathmatch":
                # Support both legacy max_buttons_pressed=1 and improved =2 checkpoints.
                candidates = [
                    {"max_buttons_pressed": 2},
                    {"max_buttons_pressed": 1},
                    {},
                ]

            chosen = []
            for extra in candidates:
                env = gymnasium.make(env_id, frame_skip=4, **extra)
                bm = getattr(env.unwrapped, "button_map", None)
                cur = [np.asarray(a, dtype=np.float32) for a in bm] if bm else []
                env.close()
                if action_n is None:
                    chosen = cur
                    break
                if cur and len(cur) == action_n:
                    chosen = cur
                    break
                if not chosen and cur:
                    chosen = cur
            self._button_map = chosen
        except Exception:
            self._button_map = []

    def _load(self):
        if self._model is not None:
            return
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"SB3 checkpoint not found: {self.path}\n"
                "Run training first:\n"
                "  uv run python train_policies.py"
            )
        try:
            sb3_mod = importlib.import_module(f"stable_baselines3.{self.algo.lower()}")
            algo_cls = getattr(sb3_mod, self.algo.upper())
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Cannot load SB3 algo '{self.algo}'. "
                "Install stable-baselines3 and verify algo name."
            ) from e
        self._model = algo_cls.load(self.path, device=self.device)
        self._load_sidecar_meta()
        self._load_button_map()

    def predict(
        self,
        obs: np.ndarray,
        available_buttons: Optional[List[str]] = None,
        game_variables: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._load()
        assert self._model is not None
        n = len(available_buttons) if available_buttons else 3

        import cv2
        # Resize to the training resolution stored in sidecar metadata.
        frame = obs if obs.shape[2] == 3 else obs[:, :, :3]
        small = cv2.resize(frame, (self._obs_shape[1], self._obs_shape[0]))  # (W, H) for cv2

        # Try MultiInputPolicy (dict obs) first, matching whatever keys the model expects
        obs_dict = None
        if hasattr(self._model, "observation_space") and hasattr(self._model.observation_space, "spaces"):
            spaces = self._model.observation_space.spaces
            obs_dict = {}
            for key in spaces:
                if key == "screen":
                    # Training used VecTransposeImage, so SB3 expects CHW.
                    if len(spaces[key].shape) == 3 and spaces[key].shape[0] in (1, 3):
                        obs_dict["screen"] = np.transpose(small, (2, 0, 1))[np.newaxis]  # (1, C, H, W)
                    else:
                        obs_dict["screen"] = small[np.newaxis]  # (1, H, W, C)
                elif key == "gamevariables":
                    dim = spaces[key].shape[-1]
                    gv = np.zeros((dim,), dtype=np.float32)
                    if game_variables is not None:
                        raw = np.asarray(game_variables, dtype=np.float32).reshape(-1)
                        ncopy = min(dim, raw.shape[0])
                        gv[:ncopy] = raw[:ncopy]
                    obs_dict["gamevariables"] = gv[np.newaxis]
                else:
                    # Unknown key — provide zeros of the right shape
                    sh = (1,) + spaces[key].shape
                    obs_dict[key] = np.zeros(sh, dtype=np.float32)

        try:
            if obs_dict is not None:
                action, _ = self._model.predict(obs_dict, deterministic=True)
            else:
                action, _ = self._model.predict(small[np.newaxis], deterministic=True)
        except Exception:
            # Final fallback: try raw array
            try:
                action, _ = self._model.predict(obs[np.newaxis], deterministic=True)
            except Exception:
                return np.zeros(n, dtype=np.float32)

        # SB3 MultiBinary action space → (n_buttons,) bool/int array
        if hasattr(action, "shape") and len(action.shape) > 1:
            action = action[0]  # un-batch

        scalar_idx: Optional[int] = None
        if isinstance(action, (int, np.integer)):
            scalar_idx = int(action)
        elif isinstance(action, np.ndarray) and action.size == 1:
            try:
                scalar_idx = int(np.asarray(action).reshape(-1)[0])
            except Exception:
                scalar_idx = None

        if scalar_idx is not None:
            idx = scalar_idx
            if self._button_map and 0 <= idx < len(self._button_map):
                mapped = self._button_map[idx]
                arr = np.zeros(n, dtype=np.float32)
                if self._train_button_names and available_buttons:
                    avail_map = {
                        self._canon_button_name(b): i for i, b in enumerate(available_buttons)
                    }
                    for j, bn in enumerate(self._train_button_names):
                        if j >= mapped.shape[0]:
                            break
                        ai = avail_map.get(self._canon_button_name(bn))
                        if ai is not None:
                            arr[ai] = mapped[j]
                    arr = self._sanitize_buttons(arr, available_buttons)
                    return arr
                if available_buttons:
                    btns = [self._canon_button_name(b) for b in available_buttons]
                    binary_ix = [i for i, b in enumerate(btns) if "DELTA" not in b]
                    if mapped.shape[0] == len(binary_ix):
                        for j, ix in enumerate(binary_ix):
                            arr[ix] = mapped[j]
                    else:
                        l = min(n, mapped.shape[0])
                        arr[:l] = mapped[:l]
                else:
                    l = min(n, mapped.shape[0])
                    arr[:l] = mapped[:l]
            else:
                arr = np.zeros(n, dtype=np.float32)
                if idx < n:
                    arr[idx] = 1.0
        else:
            arr = np.asarray(action, dtype=np.float32)
            if arr.shape[0] != n:
                # Common case: policy was trained in Gym wrapper where delta
                # buttons are separated from binary buttons; runtime rollout
                # exposes full button list including deltas.
                tmp = np.zeros(n, dtype=np.float32)
                if self._train_button_names and available_buttons and arr.shape[0] == len(self._train_button_names):
                    avail_map = {
                        self._canon_button_name(b): i for i, b in enumerate(available_buttons)
                    }
                    for j, bn in enumerate(self._train_button_names):
                        ai = avail_map.get(self._canon_button_name(bn))
                        if ai is not None:
                            tmp[ai] = arr[j]
                    arr = tmp
                    arr = self._sanitize_buttons(arr, available_buttons)
                    return arr
                if available_buttons:
                    btns = [self._canon_button_name(b) for b in available_buttons]
                    binary_ix = [i for i, b in enumerate(btns) if "DELTA" not in b]
                    if arr.shape[0] == len(binary_ix):
                        for j, ix in enumerate(binary_ix):
                            tmp[ix] = arr[j]
                    else:
                        l = min(n, arr.shape[0])
                        tmp[:l] = arr[:l]
                else:
                    l = min(n, arr.shape[0])
                    tmp[:l] = arr[:l]
                arr = tmp
        arr = self._sanitize_buttons(arr, available_buttons)
        return arr

    @staticmethod
    def _sanitize_buttons(arr: np.ndarray, available_buttons: Optional[List[str]]) -> np.ndarray:
        if available_buttons is None or arr.ndim != 1 or arr.shape[0] != len(available_buttons):
            return arr
        out = arr.copy()
        names = [SB3Policy._canon_button_name(b) for b in available_buttons]

        def _zero_dual(a: str, b: str):
            ia = [i for i, n in enumerate(names) if a in n]
            ib = [i for i, n in enumerate(names) if b in n]
            if not ia or not ib:
                return
            if out[ia[0]] > 0.5 and out[ib[0]] > 0.5:
                out[ib[0]] = 0.0

        # Remove contradictory directional commands.
        _zero_dual("MOVE_FORWARD", "MOVE_BACKWARD")
        _zero_dual("MOVE_LEFT", "MOVE_RIGHT")
        _zero_dual("TURN_LEFT", "TURN_RIGHT")

        # Keep at most one weapon-selection button.
        weapon_ix = [i for i, n in enumerate(names) if "SELECT_WEAPON" in n]
        if len(weapon_ix) > 1:
            active = [i for i in weapon_ix if out[i] > 0.5]
            for i in active[1:]:
                out[i] = 0.0

        # Avoid spammy "shoot while switching weapon".
        attack_ix = [i for i, n in enumerate(names) if "ATTACK" in n]
        if attack_ix and weapon_ix and out[attack_ix[0]] > 0.5:
            for i in weapon_ix:
                out[i] = 0.0

        return out

    def close(self):
        self._model = None


# ──────────────────────────── PyTorch ────────────────────────────

class TorchPolicy(BasePolicy):
    """Loads a PyTorch policy saved as full model or state_dict.

    Supports:
      - `torch.save(model, path)`  → loaded directly
      - `torch.save(model.state_dict(), path)` → needs arch + action_size
    """

    def __init__(
        self,
        path: str,
        action_size: Optional[int] = None,
        arch: str = "DuelQNet",
        device: str = "auto",
        name: str = "Torch",
        preprocess_resolution: tuple = (30, 45),
    ):
        self.name = name
        self.path = path
        self.action_size = action_size
        self.arch = arch
        self.device_str = device
        self.preprocess_resolution = preprocess_resolution
        self._model = None
        self._device = None

    def _resolve_device(self) -> torch.device:
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def _load(self):
        if self._model is not None:
            return
        self._device = self._resolve_device()
        try:
            obj = torch.load(self.path, map_location=self._device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Cannot load torch model from '{self.path}': {e}") from e

        if isinstance(obj, torch.nn.Module):
            self._model = obj.to(self._device).eval()
        elif isinstance(obj, dict):
            # state_dict — need architecture
            if self.arch == "DuelQNet":
                from doom_dashboard._archs import DuelQNet
                assert self.action_size is not None, (
                    "action_size must be specified for state_dict loading"
                )
                net = DuelQNet(self.action_size)
                net.load_state_dict(obj)
                self._model = net.to(self._device).eval()
            else:
                raise ValueError(f"Unknown arch '{self.arch}' for state_dict loading.")
        else:
            raise TypeError(f"Unexpected object type in checkpoint: {type(obj)}")

    def predict(
        self,
        obs: np.ndarray,
        available_buttons: Optional[List[str]] = None,
        game_variables: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        import skimage.transform
        self._load()
        assert self._model is not None and self._device is not None
        n = len(available_buttons) if available_buttons else (self.action_size or 3)

        # Pre-process: grayscale + resize to (H, W) matching training resolution
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = np.mean(obs, axis=2)
        elif obs.ndim == 3 and obs.shape[2] == 1:
            gray = obs[:, :, 0]
        else:
            gray = obs
        small = skimage.transform.resize(
            gray, self.preprocess_resolution, anti_aliasing=True
        ).astype(np.float32)
        t = torch.from_numpy(small).unsqueeze(0).unsqueeze(0).to(self._device)  # (1,1,H,W)
        with torch.no_grad():
            q = self._model(t)  # (1, n_actions)
        best = int(torch.argmax(q, dim=1).item())
        arr = np.zeros(n, dtype=np.float32)
        if best < n:
            arr[best] = 1.0
        return arr

    def close(self):
        self._model = None
        self._device = None


# ──────────────────────────── factory ────────────────────────────

def load_policy(cfg: Any) -> BasePolicy:
    """Construct a policy from a PolicyConfig (or compatible dict-like)."""
    ptype = cfg.type.lower()
    name = cfg.name

    if ptype == "random":
        return RandomPolicy(name=name)

    elif ptype == "sb3":
        assert cfg.path, "SB3 policy requires a 'path'"
        return SB3Policy(
            path=cfg.path,
            algo=cfg.algo,
            device=cfg.device,
            name=name,
        )

    elif ptype == "torch":
        assert cfg.path, "Torch policy requires a 'path'"
        return TorchPolicy(
            path=cfg.path,
            action_size=cfg.action_size,
            arch=cfg.arch,
            device=cfg.device,
            name=name,
        )

    else:
        raise ValueError(f"Unknown policy type '{ptype}'. Use 'random', 'sb3', or 'torch'.")
