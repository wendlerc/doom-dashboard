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
import os
import random
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

    def predict(self, obs: np.ndarray, available_buttons: Optional[List[str]] = None) -> np.ndarray:
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

    def predict(self, obs: np.ndarray, available_buttons: Optional[List[str]] = None) -> np.ndarray:
        self._load()
        assert self._model is not None
        n = len(available_buttons) if available_buttons else 3

        import cv2
        # Resize to training IMAGE_SHAPE (60 H × 80 W)
        frame = obs if obs.shape[2] == 3 else obs[:, :, :3]
        small = cv2.resize(frame, (80, 60))  # (W, H) for cv2

        # Try MultiInputPolicy (dict obs) first, matching whatever keys the model expects
        obs_dict = None
        if hasattr(self._model, "observation_space") and hasattr(self._model.observation_space, "spaces"):
            spaces = self._model.observation_space.spaces
            obs_dict = {}
            for key in spaces:
                if key == "screen":
                    obs_dict["screen"] = small[np.newaxis]               # (1, H, W, 3)
                elif key == "gamevariables":
                    dim = spaces[key].shape[-1]
                    obs_dict["gamevariables"] = np.zeros((1, dim), dtype=np.float32)
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

        if isinstance(action, (int, np.integer)):
            arr = np.zeros(n, dtype=np.float32)
            if action < n:
                arr[int(action)] = 1.0
        else:
            arr = np.asarray(action, dtype=np.float32)
            if arr.shape[0] != n:
                tmp = np.zeros(n, dtype=np.float32)
                l = min(n, arr.shape[0])
                tmp[:l] = arr[:l]
                arr = tmp
        return arr

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

    def predict(self, obs: np.ndarray, available_buttons: Optional[List[str]] = None) -> np.ndarray:
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
