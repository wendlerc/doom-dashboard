#!/usr/bin/env python3
"""
Behavioral Cloning (BC) pretraining from human gameplay recordings.

Trains a policy network via supervised learning on human demo .npz files,
then saves in SB3-compatible format for fine-tuning with PPO.

Usage:
    # MLP policy, compact action space
    uv run python pretrain_bc.py \
        --demo-dir human_demos/ \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --output trained_policies/bc_compact.zip \
        --epochs 50 --batch-size 128 --lr 3e-4

    # LSTM policy
    uv run python pretrain_bc.py \
        --demo-dir human_demos/ \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        --output trained_policies/bc_compact_lstm.zip \
        --policy-type lstm --epochs 50

    # Then fine-tune with PPO:
    uv run python train_overnight_dm.py \
        --init-model trained_policies/bc_compact.zip \
        --cfg doom_dashboard/scenarios/deathmatch_compact.cfg \
        ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_overnight_dm import (
    ImpalaCnnExtractor,
    build_macro_actions,
    parse_available_buttons,
    materialize_cfg,
    TRACKED_VARS,
    _canon_buttons,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DemoDataset(Dataset):
    """PyTorch dataset wrapping human demo .npz files.

    Each sample is (screen_chw_uint8, gamevars_float32, macro_action_int64).
    Frames are resized to obs_shape and converted to CHW.
    Raw button actions are mapped to the nearest macro discrete action.
    """

    def __init__(
        self,
        npz_paths: List[Path],
        macro_actions: List[np.ndarray],
        target_button_names: List[str],
        obs_shape: Tuple[int, int] = (120, 160),
        skip_noop: bool = False,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.macro_actions = macro_actions  # list of uint8 arrays, one per macro
        self.target_buttons = _canon_buttons(target_button_names)
        self.n_target = len(self.target_buttons)
        self.skip_noop = skip_noop

        # Build lookup: macro index -> button vector (float for distance calc)
        self._macro_vecs = np.stack(
            [a.astype(np.float32) for a in macro_actions], axis=0
        )  # (n_macros, n_buttons)

        # Pre-load and process all demos
        self.screens: List[np.ndarray] = []
        self.gamevars: List[np.ndarray] = []
        self.actions: List[int] = []

        n_skipped_noop = 0
        n_total = 0

        for npz_path in npz_paths:
            data = np.load(str(npz_path), allow_pickle=True)
            frames = data["frames"]     # (T, H, W, 3) uint8

            # Try loading actions from main npz; fall back to separate .actions.npz
            if "actions" in data:
                raw_actions = data["actions"]
                game_vars = data.get("game_vars", None)
            else:
                actions_path = npz_path.parent / (npz_path.stem + ".actions.npz")
                if actions_path.exists():
                    adata = np.load(str(actions_path), allow_pickle=True)
                    raw_actions = adata["actions"]
                    game_vars = adata.get("game_vars", None)
                    print(f"  [DemoDataset] loaded actions from {actions_path}")
                else:
                    print(f"  [DemoDataset] skipping {npz_path}: no actions (truncated file?)")
                    continue

            T = min(len(frames), len(raw_actions))
            if T == 0:
                print(f"  [DemoDataset] skipping empty demo: {npz_path}")
                continue

            # Detect source button layout from action dimensionality
            n_src = raw_actions.shape[1]

            # Try to load meta.json for button_names
            meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
            src_button_names = None
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    src_button_names = meta.get("button_names", None)
                except Exception:
                    pass

            # Build the mapping from source buttons to target buttons
            btn_map = self._build_button_map(n_src, src_button_names)

            for t in range(T):
                # Map raw buttons to target button space
                target_btns = self._map_buttons(raw_actions[t], btn_map)

                # Find nearest macro action
                macro_idx = self._nearest_macro(target_btns)
                n_total += 1

                # Optionally skip NOOP frames (action index 0 is always NOOP)
                if self.skip_noop and macro_idx == 0:
                    n_skipped_noop += 1
                    continue

                # Resize frame
                frame = frames[t]
                if frame.shape[0] != obs_shape[0] or frame.shape[1] != obs_shape[1]:
                    frame = cv2.resize(
                        frame, (obs_shape[1], obs_shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )

                # Convert HWC -> CHW
                screen_chw = np.transpose(frame, (2, 0, 1)).astype(np.uint8)
                self.screens.append(screen_chw)

                # Game vars (zero-pad if not available or wrong shape)
                if game_vars is not None and t < len(game_vars):
                    gv = game_vars[t].astype(np.float32)
                    # Pad/truncate to TRACKED_VARS size
                    n_vars = len(TRACKED_VARS)
                    if gv.shape[0] < n_vars:
                        gv = np.concatenate([gv, np.zeros(n_vars - gv.shape[0], dtype=np.float32)])
                    elif gv.shape[0] > n_vars:
                        gv = gv[:n_vars]
                else:
                    gv = np.zeros(len(TRACKED_VARS), dtype=np.float32)

                self.gamevars.append(gv)
                self.actions.append(macro_idx)

            print(
                f"  [DemoDataset] loaded {npz_path.name}: "
                f"{T} steps, {n_src} source buttons"
            )

        if self.skip_noop and n_skipped_noop > 0:
            print(
                f"  [DemoDataset] skipped {n_skipped_noop}/{n_total} "
                f"NOOP frames ({100*n_skipped_noop/max(1,n_total):.1f}%)"
            )

        print(f"  [DemoDataset] total samples: {len(self.actions)}")

        # Compute and report action distribution
        if self.actions:
            act_arr = np.array(self.actions)
            n_macros = len(macro_actions)
            counts = np.bincount(act_arr, minlength=n_macros)
            print("  [DemoDataset] action distribution:")
            from train_overnight_dm import build_macro_actions as _bma
            # We already have macro names from the caller, use indices
            for i in range(n_macros):
                if counts[i] > 0:
                    pct = 100.0 * counts[i] / len(self.actions)
                    print(f"    macro {i:2d}: {counts[i]:6d} ({pct:5.1f}%)")

    def _build_button_map(
        self, n_src: int, src_button_names: List[str] | None
    ) -> np.ndarray | None:
        """Build a mapping matrix from source buttons to target buttons.

        Returns an (n_target, n_src) matrix M such that
            target_btns = clip(M @ src_btns, 0, 1)
        Or None if source and target are identical.
        """
        if src_button_names is not None:
            src_canon = _canon_buttons(src_button_names)
            if src_canon == self.target_buttons:
                return None  # identity mapping
            # Build name-based mapping
            M = np.zeros((self.n_target, len(src_canon)), dtype=np.float32)
            for i, tgt_name in enumerate(self.target_buttons):
                for j, src_name in enumerate(src_canon):
                    if src_name == tgt_name:
                        M[i, j] = 1.0
            return M

        # No source button names: heuristic based on dimensions
        if n_src == self.n_target:
            return None  # assume same layout

        # Source has more buttons than target: common for fullaction -> compact
        # Build identity mapping for the first n_target columns,
        # ignoring extra source buttons
        # This is a rough heuristic; real mapping depends on cfg order.
        # Since we can't know the source layout without meta, warn and do best effort.
        print(
            f"    WARNING: source has {n_src} buttons, target has {self.n_target}. "
            f"No meta.json found; using positional mapping (may be inaccurate)."
        )
        M = np.zeros((self.n_target, n_src), dtype=np.float32)
        for i in range(min(self.n_target, n_src)):
            M[i, i] = 1.0
        return M

    def _map_buttons(self, src_btns: np.ndarray, M: np.ndarray | None) -> np.ndarray:
        """Map source button vector to target button vector."""
        if M is None:
            return src_btns[:self.n_target].astype(np.float32)
        mapped = M @ src_btns.astype(np.float32)
        return np.clip(mapped, 0.0, 1.0)

    def _nearest_macro(self, button_vec: np.ndarray) -> int:
        """Find the macro action closest to the given button vector.

        Uses Hamming distance (number of differing binary buttons),
        with a tiebreak preferring macros with more active buttons
        (so pressing forward+attack maps to ATTACK_FORWARD, not just ATTACK).
        """
        # Binarize the button vector (threshold at 0.5)
        bv = (button_vec > 0.5).astype(np.float32)

        # Compute distances to all macros
        macro_binary = (self._macro_vecs > 0.5).astype(np.float32)

        # Hamming distance
        distances = np.sum(np.abs(macro_binary - bv), axis=1)

        # Tiebreak: prefer macros with more active buttons (higher overlap)
        # among those with minimum distance
        overlap = np.sum(macro_binary * bv, axis=1)

        min_dist = distances.min()
        candidates = np.where(distances == min_dist)[0]

        if len(candidates) == 1:
            return int(candidates[0])

        # Among tied candidates, pick the one with most overlap with pressed buttons
        best = candidates[np.argmax(overlap[candidates])]
        return int(best)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        return (
            self.screens[idx],       # uint8 CHW
            self.gamevars[idx],      # float32
            np.int64(self.actions[idx]),
        )


# ---------------------------------------------------------------------------
# Sequence dataset for LSTM training
# ---------------------------------------------------------------------------

class DemoSequenceDataset(Dataset):
    """Produces fixed-length subsequences for LSTM BC training.

    Each sample is a sequence of (screens, gamevars, actions) of length seq_len.
    Sequences are drawn from contiguous segments within each demo file.
    """

    def __init__(
        self,
        base_dataset: DemoDataset,
        seq_len: int = 32,
        stride: int = 16,
        demo_boundaries: List[int] | None = None,
    ):
        super().__init__()
        self.base = base_dataset
        self.seq_len = seq_len
        self.stride = stride

        # If demo_boundaries not provided, treat entire dataset as one sequence
        if demo_boundaries is None:
            demo_boundaries = [len(base_dataset)]

        # Build valid start indices for sequences (never cross demo boundaries)
        self.starts: List[int] = []
        prev = 0
        for end in demo_boundaries:
            demo_len = end - prev
            if demo_len >= seq_len:
                for s in range(prev, end - seq_len + 1, stride):
                    self.starts.append(s)
            prev = end

        print(f"  [DemoSequenceDataset] {len(self.starts)} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        screens = np.stack(
            [self.base.screens[start + t] for t in range(self.seq_len)],
            axis=0,
        )  # (seq_len, C, H, W) uint8
        gamevars = np.stack(
            [self.base.gamevars[start + t] for t in range(self.seq_len)],
            axis=0,
        )  # (seq_len, n_vars) float32
        actions = np.array(
            [self.base.actions[start + t] for t in range(self.seq_len)],
            dtype=np.int64,
        )  # (seq_len,)
        return screens, gamevars, actions


# ---------------------------------------------------------------------------
# BC training loop (MLP)
# ---------------------------------------------------------------------------

def train_bc_mlp(
    dataset: DemoDataset,
    *,
    n_actions: int,
    obs_shape: Tuple[int, int],
    n_gamevars: int,
    cnn_type: str,
    features_dim: int,
    hidden_size: int,
    hidden_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    val_split: float = 0.1,
) -> dict:
    """Train MLP BC policy and return state_dict + metadata for SB3 injection."""

    # Train/val split
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    use_pin = device != "cpu"
    n_workers = 2 if use_pin else 0
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=use_pin)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=use_pin)

    print(f"\n  [BC-MLP] train={n_train} val={n_val} actions={n_actions}")
    print(f"  [BC-MLP] cnn={cnn_type} features_dim={features_dim} "
          f"hidden={hidden_size}x{hidden_layers} lr={lr} bs={batch_size}")

    # Build feature extractor
    obs_space = gym.spaces.Dict({
        "screen": gym.spaces.Box(0, 255, shape=(3, obs_shape[0], obs_shape[1]), dtype=np.uint8),
        "gamevars": gym.spaces.Box(-1e6, 1e6, shape=(n_gamevars,), dtype=np.float32),
    })

    if cnn_type == "impala":
        feat_ext = ImpalaCnnExtractor(obs_space, features_dim=features_dim, channels=(16, 32, 32))
    else:
        # Use SB3's built-in NatureCNN + gamevars MLP (manual reimplementation)
        feat_ext = ImpalaCnnExtractor(obs_space, features_dim=features_dim, channels=(16, 32, 32))
        print("  [BC-MLP] NOTE: using IMPALA CNN even for 'nature' to match training pipeline")

    feat_dim = feat_ext._features_dim

    # Policy head (matching SB3 MlpExtractor structure)
    pi_layers = []
    in_dim = feat_dim
    for _ in range(hidden_layers):
        pi_layers.append(nn.Linear(in_dim, hidden_size))
        pi_layers.append(nn.ReLU())
        in_dim = hidden_size
    pi_layers.append(nn.Linear(in_dim, n_actions))
    pi_head = nn.Sequential(*pi_layers)

    feat_ext = feat_ext.to(device)
    pi_head = pi_head.to(device)

    optimizer = torch.optim.Adam(
        list(feat_ext.parameters()) + list(pi_head.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        feat_ext.train()
        pi_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for screens, gv, targets in train_dl:
            screens = screens.to(device, dtype=torch.float32)
            gv = gv.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.long)

            obs = {"screen": screens, "gamevars": gv}
            feats = feat_ext(obs)
            logits = pi_head(feats)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(feat_ext.parameters()) + list(pi_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            train_loss += loss.item() * targets.shape[0]
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += targets.shape[0]
        scheduler.step()

        # --- Validate ---
        feat_ext.eval()
        pi_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for screens, gv, targets in val_dl:
                screens = screens.to(device, dtype=torch.float32)
                gv = gv.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.long)

                obs = {"screen": screens, "gamevars": gv}
                feats = feat_ext(obs)
                logits = pi_head(feats)
                loss = F.cross_entropy(logits, targets)

                val_loss += loss.item() * targets.shape[0]
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += targets.shape[0]

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        avg_train_loss = train_loss / max(1, train_total)
        avg_val_loss = val_loss / max(1, val_total)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "feat_ext": {k: v.cpu().clone() for k, v in feat_ext.state_dict().items()},
                "pi_head": {k: v.cpu().clone() for k, v in pi_head.state_dict().items()},
            }

        if epoch % max(1, epochs // 20) == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f}  "
                f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.3f}  "
                f"best_val_acc={best_val_acc:.3f}"
            )

    return {
        "feat_ext_state": best_state["feat_ext"],
        "pi_head_state": best_state["pi_head"],
        "features_dim": feat_dim,
        "hidden_size": hidden_size,
        "hidden_layers": hidden_layers,
        "n_actions": n_actions,
        "best_val_acc": best_val_acc,
    }


# ---------------------------------------------------------------------------
# BC training loop (LSTM)
# ---------------------------------------------------------------------------

def train_bc_lstm(
    seq_dataset: DemoSequenceDataset,
    *,
    n_actions: int,
    obs_shape: Tuple[int, int],
    n_gamevars: int,
    cnn_type: str,
    features_dim: int,
    hidden_size: int,
    hidden_layers: int,
    lstm_hidden_size: int,
    lstm_num_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    val_split: float = 0.1,
) -> dict:
    """Train LSTM BC policy and return state_dict + metadata."""

    n_total = len(seq_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(seq_dataset, [n_train, n_val])
    use_pin = device != "cpu"
    n_workers = 2 if use_pin else 0
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=use_pin)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=use_pin)

    print(f"\n  [BC-LSTM] train={n_train} val={n_val} actions={n_actions}")
    print(f"  [BC-LSTM] cnn={cnn_type} features_dim={features_dim} "
          f"hidden={hidden_size}x{hidden_layers} lstm={lstm_hidden_size}x{lstm_num_layers} "
          f"lr={lr} bs={batch_size}")

    obs_space = gym.spaces.Dict({
        "screen": gym.spaces.Box(0, 255, shape=(3, obs_shape[0], obs_shape[1]), dtype=np.uint8),
        "gamevars": gym.spaces.Box(-1e6, 1e6, shape=(n_gamevars,), dtype=np.float32),
    })

    feat_ext = ImpalaCnnExtractor(obs_space, features_dim=features_dim, channels=(16, 32, 32))
    feat_dim = feat_ext._features_dim

    # LSTM
    lstm = nn.LSTM(
        input_size=feat_dim,
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers,
        batch_first=True,
    )

    # Policy head after LSTM
    pi_layers = []
    in_dim = lstm_hidden_size
    for _ in range(hidden_layers):
        pi_layers.append(nn.Linear(in_dim, hidden_size))
        pi_layers.append(nn.ReLU())
        in_dim = hidden_size
    pi_layers.append(nn.Linear(in_dim, n_actions))
    pi_head = nn.Sequential(*pi_layers)

    feat_ext = feat_ext.to(device)
    lstm = lstm.to(device)
    pi_head = pi_head.to(device)

    all_params = list(feat_ext.parameters()) + list(lstm.parameters()) + list(pi_head.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None
    seq_len = seq_dataset.seq_len

    for epoch in range(1, epochs + 1):
        # --- Train ---
        feat_ext.train()
        lstm.train()
        pi_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for screens, gv, targets in train_dl:
            # screens: (B, seq_len, C, H, W) uint8
            # gv: (B, seq_len, n_vars) float32
            # targets: (B, seq_len) int64
            B, S = screens.shape[0], screens.shape[1]
            screens = screens.to(device, dtype=torch.float32).reshape(B * S, *screens.shape[2:])
            gv = gv.to(device, dtype=torch.float32).reshape(B * S, -1)
            targets = targets.to(device, dtype=torch.long).reshape(B * S)

            obs = {"screen": screens, "gamevars": gv}
            feats = feat_ext(obs)  # (B*S, feat_dim)
            feats = feats.reshape(B, S, -1)  # (B, S, feat_dim)

            lstm_out, _ = lstm(feats)  # (B, S, lstm_hidden)
            lstm_out = lstm_out.reshape(B * S, -1)
            logits = pi_head(lstm_out)  # (B*S, n_actions)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * targets.shape[0]
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += targets.shape[0]

        scheduler.step()

        # --- Validate ---
        feat_ext.eval()
        lstm.eval()
        pi_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for screens, gv, targets in val_dl:
                B, S = screens.shape[0], screens.shape[1]
                screens = screens.to(device, dtype=torch.float32).reshape(B * S, *screens.shape[2:])
                gv = gv.to(device, dtype=torch.float32).reshape(B * S, -1)
                targets = targets.to(device, dtype=torch.long).reshape(B * S)

                obs = {"screen": screens, "gamevars": gv}
                feats = feat_ext(obs).reshape(B, S, -1)
                lstm_out, _ = lstm(feats)
                lstm_out = lstm_out.reshape(B * S, -1)
                logits = pi_head(lstm_out)
                loss = F.cross_entropy(logits, targets)

                val_loss += loss.item() * targets.shape[0]
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += targets.shape[0]

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        avg_train_loss = train_loss / max(1, train_total)
        avg_val_loss = val_loss / max(1, val_total)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "feat_ext": {k: v.cpu().clone() for k, v in feat_ext.state_dict().items()},
                "lstm": {k: v.cpu().clone() for k, v in lstm.state_dict().items()},
                "pi_head": {k: v.cpu().clone() for k, v in pi_head.state_dict().items()},
            }

        if epoch % max(1, epochs // 20) == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f}  "
                f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.3f}  "
                f"best_val_acc={best_val_acc:.3f}"
            )

    return {
        "feat_ext_state": best_state["feat_ext"],
        "lstm_state": best_state["lstm"],
        "pi_head_state": best_state["pi_head"],
        "features_dim": feat_dim,
        "hidden_size": hidden_size,
        "hidden_layers": hidden_layers,
        "lstm_hidden_size": lstm_hidden_size,
        "lstm_num_layers": lstm_num_layers,
        "n_actions": n_actions,
        "best_val_acc": best_val_acc,
    }


# ---------------------------------------------------------------------------
# SB3 model creation + weight injection
# ---------------------------------------------------------------------------

def inject_bc_weights_mlp(
    bc_result: dict,
    *,
    cfg_path: str,
    obs_shape: Tuple[int, int],
    cnn_type: str,
    features_dim: int,
    hidden_size: int,
    hidden_layers: int,
    device: str,
) -> "PPO":
    """Create a PPO model and inject BC-trained weights into it."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecTransposeImage
    from train_overnight_dm import DeathmatchMacroEnv, ShapingCfg

    # Build a dummy env with the same obs/action space
    button_names = parse_available_buttons(cfg_path)
    macro_names, macro_actions = build_macro_actions(button_names)

    def make_dummy():
        return DeathmatchMacroEnv(
            cfg_path=cfg_path,
            frame_skip=4,
            obs_shape=obs_shape,
            maps=["map01"],
            bots=0,
            timelimit_minutes=0.5,
            shaping=ShapingCfg(),
            spawn_farthest=False,
            no_autoaim=False,
        )

    dummy_env = make_vec_env(make_dummy, n_envs=1)
    dummy_env = VecTransposeImage(dummy_env)

    policy_kwargs = {
        "net_arch": {
            "pi": [hidden_size] * hidden_layers,
            "vf": [hidden_size] * hidden_layers,
        },
    }
    if cnn_type == "impala":
        policy_kwargs["features_extractor_class"] = ImpalaCnnExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "channels": (16, 32, 32),
        }

    model = PPO(
        "MultiInputPolicy",
        dummy_env,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )

    # Inject BC weights into the SB3 model
    sb3_state = model.policy.state_dict()
    bc_feat = bc_result["feat_ext_state"]
    bc_pi = bc_result["pi_head_state"]

    # Map BC feature extractor -> SB3 features_extractor
    n_matched = 0
    n_total = 0
    for key, val in bc_feat.items():
        sb3_key = f"features_extractor.{key}"
        n_total += 1
        if sb3_key in sb3_state and sb3_state[sb3_key].shape == val.shape:
            sb3_state[sb3_key] = val
            n_matched += 1
        else:
            print(f"  WARNING: BC feat key '{key}' -> '{sb3_key}' not found or shape mismatch")

    # Map BC policy head -> SB3 mlp_extractor.policy_net + action_net
    # SB3 structure: mlp_extractor.policy_net has Linear+ReLU layers,
    # then action_net is the final Linear layer.
    pi_keys = list(bc_pi.keys())
    # Group by layer index: 0.weight, 0.bias, 2.weight, 2.bias, ...
    # In nn.Sequential with Linear+ReLU, Linear layers are at indices 0, 2, 4, ...
    linear_params = []
    i = 0
    while True:
        w_key = f"{i}.weight"
        b_key = f"{i}.bias"
        if w_key in bc_pi:
            linear_params.append((bc_pi[w_key], bc_pi[b_key]))
            i += 2  # skip ReLU
        else:
            break

    # The last linear_params entry is the action head, the rest are MLP layers
    if len(linear_params) > 0:
        mlp_layers = linear_params[:-1]
        action_layer = linear_params[-1]

        # Inject MLP layers into mlp_extractor.policy_net
        for layer_idx, (w, b) in enumerate(mlp_layers):
            w_key = f"mlp_extractor.policy_net.{layer_idx * 2}.weight"
            b_key = f"mlp_extractor.policy_net.{layer_idx * 2}.bias"
            n_total += 2
            if w_key in sb3_state and sb3_state[w_key].shape == w.shape:
                sb3_state[w_key] = w
                sb3_state[b_key] = b
                n_matched += 2
            else:
                print(f"  WARNING: MLP layer {layer_idx} shape mismatch")

        # Inject action head
        n_total += 2
        if "action_net.weight" in sb3_state and sb3_state["action_net.weight"].shape == action_layer[0].shape:
            sb3_state["action_net.weight"] = action_layer[0]
            sb3_state["action_net.bias"] = action_layer[1]
            n_matched += 2
        else:
            print(f"  WARNING: action_net shape mismatch "
                  f"(BC: {action_layer[0].shape}, SB3: {sb3_state.get('action_net.weight', 'missing')})")

    print(f"  [inject] matched {n_matched}/{n_total} BC params into SB3 model")

    model.policy.load_state_dict(sb3_state)
    dummy_env.close()
    return model


def inject_bc_weights_lstm(
    bc_result: dict,
    *,
    cfg_path: str,
    obs_shape: Tuple[int, int],
    cnn_type: str,
    features_dim: int,
    hidden_size: int,
    hidden_layers: int,
    lstm_hidden_size: int,
    lstm_num_layers: int,
    device: str,
) -> "RecurrentPPO":
    """Create a RecurrentPPO model and inject BC-trained weights."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecTransposeImage
    from train_overnight_dm import DeathmatchMacroEnv, ShapingCfg

    button_names = parse_available_buttons(cfg_path)
    macro_names, macro_actions = build_macro_actions(button_names)

    def make_dummy():
        return DeathmatchMacroEnv(
            cfg_path=cfg_path,
            frame_skip=4,
            obs_shape=obs_shape,
            maps=["map01"],
            bots=0,
            timelimit_minutes=0.5,
            shaping=ShapingCfg(),
            spawn_farthest=False,
            no_autoaim=False,
        )

    dummy_env = make_vec_env(make_dummy, n_envs=1)
    dummy_env = VecTransposeImage(dummy_env)

    policy_kwargs = {
        "net_arch": {
            "pi": [hidden_size] * hidden_layers,
            "vf": [hidden_size] * hidden_layers,
        },
        "lstm_hidden_size": lstm_hidden_size,
        "n_lstm_layers": lstm_num_layers,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }
    if cnn_type == "impala":
        policy_kwargs["features_extractor_class"] = ImpalaCnnExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": features_dim,
            "channels": (16, 32, 32),
        }

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        dummy_env,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )

    # Inject BC weights
    sb3_state = model.policy.state_dict()
    bc_feat = bc_result["feat_ext_state"]
    bc_lstm = bc_result["lstm_state"]
    bc_pi = bc_result["pi_head_state"]

    n_matched = 0
    n_total = 0

    # Feature extractor
    for key, val in bc_feat.items():
        sb3_key = f"features_extractor.{key}"
        n_total += 1
        if sb3_key in sb3_state and sb3_state[sb3_key].shape == val.shape:
            sb3_state[sb3_key] = val
            n_matched += 1
        else:
            print(f"  WARNING: BC feat key '{key}' -> '{sb3_key}' not found or shape mismatch")

    # LSTM weights - inject into policy LSTM (pi lstm)
    # RecurrentPPO has lstm_actor (and optionally lstm_critic)
    for key, val in bc_lstm.items():
        # Try mapping to both actor and critic LSTMs
        for prefix in ["lstm_actor"]:
            sb3_key = f"{prefix}.{key}"
            n_total += 1
            if sb3_key in sb3_state and sb3_state[sb3_key].shape == val.shape:
                sb3_state[sb3_key] = val
                n_matched += 1
            else:
                # LSTM state key naming might differ
                alt_key = f"{prefix}.{key}"
                if alt_key in sb3_state and sb3_state[alt_key].shape == val.shape:
                    sb3_state[alt_key] = val
                    n_matched += 1

    # Policy head -> action_net + mlp_extractor.policy_net
    pi_keys = list(bc_pi.keys())
    linear_params = []
    i = 0
    while True:
        w_key = f"{i}.weight"
        b_key = f"{i}.bias"
        if w_key in bc_pi:
            linear_params.append((bc_pi[w_key], bc_pi[b_key]))
            i += 2
        else:
            break

    if len(linear_params) > 0:
        mlp_layers = linear_params[:-1]
        action_layer = linear_params[-1]

        for layer_idx, (w, b) in enumerate(mlp_layers):
            w_key = f"mlp_extractor.policy_net.{layer_idx * 2}.weight"
            b_key = f"mlp_extractor.policy_net.{layer_idx * 2}.bias"
            n_total += 2
            if w_key in sb3_state and sb3_state[w_key].shape == w.shape:
                sb3_state[w_key] = w
                sb3_state[b_key] = b
                n_matched += 2
            else:
                print(f"  WARNING: LSTM MLP layer {layer_idx} shape mismatch")

        n_total += 2
        if "action_net.weight" in sb3_state and sb3_state["action_net.weight"].shape == action_layer[0].shape:
            sb3_state["action_net.weight"] = action_layer[0]
            sb3_state["action_net.bias"] = action_layer[1]
            n_matched += 2

    print(f"  [inject] matched {n_matched}/{n_total} BC params into RecurrentPPO model")

    model.policy.load_state_dict(sb3_state)
    dummy_env.close()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Behavioral Cloning pretraining from human gameplay demos."
    )
    ap.add_argument("--demo-dir", type=str, default="human_demos",
                    help="Directory containing .npz demo files")
    ap.add_argument("--output", "-o", type=str, default="trained_policies/bc_pretrained.zip",
                    help="Output path for the pretrained SB3 model (.zip)")
    ap.add_argument("--cfg", type=str, default="doom_dashboard/scenarios/deathmatch_compact.cfg",
                    help="ViZDoom cfg file (defines target button/action space)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy-type", type=str, default="mlp",
                    choices=["mlp", "lstm"],
                    help="Policy architecture: mlp or lstm")
    ap.add_argument("--cnn-type", type=str, default="impala",
                    choices=["nature", "impala"],
                    help="CNN backbone: nature or impala (default matches training)")
    ap.add_argument("--cnn-features-dim", type=int, default=256,
                    help="CNN feature output dimension")
    ap.add_argument("--policy-hidden-size", type=int, default=512,
                    help="Hidden layer size for policy MLP")
    ap.add_argument("--policy-hidden-layers", type=int, default=2,
                    help="Number of hidden layers in policy MLP")
    ap.add_argument("--obs-height", type=int, default=120)
    ap.add_argument("--obs-width", type=int, default=160)
    ap.add_argument("--lstm-hidden-size", type=int, default=256,
                    help="LSTM hidden state size (for --policy-type lstm)")
    ap.add_argument("--lstm-num-layers", type=int, default=1,
                    help="Number of LSTM layers (for --policy-type lstm)")
    ap.add_argument("--seq-len", type=int, default=32,
                    help="Sequence length for LSTM training")
    ap.add_argument("--seq-stride", type=int, default=16,
                    help="Stride between sequences for LSTM training")
    ap.add_argument("--skip-noop", action="store_true",
                    help="Skip NOOP frames during training (reduces class imbalance)")
    ap.add_argument("--device", type=str, default="auto",
                    help="Device: cuda, cpu, or auto")
    ap.add_argument("--val-split", type=float, default=0.1,
                    help="Fraction of data for validation")
    args = ap.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[pretrain_bc] device={device}")

    # Resolve cfg and parse buttons
    cfg_path = materialize_cfg(args.cfg)
    button_names = parse_available_buttons(cfg_path)
    macro_names, macro_actions = build_macro_actions(button_names)
    obs_shape = (args.obs_height, args.obs_width)

    print(f"[pretrain_bc] cfg={args.cfg}")
    print(f"[pretrain_bc] buttons ({len(button_names)}): {button_names}")
    print(f"[pretrain_bc] macros ({len(macro_actions)}): {macro_names}")
    print(f"[pretrain_bc] obs_shape={obs_shape}")

    # Find demo files
    demo_dir = Path(args.demo_dir)
    if not demo_dir.exists():
        print(f"ERROR: demo directory not found: {demo_dir}")
        sys.exit(1)

    npz_files = sorted(demo_dir.glob("*.npz"))
    if not npz_files:
        print(f"ERROR: no .npz files found in {demo_dir}")
        sys.exit(1)

    print(f"[pretrain_bc] found {len(npz_files)} demo files:")
    for f in npz_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Build dataset
    print("\n[pretrain_bc] loading demos...")
    dataset = DemoDataset(
        npz_paths=npz_files,
        macro_actions=macro_actions,
        target_button_names=button_names,
        obs_shape=obs_shape,
        skip_noop=args.skip_noop,
    )

    if len(dataset) == 0:
        print("ERROR: dataset is empty after loading demos")
        sys.exit(1)

    # Train
    print(f"\n[pretrain_bc] training {args.policy_type.upper()} BC policy...")
    t0 = time.time()

    if args.policy_type == "lstm":
        # Build demo boundaries for sequence dataset
        # Since all demos are concatenated, we need to track where each demo starts/ends
        # Re-load to get boundaries
        demo_boundaries = []
        offset = 0
        for npz_path in npz_files:
            data = np.load(str(npz_path), allow_pickle=True)
            T = min(len(data["frames"]), len(data["actions"]))
            # Account for skip_noop reducing sample count
            # (boundaries are approximate; sequences may cross skip gaps but that's OK)
            offset += T
            demo_boundaries.append(min(offset, len(dataset)))

        seq_dataset = DemoSequenceDataset(
            base_dataset=dataset,
            seq_len=args.seq_len,
            stride=args.seq_stride,
            demo_boundaries=demo_boundaries,
        )

        bc_result = train_bc_lstm(
            seq_dataset,
            n_actions=len(macro_actions),
            obs_shape=obs_shape,
            n_gamevars=len(TRACKED_VARS),
            cnn_type=args.cnn_type,
            features_dim=args.cnn_features_dim,
            hidden_size=args.policy_hidden_size,
            hidden_layers=args.policy_hidden_layers,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            val_split=args.val_split,
        )
    else:
        bc_result = train_bc_mlp(
            dataset,
            n_actions=len(macro_actions),
            obs_shape=obs_shape,
            n_gamevars=len(TRACKED_VARS),
            cnn_type=args.cnn_type,
            features_dim=args.cnn_features_dim,
            hidden_size=args.policy_hidden_size,
            hidden_layers=args.policy_hidden_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            val_split=args.val_split,
        )

    t1 = time.time()
    print(f"\n[pretrain_bc] training took {t1 - t0:.1f}s  best_val_acc={bc_result['best_val_acc']:.3f}")

    # Inject into SB3 model and save
    print(f"\n[pretrain_bc] injecting BC weights into SB3 model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.policy_type == "lstm":
        model = inject_bc_weights_lstm(
            bc_result,
            cfg_path=cfg_path,
            obs_shape=obs_shape,
            cnn_type=args.cnn_type,
            features_dim=args.cnn_features_dim,
            hidden_size=args.policy_hidden_size,
            hidden_layers=args.policy_hidden_layers,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            device=device,
        )
    else:
        model = inject_bc_weights_mlp(
            bc_result,
            cfg_path=cfg_path,
            obs_shape=obs_shape,
            cnn_type=args.cnn_type,
            features_dim=args.cnn_features_dim,
            hidden_size=args.policy_hidden_size,
            hidden_layers=args.policy_hidden_layers,
            device=device,
        )

    model.save(str(output_path))
    print(f"[pretrain_bc] saved SB3 model: {output_path}")

    # Write meta.json sidecar (matching train_overnight_dm.py format)
    cfg_basename = Path(args.cfg).stem
    if "cig" in cfg_basename:
        scenario_name = "cig_fullaction"
    elif "deathmatch" in cfg_basename:
        scenario_name = "deathmatch"
    else:
        scenario_name = "deathmatch"

    meta = {
        "scenario": scenario_name,
        "env_id": f"CFG::{cfg_basename}_macro",
        "cfg_path": str(Path(args.cfg).resolve()),
        "frame_skip": 4,
        "obs_shape": [obs_shape[0], obs_shape[1]],
        "observation_keys": ["screen", "gamevars"],
        "button_names": button_names,
        "action_button_map": [a.astype(int).tolist() for a in macro_actions],
        "action_names": macro_names,
        "action_space_n": len(macro_actions),
        "maps": ["map01"],
        "spawn_farthest": False,
        "no_autoaim": False,
        "policy_type": args.policy_type,
        "frame_stack": 1,
        "lstm_hidden_size": args.lstm_hidden_size if args.policy_type == "lstm" else None,
        "bc_pretrained": True,
        "bc_epochs": args.epochs,
        "bc_best_val_acc": float(bc_result["best_val_acc"]),
        "bc_demo_files": [f.name for f in npz_files],
        "bc_n_samples": len(dataset),
        "bc_training_time_s": float(t1 - t0),
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pretrain_bc] saved meta: {meta_path}")

    # Also save a raw BC checkpoint (for debugging/analysis)
    raw_ckpt_path = output_path.with_suffix(".bc_ckpt.pt")
    torch.save(bc_result, str(raw_ckpt_path))
    print(f"[pretrain_bc] saved raw BC checkpoint: {raw_ckpt_path}")

    print(f"\n[pretrain_bc] done! To fine-tune with PPO:")
    print(f"  uv run python train_overnight_dm.py \\")
    print(f"    --init-model {output_path} \\")
    print(f"    --cfg {args.cfg} \\")
    print(f"    --policy-type {args.policy_type} \\")
    print(f"    --cnn-type {args.cnn_type} \\")
    print(f"    ...")


if __name__ == "__main__":
    main()
