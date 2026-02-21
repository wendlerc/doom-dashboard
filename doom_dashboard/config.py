"""
Configuration dataclasses for doom-dashboard.
Load with `DashboardConfig.from_yaml(path)`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml
import vizdoom as vzd


# ──────────────────────────── helpers ────────────────────────────

BUILTIN_SCENARIOS = {
    s: os.path.join(vzd.scenarios_path, f"{s}.cfg")
    for s in [
        "basic", "simpler_basic", "rocket_basic",
        "defend_the_center", "defend_the_line",
        "health_gathering", "health_gathering_supreme",
        "my_way_home", "deadly_corridor", "deathmatch",
        "predict_position", "take_cover",
    ]
}


def resolve_cfg(name_or_path: str) -> str:
    """Return absolute path to a .cfg file.

    Accepts either a built-in scenario name (e.g. 'basic') or an
    absolute/relative path to a .cfg file.
    """
    if os.path.isabs(name_or_path) or name_or_path.endswith(".cfg"):
        if not os.path.exists(name_or_path):
            raise FileNotFoundError(f"Scenario config not found: {name_or_path}")
        return name_or_path
    if name_or_path in BUILTIN_SCENARIOS:
        return BUILTIN_SCENARIOS[name_or_path]
    raise ValueError(
        f"Unknown scenario '{name_or_path}'. "
        f"Available built-ins: {sorted(BUILTIN_SCENARIOS)}"
    )


# ──────────────────────────── dataclasses ────────────────────────

@dataclass
class ScenarioConfig:
    name: str                           # human-readable label
    cfg: str                            # built-in key or path to .cfg
    episode_timeout: Optional[int] = None  # None → use cfg default
    frame_skip: int = 4
    render_resolution: str = "RES_320X240"
    render_hud: bool = True

    def cfg_path(self) -> str:
        return resolve_cfg(self.cfg)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        return cls(
            name=d["name"],
            cfg=d["cfg"],
            episode_timeout=d.get("episode_timeout"),
            frame_skip=d.get("frame_skip", 4),
            render_resolution=d.get("render_resolution", "RES_320X240"),
            render_hud=d.get("render_hud", True),
        )


@dataclass
class PolicyConfig:
    name: str
    type: str               # "random" | "sb3" | "torch"
    path: Optional[str] = None
    algo: str = "PPO"       # for sb3: PPO / DQN / A2C …
    arch: str = "DuelQNet"  # for torch: architecture name
    action_size: Optional[int] = None  # for torch (inferred if None)
    device: str = "auto"    # "auto" | "cpu" | "cuda"

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyConfig":
        return cls(
            name=d["name"],
            type=d["type"],
            path=d.get("path"),
            algo=d.get("algo", "PPO"),
            arch=d.get("arch", "DuelQNet"),
            action_size=d.get("action_size"),
            device=d.get("device", "auto"),
        )


@dataclass
class DashboardConfig:
    samples_per_map: int = 1
    render_resolution: str = "RES_640X480"
    fps: int = 30
    output_dir: str = "samples"

    @classmethod
    def from_dict(cls, d: dict) -> "DashboardConfig":
        return cls(
            samples_per_map=d.get("samples_per_map", 1),
            render_resolution=d.get("render_resolution", "RES_640X480"),
            fps=d.get("fps", 30),
            output_dir=d.get("output_dir", "samples"),
        )


@dataclass
class DatasetConfig:
    output_dir: str = "dataset"
    total_hours: float = 2.0
    scenario_ratios: Dict[str, float] = field(default_factory=dict)
    policy_ratios: Dict[str, float] = field(default_factory=dict)
    frame_skip: int = 4
    render_resolution: str = "RES_320X240"
    render_hud: bool = False
    shard_size_mb: int = 512
    num_workers: Optional[int] = None  # None → auto (cpu_count)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            output_dir=d.get("output_dir", "dataset"),
            total_hours=d.get("total_hours", 2.0),
            scenario_ratios=d.get("scenario_ratios", {}),
            policy_ratios=d.get("policy_ratios", {}),
            frame_skip=d.get("frame_skip", 4),
            render_resolution=d.get("render_resolution", "RES_320X240"),
            render_hud=d.get("render_hud", False),
            shard_size_mb=d.get("shard_size_mb", 512),
            num_workers=d.get("num_workers"),
        )


@dataclass
class Config:
    scenarios: List[ScenarioConfig]
    policies: List[PolicyConfig]
    dashboard: DashboardConfig
    dataset: DatasetConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            scenarios=[ScenarioConfig.from_dict(s) for s in raw.get("scenarios", [])],
            policies=[PolicyConfig.from_dict(p) for p in raw.get("policies", [])],
            dashboard=DashboardConfig.from_dict(raw.get("dashboard", {})),
            dataset=DatasetConfig.from_dict(raw.get("dataset", {})),
        )

    def scenario_by_name(self, name: str) -> ScenarioConfig:
        for s in self.scenarios:
            if s.name == name:
                return s
        raise KeyError(f"Scenario '{name}' not found in config")

    def policy_by_name(self, name: str) -> PolicyConfig:
        for p in self.policies:
            if p.name == name:
                return p
        raise KeyError(f"Policy '{name}' not found in config")
