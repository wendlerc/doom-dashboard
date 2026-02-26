"""
Policy–scenario alignment helpers.

Determines whether a policy (trained for a given scenario) can be used
with a particular scenario, and infers scenario keys from configs.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from doom_dashboard.config import PolicyConfig, ScenarioConfig
from doom_dashboard.policies import SB3Policy


def infer_policy_scenario_key(pol_cfg: PolicyConfig | dict) -> Optional[str]:
    """Infer which scenario a policy was trained for (from name/path)."""
    if isinstance(pol_cfg, dict):
        name = pol_cfg.get("name", "")
        path = pol_cfg.get("path") or ""
    else:
        name = pol_cfg.name
        path = pol_cfg.path or ""
    return SB3Policy._infer_scenario_key(name, path)


def scenario_key(sc: ScenarioConfig) -> str:
    """Extract a canonical scenario key from a ScenarioConfig (for MP_SCENARIOS lookup)."""
    cfg = sc.cfg
    if cfg.endswith(".cfg"):
        stem = os.path.splitext(os.path.basename(cfg))[0]
        return stem.lower().replace(" ", "_")
    return cfg.lower().replace(" ", "_")


def policy_matches_scenario(pol_cfg: PolicyConfig | dict, scenario: ScenarioConfig) -> bool:
    """Return True if the policy can be used with this scenario."""
    if isinstance(pol_cfg, dict):
        ptype = (pol_cfg.get("type") or "").lower()
    else:
        ptype = pol_cfg.type.lower()

    if ptype == "random":
        return True

    exp = infer_policy_scenario_key(pol_cfg)
    sc_key = scenario_key(scenario)
    if exp is None:
        return True  # Unknown policy — allow (may fail at runtime)
    return exp == sc_key
