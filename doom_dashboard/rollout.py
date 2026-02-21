"""
Single-episode rollout using vizdoom.DoomGame.

Returns an EpisodeData with raw frames, actions, rewards, and metadata.
Safe to call from a subprocess — each DoomGame is self-contained.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import vizdoom as vzd

from doom_dashboard.config import ScenarioConfig
from doom_dashboard.policies import BasePolicy


# ──────────────────────────── result type ────────────────────────

@dataclass
class EpisodeData:
    frames: List[np.ndarray]        # list of (H, W, C) uint8 arrays
    actions: List[np.ndarray]       # list of (n_buttons,) float arrays
    button_names: List[str]         # names of buttons in same order as actions
    rewards: List[float]
    game_vars: List[np.ndarray]     # per-step game variable arrays
    total_reward: float
    scenario_name: str
    policy_name: str
    cfg_path: str
    frame_skip: int
    steps: int
    duration_s: float               # real elapsed seconds for a single game
    game_tics: int                  # vizdoom tic count
    metadata: dict = field(default_factory=dict)


# ──────────────────────────── resolution helper ──────────────────

_RES_MAP = {
    "RES_160X120":  vzd.ScreenResolution.RES_160X120,
    "RES_200X150":  vzd.ScreenResolution.RES_200X150,
    "RES_256X144":  vzd.ScreenResolution.RES_256X144,
    "RES_320X180":  vzd.ScreenResolution.RES_320X180,
    "RES_320X240":  vzd.ScreenResolution.RES_320X240,
    "RES_400X225":  vzd.ScreenResolution.RES_400X225,
    "RES_512X288":  vzd.ScreenResolution.RES_512X288,
    "RES_640X360":  vzd.ScreenResolution.RES_640X360,
    "RES_640X480":  vzd.ScreenResolution.RES_640X480,
    "RES_800X450":  vzd.ScreenResolution.RES_800X450,
    "RES_800X600":  vzd.ScreenResolution.RES_800X600,
    "RES_1024X576": vzd.ScreenResolution.RES_1024X576,
    "RES_1280X720": vzd.ScreenResolution.RES_1280X720,
}


def _parse_resolution(s: str) -> vzd.ScreenResolution:
    if s in _RES_MAP:
        return _RES_MAP[s]
    raise ValueError(f"Unknown resolution '{s}'. Available: {sorted(_RES_MAP)}")


# ──────────────────────────── main rollout ───────────────────────

def rollout_episode(
    scenario: ScenarioConfig,
    policy: BasePolicy,
    render_resolution: Optional[str] = None,
    frame_skip: Optional[int] = None,
    render_hud: Optional[bool] = None,
    doom_map: Optional[str] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    record_frames: bool = True,
) -> EpisodeData:
    """Run one episode and return an EpisodeData.

    Parameters
    ----------
    scenario:           Scenario config (resolved cfg path + params)
    policy:             Policy to use for action selection
    render_resolution:  Override scenario resolution (e.g. "RES_640X480")
    frame_skip:         Override scenario frame_skip
    render_hud:         Override scenario HUD rendering
    doom_map:           Optional map override (e.g. "map01")
    max_steps:          Override episode_timeout (None = use cfg)
    seed:               Random seed for reproducibility
    record_frames:      If False, skip frame capture (faster for non-video runs)
    """
    game = vzd.DoomGame()
    game.load_config(scenario.cfg_path())
    if doom_map:
        game.set_doom_map(doom_map)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)  # C×H×W uint8

    res = render_resolution or scenario.render_resolution
    hud = scenario.render_hud if render_hud is None else bool(render_hud)
    eff_frame_skip = scenario.frame_skip if frame_skip is None else int(frame_skip)
    game.set_screen_resolution(_parse_resolution(res))
    game.set_render_hud(hud)

    if max_steps is not None:
        game.set_episode_timeout(max_steps)
    elif scenario.episode_timeout is not None:
        game.set_episode_timeout(scenario.episode_timeout)

    if seed is not None:
        game.set_seed(seed)

    game.init()

    button_names: List[str] = [str(b) for b in game.get_available_buttons()]
    n_buttons = len(button_names)

    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    game_vars_list: List[np.ndarray] = []

    game.new_episode()
    t_start = time.perf_counter()
    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break

        # Build observation: convert C×H×W → H×W×C
        buf = state.screen_buffer  # np.ndarray (C, H, W) uint8
        obs = np.transpose(buf, (1, 2, 0))  # → (H, W, C)

        action_arr = policy.predict(obs, button_names)
        action_list = [bool(a) for a in action_arr]

        gv = np.array(state.game_variables, dtype=np.float32) if state.game_variables is not None else np.zeros(1)
        reward = game.make_action(action_list, eff_frame_skip)

        if record_frames:
            frames.append(obs)
        actions.append(action_arr.copy())
        rewards.append(float(reward))
        game_vars_list.append(gv)

    t_end = time.perf_counter()
    total_reward = game.get_total_reward()
    game_tics = game.get_episode_time()
    game.close()

    return EpisodeData(
        frames=frames,
        actions=actions,
        button_names=button_names,
        rewards=rewards,
        game_vars=game_vars_list,
        total_reward=total_reward,
        scenario_name=scenario.name,
        policy_name=policy.name,
        cfg_path=scenario.cfg_path(),
        frame_skip=eff_frame_skip,
        steps=len(actions),
        duration_s=t_end - t_start,
        game_tics=int(game_tics),
        metadata={
            "render_resolution": res,
            "render_hud": hud,
        },
    )
