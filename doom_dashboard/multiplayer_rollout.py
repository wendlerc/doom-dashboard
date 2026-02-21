"""
Multiplayer 1v1 rollout using subprocess.Popen.

Each episode = two independent Python subprocesses:
  host_proc: fresh Python interpreter, plays as host
  join_proc: fresh Python interpreter, plays as join (after 2.5s delay)

No CUDA/fork issues. Results returned via temp files (pickle).
Port: caller provides a unique per-worker port.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import vizdoom as vzd


BASE_PORT       = 5300          # fresh range away from all previous attempts
DEATHMATCH_MAPS = ["map01"]
CIG_MAPS        = ["map01", "map02"]

_RES_MAP = {
    "RES_160X120": vzd.ScreenResolution.RES_160X120,
    "RES_320X240": vzd.ScreenResolution.RES_320X240,
    "RES_640X480": vzd.ScreenResolution.RES_640X480,
}


@dataclass
class MultiEpisodeData:
    frames_p1:    List[np.ndarray]
    actions_p1:   List[np.ndarray]
    rewards_p1:   List[float]
    game_vars_p1: List[np.ndarray]
    frames_p2:    List[np.ndarray]
    actions_p2:   List[np.ndarray]
    rewards_p2:   List[float]
    game_vars_p2: List[np.ndarray]
    button_names: List[str]
    scenario_name: str
    map_name: str
    policy_name_p1: str
    policy_name_p2: str
    cfg_path: str
    frame_skip: int
    timelimit_minutes: float
    duration_s: float
    game_tics: int
    metadata: dict = field(default_factory=dict)


# ─── inline player script ─────────────────────────────────────────

# A standalone Python script that runs one player of a 1v1 game.
# Receives config via environment variables; writes result via pickle file.
_PLAYER_SCRIPT = r"""
import os, sys, time, pickle, json
import numpy as np
sys.path.insert(0, os.environ["DOOM_DASH_ROOT"])

is_host   = os.environ["IS_HOST"] == "1"
port      = int(os.environ["PORT"])
cfg_path  = os.environ["CFG_PATH"]
doom_map  = os.environ["DOOM_MAP"]
timelimit = float(os.environ["TIMELIMIT"])
frame_skip= int(os.environ["FRAME_SKIP"])
res       = os.environ["RESOLUTION"]
pol_json  = json.loads(os.environ["POLICY_JSON"])
out_file  = os.environ["OUT_FILE"]

import vizdoom as vzd

_RES_MAP = {
    "RES_160X120": vzd.ScreenResolution.RES_160X120,
    "RES_320X240": vzd.ScreenResolution.RES_320X240,
    "RES_640X480": vzd.ScreenResolution.RES_640X480,
}

from doom_dashboard.config import PolicyConfig
from doom_dashboard.policies import load_policy
pol = load_policy(PolicyConfig(**pol_json))

result = {"frames":[], "actions":[], "rewards":[], "game_vars":[],
          "button_names":[], "duration_s":0.0, "game_tics":0}
try:
    g = vzd.DoomGame()
    g.load_config(cfg_path)
    g.set_doom_map(doom_map)
    g.set_window_visible(False)
    g.set_mode(vzd.Mode.ASYNC_PLAYER)
    g.set_screen_format(vzd.ScreenFormat.CRCGCB)
    g.set_screen_resolution(_RES_MAP.get(res, vzd.ScreenResolution.RES_320X240))
    g.set_render_hud(False)
    if is_host:
        g.add_game_args(
            f"-host 2 -port {port} +viz_connect_timeout 120 "
            f"-deathmatch +timelimit {timelimit:.1f} "
            "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
            "+sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 0 +viz_nocheat 1"
        )
    else:
        g.add_game_args(f"-join 127.0.0.1 -port {port} +viz_connect_timeout 120")
    g.add_game_args(f"+name {'P1' if is_host else 'P2'} +colorset {0 if is_host else 3}")
    g.init()
    btns = [str(b) for b in g.get_available_buttons()]
    t0 = time.perf_counter()
    while not g.is_episode_finished():
        s = g.get_state()
        if s is None:
            g.advance_action(frame_skip)
            continue
        obs = np.transpose(s.screen_buffer, (1, 2, 0))
        gv  = np.array(s.game_variables, dtype=np.float32) if s.game_variables else np.zeros(1)
        act = pol.predict(obs, btns)
        rew = g.make_action([bool(a) for a in act], frame_skip)
        if g.is_player_dead():
            g.respawn_player()
        result["frames"].append(obs)
        result["actions"].append(act.copy())
        result["rewards"].append(float(rew))
        result["game_vars"].append(gv)
    result["button_names"] = btns
    result["duration_s"]   = time.perf_counter() - t0
    result["game_tics"]    = int(g.get_episode_time())
    g.close()
except Exception as e:
    result["error"] = str(e)

with open(out_file, "wb") as f:
    pickle.dump(result, f, protocol=4)
"""


def rollout_multiplayer_episode(
    cfg_path: str,
    scenario_name: str,
    doom_map: str,
    policy_p1_dict: dict,   # serialisable dict, NOT a loaded policy object
    policy_p2_dict: dict,
    timelimit_minutes: float = 5.0,
    frame_skip: int = 4,
    render_resolution: str = "RES_320X240",
    render_hud: bool = False,
    port: int = BASE_PORT,
) -> MultiEpisodeData:
    """
    Run a 1v1 episode. Host and join are independent subprocess.Popen()
    processes running from a fresh Python interpreter — no CUDA fork issues.
    Results are returned via temp pickle files.
    """
    tmpdir   = tempfile.mkdtemp(prefix="doom_mp_")
    out1     = os.path.join(tmpdir, "p1.pkl")
    out2     = os.path.join(tmpdir, "p2.pkl")
    script   = os.path.join(tmpdir, "player.py")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(script, "w") as f:
        f.write(_PLAYER_SCRIPT)

    common_env = {
        **os.environ,
        "DOOM_DASH_ROOT": root_dir,
        "CFG_PATH":       cfg_path,
        "DOOM_MAP":       doom_map,
        "PORT":           str(port),
        "TIMELIMIT":      str(timelimit_minutes),
        "FRAME_SKIP":     str(frame_skip),
        "RESOLUTION":     render_resolution,
    }

    host_proc = subprocess.Popen(
        [sys.executable, script],
        env={**common_env, "IS_HOST": "1",
             "POLICY_JSON": json.dumps(policy_p1_dict), "OUT_FILE": out1},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # Brief delay so host binds the port before join tries to connect
    time.sleep(3.0)

    join_proc = subprocess.Popen(
        [sys.executable, script],
        env={**common_env, "IS_HOST": "0",
             "POLICY_JSON": json.dumps(policy_p2_dict), "OUT_FILE": out2},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    timeout = timelimit_minutes * 60 + 180
    try:
        host_proc.wait(timeout=timeout)
        join_proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        host_proc.kill()
        join_proc.kill()

    # Load results
    def _load(path: str, label: str) -> dict:
        if not os.path.exists(path):
            return {"error": f"{label} result file missing"}
        with open(path, "rb") as f:
            return pickle.load(f)

    r1 = _load(out1, "host")
    r2 = _load(out2, "join")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    for r, label in [(r1, "host"), (r2, "join")]:
        if "error" in r:
            raise RuntimeError(f"Player {label} failed: {r['error']}")

    return MultiEpisodeData(
        frames_p1=r1.get("frames", []),
        actions_p1=r1.get("actions", []),
        rewards_p1=r1.get("rewards", []),
        game_vars_p1=r1.get("game_vars", []),
        frames_p2=r2.get("frames", []),
        actions_p2=r2.get("actions", []),
        rewards_p2=r2.get("rewards", []),
        game_vars_p2=r2.get("game_vars", []),
        button_names=r1.get("button_names") or r2.get("button_names", []),
        scenario_name=scenario_name,
        map_name=doom_map,
        policy_name_p1=policy_p1_dict["name"],
        policy_name_p2=policy_p2_dict["name"],
        cfg_path=cfg_path,
        frame_skip=frame_skip,
        timelimit_minutes=timelimit_minutes,
        duration_s=max(r1.get("duration_s", 0.0), r2.get("duration_s", 0.0)),
        game_tics=max(r1.get("game_tics", 0), r2.get("game_tics", 0)),
        metadata=dict(render_resolution=render_resolution, render_hud=render_hud),
    )
