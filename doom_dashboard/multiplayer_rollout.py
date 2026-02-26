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
_TRACKED_VARS = [
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.KILLCOUNT,
    vzd.GameVariable.DEATHCOUNT,
    vzd.GameVariable.HITCOUNT,
    vzd.GameVariable.HITS_TAKEN,
    vzd.GameVariable.DAMAGECOUNT,
    vzd.GameVariable.DAMAGE_TAKEN,
    vzd.GameVariable.HEALTH,
]
def _read_vars(game):
    vals = []
    for v in _TRACKED_VARS:
        try:
            vals.append(float(game.get_game_variable(v)))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)
try:
    g = vzd.DoomGame()
    g.load_config(cfg_path)
    g.set_doom_map(doom_map)
    g.set_window_visible(False)
    g.set_mode(vzd.Mode.ASYNC_PLAYER)
    # Match Gymnasium training env defaults (SB3 checkpoints were trained on RGB24).
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_screen_resolution(_RES_MAP.get(res, vzd.ScreenResolution.RES_320X240))
    g.set_render_hud(False)
    if is_host:
        # NOTE: Do NOT pass "-deathmatch" here. The -deathmatch flag tells ZDoom
        # to use deathmatch start spawn points (Thing type 11). Many ViZDoom WADs
        # (e.g. deathmatch.wad) lack these and the engine segfaults (signal 11).
        # The +sv_* CVARs below already configure deathmatch-style gameplay rules
        # (forced respawn, weapon stay, etc.) without requiring DM spawn points.
        g.add_game_args(
            f"-host 2 -port {port} +viz_connect_timeout 30 "
            f"+timelimit {timelimit:.1f} "
            "+sv_forcerespawn 1 +sv_noautoaim 0 +sv_respawnprotect 1 "
            "+sv_spawnfarthest 0 +sv_nocrouch 1 +viz_respawn_delay 0 +viz_nocheat 1"
        )
    else:
        g.add_game_args(f"-join 127.0.0.1 -port {port} +viz_connect_timeout 30")
    g.add_game_args(f"+name {'P1' if is_host else 'P2'} +colorset {0 if is_host else 3}")
    g.init()
    btns = [str(b) for b in g.get_available_buttons()]
    t0 = time.perf_counter()
    while not g.is_episode_finished():
        s = g.get_state()
        if s is None:
            g.advance_action(frame_skip)
            continue
        buf = s.screen_buffer
        if buf.ndim == 3 and buf.shape[0] in (1, 3, 4) and buf.shape[-1] not in (1, 3, 4):
            obs = np.transpose(buf, (1, 2, 0))
        else:
            obs = buf
        gv  = _read_vars(g)
        act = pol.predict(obs, btns, game_variables=gv)
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
    log1     = os.path.join(tmpdir, "p1.log")
    log2     = os.path.join(tmpdir, "p2.log")
    script   = os.path.join(tmpdir, "player.py")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Some custom cfg files reference WADs via relative paths.
    # Materialize cfg + WAD in tmpdir so both subprocesses resolve assets reliably.
    resolved_cfg_path = cfg_path
    try:
        with open(cfg_path, "r") as f:
            cfg_text = f.read()
        wad_ref = None
        for ln in cfg_text.splitlines():
            raw = ln.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.lower().startswith("doom_scenario_path"):
                wad_ref = raw.split("=", 1)[1].strip().strip("\"'")
                break
        if wad_ref:
            if os.path.isabs(wad_ref):
                wad_ok = os.path.exists(wad_ref)
            else:
                local = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), wad_ref)
                wad_ok = os.path.exists(local)
                if not wad_ok:
                    src = os.path.join(vzd.scenarios_path, wad_ref)
                    if os.path.exists(src):
                        dst_cfg = os.path.join(tmpdir, os.path.basename(cfg_path))
                        dst_wad = os.path.join(tmpdir, wad_ref)
                        import shutil
                        shutil.copy2(cfg_path, dst_cfg)
                        shutil.copy2(src, dst_wad)
                        resolved_cfg_path = dst_cfg
                        wad_ok = True
            if not wad_ok:
                raise FileNotFoundError(f"Cannot resolve doom_scenario_path '{wad_ref}' for cfg '{cfg_path}'")
    except Exception:
        # Keep legacy behavior when cfg parsing fails.
        resolved_cfg_path = cfg_path

    def _absolutize_policy_path(pol: dict) -> dict:
        out = dict(pol)
        p = out.get("path")
        if p and not os.path.isabs(p):
            out["path"] = os.path.abspath(os.path.join(root_dir, p))
        return out

    policy_p1_dict = _absolutize_policy_path(policy_p1_dict)
    policy_p2_dict = _absolutize_policy_path(policy_p2_dict)

    with open(script, "w") as f:
        f.write(_PLAYER_SCRIPT)

    common_env = {
        **os.environ,
        "DOOM_DASH_ROOT": root_dir,
        "CFG_PATH":       resolved_cfg_path,
        "DOOM_MAP":       doom_map,
        "PORT":           str(port),
        "TIMELIMIT":      str(timelimit_minutes),
        "FRAME_SKIP":     str(frame_skip),
        "RESOLUTION":     render_resolution,
    }

    with open(log1, "wb") as host_log, open(log2, "wb") as join_log:
        host_proc = subprocess.Popen(
            [sys.executable, script],
            env={**common_env, "IS_HOST": "1",
                 "POLICY_JSON": json.dumps(policy_p1_dict), "OUT_FILE": out1},
            stdout=host_log, stderr=subprocess.STDOUT,
        )

        # Brief delay so host binds the port before join tries to connect
        time.sleep(1.0)

        join_proc = subprocess.Popen(
            [sys.executable, script],
            env={**common_env, "IS_HOST": "0",
                 "POLICY_JSON": json.dumps(policy_p2_dict), "OUT_FILE": out2},
            stdout=join_log, stderr=subprocess.STDOUT,
        )

        # Keep hard timeout tight to avoid hanging on broken network handshakes.
        timeout = timelimit_minutes * 60 + 35
        t0 = time.time()
        while True:
            h = host_proc.poll()
            j = join_proc.poll()
            if h is not None and j is not None:
                break
            if time.time() - t0 > timeout:
                host_proc.kill()
                join_proc.kill()
                break
            # If one side died quickly, don't leave the other side hanging forever.
            if (h is not None and j is None) or (j is not None and h is None):
                if time.time() - t0 > 8:
                    host_proc.kill()
                    join_proc.kill()
                    break
            time.sleep(0.2)

    # Load results
    def _tail(path: str, n: int = 40) -> str:
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n:]).strip()
        except Exception:
            return ""

    def _load(path: str, label: str, log_path: str) -> dict:
        if not os.path.exists(path):
            log_tail = _tail(log_path)
            if log_tail:
                return {"error": f"{label} result file missing. log:\n{log_tail}"}
            return {"error": f"{label} result file missing"}
        with open(path, "rb") as f:
            return pickle.load(f)

    r1 = _load(out1, "host", log1)
    r2 = _load(out2, "join", log2)

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
