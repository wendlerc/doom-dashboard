#!/usr/bin/env python3
"""
Record human gameplay against trained AI policies.

The human plays in one window while AI agent(s) join as opponents.
Human inputs are captured via ViZDoom SPECTATOR mode; AI runs in a
background subprocess.

Usage:
    # Play vs the competition submission model
    uv run python record_vs_ai.py \
        --model results_latest/competition_submission_v3/bc_v2_highent_best.zip

    # Play vs a model + 1 bot on a specific map
    uv run python record_vs_ai.py \
        --model trained_policies/bc_finetune_v2_highent_best/best_model.zip \
        --bots 1 --map map01

    # Play vs 2 AI copies
    uv run python record_vs_ai.py \
        --model results_latest/competition_submission_v3/bc_v2_highent_best.zip \
        --ai-copies 2

    # Just play (no recording)
    uv run python record_vs_ai.py \
        --model results_latest/competition_submission_v3/bc_v2_highent_best.zip \
        --no-save
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import vizdoom as vzd


# ─── AI player subprocess script ───────────────────────────────────
# Runs in a fresh Python process to avoid CUDA/fork issues.
_AI_PLAYER_SCRIPT = r"""
import os, sys, time, pickle, json
import numpy as np
sys.path.insert(0, os.environ["DOOM_DASH_ROOT"])

port       = int(os.environ["PORT"])
cfg_path   = os.environ["CFG_PATH"]
doom_map   = os.environ["DOOM_MAP"]
frame_skip = int(os.environ["FRAME_SKIP"])
res        = os.environ.get("RESOLUTION", "RES_320X240")
pol_json   = json.loads(os.environ["POLICY_JSON"])
out_file   = os.environ["OUT_FILE"]
player_id  = int(os.environ.get("PLAYER_ID", "2"))

import vizdoom as vzd
from doom_dashboard.config import PolicyConfig
from doom_dashboard.policies import load_policy

_RES_MAP = {
    "RES_160X120": vzd.ScreenResolution.RES_160X120,
    "RES_320X240": vzd.ScreenResolution.RES_320X240,
    "RES_640X480": vzd.ScreenResolution.RES_640X480,
}

pol = load_policy(PolicyConfig(**pol_json))

result = {"frames":[], "actions":[], "rewards":[], "game_vars":[],
          "button_names":[], "duration_s":0.0, "game_tics":0, "error": None}

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
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_screen_resolution(_RES_MAP.get(res, vzd.ScreenResolution.RES_320X240))
    g.set_render_hud(False)
    g.add_game_args(f"-join 127.0.0.1 -port {port} +viz_connect_timeout 60")
    g.add_game_args(f"+name AI_P{player_id} +colorset {player_id}")
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
        gv = _read_vars(g)
        act = pol.predict(obs, btns, game_variables=gv)
        rew = g.make_action([bool(a) for a in act], frame_skip)
        if g.is_player_dead():
            g.respawn_player()
        # Only save a small sample of frames to reduce memory
        if len(result["frames"]) < 100:
            result["frames"].append(obs)
        result["actions"].append(act.copy())
        result["rewards"].append(float(rew))
        result["game_vars"].append(gv)
    result["button_names"] = btns
    result["duration_s"]   = time.perf_counter() - t0
    result["game_tics"]    = int(g.get_episode_time())
    g.close()
except Exception as e:
    import traceback
    result["error"] = traceback.format_exc()

with open(out_file, "wb") as f:
    pickle.dump(result, f, protocol=4)
"""


def find_free_port(start: int = 5800) -> int:
    """Find a free port starting from `start`."""
    import socket
    for p in range(start, start + 200):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    return start + random.randint(0, 199)


def launch_ai_player(
    cfg_path: str,
    doom_map: str,
    model_path: str,
    port: int,
    frame_skip: int,
    player_id: int = 2,
    deterministic: bool = False,
) -> tuple[subprocess.Popen, str]:
    """Launch an AI player subprocess that joins the game."""
    tmpdir = tempfile.mkdtemp(prefix="doom_ai_")
    out_file = os.path.join(tmpdir, f"ai_p{player_id}.pkl")
    script_file = os.path.join(tmpdir, f"ai_p{player_id}.py")

    with open(script_file, "w") as f:
        f.write(_AI_PLAYER_SCRIPT)

    # Build policy config dict
    pol_dict = {
        "name": f"AI_P{player_id}",
        "type": "sb3",
        "path": str(Path(model_path).resolve()),
        "algo": "PPO",
        "device": "cuda",
        "deterministic": deterministic,
    }

    env = {
        **os.environ,
        "DOOM_DASH_ROOT": str(Path(__file__).resolve().parent),
        "PORT": str(port),
        "CFG_PATH": str(Path(cfg_path).resolve()),
        "DOOM_MAP": doom_map,
        "FRAME_SKIP": str(frame_skip),
        "RESOLUTION": "RES_320X240",
        "POLICY_JSON": json.dumps(pol_dict),
        "OUT_FILE": out_file,
        "PLAYER_ID": str(player_id),
    }

    proc = subprocess.Popen(
        [sys.executable, script_file],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, out_file


def main():
    ap = argparse.ArgumentParser(
        description="Play against trained AI policies and record your gameplay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", "-m", required=True,
                    help="Path to trained model .zip file")
    ap.add_argument("--ai-copies", type=int, default=1,
                    help="Number of AI copies to play against (default: 1)")
    ap.add_argument("--bots", type=int, default=0,
                    help="Number of additional built-in bots (default: 0)")
    ap.add_argument("--cfg", default="doom_dashboard/scenarios/deathmatch_compact.cfg",
                    help="Scenario config file")
    ap.add_argument("--map", default="map01",
                    help="Map name (default: map01)")
    ap.add_argument("--timelimit", type=float, default=5.0,
                    help="Match length in minutes (default: 5)")
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="Frame skip for human recording (default: 1)")
    ap.add_argument("--ai-frame-skip", type=int, default=4,
                    help="Frame skip for AI policy (default: 4, matches training)")
    ap.add_argument("--resolution", default="RES_1280X720",
                    help="Human player screen resolution")
    ap.add_argument("--render-hud", action="store_true",
                    help="Show HUD overlay")
    ap.add_argument("--fullscreen", action="store_true",
                    help="Launch in fullscreen mode")
    ap.add_argument("--deterministic", action="store_true",
                    help="Use deterministic AI inference (default: stochastic)")
    ap.add_argument("--output-dir", "-o", default="human_demos",
                    help="Output directory for recordings")
    ap.add_argument("--name", default=None,
                    help="Session name (default: auto-generated)")
    ap.add_argument("--no-save", action="store_true",
                    help="Play without saving recording data")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip saving annotated replay video")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    cfg_path = str(Path(args.cfg).resolve())
    doom_map = args.map.lower()
    total_players = 1 + args.ai_copies  # human + AI players

    # Generate session name
    if args.name:
        stem = re.sub(r"[^a-z0-9_.-]+", "_", args.name.lower())
    else:
        model_name = model_path.stem.replace(".", "_")[:30]
        stem = f"vs_ai_{model_name}_{doom_map}_{int(time.time())}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find a free port for multiplayer
    port = find_free_port()

    print(f"\n{'='*60}")
    print(f"  HUMAN vs AI — Doom Deathmatch")
    print(f"{'='*60}")
    print(f"  Model:      {model_path.name}")
    print(f"  AI copies:  {args.ai_copies}")
    print(f"  Bots:       {args.bots}")
    print(f"  Map:        {doom_map}")
    print(f"  Timelimit:  {args.timelimit} min")
    print(f"  Port:       {port}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Recording:  {'OFF' if args.no_save else stem}")
    print(f"{'='*60}")
    print()
    print("  Controls: Use your normal Doom keybinds.")
    print("  Click into the game window to capture mouse/keyboard.")
    print("  The AI opponent will join after a few seconds.")
    print()

    # --- Set up human player (host) ---
    res_enum = getattr(vzd.ScreenResolution, args.resolution, None)
    if res_enum is None:
        valid = sorted([n for n in dir(vzd.ScreenResolution) if n.startswith("RES_")])
        print(f"Unknown resolution '{args.resolution}'. Valid: {valid}")
        sys.exit(1)

    tracked_vars = [
        vzd.GameVariable.FRAGCOUNT,
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.DEATHCOUNT,
        vzd.GameVariable.HITCOUNT,
        vzd.GameVariable.HITS_TAKEN,
        vzd.GameVariable.DAMAGECOUNT,
        vzd.GameVariable.DAMAGE_TAKEN,
        vzd.GameVariable.HEALTH,
    ]

    def _read_vars(game: vzd.DoomGame) -> np.ndarray:
        vals = []
        for gv in tracked_vars:
            try:
                vals.append(float(game.get_game_variable(gv)))
            except Exception:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float32)

    def _to_hwc(buf: np.ndarray) -> np.ndarray:
        if buf.ndim == 3 and buf.shape[0] in (1, 3, 4) and buf.shape[-1] not in (1, 3, 4):
            return np.transpose(buf, (1, 2, 0))
        return buf

    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.set_doom_map(doom_map)

    timeout_tics = int(float(args.timelimit) * 60 * 35)
    game.set_episode_timeout(max(4200, timeout_tics))

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(res_enum)
    game.set_render_hud(args.render_hud)
    game.set_sound_enabled(True)

    # Host multiplayer game: human + N AI players
    game.add_game_args(
        f"-host {total_players} -port {port} +viz_connect_timeout 60 "
        f"+timelimit {float(max(0.1, args.timelimit)):.2f} "
        "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
        "+sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 0 +viz_nocheat 1"
    )
    if args.fullscreen:
        game.add_game_args("+fullscreen 1")
    game.add_game_args("+name Human +colorset 2")

    print("Initializing game (waiting for AI to connect)...")
    game.init()

    # Add built-in bots if requested
    for _ in range(max(0, args.bots)):
        try:
            game.send_game_command("addbot")
        except Exception:
            pass

    # --- Launch AI player subprocess(es) ---
    ai_procs = []
    for i in range(args.ai_copies):
        # Small delay between AI joins so they don't collide
        time.sleep(0.5)
        proc, out_file = launch_ai_player(
            cfg_path=cfg_path,
            doom_map=doom_map,
            model_path=str(model_path.resolve()),
            port=port,
            frame_skip=args.ai_frame_skip,
            player_id=i + 2,
            deterministic=args.deterministic,
        )
        ai_procs.append((proc, out_file))
        print(f"  AI player {i+1} launched (PID {proc.pid})")

    print(f"\nGame started! You are playing against {args.ai_copies} AI + {args.bots} bots.")
    print("Have fun!\n")

    # --- Record human gameplay ---
    button_names = [str(b) for b in game.get_available_buttons()]
    n_buttons = len(button_names)

    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    game_vars_list: List[np.ndarray] = []

    t0 = time.perf_counter()
    try:
        while not game.is_episode_finished():
            game.advance_action(max(1, args.frame_skip))
            if game.is_player_dead():
                try:
                    game.respawn_player()
                except Exception:
                    pass
            st = game.get_state()
            if st is None:
                continue

            obs = _to_hwc(st.screen_buffer)
            act_raw = game.get_last_action()
            if act_raw is None:
                act = np.zeros(n_buttons, dtype=np.float32)
            else:
                act = np.asarray(act_raw, dtype=np.float32).reshape(-1)
                if act.shape[0] != n_buttons:
                    tmp = np.zeros(n_buttons, dtype=np.float32)
                    tmp[:min(n_buttons, act.shape[0])] = act[:min(n_buttons, act.shape[0])]
                    act = tmp
            rew = float(game.get_last_reward())
            gv = _read_vars(game)

            frames.append(obs.copy())
            actions.append(act.copy())
            rewards.append(rew)
            game_vars_list.append(gv.copy())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        total_reward = float(game.get_total_reward())
        game_tics = int(game.get_episode_time())
        game.close()
    t1 = time.perf_counter()

    # --- Wait for AI processes to finish ---
    print("\nWaiting for AI processes to finish...")
    ai_stats = []
    for proc, out_file in ai_procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        # Read AI results
        try:
            with open(out_file, "rb") as f:
                ai_result = pickle.load(f)
            if ai_result.get("error"):
                print(f"  AI error: {ai_result['error'][:200]}")
            else:
                ai_gv = ai_result.get("game_vars", [])
                if ai_gv:
                    last_gv = ai_gv[-1]
                    ai_kills = int(last_gv[1]) if len(last_gv) > 1 else 0
                    ai_deaths = int(last_gv[2]) if len(last_gv) > 2 else 0
                    ai_stats.append({"kills": ai_kills, "deaths": ai_deaths})
        except Exception as e:
            print(f"  Could not read AI results: {e}")

    # --- Print results ---
    human_gv = game_vars_list[-1] if game_vars_list else np.zeros(8)
    human_kills = int(human_gv[1]) if len(human_gv) > 1 else 0
    human_deaths = int(human_gv[2]) if len(human_gv) > 2 else 0
    human_frags = int(human_gv[0]) if len(human_gv) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  MATCH RESULTS")
    print(f"{'='*60}")
    print(f"  Duration: {t1 - t0:.1f}s ({game_tics} tics)")
    print(f"  Human:  {human_kills} kills, {human_deaths} deaths (frags: {human_frags})")
    for i, stats in enumerate(ai_stats):
        print(f"  AI #{i+1}:  {stats['kills']} kills, {stats['deaths']} deaths")
    print(f"  Frames recorded: {len(frames)}")
    print(f"{'='*60}")

    # --- Save recording ---
    if not args.no_save and frames:
        frames_np = np.stack(frames, axis=0).astype(np.uint8)
        actions_np = np.stack(actions, axis=0).astype(np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        game_vars_np = np.stack(game_vars_list, axis=0).astype(np.float32)

        npz_path = out_dir / f"{stem}.npz"
        meta_path = out_dir / f"{stem}.meta.json"

        # Save actions separately first (small, survives truncation)
        actions_path = out_dir / f"{stem}.actions.npz"
        np.savez_compressed(actions_path, actions=actions_np, rewards=rewards_np, game_vars=game_vars_np)

        # Save full data
        np.savez_compressed(npz_path, frames=frames_np, actions=actions_np,
                            rewards=rewards_np, game_vars=game_vars_np)

        meta = {
            "name": stem,
            "scenario": "deathmatch_vs_ai",
            "cfg_path": cfg_path,
            "map": doom_map,
            "timelimit_minutes": args.timelimit,
            "frame_skip": args.frame_skip,
            "resolution": args.resolution,
            "render_hud": args.render_hud,
            "bots": args.bots,
            "ai_model": str(model_path.resolve()),
            "ai_copies": args.ai_copies,
            "ai_deterministic": args.deterministic,
            "button_names": button_names,
            "steps": len(frames),
            "duration_s": float(t1 - t0),
            "game_tics": game_tics,
            "total_reward": total_reward,
            "human_kills": human_kills,
            "human_deaths": human_deaths,
            "ai_stats": ai_stats,
            "vars_layout": [
                "FRAGCOUNT", "KILLCOUNT", "DEATHCOUNT", "HITCOUNT",
                "HITS_TAKEN", "DAMAGECOUNT", "DAMAGE_TAKEN", "HEALTH",
            ],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Recording saved:")
        print(f"    data:  {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"    meta:  {meta_path}")

        # Save replay video
        if not args.no_video:
            video_path = out_dir / f"{stem}.mp4"
            h, w = frames[0].shape[:2]
            fps = max(1, round(35.0 / max(1, args.frame_skip)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"    video: {video_path}")

    elif not frames:
        print("\n  No frames recorded (game ended immediately?).")
    else:
        print("\n  Recording skipped (--no-save).")

    # Clean up temp files
    for _, out_file in ai_procs:
        try:
            os.unlink(out_file)
            os.rmdir(os.path.dirname(out_file))
        except Exception:
            pass

    print("\nGG!")


if __name__ == "__main__":
    main()
