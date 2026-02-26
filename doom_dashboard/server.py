"""Flask backend for the doom-dashboard."""
from __future__ import annotations

import io
import json
import os
import random
import socket
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, request, send_from_directory

from doom_dashboard.config import Config
from doom_dashboard.alignment import policy_matches_scenario, infer_policy_scenario_key, scenario_key


def create_app(config: Config) -> Flask:
    samples_dir = Path(config.dashboard.output_dir)

    app = Flask(__name__, static_folder=None)

    # ── static dashboard files ────────────────────────────────────
    static_dir = Path(__file__).parent / "static"

    @app.route("/")
    def index():
        return send_from_directory(str(static_dir), "index.html")

    @app.route("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(str(static_dir), filename)

    # ── sample videos ─────────────────────────────────────────────
    @app.route("/videos/<path:filename>")
    def serve_video(filename: str):
        return send_from_directory(str(samples_dir), filename)

    # ── samples index ─────────────────────────────────────────────
    @app.route("/api/samples")
    def api_samples():
        index_file = samples_dir / "index.json"
        if not index_file.exists():
            return jsonify({"samples": [], "message": "No samples generated yet. Run: generate-samples"}), 200
        with open(index_file) as f:
            data = json.load(f)
        return jsonify({"samples": data})

    # ── config schema ─────────────────────────────────────────────
    @app.route("/api/config")
    def api_config():
        return jsonify({
            "scenarios": [
                {"name": s.name, "cfg": s.cfg, "frame_skip": s.frame_skip}
                for s in config.scenarios
            ],
            "policies": [
                {
                    "name": p.name,
                    "type": p.type,
                    "path": p.path,
                    "expected_scenario": infer_policy_scenario_key(p),
                }
                for p in config.policies
            ],
            "dataset": {
                "output_dir": config.dataset.output_dir,
                "total_hours": config.dataset.total_hours,
                "scenario_ratios": config.dataset.scenario_ratios,
                "policy_ratios": config.dataset.policy_ratios,
                "frame_skip": config.dataset.frame_skip,
                "render_resolution": config.dataset.render_resolution,
                "shard_size_mb": config.dataset.shard_size_mb,
                "num_workers": config.dataset.num_workers,
            },
        })

    # ── dataset generation job ────────────────────────────────────
    _job_state: Dict[str, Any] = {"running": False, "hours_done": 0.0, "total": 0.0, "log": []}
    _job_lock = threading.Lock()

    @app.route("/api/launch-dataset", methods=["POST"])
    def launch_dataset():
        with _job_lock:
            if _job_state["running"]:
                return jsonify({"ok": False, "error": "A job is already running"}), 409

        body = request.json or {}

        # Patch config in-place with request overrides
        ds = config.dataset
        ds.total_hours = float(body.get("total_hours", ds.total_hours))
        ds.scenario_ratios = body.get("scenario_ratios", ds.scenario_ratios)
        ds.policy_ratios = body.get("policy_ratios", ds.policy_ratios)
        ds.frame_skip = int(body.get("frame_skip", ds.frame_skip))
        ds.render_resolution = body.get("render_resolution", ds.render_resolution)
        ds.shard_size_mb = int(body.get("shard_size_mb", ds.shard_size_mb))
        ds.num_workers = body.get("num_workers") or ds.num_workers
        ds.output_dir = body.get("output_dir", ds.output_dir)

        _job_state["running"] = True
        _job_state["hours_done"] = 0.0
        _job_state["total"] = ds.total_hours
        _job_state["log"] = [f"Dataset job started: {ds.total_hours:.2f} h"]

        def _progress(done, total):
            with _job_lock:
                _job_state["hours_done"] = done
                _job_state["log"].append(f"Progress: {done:.3f}/{total:.2f} h")

        def _run():
            from doom_dashboard.dataset_gen import generate_dataset
            try:
                generate_dataset(config=config, progress_callback=_progress)
                with _job_lock:
                    _job_state["log"].append("Job completed successfully.")
            except Exception as e:
                with _job_lock:
                    _job_state["log"].append(f"Job failed: {e}")
            finally:
                with _job_lock:
                    _job_state["running"] = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return jsonify({"ok": True})

    @app.route("/api/dataset-status")
    def dataset_status():
        with _job_lock:
            return jsonify(dict(_job_state))

    # ── SSE progress stream ───────────────────────────────────────
    @app.route("/api/dataset-stream")
    def dataset_stream():
        def _gen():
            last_log_len = 0
            while True:
                with _job_lock:
                    state = dict(_job_state)
                    new_logs = state["log"][last_log_len:]
                    last_log_len = len(state["log"])
                for msg in new_logs:
                    yield f"data: {json.dumps({'log': msg, 'hours_done': state['hours_done'], 'total': state['total'], 'running': state['running']})}\n\n"
                if not state["running"] and last_log_len > 0:
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                import time
                time.sleep(0.5)

        return Response(_gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── preview generator ─────────────────────────────────────────
    previews_dir = Path(samples_dir).parent / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    def _find_preview_port(start: int = 7200, end: int = 7600, step: int = 10) -> int:
        for p in range(start, end, step):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                pass
            finally:
                s.close()
        raise RuntimeError(f"No free preview port in range [{start}, {end})")

    @app.route("/previews/<path:filename>")
    def serve_preview(filename: str):
        return send_from_directory(str(previews_dir), filename)

    _preview_state: Dict[str, Any] = {"running": False, "video": None, "error": None, "meta": None}
    _preview_lock = threading.Lock()

    @app.route("/api/generate-preview", methods=["POST"])
    def generate_preview():
        with _preview_lock:
            if _preview_state["running"]:
                return jsonify({"ok": False, "error": "A preview is already generating"}), 409

        body = request.json or {}
        scenario_name   = body.get("scenario")
        policy_name     = body.get("policy")
        frame_skip      = max(1, int(body.get("frame_skip", 4)))
        resolution      = body.get("resolution", "RES_320X240")
        timelimit_secs  = max(5, min(int(body.get("timelimit_secs", 120)), 1800))
        render_hud      = bool(body.get("render_hud", False))
        map_name        = (body.get("map") or "").strip() or None

        # Validate
        sc = next((s for s in config.scenarios if s.name == scenario_name), None)
        pol_cfg = next((p for p in config.policies if p.name == policy_name), None)
        if sc is None or pol_cfg is None:
            return jsonify({"ok": False, "error": f"Unknown scenario '{scenario_name}' or policy '{policy_name}'"}), 400
        if pol_cfg.type != "random" and not policy_matches_scenario(pol_cfg, sc):
            return jsonify({"ok": False, "error": "Selected policy is not aligned with that scenario's training setup."}), 400

        with _preview_lock:
            _preview_state["running"] = True
            _preview_state["video"]   = None
            _preview_state["error"]   = None
            _preview_state["meta"]    = None

        def _run():
            from doom_dashboard.policies import load_policy
            from doom_dashboard.rollout import rollout_episode, EpisodeData
            from doom_dashboard.multiplayer_rollout import rollout_multiplayer_episode
            from doom_dashboard.annotate import annotate_and_encode
            import time as _time
            policy = None
            try:
                sc_key = scenario_key(sc)
                if sc_key in {"multi_duel", "cig"}:
                    from doom_dashboard.mp_dataset_gen import MP_SCENARIOS
                    mp_meta = MP_SCENARIOS["multi_duel" if sc_key == "multi_duel" else "cig_fullaction"]
                    chosen_map = map_name or random.choice(mp_meta["maps"])
                    p1 = dict(
                        name=pol_cfg.name,
                        type=pol_cfg.type,
                        path=pol_cfg.path,
                        algo=pol_cfg.algo,
                        arch=pol_cfg.arch,
                        action_size=pol_cfg.action_size,
                        device=pol_cfg.device,
                    )
                    p2 = dict(name="Random", type="random", path=None, algo="PPO", arch="DuelQNet", action_size=None, device="auto")
                    ep_mp = rollout_multiplayer_episode(
                        cfg_path=sc.cfg_path(),
                        scenario_name=sc_key,
                        doom_map=chosen_map,
                        policy_p1_dict=p1,
                        policy_p2_dict=p2,
                        timelimit_minutes=float(timelimit_secs) / 60.0,
                        frame_skip=frame_skip,
                        render_resolution=resolution,
                        render_hud=render_hud,
                        port=_find_preview_port(),
                    )
                    ep = EpisodeData(
                        frames=ep_mp.frames_p1,
                        actions=ep_mp.actions_p1,
                        button_names=ep_mp.button_names,
                        rewards=ep_mp.rewards_p1,
                        game_vars=ep_mp.game_vars_p1,
                        total_reward=float(sum(ep_mp.rewards_p1)),
                        scenario_name=sc.name,
                        policy_name=f"{pol_cfg.name} vs Random",
                        cfg_path=ep_mp.cfg_path,
                        frame_skip=frame_skip,
                        steps=len(ep_mp.actions_p1),
                        duration_s=float(ep_mp.duration_s),
                        game_tics=int(ep_mp.game_tics),
                        metadata=dict(mode="multiplayer", opponent="Random"),
                    )
                else:
                    policy = load_policy(pol_cfg)
                    ep = rollout_episode(
                        scenario=sc,
                        policy=policy,
                        render_resolution=resolution,
                        frame_skip=frame_skip,
                        max_steps=int(timelimit_secs * 35 / frame_skip),
                        render_hud=render_hud,
                        doom_map=map_name,
                    )
                fname = f"preview_{scenario_name}__{policy_name}_{int(_time.time())}.mp4".lower().replace(" ", "_").replace("-", "")
                out_path = previews_dir / fname
                # Match encoded playback speed to in-game time at current frame_skip.
                preview_fps = max(1, round(35.0 / max(1, frame_skip), 2))
                annotate_and_encode(ep, str(out_path), fps=preview_fps)
                with _preview_lock:
                    _preview_state["video"] = fname
                    _preview_state["meta"]  = {
                        "scenario": scenario_name, "policy": policy_name,
                        "steps": ep.steps, "total_reward": ep.total_reward,
                        "game_tics": ep.game_tics, "duration_s": ep.duration_s,
                        "frame_skip": frame_skip, "resolution": resolution,
                        "map": map_name,
                        "fps": preview_fps,
                        "button_names": ep.button_names,
                    }
            except Exception as e:
                with _preview_lock:
                    _preview_state["error"] = str(e)
            finally:
                if policy is not None:
                    policy.close()
                with _preview_lock:
                    _preview_state["running"] = False

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/preview-status")
    def preview_status():
        with _preview_lock:
            return jsonify(dict(_preview_state))

    return app
