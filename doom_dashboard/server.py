"""Flask backend for the doom-dashboard."""
from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, request, send_from_directory

from doom_dashboard.config import Config


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
                {"name": p.name, "type": p.type, "path": p.path}
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
        frame_skip      = int(body.get("frame_skip", 4))
        resolution      = body.get("resolution", "RES_320X240")
        timelimit_secs  = int(body.get("timelimit_secs", 30))
        render_hud      = bool(body.get("render_hud", False))
        map_name        = (body.get("map") or "").strip() or None

        # Validate
        sc = next((s for s in config.scenarios if s.name == scenario_name), None)
        pol_cfg = next((p for p in config.policies if p.name == policy_name), None)
        if sc is None or pol_cfg is None:
            return jsonify({"ok": False, "error": f"Unknown scenario '{scenario_name}' or policy '{policy_name}'"}), 400

        with _preview_lock:
            _preview_state["running"] = True
            _preview_state["video"]   = None
            _preview_state["error"]   = None
            _preview_state["meta"]    = None

        def _run():
            from doom_dashboard.policies import load_policy
            from doom_dashboard.rollout import rollout_episode
            from doom_dashboard.annotate import annotate_and_encode
            import time as _time
            policy = None
            try:
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
