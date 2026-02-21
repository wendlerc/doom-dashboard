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

    return app
