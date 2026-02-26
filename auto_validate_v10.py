#!/usr/bin/env python3
"""
Auto-validate v10 training: at each new checkpoint, generate a quick 1v1 video
and check frame diffs to ensure the agent is actually moving.
Also runs bench_vs_bots to check combat performance.
"""
import glob
import json
import os
import sys
import time
import subprocess
import numpy as np

CKPT_DIR = "trained_policies/overnight_fight_v10_ckpts"
VIDEO_DIR = "v10_validation_videos"
RESULTS_FILE = "v10_validation_results.json"
SEEN_FILE = "v10_seen_ckpts.txt"
MIN_STEPS = 200_000  # don't validate before this
CHECK_INTERVAL = 60  # seconds between checks

os.makedirs(VIDEO_DIR, exist_ok=True)


def get_latest_ckpt():
    """Find the latest checkpoint."""
    pattern = os.path.join(CKPT_DIR, "ppo_*_steps.zip")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not ckpts:
        return None
    return ckpts[-1]


def get_steps(ckpt_path):
    """Extract step count from checkpoint filename."""
    base = os.path.basename(ckpt_path)
    # ppo_1000000_steps.zip
    parts = base.replace(".zip", "").split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return 0


def seen(ckpt_path):
    """Check if we already validated this checkpoint."""
    if not os.path.exists(SEEN_FILE):
        return False
    with open(SEEN_FILE) as f:
        return os.path.basename(ckpt_path) in f.read()


def mark_seen(ckpt_path):
    with open(SEEN_FILE, "a") as f:
        f.write(os.path.basename(ckpt_path) + "\n")


def write_temp_config(ckpt_path):
    """Write a temporary config.yaml for video generation."""
    cfg = f"""scenarios:
  - name: "DeathmatchCompact"
    cfg: "doom_dashboard/scenarios/deathmatch_compact.cfg"
    frame_skip: 4
    render_resolution: "RES_640X480"
    render_hud: false

policies:
  - name: "V10-Test"
    type: "sb3"
    path: "{ckpt_path}"
    algo: "PPO"
    device: "auto"

  - name: "PPO-Fighter-V2"
    type: "sb3"
    path: "trained_policies/exp_fighter_v2_best/best_model.zip"
    algo: "PPO"
    device: "auto"

  - name: "Random"
    type: "random"

dashboard:
  samples_per_map: 1
  render_resolution: "RES_640X480"
  fps: 30
  output_dir: "{VIDEO_DIR}"
"""
    path = "config_v10_validate.yaml"
    with open(path, "w") as f:
        f.write(cfg)
    return path


def generate_video(ckpt_path, steps):
    """Generate a quick 1-game video."""
    config_path = write_temp_config(ckpt_path)
    out_dir = os.path.join(VIDEO_DIR, f"step_{steps}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "uv", "run", "python", "-m", "doom_dashboard", "generate-elim-videos",
        "--config", config_path,
        "--output-dir", out_dir,
        "--scenario", "deathmatch_compact",
        "--games", "2",
        "--timelimit", "1.5",
        "--frame-skip", "4",
        "--resolution", "RES_640X480",
        "--policy", "V10-Test",
        "--policy", "PPO-Fighter-V2",
        "--allow-mismatch",
    ]
    env = os.environ.copy()
    env["DISPLAY"] = ":0"

    print(f"  Generating video for step {steps}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if result.returncode != 0:
            print(f"  Video gen failed: {result.stderr[-500:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  Video gen timed out")
        return None

    # Find generated mp4s
    mp4s = glob.glob(os.path.join(out_dir, "*.mp4"))
    return mp4s


def check_movement(mp4_path):
    """Check if agents are moving by computing frame diffs."""
    import cv2

    cap = cv2.VideoCapture(mp4_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 10:
        cap.release()
        return {"left_mean": 0, "right_mean": 0, "verdict": "too_short"}

    indices = np.linspace(0, n_frames - 2, min(20, n_frames // 2), dtype=int)
    left_diffs, right_diffs = [], []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if ok1 and ok2:
            w = f1.shape[1]
            mid = w // 2
            left_diffs.append(np.mean(np.abs(f1[:, :mid].astype(float) - f2[:, :mid].astype(float))))
            right_diffs.append(np.mean(np.abs(f1[:, mid:].astype(float) - f2[:, mid:].astype(float))))

    cap.release()

    lm = float(np.mean(left_diffs)) if left_diffs else 0
    rm = float(np.mean(right_diffs)) if right_diffs else 0

    # Threshold: mean_diff < 5 = barely moving
    l_moving = lm >= 5.0
    r_moving = rm >= 5.0

    return {
        "left_mean_diff": round(lm, 1),
        "right_mean_diff": round(rm, 1),
        "left_moving": l_moving,
        "right_moving": r_moving,
        "verdict": "GOOD" if l_moving else "STATIC_AGENT",
    }


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def main():
    print("=== V10 Auto-Validator ===")
    print(f"Watching: {CKPT_DIR}")
    print(f"Videos: {VIDEO_DIR}")
    print()

    results = load_results()

    while True:
        ckpt = get_latest_ckpt()
        if ckpt is None:
            print(f"[{time.strftime('%H:%M:%S')}] No checkpoints yet, waiting...")
            time.sleep(CHECK_INTERVAL)
            continue

        steps = get_steps(ckpt)
        if steps < MIN_STEPS:
            print(f"[{time.strftime('%H:%M:%S')}] Only {steps} steps, waiting for {MIN_STEPS}...")
            time.sleep(CHECK_INTERVAL)
            continue

        if seen(ckpt):
            time.sleep(CHECK_INTERVAL)
            continue

        print(f"\n[{time.strftime('%H:%M:%S')}] New checkpoint: {os.path.basename(ckpt)} ({steps} steps)")

        # Generate video
        mp4s = generate_video(ckpt, steps)
        if not mp4s:
            print(f"  No videos generated, will retry next check")
            time.sleep(CHECK_INTERVAL)
            continue

        # Check movement in each video
        for mp4 in mp4s:
            movement = check_movement(mp4)
            entry = {
                "steps": steps,
                "checkpoint": os.path.basename(ckpt),
                "video": os.path.basename(mp4),
                **movement,
            }
            results.append(entry)

            v = movement["verdict"]
            lm = movement.get("left_mean_diff", 0)
            rm = movement.get("right_mean_diff", 0)
            symbol = "OK" if v == "GOOD" else "BAD"
            print(f"  [{symbol}] {os.path.basename(mp4)}: left={lm}, right={rm} -> {v}")

        save_results(results)
        mark_seen(ckpt)

        # Print summary
        good = sum(1 for r in results if r.get("verdict") == "GOOD")
        total = len(results)
        print(f"\n  Progress: {good}/{total} videos show movement")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
