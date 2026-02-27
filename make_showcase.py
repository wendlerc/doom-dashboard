#!/usr/bin/env python3
"""
Generate showcase videos: trained model side-by-side with random baseline.
Uses frame_skip=1 for smooth video with no skipped frames.

Usage:
    uv run python make_showcase.py \
        results_latest/competition_submission_v3/bc_v2_highent_best.zip \
        --output results_latest/competition_submission_v3/showcase_vs_random.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from train_overnight_dm import DeathmatchMacroEnv, ShapingCfg, build_macro_actions, parse_available_buttons


def run_episode(model, is_recurrent, env, action_names, max_steps=2100,
                deterministic=False, frame_skip_video=1, frame_skip_action=4,
                label="Model"):
    """Run one episode, recording EVERY frame (frame_skip=1 for video).

    The policy acts every `frame_skip_action` frames (matching training),
    but we record every single frame for smooth video.
    """
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    frames = []
    actions_taken = []
    current_action = 0  # NOOP
    steps_since_action = 0

    for step in range(max_steps):
        # Record frame BEFORE stepping
        frame = obs["screen"].copy()

        # Get game vars for overlay
        gv = obs["gamevars"]
        kills = int(gv[1]) if len(gv) > 1 else 0
        deaths = int(gv[2]) if len(gv) > 2 else 0
        health = int(gv[7]) if len(gv) > 7 else 0

        # Decide action every frame_skip_action steps
        if steps_since_action == 0:
            if model is not None:
                obs_dict = {
                    "screen": np.transpose(obs["screen"], (2, 0, 1))[None],
                    "gamevars": obs["gamevars"][None],
                }
                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs_dict, state=lstm_states, episode_start=episode_starts,
                        deterministic=deterministic
                    )
                    episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs_dict, deterministic=deterministic)
                current_action = int(action.item()) if hasattr(action, 'item') else int(action[0])
            else:
                # Random policy
                current_action = env.action_space.sample()

        actions_taken.append(current_action)
        act_name = action_names[current_action] if current_action < len(action_names) else f"act_{current_action}"

        # Annotate frame
        h, w = frame.shape[:2]
        # Semi-transparent header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Label
        cv2.putText(frame, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Stats
        cv2.putText(frame, f"K:{kills}  D:{deaths}  HP:{health}", (5, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # Action
        cv2.putText(frame, act_name, (w - 5 - len(act_name) * 7, 14),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        # Step counter
        cv2.putText(frame, f"t={step}", (w - 50, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        frames.append(frame)

        # Step environment (1 tic at a time for smooth video)
        obs, _, term, trunc, _ = env.step(current_action)
        steps_since_action += 1
        if steps_since_action >= frame_skip_action:
            steps_since_action = 0

        if term or trunc:
            # Record final frame
            final_frame = obs["screen"].copy()
            final_gv = obs["gamevars"]
            final_kills = int(final_gv[1]) if len(final_gv) > 1 else 0
            final_deaths = int(final_gv[2]) if len(final_gv) > 2 else 0
            overlay = final_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
            final_frame = cv2.addWeighted(overlay, 0.5, final_frame, 0.5, 0)
            cv2.putText(final_frame, f"{label} - FINAL", (5, 14),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(final_frame, f"K:{final_kills}  D:{final_deaths}", (5, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            frames.append(final_frame)
            break

    # Final stats
    final_gv = obs["gamevars"]
    kills = int(final_gv[1]) if len(final_gv) > 1 else 0
    deaths = int(final_gv[2]) if len(final_gv) > 2 else 0
    unique_actions = len(set(actions_taken))

    return frames, kills, deaths, unique_actions, Counter(actions_taken)


def make_side_by_side(frames_left, frames_right, h, w):
    """Combine two frame lists into side-by-side video frames."""
    combined = []
    max_len = max(len(frames_left), len(frames_right))
    # Add a thin separator bar
    sep_w = 2

    for i in range(max_len):
        fl = frames_left[min(i, len(frames_left) - 1)]
        fr = frames_right[min(i, len(frames_right) - 1)]
        # Resize both to same height if needed
        fl = cv2.resize(fl, (w, h))
        fr = cv2.resize(fr, (w, h))
        sep = np.zeros((h, sep_w, 3), dtype=np.uint8)
        sep[:, :] = [128, 128, 128]  # gray separator
        combined.append(np.concatenate([fl, sep, fr], axis=1))

    return combined


def main():
    ap = argparse.ArgumentParser(description="Generate side-by-side showcase: Model vs Random")
    ap.add_argument("model", help="Path to model .zip file")
    ap.add_argument("--cfg", default="doom_dashboard/scenarios/deathmatch_compact.cfg")
    ap.add_argument("--map", default="map01")
    ap.add_argument("--bots", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    ap.add_argument("--max-steps", type=int, default=2100,
                    help="Max steps per episode (2100 @ 35fps = 60s)")
    ap.add_argument("--frame-skip-action", type=int, default=4,
                    help="How often the policy chooses a new action (matches training)")
    ap.add_argument("--output", "-o", default="results_latest/showcase_vs_random.mp4")
    ap.add_argument("--fps", type=int, default=35, help="Video FPS (35 = real-time Doom)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--model-only", action="store_true",
                    help="Only show model (no side-by-side with random)")
    ap.add_argument("--model2", default=None,
                    help="Second model for model-vs-model comparison (replaces random)")
    ap.add_argument("--label1", default="Trained Agent", help="Label for first model")
    ap.add_argument("--label2", default=None, help="Label for second model / random")
    args = ap.parse_args()

    # Load model
    model_path = Path(args.model)
    print(f"Loading model: {model_path.name}")
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(model_path), device=args.device)
        is_recurrent = True
    except Exception:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device=args.device)
        is_recurrent = False
    print(f"  Type: {'RecurrentPPO' if is_recurrent else 'PPO'}, ent_coef={model.ent_coef}")

    # Get action names
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        action_names = meta.get("action_names", [])
    else:
        btn_names = parse_available_buttons(args.cfg)
        action_names, _ = build_macro_actions(btn_names)

    # Load optional second model
    model2 = None
    is_recurrent2 = False
    action_names2 = action_names  # default: same actions
    if args.model2:
        model2_path = Path(args.model2)
        print(f"Loading model2: {model2_path.name}")
        try:
            from sb3_contrib import RecurrentPPO
            model2 = RecurrentPPO.load(str(model2_path), device=args.device)
            is_recurrent2 = True
        except Exception:
            from stable_baselines3 import PPO
            model2 = PPO.load(str(model2_path), device=args.device)
            is_recurrent2 = False
        print(f"  Type: {'RecurrentPPO' if is_recurrent2 else 'PPO'}, ent_coef={model2.ent_coef}")
        meta2_path = model2_path.with_suffix(".meta.json")
        if meta2_path.exists():
            with open(meta2_path) as f:
                meta2 = json.load(f)
            action_names2 = meta2.get("action_names", action_names)

    all_combined_frames = []

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep+1}/{args.episodes} ---")

        # Run trained model
        print("  Running trained model...")
        env_model = DeathmatchMacroEnv(
            cfg_path=args.cfg, obs_shape=(120, 160), frame_skip=1,
            bots=args.bots, maps=[args.map], timelimit_minutes=2.0,
            shaping=ShapingCfg(), spawn_farthest=True, no_autoaim=False,
        )
        model_frames, model_kills, model_deaths, model_unique, model_dist = run_episode(
            model, is_recurrent, env_model, action_names,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            frame_skip_action=args.frame_skip_action,
            label=args.label1,
        )
        env_model.close()
        print(f"  Model: {len(model_frames)} frames, K:{model_kills} D:{model_deaths}, "
              f"{model_unique} unique actions")

        if not args.model_only:
            # Run second agent (model2 or random baseline)
            if model2 is not None:
                right_label = args.label2 or "Model 2"
                print(f"  Running second model ({right_label})...")
                env_right = DeathmatchMacroEnv(
                    cfg_path=args.cfg, obs_shape=(120, 160), frame_skip=1,
                    bots=args.bots, maps=[args.map], timelimit_minutes=2.0,
                    shaping=ShapingCfg(), spawn_farthest=True, no_autoaim=False,
                )
                random_frames, random_kills, random_deaths, random_unique, random_dist = run_episode(
                    model2, is_recurrent2, env_right, action_names2,
                    max_steps=args.max_steps,
                    deterministic=args.deterministic,
                    frame_skip_action=args.frame_skip_action,
                    label=right_label,
                )
                env_right.close()
            else:
                right_label = args.label2 or "Random Agent"
                print(f"  Running random baseline...")
                env_right = DeathmatchMacroEnv(
                    cfg_path=args.cfg, obs_shape=(120, 160), frame_skip=1,
                    bots=args.bots, maps=[args.map], timelimit_minutes=2.0,
                    shaping=ShapingCfg(), spawn_farthest=True, no_autoaim=False,
                )
                random_frames, random_kills, random_deaths, random_unique, random_dist = run_episode(
                    None, False, env_right, action_names,
                    max_steps=args.max_steps,
                    frame_skip_action=args.frame_skip_action,
                    label=right_label,
                )
                env_right.close()
            print(f"  Right: {len(random_frames)} frames, K:{random_kills} D:{random_deaths}, "
                  f"{random_unique} unique actions")

            # Combine side by side
            h, w = 120, 160
            combined = make_side_by_side(model_frames, random_frames, h, w)
            all_combined_frames.extend(combined)
        else:
            all_combined_frames.extend(model_frames)

    # Write video
    if all_combined_frames:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        h, w = all_combined_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for f in all_combined_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        duration_s = len(all_combined_frames) / args.fps
        print(f"\nSaved: {args.output}")
        print(f"  {len(all_combined_frames)} frames, {duration_s:.1f}s @ {args.fps}fps")
        print(f"  Resolution: {w}x{h}")


if __name__ == "__main__":
    main()
