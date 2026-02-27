#!/usr/bin/env python3
"""Generate a test video using stochastic inference to verify diverse behavior."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from train_overnight_dm import DeathmatchMacroEnv, ShapingCfg, build_macro_actions, parse_available_buttons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to model .zip")
    ap.add_argument("--cfg", default="doom_dashboard/scenarios/deathmatch_compact.cfg")
    ap.add_argument("--steps", type=int, default=1500, help="Max steps per episode")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", default="results_latest/test_stochastic.mp4")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bots", type=int, default=1)
    args = ap.parse_args()

    # Load model
    model_path = str(args.model)
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, device=args.device)
        is_recurrent = True
        print(f"Loaded RecurrentPPO, ent_coef={model.ent_coef}")
    except Exception:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=args.device)
        is_recurrent = False
        print(f"Loaded PPO, ent_coef={model.ent_coef}")

    # Get action names from meta or cfg
    meta_path = Path(args.model).with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        action_names = meta.get("action_names", [])
    else:
        btn_names = parse_available_buttons(args.cfg)
        action_names, _ = build_macro_actions(btn_names)

    # Create env
    env = DeathmatchMacroEnv(
        cfg_path=args.cfg,
        obs_shape=(120, 160),
        frame_skip=4,
        bots=args.bots,
        maps=["map01"],
        timelimit_minutes=2.0,
        shaping=ShapingCfg(),
        spawn_farthest=True,
        no_autoaim=False,
    )

    all_frames = []
    all_actions = []
    total_kills = 0
    total_deaths = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        ep_actions = []
        frames = []

        for step in range(args.steps):
            obs_dict = {
                "screen": np.transpose(obs["screen"], (2, 0, 1))[None],
                "gamevars": obs["gamevars"][None],
            }

            if is_recurrent:
                action, lstm_states = model.predict(
                    obs_dict, state=lstm_states, episode_start=episode_starts,
                    deterministic=args.deterministic
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs_dict, deterministic=args.deterministic)

            act_idx = int(action.item()) if hasattr(action, 'item') else int(action[0])
            ep_actions.append(act_idx)

            # Annotate frame with action name
            frame = obs["screen"].copy()
            act_name = action_names[act_idx] if act_idx < len(action_names) else f"act_{act_idx}"
            # Add text overlay
            cv2.putText(frame, act_name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            # Add game vars overlay
            gv = obs["gamevars"]
            kills = int(gv[1]) if len(gv) > 1 else 0  # KILLCOUNT
            deaths = int(gv[2]) if len(gv) > 2 else 0  # DEATHCOUNT
            cv2.putText(frame, f"K:{kills} D:{deaths}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            frames.append(frame)

            obs, _, term, trunc, info = env.step(act_idx)
            if term or trunc:
                break

        # Episode stats
        action_counter = Counter(ep_actions)
        n_unique = len(action_counter)
        top3 = action_counter.most_common(3)
        top3_str = ", ".join(f"{action_names[a]}:{c}" for a, c in top3)

        # Get final kills/deaths
        final_kills = int(obs["gamevars"][1]) if len(obs["gamevars"]) > 1 else 0
        final_deaths = int(obs["gamevars"][2]) if len(obs["gamevars"]) > 2 else 0
        total_kills += final_kills
        total_deaths += final_deaths

        print(f"Episode {ep+1}: {len(ep_actions)} steps, {n_unique} unique actions, "
              f"K:{final_kills} D:{final_deaths} | top3: {top3_str}")

        all_frames.extend(frames)
        all_actions.extend(ep_actions)

    env.close()

    # Overall stats
    overall = Counter(all_actions)
    print(f"\n=== Overall: {args.episodes} episodes, {total_kills} kills, {total_deaths} deaths ===")
    print(f"Unique actions: {len(overall)}/{len(action_names)}")
    for idx, count in overall.most_common(10):
        name = action_names[idx] if idx < len(action_names) else f"act_{idx}"
        print(f"  {name}: {count} ({100*count/len(all_actions):.1f}%)")

    # Write video
    if all_frames:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 30, (w, h))
        for f in all_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\nSaved video: {args.output} ({len(all_frames)} frames)")


if __name__ == "__main__":
    main()
