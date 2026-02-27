#!/usr/bin/env python3
"""Diagnose policy collapse: check action distributions, entropy, and stochastic vs deterministic."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import vizdoom as vzd

# Reuse macro action builder from training
from train_overnight_dm import build_macro_actions, DeathmatchMacroEnv, ShapingCfg, parse_available_buttons


def load_model(model_path: str, device="cuda"):
    """Load model, auto-detect RecurrentPPO vs PPO."""
    model_path = str(model_path)
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, device=device)
        return model, True
    except Exception:
        pass
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)
        return model, False
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)


def get_action_names(meta_path: str):
    """Load action names from meta.json."""
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("action_names", [f"action_{i}" for i in range(meta.get("action_space_n", 19))])


def analyze_logits(model, env, is_recurrent, n_steps=200):
    """Run model and collect raw logits/action distributions."""
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    all_logits = []
    all_probs = []
    det_actions = []
    stoch_actions = []

    for step in range(n_steps):
        obs_dict = {
            "screen": np.transpose(obs["screen"], (2, 0, 1))[None],
            "gamevars": obs["gamevars"][None],
        }
        obs_tensor = {}
        for k, v in obs_dict.items():
            obs_tensor[k] = torch.as_tensor(v, dtype=torch.float32).to(model.device)

        # Get raw logits from the policy network
        with torch.no_grad():
            if is_recurrent:
                # For RecurrentPPO, we need to get the distribution
                if lstm_states is not None:
                    th_states = (
                        torch.as_tensor(lstm_states[0]).to(model.device),
                        torch.as_tensor(lstm_states[1]).to(model.device),
                    )
                else:
                    th_states = None
                ep_starts = torch.as_tensor(episode_starts).float().to(model.device)

                # Get deterministic action
                det_action, new_lstm = model.predict(
                    obs_dict, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
                # Get stochastic action (re-run)
                stoch_action, _ = model.predict(
                    obs_dict, state=lstm_states, episode_start=episode_starts, deterministic=False
                )

                # Try to get logits via policy forward pass
                try:
                    features = model.policy.extract_features(obs_tensor)
                    if hasattr(model.policy, 'lstm_actor'):
                        # pi features through LSTM
                        pi_features = features  # simplified
                    latent_pi, latent_vf, new_states = model.policy._process_sequence(
                        features, th_states, ep_starts, model.policy.lstm_actor
                    )
                    dist = model.policy.get_distribution(
                        model.policy.mlp_extractor.forward_actor(latent_pi)
                    )
                    logits = dist.distribution.logits
                    probs = torch.softmax(logits, dim=-1)
                    all_logits.append(logits.cpu().numpy().flatten())
                    all_probs.append(probs.cpu().numpy().flatten())
                except Exception as e:
                    if step == 0:
                        print(f"  [warn] Could not extract logits: {e}")

                lstm_states = new_lstm
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                det_action, _ = model.predict(obs_dict, deterministic=True)
                stoch_action, _ = model.predict(obs_dict, deterministic=False)

                try:
                    features = model.policy.extract_features(obs_tensor)
                    latent_pi = model.policy.mlp_extractor.forward_actor(features)
                    dist = model.policy.get_distribution(latent_pi)
                    logits = dist.distribution.logits
                    probs = torch.softmax(logits, dim=-1)
                    all_logits.append(logits.cpu().numpy().flatten())
                    all_probs.append(probs.cpu().numpy().flatten())
                except Exception as e:
                    if step == 0:
                        print(f"  [warn] Could not extract logits: {e}")

        det_actions.append(int(det_action.item()) if hasattr(det_action, 'item') else int(det_action[0]))
        stoch_actions.append(int(stoch_action.item()) if hasattr(stoch_action, 'item') else int(stoch_action[0]))

        # Step env with stochastic action (more diverse exploration)
        action_to_take = stoch_actions[-1]
        obs, _, term, trunc, _ = env.step(action_to_take)
        if term or trunc:
            obs, _ = env.reset()
            if is_recurrent:
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)

    return det_actions, stoch_actions, all_logits, all_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to model .zip")
    ap.add_argument("--cfg", default="doom_dashboard/scenarios/deathmatch_compact.cfg")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model_path = Path(args.model)
    meta_path = model_path.with_suffix(".meta.json")

    # Load action names
    action_names = None
    if meta_path.exists():
        action_names = get_action_names(str(meta_path))
    else:
        # Try to get from cfg
        game = vzd.DoomGame()
        game.load_config(args.cfg)
        btn_names = [str(b).split(".")[-1] for b in game.get_available_buttons()]
        macro_names, _ = build_macro_actions(btn_names)
        action_names = macro_names
        game.close()

    print(f"\n{'='*60}")
    print(f"Diagnosing: {model_path.name}")
    print(f"{'='*60}")

    # Load model
    model, is_recurrent = load_model(str(model_path), device=args.device)
    model_type = "RecurrentPPO (LSTM)" if is_recurrent else "PPO"
    print(f"Model type: {model_type}")

    # Check entropy coefficient from model
    if hasattr(model, 'ent_coef'):
        print(f"Entropy coefficient: {model.ent_coef}")

    # Create env
    env = DeathmatchMacroEnv(
        cfg_path=args.cfg,
        obs_shape=(120, 160),
        frame_skip=4,
        bots=1,
        maps=["map01"],
        timelimit_minutes=2.0,
        shaping=ShapingCfg(),
        spawn_farthest=True,
        no_autoaim=False,
    )

    print(f"\nRunning {args.steps} steps...")
    det_actions, stoch_actions, all_logits, all_probs = analyze_logits(
        model, env, is_recurrent, n_steps=args.steps
    )
    env.close()

    # --- Report ---
    n_actions = len(action_names) if action_names else 19

    # Deterministic action distribution
    det_counter = Counter(det_actions)
    print(f"\n--- Deterministic Actions ({len(det_actions)} steps) ---")
    for idx, count in det_counter.most_common(5):
        name = action_names[idx] if action_names and idx < len(action_names) else f"action_{idx}"
        print(f"  {name}: {count} ({100*count/len(det_actions):.1f}%)")
    det_unique = len(det_counter)
    print(f"  Unique actions used: {det_unique}/{n_actions}")

    # Stochastic action distribution
    stoch_counter = Counter(stoch_actions)
    print(f"\n--- Stochastic Actions ({len(stoch_actions)} steps) ---")
    for idx, count in stoch_counter.most_common(10):
        name = action_names[idx] if action_names and idx < len(action_names) else f"action_{idx}"
        print(f"  {name}: {count} ({100*count/len(stoch_actions):.1f}%)")
    stoch_unique = len(stoch_counter)
    print(f"  Unique actions used: {stoch_unique}/{n_actions}")

    # Logit analysis
    if all_logits:
        logits_arr = np.array(all_logits)
        probs_arr = np.array(all_probs)

        # Average probability distribution
        mean_probs = probs_arr.mean(axis=0)
        print(f"\n--- Average Action Probabilities ---")
        sorted_idxs = np.argsort(mean_probs)[::-1]
        for i, idx in enumerate(sorted_idxs[:10]):
            name = action_names[idx] if action_names and idx < len(action_names) else f"action_{idx}"
            print(f"  {name}: {mean_probs[idx]:.4f} ({100*mean_probs[idx]:.2f}%)")

        # Entropy analysis
        # H = -sum(p * log(p))
        eps = 1e-8
        entropies = -np.sum(probs_arr * np.log(probs_arr + eps), axis=1)
        max_entropy = np.log(n_actions)
        print(f"\n--- Entropy Analysis ---")
        print(f"  Mean entropy: {entropies.mean():.4f} (max possible: {max_entropy:.4f})")
        print(f"  Min entropy:  {entropies.min():.4f}")
        print(f"  Max entropy:  {entropies.max():.4f}")
        print(f"  Entropy ratio: {entropies.mean()/max_entropy:.4f} (1.0 = uniform)")

        # Top action dominance
        top_prob = probs_arr.max(axis=1)
        print(f"\n--- Top Action Dominance ---")
        print(f"  Mean top-action probability: {top_prob.mean():.4f}")
        print(f"  Min top-action probability:  {top_prob.min():.4f}")
        print(f"  Max top-action probability:  {top_prob.max():.4f}")

    # Verdict
    print(f"\n{'='*60}")
    if det_unique <= 2:
        print("VERDICT: SEVERE POLICY COLLAPSE - deterministic mode uses <=2 actions")
        if stoch_unique >= 5:
            print("  -> Stochastic mode is much more diverse! Use deterministic=False for inference.")
        else:
            print("  -> Even stochastic mode is collapsed. Need to retrain with higher entropy.")
    elif det_unique <= 5:
        print("VERDICT: MODERATE COLLAPSE - limited action diversity")
    else:
        print("VERDICT: Policy looks reasonably diverse")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
