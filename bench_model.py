#!/usr/bin/env python3
"""Quick standalone benchmark: load a model and run bench_duel_vs_random."""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to model .zip")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--timelimit", type=float, default=2.0)
    ap.add_argument("--cfg", default="doom_dashboard/scenarios/deathmatch_compact.cfg")
    ap.add_argument("--maps", default="map01")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--resolution", default="320x240")
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    from train_overnight_dm import bench_duel_vs_random, materialize_cfg

    cfg_path = materialize_cfg(args.cfg)
    maps = [m.strip() for m in args.maps.split(",")]

    print(f"Benchmarking {args.model} ({args.episodes} episodes)...")
    result = bench_duel_vs_random(
        model_path=args.model,
        cfg_path=cfg_path,
        scenario_name="deathmatch",
        maps=maps,
        frame_skip=args.frame_skip,
        timelimit_minutes=args.timelimit,
        episodes=args.episodes,
        resolution=args.resolution,
    )

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Episodes: {result['episodes']}")
    print(f"Frags: {result['model_frag_mean']:.2f}")
    print(f"Deaths: {result['model_death_mean']:.2f}")
    print(f"Kills: {result['model_kill_mean']:.2f}")
    print(f"Damage: {result['model_damage_mean']:.1f}")
    print(f"Damage Taken: {result['model_damage_taken_mean']:.1f}")
    print(f"Damage Ratio: {result['damage_ratio']:.3f}")
    print(f"Win Rate: {result['win_rate_vs_random_net']:.3f}")
    print(f"Net Gap: {result['mean_net_gap']:.2f}")
    gate = "PASS" if result['pass'] else "FAIL"
    print(f"Gate: {gate}")
    print(f"{'='*60}")

    out_path = args.out or str(Path(args.model).with_suffix('.bench.json'))
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
