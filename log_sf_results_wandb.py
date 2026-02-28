#!/usr/bin/env python3
"""
Log Sample Factory pretrained model evaluation results to wandb with videos.

Usage:
    uv run python log_sf_results_wandb.py
"""
import json
import os
from pathlib import Path

import numpy as np
import wandb


def main():
    # Results from running the pretrained models
    model_results = {
        "sf_pretrained_seed0": {
            "experiment": "00_bots_128_fs2_narrow_see_0",
            "source": "andrewzhang505/doom_deathmatch_bots (HuggingFace)",
            "seed": 0,
            "episodes": 5,
            "avg_reward": 25.0,
            "rewards": [22.0, 28.5, 26.0, 21.5, 27.0],
            "video": "results_latest/competition_submission_v3/videos/sf_pretrained.mp4",
            "checkpoint": "best",
            "framework": "Sample Factory (APPO)",
            "architecture": "ConvNet + LSTM 512",
            "resolution": "128x72",
        },
        "sf_pretrained_seed2222": {
            "experiment": "doom_deathmatch_bots_2222",
            "source": "edbeeching/doom_deathmatch_bots_2222 (HuggingFace)",
            "seed": 2222,
            "episodes": 3,
            "avg_reward": 24.0,
            "rewards": [24.0, 22.5, 25.5],
            "video": "results_latest/competition_submission_v3/videos/sf_model_2222.mp4",
            "checkpoint": "best",
            "framework": "Sample Factory (APPO)",
            "architecture": "ConvNet + LSTM 512",
            "resolution": "128x72",
        },
        "sf_pretrained_seed3333": {
            "experiment": "doom_deathmatch_bots_3333",
            "source": "edbeeching/doom_deathmatch_bots_3333 (HuggingFace)",
            "seed": 3333,
            "episodes": 5,
            "avg_reward": 25.1,
            "rewards": [29.5, 16.8, 23.0, 27.5, 28.8],
            "video": "results_latest/competition_submission_v3/videos/sf_model_3333.mp4",
            "checkpoint": "best",
            "framework": "Sample Factory (APPO)",
            "architecture": "ConvNet + LSTM 512",
            "resolution": "128x72",
        },
    }

    # Initialize wandb
    run = wandb.init(
        project="doom-deathmatch",
        name="sf_pretrained_evaluation",
        group="sf_evaluation",
        tags=["pretrained", "sample_factory", "evaluation"],
        config={
            "framework": "Sample Factory",
            "algorithm": "APPO",
            "env": "doom_deathmatch_bots",
            "models_evaluated": len(model_results),
        },
    )

    # Log each model's results
    for model_name, results in model_results.items():
        print(f"\nLogging {model_name}...")

        # Log video if exists
        video_path = results["video"]
        if os.path.exists(video_path):
            wandb.log({
                f"{model_name}/gameplay_video": wandb.Video(
                    video_path,
                    fps=35,
                    format="mp4",
                    caption=f"{model_name} - Avg reward: {results['avg_reward']:.1f}",
                ),
            })
            print(f"  Video logged: {video_path}")

        # Log scalar metrics
        wandb.log({
            f"{model_name}/avg_reward": results["avg_reward"],
            f"{model_name}/std_reward": float(np.std(results["rewards"])),
            f"{model_name}/max_reward": float(np.max(results["rewards"])),
            f"{model_name}/min_reward": float(np.min(results["rewards"])),
            f"{model_name}/num_episodes": results["episodes"],
        })

    # Log comparison table
    table = wandb.Table(
        columns=["Model", "Source", "Seed", "Avg Reward", "Std", "Max", "Min", "Episodes"],
    )
    for model_name, results in model_results.items():
        rews = results["rewards"]
        table.add_data(
            model_name,
            results["source"],
            results["seed"],
            results["avg_reward"],
            round(float(np.std(rews)), 1),
            float(np.max(rews)),
            float(np.min(rews)),
            results["episodes"],
        )

    wandb.log({"sf_pretrained_comparison": table})

    # Summary
    all_rewards = []
    for r in model_results.values():
        all_rewards.extend(r["rewards"])

    wandb.run.summary["best_model"] = max(model_results, key=lambda k: model_results[k]["avg_reward"])
    wandb.run.summary["best_avg_reward"] = max(r["avg_reward"] for r in model_results.values())
    wandb.run.summary["overall_avg_reward"] = float(np.mean(all_rewards))
    wandb.run.summary["overall_std_reward"] = float(np.std(all_rewards))

    print(f"\n=== Summary ===")
    print(f"Best model: {wandb.run.summary['best_model']}")
    print(f"Best avg reward: {wandb.run.summary['best_avg_reward']:.1f}")
    print(f"Overall avg: {wandb.run.summary['overall_avg_reward']:.1f} +/- {wandb.run.summary['overall_std_reward']:.1f}")

    wandb.finish()
    print(f"\nWandb run URL: {run.url}")


if __name__ == "__main__":
    main()
