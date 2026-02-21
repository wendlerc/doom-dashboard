"""
CLI entry point for doom-dashboard.

Commands
────────
  generate-samples   Run rollouts and produce annotated MP4 sample videos
  serve              Start the Flask dashboard server
  generate-dataset   Run the parallelized WebDataset generation pipeline
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import click

from doom_dashboard.config import Config
from doom_dashboard.policies import load_policy
from doom_dashboard.rollout import rollout_episode
from doom_dashboard.annotate import annotate_and_encode


DEFAULT_CONFIG = os.path.join(os.getcwd(), "config.yaml")


# ─── helpers ─────────────────────────────────────────────────────

def _load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        click.echo(f"Config not found: {config_path}", err=True)
        sys.exit(1)
    return Config.from_yaml(config_path)


# ─── CLI group ───────────────────────────────────────────────────

@click.group()
def cli():
    """Doom Policy Dashboard & Dataset Generator."""
    pass


# ─── generate-samples ────────────────────────────────────────────

@cli.command("generate-samples")
@click.option("--config", "-c", default=DEFAULT_CONFIG, show_default=True,
              help="Path to config.yaml")
@click.option("--output-dir", "-o", default=None,
              help="Override output directory from config")
@click.option("--dry-run", is_flag=True,
              help="Print plan without running rollouts")
def generate_samples(config: str, output_dir: Optional[str], dry_run: bool):
    """Generate one annotated MP4 sample per scenario (for the dashboard)."""
    cfg = _load_config(config)
    out_dir = Path(output_dir or cfg.dashboard.output_dir)
    fps = cfg.dashboard.fps
    res = cfg.dashboard.render_resolution

    plan = []
    for scenario in cfg.scenarios:
        for pol_cfg in cfg.policies:
            plan.append((scenario, pol_cfg))

    if dry_run:
        click.echo(f"Would generate {len(plan)} sample videos → {out_dir}/")
        for sc, pc in plan:
            click.echo(f"  {sc.name} × {pc.name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    index_entries = []

    from tqdm import tqdm
    for scenario, pol_cfg in tqdm(plan, desc="Generating samples"):
        policy = load_policy(pol_cfg)
        try:
            ep = rollout_episode(
                scenario=scenario,
                policy=policy,
                render_resolution=res,
            )
            fname = f"{scenario.name.lower().replace(' ', '_')}__{pol_cfg.name.lower().replace(' ', '_')}.mp4"
            video_path = out_dir / fname
            annotate_and_encode(ep, str(video_path), fps=fps)

            entry = {
                "scenario": scenario.name,
                "policy": pol_cfg.name,
                "video": fname,
                "total_reward": ep.total_reward,
                "steps": ep.steps,
                "game_tics": ep.game_tics,
                "duration_s": ep.duration_s,
                "button_names": ep.button_names,
                "game_vars_final": ep.game_vars[-1].tolist() if ep.game_vars else [],
            }
            index_entries.append(entry)
            tqdm.write(f"  ✓ {scenario.name} × {pol_cfg.name}  reward={ep.total_reward:.1f}  steps={ep.steps}")
        except Exception as exc:
            tqdm.write(f"  ✗ {scenario.name} × {pol_cfg.name}: {exc}")
        finally:
            policy.close()

    # Write samples index JSON consumed by the Flask server
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index_entries, f, indent=2)

    click.echo(f"\n✓ {len(index_entries)} sample videos → {out_dir}/")


# ─── serve ───────────────────────────────────────────────────────

@cli.command("serve")
@click.option("--config", "-c", default=DEFAULT_CONFIG, show_default=True)
@click.option("--port", "-p", default=8080, show_default=True, type=int)
@click.option("--host", default="0.0.0.0", show_default=True)
def serve(config: str, port: int, host: str):
    """Start the Flask dashboard server."""
    from doom_dashboard.server import create_app
    cfg = _load_config(config)

    # Resolve relative dirs to absolute using the config file's parent directory
    config_dir = Path(os.path.abspath(config)).parent
    cfg.dashboard.output_dir = str(config_dir / cfg.dashboard.output_dir)
    cfg.dataset.output_dir   = str(config_dir / cfg.dataset.output_dir)

    app = create_app(cfg)
    click.echo(f"Dashboard running at http://{host}:{port}")
    click.echo(f"  Samples dir: {cfg.dashboard.output_dir}")
    app.run(host=host, port=port, debug=False, threaded=True)



# ─── generate-dataset ────────────────────────────────────────────

@cli.command("generate-dataset")
@click.option("--config", "-c", default=DEFAULT_CONFIG, show_default=True)
@click.option("--output-dir", "-o", default=None,
              help="Override output directory from config")
@click.option("--hours", default=None, type=float,
              help="Override total gameplay hours from config")
@click.option("--workers", default=None, type=int,
              help="Number of parallel rollout workers (default: auto)")
def generate_dataset(config: str, output_dir: Optional[str], hours: Optional[float],
                     workers: Optional[int]):
    """Run parallelized (single-player) WebDataset generation."""
    from doom_dashboard.dataset_gen import generate_dataset as gen

    cfg = _load_config(config)
    out = gen(
        config=cfg,
        target_hours=hours,
        num_workers=workers,
        output_dir=output_dir,
    )
    click.echo(f"\n✓ Dataset written to: {out}")
    shards = sorted(glob.glob(os.path.join(out, "*.tar")))
    click.echo(f"  Shards: {len(shards)}")
    for s in shards[:5]:
        size_mb = os.path.getsize(s) / 1024 / 1024
        click.echo(f"    {os.path.basename(s)}  ({size_mb:.1f} MB)")
    if len(shards) > 5:
        click.echo(f"    … and {len(shards) - 5} more")


@cli.command("generate-mp-dataset")
@click.option("--config", "-c", default=DEFAULT_CONFIG, show_default=True)
@click.option("--output-dir", "-o", default="mp_dataset",
              help="Output directory for multiplayer WebDataset shards")
@click.option("--hours", default=200.0, type=float, show_default=True,
              help="Target total gameplay hours")
@click.option("--workers", default=None, type=int,
              help="Parallel game pairs (default: auto, max 32)")
@click.option("--random-ratio", default=0.05, type=float, show_default=True,
              help="Fraction of policy-slots filled by random policy")
@click.option("--timelimit", default=5.0, type=float, show_default=True,
              help="Default game timelimit in minutes per episode")
@click.option("--frame-skip", default=4, type=int, show_default=True)
@click.option("--resolution", default="RES_320X240", show_default=True)
@click.option("--shard-mb", default=512, type=int, show_default=True)
def generate_mp_dataset(
    config: str, output_dir: str, hours: float, workers: Optional[int],
    random_ratio: float, timelimit: float, frame_skip: int,
    resolution: str, shard_mb: int,
):
    """Generate parallelized 1v1 multiplayer WebDataset."""
    from doom_dashboard.mp_dataset_gen import (
        generate_multiplayer_dataset, MP_SCENARIOS,
    )

    cfg = _load_config(config)
    pol_dicts = [
        dict(name=p.name, type=p.type, path=p.path,
             algo=p.algo, arch=p.arch, action_size=p.action_size, device=p.device)
        for p in cfg.policies
    ]

    # Filter to policies whose checkpoints actually exist
    valid = []
    for d in pol_dicts:
        if d["type"] == "random":
            valid.append(d)
        elif d["path"] and os.path.exists(d["path"]):
            valid.append(d)
        else:
            click.echo(f"  ⚠ Skipping '{d['name']}' — checkpoint not found: {d['path']}")
    if not valid:
        click.echo("No valid policies found. Run train_policies.py first.", err=True)
        sys.exit(1)

    # Override timelimit in MP_SCENARIOS
    for sc in MP_SCENARIOS.values():
        sc["timelimit_min"] = timelimit

    click.echo(f"Generating {hours:.0f} h of 1v1 multiplayer gameplay → {output_dir}/")
    click.echo(f"  Policies ({len(valid)}): {[d['name'] for d in valid]}")
    click.echo(f"  Random ratio: {random_ratio:.0%}")
    click.echo(f"  Scenarios: {list(MP_SCENARIOS.keys())}")

    out = generate_multiplayer_dataset(
        output_dir=output_dir,
        total_hours=hours,
        policy_dicts=valid,
        random_policy_ratio=random_ratio,
        frame_skip=frame_skip,
        render_resolution=resolution,
        shard_size_mb=shard_mb,
        num_workers=workers,
    )
    shards = sorted(glob.glob(os.path.join(out, "mp-shard-*.tar")))
    click.echo(f"\n✓ {len(shards)} shards written to: {out}")



def main():
    cli()


if __name__ == "__main__":
    main()
