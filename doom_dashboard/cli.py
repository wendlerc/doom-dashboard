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
import random
import re
import socket
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import click
import numpy as np
from PIL import Image, ImageDraw

from doom_dashboard.config import Config, PolicyConfig
from doom_dashboard.alignment import policy_matches_scenario, infer_policy_scenario_key
from doom_dashboard.policies import load_policy
from doom_dashboard.rollout import rollout_episode
from doom_dashboard.annotate import annotate_and_encode
from doom_dashboard.annotate import annotate_frame


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
@click.option("--min-seconds", default=30.0, show_default=True, type=float,
              help="Minimum output duration per clip (seconds)")
@click.option("--dry-run", is_flag=True,
              help="Print plan without running rollouts")
def generate_samples(config: str, output_dir: Optional[str], min_seconds: float, dry_run: bool):
    """Generate one annotated MP4 sample per scenario (for the dashboard)."""
    cfg = _load_config(config)
    out_dir = Path(output_dir or cfg.dashboard.output_dir)
    res = cfg.dashboard.render_resolution

    plan = []
    skipped = 0
    for scenario in cfg.scenarios:
        for pol_cfg in cfg.policies:
            if pol_cfg.type == "random" or policy_matches_scenario(pol_cfg, scenario):
                plan.append((scenario, pol_cfg))
            else:
                skipped += 1

    if dry_run:
        click.echo(f"Would generate {len(plan)} sample videos → {out_dir}/")
        if skipped:
            click.echo(f"  (skipped {skipped} mismatched scenario/policy pairs)")
        for sc, pc in plan:
            click.echo(f"  {sc.name} × {pc.name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    index_entries = []

    from tqdm import tqdm
    for scenario, pol_cfg in tqdm(plan, desc="Generating samples"):
        policy = load_policy(pol_cfg)
        try:
            target_secs = max(1.0, float(min_seconds))
            # Keep playback close to real in-game time at this frame_skip.
            sample_fps = max(1, round(35.0 / max(1, scenario.frame_skip), 2))
            target_frames = int(target_secs * sample_fps)

            episodes = []
            total_frames = 0
            total_steps = 0
            total_reward = 0.0
            total_tics = 0
            total_runtime = 0.0
            safety_cap = 64
            while total_frames < target_frames and len(episodes) < safety_cap:
                ep_i = rollout_episode(
                    scenario=scenario,
                    policy=policy,
                    render_resolution=res,
                )
                episodes.append(ep_i)
                total_frames += len(ep_i.frames)
                total_steps += ep_i.steps
                total_reward += float(ep_i.total_reward)
                total_tics += int(ep_i.game_tics)
                total_runtime += float(ep_i.duration_s)

            if not episodes:
                raise RuntimeError("No episode data was generated.")

            ep0 = episodes[0]
            ep = ep0.__class__(
                frames=[f for e in episodes for f in e.frames],
                actions=[a for e in episodes for a in e.actions],
                button_names=ep0.button_names,
                rewards=[r for e in episodes for r in e.rewards],
                game_vars=[g for e in episodes for g in e.game_vars],
                total_reward=total_reward,
                scenario_name=ep0.scenario_name,
                policy_name=ep0.policy_name,
                cfg_path=ep0.cfg_path,
                frame_skip=ep0.frame_skip,
                steps=total_steps,
                duration_s=total_runtime,
                game_tics=total_tics,
                metadata=dict(ep0.metadata or {}, stitched_episodes=len(episodes)),
            )
            fname = f"{scenario.name.lower().replace(' ', '_')}__{pol_cfg.name.lower().replace(' ', '_')}.mp4"
            video_path = out_dir / fname
            annotate_and_encode(ep, str(video_path), fps=sample_fps)

            entry = {
                "scenario": scenario.name,
                "policy": pol_cfg.name,
                "video": fname,
                "total_reward": ep.total_reward,
                "steps": ep.steps,
                "game_tics": ep.game_tics,
                "duration_s": ep.duration_s,
                "clip_seconds": round(len(ep.frames) / float(sample_fps), 2),
                "clip_fps": sample_fps,
                "stitched_episodes": int(ep.metadata.get("stitched_episodes", 1)) if ep.metadata else 1,
                "button_names": ep.button_names,
                "game_vars_final": ep.game_vars[-1].tolist() if ep.game_vars else [],
            }
            index_entries.append(entry)
            tqdm.write(
                f"  ✓ {scenario.name} × {pol_cfg.name}  reward={ep.total_reward:.1f}  "
                f"steps={ep.steps}  clip={entry['clip_seconds']:.1f}s"
            )
        except Exception as exc:
            tqdm.write(f"  ✗ {scenario.name} × {pol_cfg.name}: {exc}")
        finally:
            policy.close()

    # Write samples index JSON consumed by the Flask server
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index_entries, f, indent=2)

    click.echo(f"\n✓ {len(index_entries)} sample videos → {out_dir}/")
    if skipped:
        click.echo(f"  Skipped {skipped} mismatched scenario/policy pairs")


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
@click.option("--allow-mismatch", is_flag=True,
              help="Allow policies to run in scenarios they were not trained on.")
def generate_mp_dataset(
    config: str, output_dir: str, hours: float, workers: Optional[int],
    random_ratio: float, timelimit: float, frame_skip: int,
    resolution: str, shard_mb: int, allow_mismatch: bool,
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
        elif not allow_mismatch:
            exp = infer_policy_scenario_key(PolicyConfig(**d))
            click.echo(f"  ⚠ Skipping '{d['name']}' — trained for '{exp or 'unknown'}', not multiplayer scenarios")
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


def _policy_dict_from_cfg(cfg: Config, name: str) -> Optional[dict]:
    for p in cfg.policies:
        if p.name == name:
            return dict(
                name=p.name,
                type=p.type,
                path=p.path,
                algo=p.algo,
                arch=p.arch,
                action_size=p.action_size,
                device=p.device,
            )
    return None


def _discover_extra_elim_policies(root: Path) -> List[dict]:
    out: List[dict] = []
    pats = [
        str(root / "trained_policies" / "defend_the_center_elim_*.zip"),
        str(root / "trained_policies" / "deadly_corridor_elim_*.zip"),
        str(root / "trained_policies" / "multi_duel_elim_*.zip"),
        str(root / "trained_policies" / "elimination_*_elim_*.zip"),
        str(root / "trained_policies" / "deathmatch_selfplay*.zip"),
        str(root / "trained_policies" / "deathmatch_dm_mp_*.zip"),
    ]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            stem = Path(p).stem
            out.append(
                dict(
                    name=f"PPO-{stem}",
                    type="sb3",
                    path=str(Path(p)),
                    algo="PPO",
                    arch="DuelQNet",
                    action_size=None,
                    device="auto",
                )
            )
    return out


def _encode_side_by_side_mp4(ep, out_path: str, fps: float) -> None:
    import imageio

    t = min(len(ep.frames_p1), len(ep.frames_p2), len(ep.actions_p1), len(ep.actions_p2))
    if t <= 0:
        raise RuntimeError("No frames to encode for side-by-side video.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wr = imageio.get_writer(
        out_path, fps=fps, codec="libx264", output_params=["-crf", "22", "-preset", "fast"]
    )

    r1 = 0.0
    r2 = 0.0
    for i in range(t):
        r1 += float(ep.rewards_p1[i])
        r2 += float(ep.rewards_p2[i])
        left = annotate_frame(
            frame=ep.frames_p1[i],
            action=ep.actions_p1[i],
            button_names=ep.button_names,
            reward=float(ep.rewards_p1[i]),
            step=i + 1,
            total_steps=t,
            scenario_name=f"P1 · {ep.policy_name_p1}",
            policy_name=ep.scenario_name,
            total_reward_so_far=r1,
        )
        right = annotate_frame(
            frame=ep.frames_p2[i],
            action=ep.actions_p2[i],
            button_names=ep.button_names,
            reward=float(ep.rewards_p2[i]),
            step=i + 1,
            total_steps=t,
            scenario_name=f"P2 · {ep.policy_name_p2}",
            policy_name=ep.scenario_name,
            total_reward_so_far=r2,
        )
        if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
            right = np.array(Image.fromarray(right).resize((left.shape[1], left.shape[0])))
        frame = np.concatenate([left, right], axis=1)

        # Small center overlay with match context.
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([frame.shape[1] // 2 - 180, 6, frame.shape[1] // 2 + 180, 24], fill=(0, 0, 0, 180))
        draw.text((frame.shape[1] // 2 - 170, 8), f"{ep.scenario_name} · {ep.map_name}", fill=(220, 220, 220))
        wr.append_data(np.array(img))

    wr.close()


def _find_free_port(start: int = 6200, end: int = 6400, step: int = 10) -> int:
    for p in range(start, end, step):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            return p
        except OSError:
            pass
        finally:
            s.close()
    raise RuntimeError(f"No free port found in range [{start}, {end})")


@cli.command("generate-elim-videos")
@click.option("--config", "-c", default=DEFAULT_CONFIG, show_default=True)
@click.option("--output-dir", "-o", default="elim_videos", show_default=True,
              help="Output directory for side-by-side elimination videos")
@click.option("--scenario", default="multi_duel", show_default=True,
              type=click.Choice(["multi_duel", "deathmatch", "cig", "cig_fullaction", "deathmatch_compact", "deathmatch_nomonsters"]))
@click.option("--games", default=6, show_default=True, type=int,
              help="Number of match videos to generate")
@click.option("--timelimit", default=3.0, show_default=True, type=float,
              help="Timelimit minutes per game")
@click.option("--frame-skip", default=4, show_default=True, type=int)
@click.option("--resolution", default="RES_640X480", show_default=True)
@click.option("--policy", "policy_names", multiple=True,
              help="Policy names to include; can repeat. If omitted, auto-picks elimination policies.")
@click.option("--map", "maps_filter", multiple=True,
              help="Restrict to specific map(s), e.g. --map map01 --map map02.")
@click.option("--append-index/--replace-index", default=True, show_default=True,
              help="Append to existing index.json in output dir (dashboard-friendly) or replace it.")
@click.option("--max-attempts", default=3, show_default=True, type=int,
              help="Retry attempts per requested game if a multiplayer rollout fails.")
@click.option("--allow-mismatch", is_flag=True,
              help="Allow policies outside the scenario they were trained for.")
def generate_elim_videos(
    config: str,
    output_dir: str,
    scenario: str,
    games: int,
    timelimit: float,
    frame_skip: int,
    resolution: str,
    policy_names: Tuple[str, ...],
    maps_filter: Tuple[str, ...],
    append_index: bool,
    max_attempts: int,
    allow_mismatch: bool,
):
    """Generate side-by-side 1v1 elimination match videos."""
    from doom_dashboard.mp_dataset_gen import MP_SCENARIOS
    from doom_dashboard.multiplayer_rollout import rollout_multiplayer_episode, BASE_PORT

    cfg = _load_config(config)
    cfg_path = MP_SCENARIOS[scenario]["cfg"]
    scenario_maps = list(MP_SCENARIOS[scenario]["maps"])
    if maps_filter:
        wanted = [m.strip().lower() for m in maps_filter if m.strip()]
        allowed = {m.lower() for m in scenario_maps}
        bad = [m for m in wanted if m not in allowed]
        if bad:
            raise click.ClickException(
                f"Unsupported map(s) for scenario '{scenario}': {bad}. "
                f"Allowed: {scenario_maps}"
            )
        maps = wanted
    else:
        maps = scenario_maps
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(os.path.abspath(config)).parent

    available: Dict[str, dict] = {}
    for p in cfg.policies:
        available[p.name] = dict(
            name=p.name, type=p.type, path=p.path, algo=p.algo, arch=p.arch,
            action_size=p.action_size, device=p.device,
        )
    for d in _discover_extra_elim_policies(root):
        available.setdefault(d["name"], d)

    if policy_names:
        selected = []
        for n in policy_names:
            if n not in available:
                raise click.ClickException(f"Unknown policy '{n}'.")
            selected.append(available[n])
    else:
        auto = [n for n in available if ("DefendTheCenter" in n or "DeadlyCorridor" in n or "_elim_" in n)]
        auto = sorted(auto)
        if not auto:
            auto = [n for n in available if n.startswith("PPO-")]
        selected = [available[n] for n in auto[:8]]

    if len(selected) < 2:
        raise click.ClickException("Need at least 2 policies for head-to-head elimination videos.")

    if not allow_mismatch:
        filtered = []
        for p in selected:
            if p.get("type") == "random":
                filtered.append(p)
                continue
            exp = infer_policy_scenario_key(PolicyConfig(**p))
            # For MP sample generation, default to strict scenario matching.
            if exp is None or exp == scenario:
                filtered.append(p)
            else:
                click.echo(f"  ⚠ skipping {p['name']} (trained for {exp}, requested {scenario})")
        selected = filtered
        if len(selected) < 2:
            for name in sorted(available):
                p = available[name]
                if any(x.get("name") == p.get("name") for x in selected):
                    continue
                if p.get("type") != "random":
                    exp = infer_policy_scenario_key(PolicyConfig(**p))
                    if exp is not None and exp != scenario:
                        continue
                selected.append(p)
                if len(selected) >= 2:
                    break
        if len(selected) < 2 and "Random" in available:
            if all(p.get("name") != "Random" for p in selected):
                selected.append(available["Random"])
        if len(selected) < 2:
            raise click.ClickException(
                "Need at least 2 compatible policies after alignment filtering. "
                "Use --allow-mismatch to override."
            )

    pairs: List[Tuple[dict, dict]] = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairs.append((selected[i], selected[j]))
    if not pairs:
        pairs = [(selected[0], selected[1])]

    fps = max(1, round(35.0 / max(1, frame_skip), 2))
    entries = []

    click.echo(f"Generating {games} elimination videos in '{out_dir}'")
    click.echo(f"  scenario={scenario} timelimit={timelimit}m frame_skip={frame_skip} fps={fps} res={resolution}")
    click.echo(f"  maps={maps}")
    click.echo(f"  policies={[p['name'] for p in selected]}")

    from tqdm import tqdm
    made = 0
    attempts_total = 0
    pbar = tqdm(total=games, desc="Elim Videos")
    while made < games:
        attempts_total += 1
        idx = made
        p1, p2 = pairs[attempts_total % len(pairs)]
        doom_map = random.choice(maps)
        ep = None
        err = None
        for _ in range(max(1, max_attempts)):
            try:
                ep = rollout_multiplayer_episode(
                    cfg_path=cfg_path,
                    scenario_name=scenario,
                    doom_map=doom_map,
                    policy_p1_dict=p1,
                    policy_p2_dict=p2,
                    timelimit_minutes=timelimit,
                    frame_skip=frame_skip,
                    render_resolution=resolution,
                    render_hud=False,
                    port=_find_free_port(start=6200, end=7000, step=10),
                )
                err = None
                break
            except Exception as e:
                err = str(e)
                continue
        if ep is None:
            tqdm.write(f"  ✗ failed after {max_attempts} attempts: {p1['name']} vs {p2['name']} ({err})")
            if attempts_total > games * max(2, max_attempts):
                break
            continue
        fname = (
            f"{scenario}__{doom_map}__{p1['name'].lower().replace(' ', '_')}__vs__"
            f"{p2['name'].lower().replace(' ', '_')}__g{idx+1:02d}.mp4"
        )
        out_path = out_dir / fname
        _encode_side_by_side_mp4(ep, str(out_path), fps=fps)

        r1 = float(np.sum(np.asarray(ep.rewards_p1, dtype=np.float32)))
        r2 = float(np.sum(np.asarray(ep.rewards_p2, dtype=np.float32)))
        e = {
            "video": fname,
            "scenario": scenario,
            "map": doom_map,
            "policy_p1": p1["name"],
            "policy_p2": p2["name"],
            "total_reward_p1": r1,
            "total_reward_p2": r2,
            "frames": min(len(ep.frames_p1), len(ep.frames_p2)),
            "fps": fps,
            "clip_seconds": round(min(len(ep.frames_p1), len(ep.frames_p2)) / float(fps), 2),
            "game_tics": int(ep.game_tics),
            # Dashboard card compatibility fields:
            "policy": f"{p1['name']} vs {p2['name']}",
            "total_reward": float(r1 - r2),
            "steps": int(min(len(ep.frames_p1), len(ep.frames_p2))),
            "duration_s": float(min(len(ep.frames_p1), len(ep.frames_p2)) / float(fps)),
        }
        entries.append(e)
        tqdm.write(
            f"  ✓ {p1['name']} vs {p2['name']}  map={doom_map}  "
            f"R1={r1:.1f} R2={r2:.1f} clip={e['clip_seconds']:.1f}s"
        )
        made += 1
        pbar.update(1)
    pbar.close()

    index_path = out_dir / "index.json"
    if append_index and index_path.exists():
        try:
            with open(index_path) as f:
                prior = json.load(f)
            if not isinstance(prior, list):
                prior = []
        except Exception:
            prior = []
    else:
        prior = []
    merged = prior + entries
    with open(index_path, "w") as f:
        json.dump(merged, f, indent=2)
    click.echo(f"\n✓ Wrote {len(entries)} videos to {out_dir} (index size: {len(merged)})")


@cli.command("record-human-demo")
@click.option("--config", "-c", default=DEFAULT_CONFIG, show_default=True)
@click.option("--output-dir", "-o", default="human_demos", show_default=True,
              help="Where to save recorded demo files.")
@click.option("--name", default=None,
              help="Optional run name; default: auto timestamped stem.")
@click.option("--scenario", default="deathmatch_compact", show_default=True,
              type=click.Choice(["deathmatch_compact", "deathmatch_nomonsters",
                                 "deathmatch_fullaction", "cig_fullaction", "cig", "deathmatch", "multi_duel"]))
@click.option("--map", "doom_map", default="", show_default=True,
              help="Map name (e.g. map01); empty = random map from scenario.")
@click.option("--timelimit", default=5.0, show_default=True, type=float,
              help="Match length in minutes.")
@click.option("--frame-skip", default=1, show_default=True, type=int,
              help="Frames to advance per step while recording.")
@click.option("--resolution", default="RES_1280X720", show_default=True,
              help="Render resolution for the human play window.")
@click.option("--bots", default=1, show_default=True, type=int,
              help="Number of Doom bots to add to the match.")
@click.option("--render-hud/--no-render-hud", default=False, show_default=True)
@click.option("--fullscreen/--no-fullscreen", default=False, show_default=True,
              help="Launch game in fullscreen mode.")
@click.option("--record-audio/--no-record-audio", default=True, show_default=True,
              help="Capture game audio into the output video (requires OpenAL; disables live sound during play).")
@click.option("--save-video/--no-save-video", default=True, show_default=True)
@click.option("--fps", default=35.0, show_default=True, type=float,
              help="Output FPS for saved demo video.")
def record_human_demo(
    config: str,
    output_dir: str,
    name: Optional[str],
    scenario: str,
    doom_map: str,
    timelimit: float,
    frame_skip: int,
    resolution: str,
    bots: int,
    render_hud: bool,
    fullscreen: bool,
    record_audio: bool,
    save_video: bool,
    fps: float,
):
    """Record a human-played deathmatch demo with actions/rewards/frames."""
    import time

    import vizdoom as vzd

    from doom_dashboard.annotate import annotate_and_encode
    from doom_dashboard.mp_dataset_gen import MP_SCENARIOS
    from doom_dashboard.rollout import EpisodeData

    _ = _load_config(config)  # keep UX consistent with other commands

    if scenario not in MP_SCENARIOS:
        raise click.ClickException(f"Unknown scenario '{scenario}'.")

    cfg_path = MP_SCENARIOS[scenario]["cfg"]
    maps = MP_SCENARIOS[scenario]["maps"]
    map_name = (doom_map or "").strip().lower() or random.choice(maps)
    if map_name.lower() not in {m.lower() for m in maps}:
        click.echo(f"  ⚠ map '{map_name}' not in default pool {maps}; attempting anyway.")

    res_enum = getattr(vzd.ScreenResolution, resolution, None)
    if res_enum is None:
        valid = sorted([n for n in dir(vzd.ScreenResolution) if n.startswith("RES_")])
        raise click.ClickException(f"Unknown resolution '{resolution}'. Try one of: {valid}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if name:
        stem = re.sub(r"[^a-z0-9_.-]+", "_", name.lower())
    else:
        stem = f"human_{scenario}_{map_name}_{int(time.time())}"

    click.echo("\nHuman demo recorder starting.")
    click.echo("Short action tutorial:")
    click.echo("  Movement actions: MOVE_FORWARD / MOVE_BACKWARD / MOVE_LEFT / MOVE_RIGHT")
    click.echo("  Combat actions: ATTACK, USE, weapon select (1-6 / next / prev)")
    click.echo("  Camera actions: TURN_LEFT/RIGHT (+ delta turn/look channels)")
    click.echo("  Controls are your in-game keybinds. If needed: ESC -> Options -> Customize Controls.")
    click.echo("  Tip: click into the game window so keyboard/mouse input is captured.\n")

    tracked_vars = [
        vzd.GameVariable.FRAGCOUNT,
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.DEATHCOUNT,
        vzd.GameVariable.HITCOUNT,
        vzd.GameVariable.HITS_TAKEN,
        vzd.GameVariable.DAMAGECOUNT,
        vzd.GameVariable.DAMAGE_TAKEN,
        vzd.GameVariable.HEALTH,
    ]

    def _read_vars(game: vzd.DoomGame) -> np.ndarray:
        vals: List[float] = []
        for gv in tracked_vars:
            try:
                vals.append(float(game.get_game_variable(gv)))
            except Exception:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float32)

    def _to_hwc(buf: np.ndarray) -> np.ndarray:
        if buf.ndim == 3 and buf.shape[0] in (1, 3, 4) and buf.shape[-1] not in (1, 3, 4):
            return np.transpose(buf, (1, 2, 0))
        return buf

    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.set_doom_map(map_name)
    # Override episode_timeout to match our timelimit (cfg often has 4200 = 2 min)
    timeout_tics = int(float(timelimit) * 60 * 35)  # minutes -> seconds -> tics @ 35 tics/sec
    game.set_episode_timeout(max(4200, timeout_tics))
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(res_enum)
    game.set_render_hud(render_hud)
    # Disable sound when not recording audio — avoids OpenAL init that can crash on macOS/XQuartz
    game.set_sound_enabled(record_audio)
    if record_audio:
        game.set_audio_buffer_enabled(True)
        game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)
        game.set_audio_buffer_size(max(1, int(frame_skip)))
        # Workaround for OpenAL 1.19 "no audio in buffer" bug on some systems
        game.add_game_args("+snd_efx 0")
    # NOTE: Do NOT pass "-deathmatch" here.  The flag tells ZDoom to use
    # deathmatch start spawn points (Thing type 11); many ViZDoom WADs
    # lack these and the engine segfaults.  The +sv_* CVARs already
    # configure deathmatch-style gameplay rules.
    game.add_game_args(
        f"-host 1 +timelimit {float(max(0.1, timelimit)):.2f} "
        "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
        "+sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 0 +viz_nocheat 1"
    )
    if fullscreen:
        game.add_game_args("+fullscreen 1")
    game.add_game_args("+name HumanRecorder +colorset 2")
    game.init()

    for _ in range(max(0, int(bots))):
        try:
            game.send_game_command("addbot")
        except Exception:
            pass

    button_names: List[str] = [str(b) for b in game.get_available_buttons()]
    n_buttons = len(button_names)
    click.echo(f"Recording '{stem}' on {scenario}/{map_name} at {resolution} with {n_buttons} actions.")
    click.echo(f"Buttons: {button_names}")

    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    game_vars: List[np.ndarray] = []
    audio_slices: List[np.ndarray] = []

    t0 = time.perf_counter()
    try:
        while not game.is_episode_finished():
            game.advance_action(max(1, int(frame_skip)))
            if game.is_player_dead():
                try:
                    game.respawn_player()
                except Exception:
                    pass
            st = game.get_state()
            if st is None:
                continue

            obs = _to_hwc(st.screen_buffer)
            act_raw = game.get_last_action()
            if act_raw is None:
                act = np.zeros(n_buttons, dtype=np.float32)
            else:
                act = np.asarray(act_raw, dtype=np.float32).reshape(-1)
                if act.shape[0] != n_buttons:
                    tmp = np.zeros(n_buttons, dtype=np.float32)
                    l = min(n_buttons, act.shape[0])
                    tmp[:l] = act[:l]
                    act = tmp
            rew = float(game.get_last_reward())
            gv = _read_vars(game)

            frames.append(obs.copy())
            actions.append(act.copy())
            rewards.append(rew)
            game_vars.append(gv.copy())
            if record_audio and hasattr(st, "audio_buffer") and st.audio_buffer is not None:
                try:
                    ab = np.asarray(st.audio_buffer)
                    if ab.size > 0:
                        audio_slices.append(ab.copy())
                except Exception:
                    pass
    finally:
        total_reward = float(game.get_total_reward())
        game_tics = int(game.get_episode_time())
        game.close()
    t1 = time.perf_counter()

    if frames:
        frames_np = np.stack(frames, axis=0).astype(np.uint8)
    else:
        frames_np = np.zeros((0, 1, 1, 3), dtype=np.uint8)
    if actions:
        actions_np = np.stack(actions, axis=0).astype(np.float32)
    else:
        actions_np = np.zeros((0, n_buttons), dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    if game_vars:
        game_vars_np = np.stack(game_vars, axis=0).astype(np.float32)
    else:
        game_vars_np = np.zeros((0, len(tracked_vars)), dtype=np.float32)

    npz_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.meta.json"
    audio_np = None
    if record_audio and audio_slices:
        try:
            audio_np = np.concatenate(audio_slices, axis=0)
        except Exception:
            audio_np = None
    # Save actions/rewards/game_vars separately first (small files, survives truncation)
    actions_path = out_dir / f"{stem}.actions.npz"
    np.savez_compressed(actions_path, actions=actions_np, rewards=rewards_np, game_vars=game_vars_np)
    click.echo(f"  Actions saved: {actions_path} ({actions_path.stat().st_size / 1024:.0f} KB)")

    # Save frames (may be large at high resolution)
    save_dict = dict(
        frames=frames_np,
        actions=actions_np,
        rewards=rewards_np,
        game_vars=game_vars_np,
    )
    if audio_np is not None and audio_np.size > 0:
        save_dict["audio"] = audio_np
        save_dict["audio_sample_rate"] = 22050
    np.savez_compressed(npz_path, **save_dict)

    meta = {
        "name": stem,
        "scenario": scenario,
        "cfg_path": cfg_path,
        "map": map_name,
        "timelimit_minutes": float(timelimit),
        "frame_skip": int(frame_skip),
        "resolution": resolution,
        "render_hud": bool(render_hud),
        "bots": int(max(0, bots)),
        "button_names": button_names,
        "steps": int(actions_np.shape[0]),
        "duration_s": float(t1 - t0),
        "game_tics": int(game_tics),
        "total_reward": float(total_reward),
        "vars_layout": [
            "FRAGCOUNT",
            "KILLCOUNT",
            "DEATHCOUNT",
            "HITCOUNT",
            "HITS_TAKEN",
            "DAMAGECOUNT",
            "DAMAGE_TAKEN",
            "HEALTH",
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    video_path = None
    if save_video and frames:
        video_path = out_dir / f"{stem}.mp4"
        ep = EpisodeData(
            frames=frames,
            actions=actions,
            button_names=button_names,
            rewards=rewards,
            game_vars=game_vars,
            total_reward=float(total_reward),
            scenario_name=f"Human-{scenario}",
            policy_name="HumanPlayer",
            cfg_path=cfg_path,
            frame_skip=int(frame_skip),
            steps=len(actions),
            duration_s=float(t1 - t0),
            game_tics=int(game_tics),
            metadata={"map": map_name, "resolution": resolution, "bots": int(max(0, bots))},
        )
        eff_fps = float(fps) if fps > 0 else max(1, round(35.0 / max(1, frame_skip), 2))
        annotate_and_encode(
            ep, str(video_path), fps=eff_fps,
            audio=audio_np if (audio_np is not None and audio_np.size > 0 and audio_np.max() != 0) else None,
            audio_sample_rate=22050,
        )

    click.echo("\n✓ Human demo saved")
    click.echo(f"  data: {npz_path}")
    click.echo(f"  meta: {meta_path}")
    if video_path is not None:
        click.echo(f"  video: {video_path}")


def main():
    cli()


if __name__ == "__main__":
    main()
