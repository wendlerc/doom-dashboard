"""
Parallelized 1v1 multiplayer dataset generator.

Spawns M game-pairs in parallel. Each pair = host subprocess + join subprocess
sharing a unique port.  Results written as WebDataset shards with both
players' data per sample.

Port allocation: worker_id → BASE_PORT + worker_id * 10
  (10 ports per game leaves room for VizDoom's internal range)

Policy sampling:
  - For each game, P1 policy is chosen according to policy_ratios
  - P2 policy is chosen independently (same distribution)
  - This naturally produces:  random vs random, trained vs trained, mixed 

Shard schema per sample (one sample = one 1v1 episode):
  frames_p1.npy   (T, H, W, 3) uint8
  actions_p1.npy  (T, n_buttons) bool
  rewards_p1.npy  (T,) float32
  frames_p2.npy   (T, H, W, 3) uint8
  actions_p2.npy  (T, n_buttons) bool
  rewards_p2.npy  (T,) float32
  meta.json       {scenario, map, policies, button_names, game_tics, ...}
"""
from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import vizdoom as vzd

from doom_dashboard.multiplayer_rollout import (
    MultiEpisodeData,
    rollout_multiplayer_episode,
    BASE_PORT,
    DEATHMATCH_MAPS,
    CIG_MAPS,
)


# ─── helpers ─────────────────────────────────────────────────────

def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _sample_policy(policy_names: List[str], weights: List[float]) -> str:
    return random.choices(policy_names, weights=weights, k=1)[0]


# ─── multiplayer scenario registry ───────────────────────────────

# Scenarios suitable for 1v1 deathmatch with their WAD map lists.
# 'base_cfg' keys map to doom_dashboard/config.py BUILTIN_SCENARIOS.
MP_SCENARIOS = {
    "deathmatch": {
        "cfg": str(os.path.join(vzd.scenarios_path, "deathmatch.cfg")),
        "maps": ["map01"],
        "timelimit_min": 5.0,
    },
    "cig": {
        "cfg": str(os.path.join(vzd.scenarios_path, "cig.cfg")),
        "maps": CIG_MAPS,
        "timelimit_min": 5.0,
    },
    "multi_duel": {
        "cfg": str(os.path.join(vzd.scenarios_path, "multi_duel.cfg")),
        "maps": ["map01"],
        "timelimit_min": 3.0,
    },
}


# ─── worker function ──────────────────────────────────────────────

def _mp_worker_fn(
    worker_id: int,
    n_episodes: int,
    scenario_names: List[str],
    scenario_weights: List[float],
    mp_scenario_meta: Dict,
    policy_names: List[str],
    policy_weights: List[float],
    policy_dicts: List[dict],
    output_dir: str,
    shard_size_mb: int,
    frame_skip: int,
    render_resolution: str,
    render_hud: bool,
    progress_queue,           # Manager().Queue()
    target_secs: float,
):
    """Run inside a spawned subprocess. Launches episodes via subprocess pairs."""
    import webdataset as wds
    from doom_dashboard.multiplayer_rollout import rollout_multiplayer_episode, BASE_PORT

    port = BASE_PORT + worker_id * 10
    pol_by_name = {d["name"]: d for d in policy_dicts}

    shard_pat   = os.path.join(output_dir, f"mp-shard-{worker_id:05d}-%05d.tar")
    shard_bytes = shard_size_mb * 1024 * 1024
    accumulated_game_secs = 0.0

    # Normalise weights
    total_w = sum(policy_weights) or 1.0
    ok_weights = [w / total_w for w in policy_weights]

    with wds.ShardWriter(shard_pat, maxsize=shard_bytes) as writer:
        for ep_idx in range(n_episodes):
            if accumulated_game_secs >= target_secs:
                break

            sc_name  = random.choices(scenario_names, weights=scenario_weights, k=1)[0]
            sc_meta  = mp_scenario_meta[sc_name]
            doom_map = random.choice(sc_meta["maps"])
            timelimit = sc_meta["timelimit_min"]

            pol_name_p1 = _sample_policy(policy_names, ok_weights)
            pol_name_p2 = _sample_policy(policy_names, ok_weights)
            pol_dict_p1 = pol_by_name[pol_name_p1]
            pol_dict_p2 = pol_by_name[pol_name_p2]

            try:
                ep: MultiEpisodeData = rollout_multiplayer_episode(
                    cfg_path=sc_meta["cfg"],
                    scenario_name=sc_name,
                    doom_map=doom_map,
                    policy_p1_dict=pol_dict_p1,
                    policy_p2_dict=pol_dict_p2,
                    timelimit_minutes=timelimit,
                    frame_skip=frame_skip,
                    render_resolution=render_resolution,
                    render_hud=render_hud,
                    port=port,
                )
            except Exception as exc:
                progress_queue.put(("error", worker_id, sc_name, str(exc)))
                time.sleep(1)
                continue

            if not ep.frames_p1 or not ep.frames_p2:
                progress_queue.put(("skip", worker_id, sc_name, "empty frames"))
                continue

            f1 = np.stack(ep.frames_p1, axis=0)
            a1 = np.stack(ep.actions_p1, axis=0)
            r1 = np.array(ep.rewards_p1, dtype=np.float32)
            f2 = np.stack(ep.frames_p2, axis=0)
            a2 = np.stack(ep.actions_p2, axis=0)
            r2 = np.array(ep.rewards_p2, dtype=np.float32)

            ep_id = str(uuid.uuid4())
            meta = {
                "episode_id": ep_id,
                "scenario": sc_name,
                "map": doom_map,
                "policy_p1": pol_name_p1,
                "policy_p2": pol_name_p2,
                "button_names": ep.button_names,
                "frame_skip": frame_skip,
                "render_resolution": render_resolution,
                "timelimit_minutes": timelimit,
                "steps_p1": len(ep.frames_p1),
                "steps_p2": len(ep.frames_p2),
                "total_reward_p1": float(np.sum(r1)),
                "total_reward_p2": float(np.sum(r2)),
                "game_tics": ep.game_tics,
                "duration_s": ep.duration_s,
                "timestamp": time.time(),
                "worker_id": worker_id,
            }
            writer.write({
                "__key__": f"ep_{ep_id}",
                "frames_p1.npy":  _npy_bytes(f1),
                "actions_p1.npy": _npy_bytes(a1),
                "rewards_p1.npy": _npy_bytes(r1),
                "frames_p2.npy":  _npy_bytes(f2),
                "actions_p2.npy": _npy_bytes(a2),
                "rewards_p2.npy": _npy_bytes(r2),
                "meta.json":      json.dumps(meta).encode(),
            })

            game_secs = ep.game_tics / 35.0
            accumulated_game_secs += game_secs
            progress_queue.put(("done", worker_id, sc_name, pol_name_p1, pol_name_p2, game_secs))

    progress_queue.put(("worker_done", worker_id, accumulated_game_secs))



# ─── public entry point ───────────────────────────────────────────

def generate_multiplayer_dataset(
    output_dir: str,
    total_hours: float,
    policy_dicts: List[dict],          # from config.policies
    random_policy_ratio: float = 0.05,
    scenario_ratios: Optional[Dict[str, float]] = None,
    frame_skip: int = 4,
    render_resolution: str = "RES_320X240",
    render_hud: bool = False,
    shard_size_mb: int = 512,
    num_workers: Optional[int] = None,
    progress_callback=None,
) -> str:
    """
    Generate a multiplayer WebDataset with 1v1 games.

    Parameters
    ----------
    output_dir:           Where to write .tar shards
    total_hours:          Target total gameplay hours
    policy_dicts:         List of serialisable policy config dicts
    random_policy_ratio:  Fraction of policy-slots filled by Random (default 5%)
    scenario_ratios:      {scenario_name: weight} — defaults to uniform
    frame_skip, render_resolution, render_hud: game settings
    shard_size_mb:        Max shard size in MB
    num_workers:          Parallel game pairs (default: auto, max 32)
    progress_callback:    Callable(hours_done, total_hours)
    """
    import webdataset  # verify installed

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build scenario list
    sc_names = list(MP_SCENARIOS.keys())
    if scenario_ratios:
        sc_names = [k for k in scenario_ratios if k in MP_SCENARIOS]
    sc_weights = [scenario_ratios.get(s, 1.0) for s in sc_names] if scenario_ratios else [1.0] * len(sc_names)

    assert sc_names, f"No valid multiplayer scenarios found. Valid: {list(MP_SCENARIOS)}"

    # Build policy sampling distribution
    random_pols  = [p for p in policy_dicts if p["type"] == "random"]
    trained_pols = [p for p in policy_dicts if p["type"] != "random"]

    if not random_pols:
        random_pols = [{"name": "Random", "type": "random", "path": None,
                        "algo": "PPO", "arch": "DuelQNet", "action_size": None, "device": "auto"}]

    pol_list: List[dict] = []
    pol_weights: List[float] = []

    if trained_pols:
        trained_share = 1.0 - random_policy_ratio
        per_trained = trained_share / len(trained_pols)
        for p in trained_pols:
            pol_list.append(p)
            pol_weights.append(per_trained)
    else:
        random_policy_ratio = 1.0

    rnd_per = random_policy_ratio / len(random_pols)
    for p in random_pols:
        pol_list.append(p)
        pol_weights.append(rnd_per)

    pol_names = [p["name"] for p in pol_list]

    # Work distribution
    n_workers = min(num_workers or mp.cpu_count(), 32)
    mean_timelimit_s = sum(MP_SCENARIOS[s]["timelimit_min"] for s in sc_names) / len(sc_names) * 60
    total_secs = total_hours * 3600.0
    n_eps_total = max(n_workers, math.ceil(total_secs / mean_timelimit_s * 1.2))
    eps_per_worker = math.ceil(n_eps_total / n_workers)

    print(f"[mp_dataset_gen] {total_hours:.1f} h target, {n_workers} game-pair workers, "
          f"~{n_eps_total} episodes planned ({eps_per_worker}/worker)")
    print(f"  Scenarios: {sc_names}")
    print(f"  Policy distribution: "
          + ", ".join(f"{n}={w:.1%}" for n, w in zip(pol_names, pol_weights)))

    # Use Manager().Queue() so it can be pickled across spawn boundaries
    ctx = mp.get_context("spawn")
    with mp.Manager() as manager:
        progress_queue = manager.Queue()

        procs = []
        for wid in range(n_workers):
            p = ctx.Process(
                target=_mp_worker_fn,
                name=f"mp-worker-{wid}",
                args=(
                    wid,
                    eps_per_worker,
                    sc_names,
                    sc_weights,
                    {s: MP_SCENARIOS[s] for s in sc_names},
                    pol_names,
                    pol_weights,
                    pol_list,
                    str(out_dir),
                    shard_size_mb,
                    frame_skip,
                    render_resolution,
                    render_hud,
                    progress_queue,
                    total_secs,
                ),
            )
            p.start()
            procs.append(p)

        accumulated_secs = 0.0
        workers_done = 0

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_hours, unit="h", desc="MP Dataset", ncols=90)
        except ImportError:
            pbar = None

        while workers_done < n_workers:
            try:
                msg = progress_queue.get(timeout=5.0)
            except Exception:
                if all(not p.is_alive() for p in procs):
                    break
                continue

            kind = msg[0]
            if kind == "done":
                _, wid, sc, pol1, pol2, game_secs = msg
                accumulated_secs += game_secs
                hours_done = accumulated_secs / 3600.0
                if pbar:
                    pbar.n = min(hours_done, total_hours)
                    pbar.set_postfix(sc=sc, p1=pol1[:8], p2=pol2[:8])
                    pbar.refresh()
                if progress_callback:
                    progress_callback(hours_done, total_hours)
                if accumulated_secs >= total_secs:
                    for p in procs:
                        p.terminate()
                    break
            elif kind == "worker_done":
                workers_done += 1
            elif kind == "error":
                _, wid, sc, err = msg
                print(f"\n[worker {wid}] error on {sc}: {err}")

        if pbar:
            pbar.close()

        for p in procs:
            p.join(timeout=10)

    shards = sorted(out_dir.glob("mp-shard-*.tar"))
    print(f"\n[mp_dataset_gen] Done. {accumulated_secs/3600:.3f} h collected, "
          f"{len(shards)} shards in '{out_dir}'")
    return str(out_dir)

