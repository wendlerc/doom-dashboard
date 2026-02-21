"""
Parallelized WebDataset generator.

Architecture
────────────
  ┌─ Main ──────────────────────────────────────────────────────────┐
  │  1. Compute work schedule (scenario × policy combos + counts)   │
  │  2. If GPU policy: spawn InferenceServer in a daemon thread      │
  │  3. Spawn N rollout workers via multiprocessing.Pool             │
  │     N = min(os.cpu_count(), 32) by default                      │
  │  4. Each worker writes its own shards independently              │
  │  5. Progress bar updates via mp.Manager().Queue                  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─ Worker (per CPU core) ─────────────────────────────────────────┐
  │  • Creates its own DoomGame (VizDoom is NOT thread-safe)         │
  │  • For RandomPolicy: carries its own policy object               │
  │  • For GPU policy:  sends obs to InferenceServer, waits for act  │
  │  • Writes shard-{worker_id:05d}-{shard_idx:05d}.tar via          │
  │    webdataset.ShardWriter                                         │
  └─────────────────────────────────────────────────────────────────┘

  ┌─ InferenceServer (daemon thread, runs policy on GPU) ───────────┐
  │  • Receives (worker_id, obs_bytes) from req_queue                │
  │  • Batches a tick worth of requests, runs forward() on GPU       │
  │  • Sends (worker_id, action) back via res_queues[worker_id]      │
  └─────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import os
import queue
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from doom_dashboard.config import Config, DatasetConfig, PolicyConfig, ScenarioConfig
from doom_dashboard.policies import load_policy, RandomPolicy, BasePolicy
from doom_dashboard.rollout import rollout_episode, EpisodeData


# ──────────────────────────────────────────────────────────────────
# Serialisation helpers (shared-mem friendly)
# ──────────────────────────────────────────────────────────────────

def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────
# GPU Inference Server (daemon thread)
# ──────────────────────────────────────────────────────────────────

class InferenceServer(threading.Thread):
    """Single GPU thread that batches obs from N workers and returns actions.

    Workers send: (worker_id, obs_bytes, button_names_json)
    Server replies via per-worker response queues: action_array_bytes
    """

    # sentinel to shut down
    STOP = "__STOP__"

    def __init__(
        self,
        policy_cfg: PolicyConfig,
        req_queue: "mp.Queue[Any]",
        res_queues: Dict[int, "mp.Queue[Any]"],
        max_batch_size: int = 32,
        batch_timeout_ms: float = 5.0,
    ):
        super().__init__(daemon=True, name="inference-server")
        self.policy_cfg = policy_cfg
        self.req_queue = req_queue
        self.res_queues = res_queues
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0
        self._policy: Optional[BasePolicy] = None

    def run(self):
        # Load policy on GPU in this thread
        self._policy = load_policy(self.policy_cfg)
        print(f"[InferenceServer] Policy '{self.policy_cfg.name}' loaded on GPU.")

        while True:
            # Collect a micro-batch
            requests: List[Tuple[int, np.ndarray, List[str]]] = []
            deadline = time.perf_counter() + self.batch_timeout_s
            while len(requests) < self.max_batch_size:
                try:
                    timeout = max(0, deadline - time.perf_counter())
                    item = self.req_queue.get(timeout=timeout)
                    if item == self.STOP:
                        return
                    requests.append(item)  # (worker_id, obs, button_names)
                except queue.Empty:
                    break

            if not requests:
                continue

            # Dispatch individually (batched GPU inference is only meaningful
            # for large Torch models; SB3 predict() handles its own batching)
            for wid, obs, btn_names in requests:
                action = self._policy.predict(obs, btn_names)
                self.res_queues[wid].put(action)

        if self._policy:
            self._policy.close()


# ──────────────────────────────────────────────────────────────────
# Proxy policy used by workers to talk to InferenceServer
# ──────────────────────────────────────────────────────────────────

class RemotePolicy(BasePolicy):
    """Worker-side stub: sends obs to GPU server, blocks for action."""

    def __init__(self, name: str, worker_id: int,
                 req_queue: "mp.Queue", res_queue: "mp.Queue"):
        self.name = name
        self._wid = worker_id
        self._req = req_queue
        self._res = res_queue

    def predict(self, obs: np.ndarray, available_buttons=None) -> np.ndarray:
        self._req.put((self._wid, obs, available_buttons or []))
        return self._res.get()


# ──────────────────────────────────────────────────────────────────
# Per-worker entry point (called in subprocess)
# ──────────────────────────────────────────────────────────────────

def _worker_fn(
    worker_id: int,
    work_items: List[Tuple[str, str]],   # [(scenario_name, policy_name), ...]
    scenarios: List[dict],               # serialisable ScenarioConfig dicts
    policies: List[dict],                # serialisable PolicyConfig dicts
    output_dir: str,
    shard_size_mb: int,
    frame_skip: int,
    render_resolution: str,
    render_hud: bool,
    progress_queue: "mp.Queue",
    # GPU server queues (None if not used)
    gpu_req_queue: Optional["mp.Queue"],
    gpu_res_queue: Optional["mp.Queue"],
):
    """Run inside a subprocess. Each worker owns its own DoomGame."""
    import webdataset as wds
    from doom_dashboard.config import ScenarioConfig, PolicyConfig
    from doom_dashboard.policies import load_policy, RandomPolicy

    # Rebuild config objects from dicts
    sc_map = {d["name"]: ScenarioConfig.from_dict(d) for d in scenarios}
    pol_map = {d["name"]: PolicyConfig.from_dict(d) for d in policies}

    # Build per-policy loader (or remote stub)
    policy_cache: Dict[str, BasePolicy] = {}

    def get_policy(pname: str) -> BasePolicy:
        if pname not in policy_cache:
            pcfg = pol_map[pname]
            if gpu_req_queue is not None and pcfg.type != "random":
                # Use GPU inference server
                policy_cache[pname] = RemotePolicy(
                    name=pname,
                    worker_id=worker_id,
                    req_queue=gpu_req_queue,
                    res_queue=gpu_res_queue,
                )
            else:
                policy_cache[pname] = load_policy(pcfg)
        return policy_cache[pname]

    # Shard writer for this worker
    shard_pat = os.path.join(
        output_dir, f"shard-{worker_id:05d}-%05d.tar"
    )
    shard_bytes = shard_size_mb * 1024 * 1024

    with wds.ShardWriter(shard_pat, maxsize=shard_bytes) as writer:
        for scenario_name, policy_name in work_items:
            sc = sc_map[scenario_name]
            # override dataset-level frame_skip / resolution / hud
            sc_override = ScenarioConfig(
                name=sc.name,
                cfg=sc.cfg,
                episode_timeout=sc.episode_timeout,
                frame_skip=frame_skip,
                render_resolution=render_resolution,
                render_hud=render_hud,
            )
            policy = get_policy(policy_name)

            ep = rollout_episode(
                scenario=sc_override,
                policy=policy,
                record_frames=True,
            )

            if len(ep.frames) == 0:
                progress_queue.put(("skip", worker_id, scenario_name, policy_name, 0.0))
                continue

            frames_arr = np.stack(ep.frames, axis=0)   # (T, H, W, C) uint8
            actions_arr = np.stack(ep.actions, axis=0)  # (T, n_buttons)
            rewards_arr = np.array(ep.rewards, dtype=np.float32)

            meta = {
                "episode_id": str(uuid.uuid4()),
                "scenario": scenario_name,
                "policy": policy_name,
                "cfg_path": ep.cfg_path,
                "frame_skip": ep.frame_skip,
                "render_resolution": render_resolution,
                "render_hud": render_hud,
                "button_names": ep.button_names,
                "steps": ep.steps,
                "total_reward": ep.total_reward,
                "game_tics": ep.game_tics,
                "duration_s": ep.duration_s,
                "timestamp": time.time(),
                "worker_id": worker_id,
            }
            ep_id = meta["episode_id"]

            writer.write({
                "__key__": f"ep_{ep_id}",
                "frames.npy": _npy_bytes(frames_arr),
                "actions.npy": _npy_bytes(actions_arr),
                "rewards.npy": _npy_bytes(rewards_arr),
                "meta.json": json.dumps(meta).encode(),
            })

            game_secs = ep.game_tics / 35.0  # VizDoom runs at ~35 tics/s
            progress_queue.put(("done", worker_id, scenario_name, policy_name, game_secs))

    # Signal done
    progress_queue.put(("worker_done", worker_id, None, None, 0.0))

    for p in policy_cache.values():
        p.close()


# ──────────────────────────────────────────────────────────────────
# Work schedule
# ──────────────────────────────────────────────────────────────────

def _build_work_schedule(
    dataset_cfg: DatasetConfig,
    scenario_cfgs: List[ScenarioConfig],
    policy_cfgs: List[PolicyConfig],
) -> List[Tuple[str, str]]:
    """Return a flat list of (scenario_name, policy_name) rollout tasks.

    Sizes are estimated from total_hours × ratios. We over-provision by 10%
    and stop workers once target is hit (they check progress_queue).
    """
    total_secs = dataset_cfg.total_hours * 3600.0

    sc_names = [s.name for s in scenario_cfgs]
    pol_names = [p.name for p in policy_cfgs]

    # default to uniform if not specified
    sc_ratios = dataset_cfg.scenario_ratios or {n: 1.0 / len(sc_names) for n in sc_names}
    pol_ratios = dataset_cfg.policy_ratios or {n: 1.0 / len(pol_names) for n in pol_names}

    # Normalise
    sc_tot = sum(sc_ratios.values()) or 1.0
    pol_tot = sum(pol_ratios.values()) or 1.0
    sc_ratios = {k: v / sc_tot for k, v in sc_ratios.items()}
    pol_ratios = {k: v / pol_tot for k, v in pol_ratios.items()}

    # Estimate episode duration per scenario from timeout + frame_skip
    # VizDoom ~35 tics per second
    TIC_RATE = 35.0
    schedule: List[Tuple[str, str]] = []

    for sc in scenario_cfgs:
        if sc.name not in sc_ratios:
            continue
        timeout = sc.episode_timeout or 2100  # fallback
        ep_secs = (timeout / dataset_cfg.frame_skip) / (TIC_RATE / dataset_cfg.frame_skip)
        ep_secs = timeout / TIC_RATE  # real game seconds

        sc_secs = total_secs * sc_ratios[sc.name]
        for pol in policy_cfgs:
            if pol.name not in pol_ratios:
                continue
            pol_secs = sc_secs * pol_ratios[pol.name]
            n_eps = max(1, math.ceil(pol_secs / ep_secs * 1.15))  # 15% buffer
            schedule.extend([(sc.name, pol.name)] * n_eps)

    random.shuffle(schedule)
    return schedule


# ──────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────

def generate_dataset(
    config: Config,
    target_hours: Optional[float] = None,
    num_workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    progress_callback=None,
) -> str:
    """Generate a WebDataset from the given config.

    Parameters
    ----------
    config:             Full Config object
    target_hours:       Override config.dataset.total_hours
    num_workers:        Override config.dataset.num_workers
    output_dir:         Override config.dataset.output_dir
    progress_callback:  Callable(hours_done, total_hours) — called from main thread

    Returns
    -------
    output_dir path
    """
    ds_cfg = config.dataset
    if target_hours is not None:
        ds_cfg.total_hours = target_hours
    if output_dir is not None:
        ds_cfg.output_dir = output_dir

    out_dir = Path(ds_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = config.scenarios
    policies = config.policies

    # Filter to scenarios/policies mentioned in ratios (or use all)
    if ds_cfg.scenario_ratios:
        scenarios = [s for s in scenarios if s.name in ds_cfg.scenario_ratios]
    if ds_cfg.policy_ratios:
        policies = [p for p in policies if p.name in ds_cfg.policy_ratios]

    assert scenarios, "No scenarios selected for dataset generation"
    assert policies, "No policies selected for dataset generation"

    work = _build_work_schedule(ds_cfg, scenarios, policies)
    n_workers = num_workers or ds_cfg.num_workers or min(mp.cpu_count(), 32)
    print(f"[dataset_gen] {len(work)} episodes planned, {n_workers} workers, "
          f"target {ds_cfg.total_hours:.2f} h")

    # Decide if we need a GPU inference server
    has_gpu_policy = any(p.type != "random" for p in policies)
    gpu_req_queue: Optional[mp.Queue] = None
    res_queues: Dict[int, mp.Queue] = {}
    server: Optional[InferenceServer] = None

    if has_gpu_policy:
        gpu_req_queue = mp.Queue(maxsize=n_workers * 4)
        res_queues = {i: mp.Queue(maxsize=4) for i in range(n_workers)}
        # Use the first non-random policy for the server
        gpu_pol_cfg = next(p for p in policies if p.type != "random")
        server = InferenceServer(gpu_pol_cfg, gpu_req_queue, res_queues)
        server.start()
        print(f"[dataset_gen] GPU inference server started for '{gpu_pol_cfg.name}'")

    # Split work across workers
    chunks: List[List[Tuple[str, str]]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(work):
        chunks[i % n_workers].append(item)

    progress_queue: mp.Queue = mp.Queue()

    # Serialise config objects to plain dicts (picklable)
    sc_dicts = [dict(
        name=s.name, cfg=s.cfg,
        episode_timeout=s.episode_timeout,
        frame_skip=s.frame_skip,
        render_resolution=s.render_resolution,
        render_hud=s.render_hud,
    ) for s in config.scenarios]
    pol_dicts = [dict(
        name=p.name, type=p.type, path=p.path,
        algo=p.algo, arch=p.arch, action_size=p.action_size, device=p.device,
    ) for p in config.policies]

    pool_args = [
        (
            wid,
            chunks[wid],
            sc_dicts,
            pol_dicts,
            str(out_dir),
            ds_cfg.shard_size_mb,
            ds_cfg.frame_skip,
            ds_cfg.render_resolution,
            ds_cfg.render_hud,
            progress_queue,
            gpu_req_queue,
            res_queues.get(wid),
        )
        for wid in range(n_workers)
    ]

    # Progress tracking
    total_target_secs = ds_cfg.total_hours * 3600.0
    accumulated_secs = 0.0
    workers_done = 0

    ctx = mp.get_context("spawn")  # spawn avoids CUDA fork issues

    with ctx.Pool(processes=n_workers, initializer=os.setsid) as pool:
        async_results = [pool.apply_async(_worker_fn, args=a) for a in pool_args]

        # Progress loop
        try:
            from tqdm import tqdm
            pbar = tqdm(total=ds_cfg.total_hours, unit="h", desc="Dataset", ncols=90)
        except ImportError:
            pbar = None

        while workers_done < n_workers:
            try:
                msg = progress_queue.get(timeout=2.0)
            except Exception:
                continue

            kind = msg[0]
            if kind == "done":
                _, wid, sc, pol, game_secs = msg
                accumulated_secs += game_secs
                hours_done = accumulated_secs / 3600.0
                if pbar:
                    pbar.n = min(hours_done, ds_cfg.total_hours)
                    pbar.set_postfix(w=wid, sc=sc, pol=pol)
                    pbar.refresh()
                if progress_callback:
                    progress_callback(hours_done, ds_cfg.total_hours)
                # Early stop: cancel remaining workers
                if accumulated_secs >= total_target_secs:
                    pool.terminate()
                    break
            elif kind == "worker_done":
                workers_done += 1
            # "skip" → ignore

        if pbar:
            pbar.n = min(ds_cfg.total_hours, ds_cfg.total_hours)
            pbar.close()

    # Shut down GPU server
    if server is not None:
        gpu_req_queue.put(InferenceServer.STOP)
        server.join(timeout=5)

    print(f"[dataset_gen] Done. Data in '{out_dir}'. "
          f"Collected {accumulated_secs / 3600:.3f} h of gameplay.")
    return str(out_dir)
