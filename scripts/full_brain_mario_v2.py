"""Full-connectome Mario experiment for Meta-Brain V2.

This experiment is deliberately conservative about its claim:

* it refuses to run when the loaded graph is not the full MaleCNS graph;
* the connectome is a fixed, sparse recurrent reservoir;
* only a linear readout of the motor neurons is trained (PPO, see
  :mod:`meta_brain.mario_rl`);
* a run is called a success only when the post-training evaluation reaches
  the level without a death in the requested evaluation episodes.

The Colab notebook ``notebooks/colab_full_brain_mario_v2.ipynb`` runs the same
code (it embeds ``meta_brain/mario_rl.py``); this script is the local CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_full_connectome(data_dir: Path, min_neurons: int, min_edges: int) -> Dict[str, Any]:
    """Load MaleCNS v1.0 with :mod:`meta_brain.graph` and build signed reservoir weights.

    Keeps ``Traced`` bodies, sums any repeated weight rows into one edge per pair
    and signs each edge by the presynaptic neurotransmitter.
    """
    from meta_brain.graph import load_connectome

    graph = load_connectome(data_dir)
    if graph.n < min_neurons or graph.e < min_edges:
        raise RuntimeError(
            "Full-brain guard failed: loaded data has "
            f"{graph.n:,} neurons and {graph.e:,} edges, but the "
            f"experiment requires at least {min_neurons:,} neurons and {min_edges:,} edges."
        )
    pre = graph.pre.astype(np.int64)
    post = graph.post.astype(np.int64)
    weights = np.log1p(graph.count).astype(np.float32)
    # Normalize incoming connection energy so the full graph is numerically stable.
    indeg_sq = np.bincount(post, weights * weights, graph.n).astype(np.float32)
    weights = weights / np.sqrt(np.maximum(indeg_sq[post], 1e-6))
    weights = (weights * graph.sign()[pre]).astype(np.float32)

    sensory = graph.sensory_indices().astype(np.int64)
    motor = graph.motor_indices().astype(np.int64)
    if len(sensory) == 0 or len(motor) == 0:
        raise ValueError("Could not identify sensory and motor neurons from metadata.")

    print(
        f"FULL CONNECTOME VERIFIED: neurons={graph.n:,}, edges={graph.e:,}, "
        f"sensory={len(sensory):,}, motor={len(motor):,}"
    )
    return {
        "files": graph.report.files,
        "edge_accounting": asdict(graph.report),
        "n": graph.n,
        "e": graph.e,
        "pre": pre,
        "post": post,
        "weights": weights,
        "sensory": sensory,
        "motor": motor,
    }


def main() -> None:
    from meta_brain.mario_rl import (MarioSettings, PPOConfig, compare_vision_conditions, make_mario_env,
                                     run_experiment)
    from meta_brain.sparse_ops import SparseStructure

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/connectome")
    parser.add_argument("--out-dir", default="results/full_brain_mario_v2")
    parser.add_argument("--updates", type=int, default=int(os.getenv("MB_MARIO_UPDATES", "300")))
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--micro-steps", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--conditions", default="real,zero",
                        help="Comma-separated vision conditions: real, zero (blind), clock (blind with a timer).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-neurons", type=int, default=100000)
    parser.add_argument("--min-edges", type=int, default=5000000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Full-brain Mario V2 requires CUDA; refusing a misleading CPU run.")
    seed_all(args.seed)
    device = torch.device("cuda")
    graph = load_full_connectome(Path(args.data_dir), args.min_neurons, args.min_edges)
    probe, actions = make_mario_env("1-1", "right_only")
    probe.close()
    structure = SparseStructure.from_edges(graph["pre"], graph["post"], graph["n"], device=device,
                                           index_dtype=torch.int32)
    info = {k: graph[k] for k in ("files", "edge_accounting")}
    info.update(neurons=graph["n"], edges=graph["e"], sensory=len(graph["sensory"]), motor=len(graph["motor"]))
    runs = {}
    for condition in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        print(f"===== vision = {condition} =====")
        runs[condition] = run_experiment(
            structure, torch.as_tensor(graph["weights"]), graph["sensory"], graph["motor"],
            env_factory=lambda: make_mario_env("1-1", "right_only")[0],
            out_dir=Path(args.out_dir) / condition,
            settings=MarioSettings(n_envs=args.envs, micro_steps=args.micro_steps,
                                   eval_episodes=args.eval_episodes, seed=args.seed, vision=condition),
            ppo=PPOConfig(updates=args.updates), graph_info=info, n_actions=len(actions),
        )
    comparison = compare_vision_conditions(runs, args.out_dir)
    print(json.dumps({"out_dir": args.out_dir, "claim_status": {c: r["claim_status"] for c, r in runs.items()},
                      "comparison": comparison}, indent=2, default=float))


if __name__ == "__main__":
    main()
