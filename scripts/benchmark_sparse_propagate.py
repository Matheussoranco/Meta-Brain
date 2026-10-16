"""Milestone M1 benchmark: BPTT through the full connectome with SparsePropagate.

Runs the planned layer-1 workload (see :mod:`meta_brain.benchmark`) and
reports time and peak memory for the custom operator versus the
plain-autograd gather/scatter reference.

Examples::

    # Synthetic graph at MaleCNS (Traced) scale, no data needed
    python scripts/benchmark_sparse_propagate.py --frames 4 --micro-steps 8 --batch 8

    # Real connectome (built and cached on first use)
    python scripts/benchmark_sparse_propagate.py --data-dir data/connectome

    # Both operators, each in a fresh process so peak memory is clean
    python scripts/benchmark_sparse_propagate.py --sweep --out results/benchmarks/m1.json

Go criterion for M1: on CUDA, ``--micro-steps 8 --batch 8 --frames 16`` with
per-frame checkpointing stays under 16 GB.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_brain.benchmark import bptt_workload, prepare_tensors, synthetic_graph  # noqa: E402

# MaleCNS v1.0 with status == Traced: 165,122 neurons, 25,563,096 edges (measured on Colab).
DEFAULT_NEURONS = 165_122
DEFAULT_EDGES = 25_563_096


def real_graph(data_dir: str) -> dict:
    from meta_brain.graph import load_connectome

    g = load_connectome(data_dir)
    readout = g.descending_indices()
    if len(readout) == 0:
        readout = g.motor_indices()
    return dict(n=g.n, pre=g.pre, post=g.post, count=g.count, cell_type=g.cell_type,
                n_types=g.n_types, sign=g.sign(), readout=readout, source=str(data_dir))


def run(args) -> dict:
    device = torch.device(args.device)
    t0 = time.perf_counter()
    graph = real_graph(args.data_dir) if args.data_dir else synthetic_graph(
        args.neurons, args.edges, args.types, args.seed)
    tensors = prepare_tensors(graph["n"], graph["pre"], graph["post"], graph["count"], graph["sign"],
                              graph["cell_type"], graph["n_types"], graph["readout"], device,
                              index_dtype=getattr(torch, args.index_dtype))
    tensors["structure"].edge_chunk = args.edge_chunk
    setup_s = time.perf_counter() - t0
    row = bptt_workload(**tensors, op=args.op, batch=args.batch, frames=args.frames,
                        micro_steps=args.micro_steps, use_checkpoint=args.checkpoint,
                        iters=args.iters, warmup=args.warmup, seed=args.seed)
    row.update(graph_source=graph["source"], device=str(device), torch=torch.__version__,
               device_name=(torch.cuda.get_device_name(device) if device.type == "cuda"
                            else platform.processor()),
               setup_s=round(setup_s, 2))
    return row


def sweep() -> list:
    passthrough, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
        elif a in ("--out", "--op"):
            skip = True
        elif a not in ("--sweep", "--json") and not a.startswith(("--out=", "--op=")):
            passthrough.append(a)
    rows = []
    for op in ("custom", "reference"):
        cmd = [sys.executable, __file__, "--op", op, "--json"] + passthrough
        print("$", " ".join(cmd[1:]), flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            rows.append({"op": op, "status": "ERROR", "error": " | ".join(tail)})
        else:
            rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="Use the real connectome instead of a synthetic graph.")
    p.add_argument("--neurons", type=int, default=DEFAULT_NEURONS)
    p.add_argument("--edges", type=int, default=DEFAULT_EDGES)
    p.add_argument("--types", type=int, default=8_000)
    p.add_argument("--op", choices=("custom", "reference"), default="custom")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--micro-steps", type=int, default=8)
    p.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    p.add_argument("--index-dtype", choices=("int32", "int64"), default="int32")
    p.add_argument("--edge-chunk", type=int, default=1 << 22)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", action="store_true", help="Run custom and reference in separate processes.")
    p.add_argument("--json", action="store_true", help="Print only the result JSON.")
    p.add_argument("--out", default=None, help="Write results JSON here.")
    args = p.parse_args()

    rows = sweep() if args.sweep else [run(args)]
    if not args.sweep:
        print(json.dumps(rows[0]) if args.json else json.dumps(rows[0], indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
