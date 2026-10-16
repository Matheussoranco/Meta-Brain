"""The milestone-M1 workload: BPTT through the whole connectome.

Shared by ``scripts/benchmark_sparse_propagate.py`` and the Colab notebook.
The workload is the planned layer-1 model: leaky rate dynamics with per-type
``tau`` and ``v0`` and a per-type-pair gain, ``K`` micro-steps per frame,
``T`` frames, batch ``B``, optional per-frame gradient checkpointing.
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .graph import merge_duplicates
from .sparse_ops import SparseStructure, reference_propagate, sparse_propagate

GIB = 1024 ** 3
OPS: Dict[str, Callable] = {"custom": sparse_propagate, "reference": reference_propagate}


def synthetic_graph(n: int, e: int, n_types: int, seed: int = 0, n_readout: int = 1300) -> dict:
    """Heavy-tailed random graph with connectome-like degree spread.

    Types are random, so almost every edge gets its own type pair; the real
    connectome has far fewer pairs. Use it for timing, not parameter counts.
    """
    rng = np.random.default_rng(seed)
    w_out = rng.pareto(1.3, n) + 1
    w_in = rng.pareto(1.3, n) + 1
    m = int(e * 1.15)  # oversample to compensate for merged duplicates and self-loops
    pre = rng.choice(n, m, p=w_out / w_out.sum())
    post = rng.choice(n, m, p=w_in / w_in.sum())
    keep = pre != post
    count = rng.geometric(0.15, m).astype(np.float32)
    pre, post, count, _ = merge_duplicates(pre[keep], post[keep], count[keep], n)
    if len(pre) > e:
        idx = np.sort(rng.choice(len(pre), e, replace=False))
        pre, post, count = pre[idx], post[idx], count[idx]
    return dict(n=n, pre=pre, post=post, count=count,
                cell_type=rng.integers(0, n_types, n).astype(np.int32), n_types=n_types,
                sign=np.where(rng.random(n) < 0.3, -1.0, 1.0).astype(np.float32),
                readout=rng.choice(n, min(n_readout, n), replace=False), source="synthetic")


def base_weights(pre: np.ndarray, post: np.ndarray, count: np.ndarray, sign: np.ndarray, n: int) -> np.ndarray:
    """Signed log counts normalized by incoming energy (as in the V2 experiment)."""
    w = np.log1p(count).astype(np.float32)
    energy = np.bincount(post, w * w, n).astype(np.float32)
    w = w / np.sqrt(np.maximum(energy[post], 1e-6))
    return w * sign[pre]


def process_peak_bytes() -> int:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or getattr(info, "rss", 0))
    except ImportError:
        import resource  # Linux / macOS

        scale = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(exc: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", MemoryError)
    return isinstance(exc, (oom, MemoryError)) or "out of memory" in str(exc).lower()


def bptt_workload(structure: SparseStructure, base: torch.Tensor, edge_pair: torch.Tensor,
                  n_pairs: int, cell_type: torch.Tensor, n_types: int, readout: torch.Tensor,
                  op: str = "custom", batch: int = 8, frames: int = 4, micro_steps: int = 8,
                  use_checkpoint: bool = True, iters: int = 2, warmup: int = 1, seed: int = 0,
                  dt: float = 0.25, gain: float = 1.6) -> dict:
    """Time forward + backward of the layer-1 workload; returns a result row.

    All tensors must already live on ``structure.device``. On CUDA, peak
    memory is ``torch.cuda.max_memory_allocated`` over the whole call
    (including the graph tensors already resident). An out-of-memory error is
    caught and reported as ``status='OOM'``.
    """
    device = structure.device
    propagate_fn = OPS[op]
    row = {"op": op, "batch": batch, "frames": frames, "micro_steps": micro_steps,
           "checkpoint": bool(use_checkpoint), "neurons": structure.n, "edges": structure.e,
           "type_pairs": int(n_pairs), "cell_types": int(n_types)}
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        log_tau = torch.zeros(n_types, device=device, requires_grad=True)
        v0 = torch.zeros(n_types, device=device, requires_grad=True)
        pair_gain = torch.zeros(n_pairs, device=device, requires_grad=True)
        params = [log_tau, v0, pair_gain]
        drive = (torch.randn(frames, structure.n, batch, generator=gen) * 0.1).to(device)

        def frame(v: torch.Tensor, values: torch.Tensor, alpha: torch.Tensor, rest: torch.Tensor,
                  ext: torch.Tensor) -> torch.Tensor:
            for _ in range(micro_steps):
                # Saturating rate: with ReLU alone the real connectome's gain-1.6
                # dynamics diverge within ~100 micro-steps (non-finite gradients).
                rec = propagate_fn(torch.tanh(F.relu(v)), values, structure)
                v = v + alpha * (-v + rest + gain * rec + ext)
            return v

        timings = []
        grad_norm = 0.0
        for it in range(warmup + iters):
            _sync(device)
            t_start = time.perf_counter()
            values = base * F.softplus(pair_gain + 0.5413)[edge_pair]  # softplus(0.5413) ~= 1
            alpha = (dt / (1.0 + torch.exp(log_tau)))[cell_type].unsqueeze(1)
            rest = v0[cell_type].unsqueeze(1)
            v = torch.zeros(structure.n, batch, device=device)
            loss = torch.zeros((), device=device)
            for t in range(frames):
                if use_checkpoint:
                    v = checkpoint(frame, v, values, alpha, rest, drive[t], use_reentrant=False)
                else:
                    v = frame(v, values, alpha, rest, drive[t])
                loss = loss + F.relu(v.index_select(0, readout)).square().mean()
            _sync(device)
            t_fwd = time.perf_counter()
            loss.backward()
            _sync(device)
            t_bwd = time.perf_counter()
            grads = [p.grad for p in params]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("missing or non-finite gradient")
            grad_norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads)))
            for p in params:
                p.grad = None
            if it >= warmup:
                timings.append((t_fwd - t_start, t_bwd - t_fwd))
            del values, alpha, rest, v, loss
    except (RuntimeError, MemoryError) as exc:
        if not _is_oom(exc):
            raise
        row.update(status="OOM", error=str(exc).splitlines()[0][:200])
        if device.type == "cuda":
            row["peak_memory_gib"] = round(torch.cuda.max_memory_allocated(device) / GIB, 3)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return row

    fwd = float(np.median([a for a, _ in timings]))
    bwd = float(np.median([b for _, b in timings]))
    steps = frames * micro_steps
    peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else process_peak_bytes())
    row.update(
        status="OK",
        forward_s=round(fwd, 4),
        backward_s=round(bwd, 4),
        ms_per_micro_step=round(1000 * (fwd + bwd) / steps, 3),
        grad_norm=grad_norm,
        trainable_params=int(n_types * 2 + n_pairs),
        peak_memory_gib=round(peak / GIB, 3),
        peak_memory_kind="cuda.max_memory_allocated" if device.type == "cuda" else "process peak RSS",
    )
    return row


def prepare_tensors(n: int, pre: np.ndarray, post: np.ndarray, count: np.ndarray, sign: np.ndarray,
                    cell_type: np.ndarray, n_types: int, readout: np.ndarray, device: torch.device | str,
                    edge_pair: Optional[np.ndarray] = None, n_pairs: Optional[int] = None,
                    index_dtype: torch.dtype = torch.int32) -> dict:
    """Move a graph onto ``device`` in the form :func:`bptt_workload` expects."""
    from .graph import type_pair_index

    if edge_pair is None:
        edge_pair, pair_types = type_pair_index(pre, post, cell_type, n_types)
        n_pairs = len(pair_types)
    device = torch.device(device)
    structure = SparseStructure.from_edges(pre, post, n, device=device, index_dtype=index_dtype)
    return dict(
        structure=structure,
        base=torch.as_tensor(base_weights(pre, post, count, sign, n), device=device),
        edge_pair=torch.as_tensor(edge_pair, device=device, dtype=torch.long),
        n_pairs=int(n_pairs),
        cell_type=torch.as_tensor(cell_type, device=device, dtype=torch.long),
        n_types=int(n_types),
        readout=torch.as_tensor(readout, device=device, dtype=torch.long),
    )
