"""Null-model graphs for the connectome controls (milestone M0).

Each generator returns new ``(pre, post, count)`` arrays with the same number
of neurons and, as far as possible, the same number of edges as the input.
Signs live on neurons (via the presynaptic NT), so they are preserved
automatically by any generator that keeps the presynaptic endpoint of an edge.

* :func:`degree_preserving_shuffle` permutes postsynaptic endpoints across all
  edges: every neuron keeps its exact in- and out-degree, and its out-strength.
* :func:`type_preserving_shuffle` rewires only within a ``(pre_type,
  post_type)`` block, by repeated edge swaps: the type-to-type connectivity
  table and each neuron's per-block degrees are kept; only individual wiring
  (e.g. retinotopy) is destroyed. Every swap keeps the graph simple.
* :func:`random_sparse_graph` keeps N, E and the weight multiset but nothing
  else.

The degree-preserving shuffle and the random graph create some self-loops and
duplicate edges. They are repaired with degree-preserving swaps; whatever
cannot be repaired within ``max_rounds`` is dropped (self-loops) or merged
(duplicates) and reported in :class:`ShuffleReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .graph import merge_duplicates, sort_edges


@dataclass
class ShuffleReport:
    method: str
    edges_in: int
    edges_out: int
    repair_rounds: int
    residual_self_loops_dropped: int
    residual_duplicates_merged: int
    swap_sweeps: int = 0
    swaps_accepted: int = 0

    @property
    def exact(self) -> bool:
        return self.residual_self_loops_dropped == 0 and self.residual_duplicates_merged == 0


Edges = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _bad_edges(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    """Mask of self-loops and every-but-the-first copy of a duplicate edge."""
    key = post.astype(np.int64) * n + pre.astype(np.int64)
    _, first = np.unique(key, return_index=True)
    dup = np.ones(len(key), dtype=bool)
    dup[first] = False
    return dup | (pre == post)


def _repair(pre: np.ndarray, post: np.ndarray, group: np.ndarray, n: int,
            rng: np.random.Generator, max_rounds: int) -> Tuple[np.ndarray, int]:
    """Swap the postsynaptic endpoint of bad edges with random edges of the same group.

    A swap between two edges of the same group preserves every per-group in-
    and out-degree, so the repair never breaks the shuffle's invariants. A swap
    is only accepted when both resulting edges are new and not self-loops, so
    the number of bad edges never increases.
    """
    pre64 = pre.astype(np.int64)
    post = post.astype(np.int64)
    order = np.argsort(group, kind="stable")
    groups, start, size = np.unique(group[order], return_index=True, return_counts=True)
    gpos = np.searchsorted(groups, group)
    rounds = stalled = 0
    for rounds in range(1, max_rounds + 1):
        bad = np.flatnonzero(_bad_edges(pre64, post, n))
        if len(bad) == 0:
            return post.astype(np.int32), rounds - 1
        existing = np.sort(post * n + pre64)

        def present(k: np.ndarray) -> np.ndarray:
            i = np.minimum(np.searchsorted(existing, k), len(existing) - 1)
            return existing[i] == k

        # Hub-to-hub duplicates rarely find a valid partner at random, so try
        # several candidates per bad edge and keep the first valid one.
        c = int(np.clip((1 << 20) // len(bad), 4, 512))
        bb = np.repeat(bad, c)
        gg = gpos[bb]
        pp = order[start[gg] + (rng.random(len(bb)) * size[gg]).astype(np.int64)]
        valid = ((bb != pp) & (pre64[bb] != post[pp]) & (pre64[pp] != post[bb])
                 & ~present(post[pp] * n + pre64[bb]) & ~present(post[bb] * n + pre64[pp]))
        valid = valid.reshape(len(bad), c)
        has = valid.any(axis=1)
        b = bad[has]
        p = pp.reshape(len(bad), c)[has, valid[has].argmax(axis=1)]

        # Every edge may take part in at most one swap per round, and two
        # accepted swaps must not create the same edge.
        _, first = np.unique(np.concatenate([b, p]), return_index=True)
        once = np.zeros(2 * len(b), dtype=bool)
        once[first] = True
        ok = once[: len(b)] & once[len(b):]
        b, p = b[ok], p[ok]
        new_b, new_p = post[p] * n + pre64[b], post[b] * n + pre64[p]
        _, inv, counts = np.unique(np.concatenate([new_b, new_p]), return_inverse=True,
                                   return_counts=True)
        clash = counts[inv] > 1
        ok = ~(clash[: len(b)] | clash[len(b):])
        b, p = b[ok], p[ok]
        post[b], post[p] = post[p], post[b].copy()
        # No valid partner among hundreds of candidates for several rounds: the
        # leftovers are (almost surely) infeasible within their block.
        stalled = stalled + 1 if len(b) == 0 else 0
        if stalled >= 5:
            break
    return post.astype(np.int32), rounds


def _finish(method: str, pre: np.ndarray, post: np.ndarray, count: np.ndarray, n: int,
            rounds: int) -> Tuple[Edges, ShuffleReport]:
    loops = pre == post
    pre, post, count = pre[~loops], post[~loops], count[~loops]
    edges_before = len(pre)
    pre, post, count, merged = merge_duplicates(pre, post, count, n)
    report = ShuffleReport(method, edges_before + int(loops.sum()), len(pre), rounds,
                           int(loops.sum()), merged)
    return (pre, post, count), report


def degree_preserving_shuffle(pre: np.ndarray, post: np.ndarray, count: np.ndarray, n: int,
                              seed: Optional[int] = None, max_rounds: int = 100
                              ) -> Tuple[Edges, ShuffleReport]:
    rng = np.random.default_rng(seed)
    group = np.zeros(len(pre), dtype=np.int64)
    new_post = rng.permutation(post)
    new_post, rounds = _repair(pre, new_post, group, n, rng, max_rounds)
    return _finish("degree_preserving", pre.copy(), new_post, count.copy(), n, rounds)


def _swap_sweeps(pre: np.ndarray, post: np.ndarray, group: np.ndarray, n: int,
                 rng: np.random.Generator, sweeps: int) -> Tuple[np.ndarray, int]:
    """Edge-swap rewiring: exchange postsynaptic endpoints of random edge pairs in the same group.

    Each sweep pairs every edge with a random partner of its group and accepts
    the swaps whose two new edges are neither self-loops nor already present
    (and do not collide with another accepted swap). The graph stays simple and
    every per-group in- and out-degree is preserved after every sweep, so no
    repair is needed.
    """
    pre64 = pre.astype(np.int64)
    post = post.astype(np.int64)
    accepted = 0
    if len(pre) < 2:
        return post.astype(np.int32), 0
    # Group id in the high bits, a fresh random number in the low bits: one argsort
    # gives a random order within every group.
    g_hi = np.unique(group, return_inverse=True)[1].astype(np.int64) << 31
    for _ in range(sweeps):
        order = np.argsort(g_hi | rng.integers(0, 1 << 31, len(pre)), kind="stable")
        a, b = order[0:-1:2], order[1::2]
        same = group[a] == group[b]
        a, b = a[same], b[same]

        existing = np.sort(post * n + pre64)

        def present(k: np.ndarray) -> np.ndarray:
            i = np.minimum(np.searchsorted(existing, k), len(existing) - 1)
            return existing[i] == k

        new_a, new_b = post[b] * n + pre64[a], post[a] * n + pre64[b]
        ok = ((post[a] != post[b]) & (pre64[a] != post[b]) & (pre64[b] != post[a])
              & ~present(new_a) & ~present(new_b))
        a, b, new_a, new_b = a[ok], b[ok], new_a[ok], new_b[ok]
        _, inv, counts = np.unique(np.concatenate([new_a, new_b]), return_inverse=True, return_counts=True)
        clash = counts[inv] > 1
        keep = ~(clash[: len(a)] | clash[len(a):])
        a, b = a[keep], b[keep]
        post[a], post[b] = post[b], post[a].copy()
        accepted += len(a)
    return post.astype(np.int32), accepted


def type_preserving_shuffle(pre: np.ndarray, post: np.ndarray, count: np.ndarray, n: int,
                            cell_type: np.ndarray, seed: Optional[int] = None,
                            sweeps: int = 10) -> Tuple[Edges, ShuffleReport]:
    """Rewire within ``(pre_type, post_type)`` blocks by repeated edge swaps.

    Permute-then-repair does not scale here: the real connectome has millions
    of small, dense type-pair blocks where a random permutation creates many
    duplicates that no swap can fix. Swapping keeps the graph valid at every
    step instead. Dense blocks simply accept few swaps, which shows up as a
    higher ``overlap with the original`` rather than as dropped edges.
    """
    rng = np.random.default_rng(seed)
    n_types = int(cell_type.max()) + 1 if len(cell_type) else 1
    group = cell_type[pre].astype(np.int64) * n_types + cell_type[post].astype(np.int64)
    new_post, accepted = _swap_sweeps(pre, post, group, n, rng, sweeps)
    edges, report = _finish("type_preserving", pre.copy(), new_post, count.copy(), n, 0)
    report.swap_sweeps, report.swaps_accepted = sweeps, accepted
    return edges, report


def random_sparse_graph(n: int, e: int, count: np.ndarray, seed: Optional[int] = None,
                        max_rounds: int = 100) -> Tuple[Edges, ShuffleReport]:
    """Uniform random directed graph with ``e`` edges and a permuted weight multiset."""
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, e, dtype=np.int64)
    post = rng.integers(0, n, e, dtype=np.int64)
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        bad = np.flatnonzero(_bad_edges(pre, post, n))
        if len(bad) == 0:
            rounds -= 1
            break
        pre[bad] = rng.integers(0, n, len(bad))
        post[bad] = rng.integers(0, n, len(bad))
    w = rng.permutation(count).astype(np.float32)
    return _finish("random_sparse", pre.astype(np.int32), post.astype(np.int32), w, n, rounds)


def in_out_degrees(pre: np.ndarray, post: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.bincount(post, minlength=n), np.bincount(pre, minlength=n)


__all__ = [
    "ShuffleReport",
    "degree_preserving_shuffle",
    "type_preserving_shuffle",
    "random_sparse_graph",
    "in_out_degrees",
    "sort_edges",
]
