import numpy as np
import pytest

from meta_brain.graph import merge_duplicates, type_pair_index
from meta_brain.shuffle import (degree_preserving_shuffle, in_out_degrees, random_sparse_graph,
                                type_preserving_shuffle)


def random_graph(n=400, e=4000, n_types=12, seed=0):
    rng = np.random.default_rng(seed)
    # Heavy-tailed degrees, like a connectome.
    w_out = rng.pareto(1.5, n) + 1
    w_in = rng.pareto(1.5, n) + 1
    pre = rng.choice(n, e, p=w_out / w_out.sum())
    post = rng.choice(n, e, p=w_in / w_in.sum())
    keep = pre != post
    count = rng.integers(1, 50, e).astype(np.float32)
    pre, post, count, _ = merge_duplicates(pre[keep], post[keep], count[keep], n)
    cell_type = rng.integers(0, n_types, n).astype(np.int32)
    return pre, post, count, cell_type, n


def assert_simple(pre, post, n):
    assert not np.any(pre == post)
    key = post.astype(np.int64) * n + pre
    assert len(np.unique(key)) == len(key)


def test_degree_preserving_shuffle_keeps_degrees_and_strength():
    pre, post, count, _, n = random_graph()
    (p2, q2, c2), rep = degree_preserving_shuffle(pre, post, count, n, seed=1)
    assert rep.exact, rep
    assert len(p2) == len(pre)
    assert_simple(p2, q2, n)
    for a, b in zip(in_out_degrees(pre, post, n), in_out_degrees(p2, q2, n)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.bincount(pre, count, n), np.bincount(p2, c2, n))
    # It actually rewires.
    old = set(zip(pre.tolist(), post.tolist()))
    new = set(zip(p2.tolist(), q2.tolist()))
    # Hubs keep some edges by chance; most of the wiring must still change.
    assert len(old & new) < 0.5 * len(old)


def test_type_preserving_shuffle_keeps_type_pair_table_and_block_degrees():
    pre, post, count, cell_type, n = random_graph(n_types=6)
    (p2, q2, c2), rep = type_preserving_shuffle(pre, post, count, n, cell_type, seed=2)
    assert rep.exact, rep
    assert_simple(p2, q2, n)

    def block_table(p, q, c):
        ep, pairs = type_pair_index(p, q, cell_type, int(cell_type.max()) + 1)
        return {tuple(t): (int((ep == i).sum()), float(c[ep == i].sum())) for i, t in enumerate(pairs)}

    assert block_table(pre, post, count) == block_table(p2, q2, c2)
    for a, b in zip(in_out_degrees(pre, post, n), in_out_degrees(p2, q2, n)):
        np.testing.assert_array_equal(a, b)
    # Per-neuron in-degree from each presynaptic type is preserved as well.
    n_types = int(cell_type.max()) + 1
    def in_by_type(p, q):
        return np.bincount(q.astype(np.int64) * n_types + cell_type[p], minlength=n * n_types)
    np.testing.assert_array_equal(in_by_type(pre, post), in_by_type(p2, q2))
    assert set(zip(pre.tolist(), post.tolist())) != set(zip(p2.tolist(), q2.tolist()))


def test_shuffles_are_seeded():
    pre, post, count, cell_type, n = random_graph()
    a, _ = degree_preserving_shuffle(pre, post, count, n, seed=7)
    b, _ = degree_preserving_shuffle(pre, post, count, n, seed=7)
    c, _ = degree_preserving_shuffle(pre, post, count, n, seed=8)
    # Compare presynaptic endpoints: in CSR order the post column is fixed by
    # the preserved in-degrees.
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_unrepairable_block_is_reported_not_hidden():
    # Two neurons, one type: the only simple rewiring of 0->1, 1->0 is itself,
    # and a 3-edge star cannot be rewired without duplicates.
    pre = np.array([0, 0, 0], np.int32)
    post = np.array([1, 2, 3], np.int32)
    (p2, q2, _), rep = degree_preserving_shuffle(pre, post, np.ones(3, np.float32), 4, seed=0)
    assert rep.exact and sorted(q2.tolist()) == [1, 2, 3]

    pre = np.array([0, 1], np.int32)
    post = np.array([1, 0], np.int32)
    (p2, q2, _), rep = degree_preserving_shuffle(pre, post, np.ones(2, np.float32), 2, seed=0,
                                                 max_rounds=5)
    assert len(p2) + rep.residual_self_loops_dropped + rep.residual_duplicates_merged == 2


def test_random_sparse_graph_matches_size_and_weights():
    pre, post, count, _, n = random_graph()
    (p2, q2, c2), rep = random_sparse_graph(n, len(pre), count, seed=3)
    assert rep.exact
    assert len(p2) == len(pre)
    assert_simple(p2, q2, n)
    np.testing.assert_allclose(np.sort(c2), np.sort(count))


@pytest.mark.parametrize("fn", ["degree", "type"])
def test_output_is_csr_sorted(fn):
    pre, post, count, cell_type, n = random_graph()
    if fn == "degree":
        (p2, q2, _), _ = degree_preserving_shuffle(pre, post, count, n, seed=0)
    else:
        (p2, q2, _), _ = type_preserving_shuffle(pre, post, count, n, cell_type, seed=0)
    key = q2.astype(np.int64) * n + p2
    assert np.all(np.diff(key) > 0)


def test_type_preserving_shuffle_never_drops_edges_in_dense_blocks():
    # Type 0 -> type 1 is a complete bipartite block: no rewiring is possible,
    # so swaps must be rejected rather than creating duplicates.
    cell_type = np.array([0, 0, 0, 1, 1, 1, 2, 2], np.int32)
    pre = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 6, 7], np.int32)
    post = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1], np.int32)
    pre, post, count, _ = merge_duplicates(pre, post, np.ones(len(pre), np.float32), 8)
    (p2, q2, _), rep = type_preserving_shuffle(pre, post, count, 8, cell_type, seed=0, sweeps=20)
    assert rep.exact and rep.swap_sweeps == 20
    assert set(zip(p2.tolist(), q2.tolist())) == set(zip(pre.tolist(), post.tolist()))


def test_type_preserving_shuffle_reports_swaps():
    pre, post, count, cell_type, n = random_graph(n_types=6)
    _, rep = type_preserving_shuffle(pre, post, count, n, cell_type, seed=0, sweeps=3)
    assert rep.exact and 2 * rep.swaps_accepted > len(pre)  # each swap moves two edges
