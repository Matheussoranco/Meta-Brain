import numpy as np
import pytest
import torch

from meta_brain.graph import merge_duplicates
from meta_brain.sparse_ops import (SparseStructure, _edge_grad, propagate, reference_propagate,
                                   sparse_propagate)


def make_structure(n=60, e=500, seed=0, **kw):
    rng = np.random.default_rng(seed)
    pre, post, count, _ = merge_duplicates(rng.integers(0, n, e), rng.integers(0, n, e),
                                           np.ones(e, np.float32), n)
    return SparseStructure.from_edges(pre, post, n, **kw)


@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("edge_chunk", [7, 1 << 20])
@pytest.mark.parametrize("kernel", ["sddmm", "gather"])
def test_gradcheck(index_dtype, edge_chunk, kernel):
    s = make_structure(n=25, e=120, index_dtype=index_dtype, edge_chunk=edge_chunk, edge_grad_kernel=kernel)
    x = torch.randn(s.n, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(s.e, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, v: sparse_propagate(x, v, s), (x, v))


def test_matches_reference_forward_and_backward():
    s = make_structure(edge_chunk=33)
    x = torch.randn(s.n, 4, requires_grad=True)
    v = torch.randn(s.e, requires_grad=True)
    g = torch.randn(s.n, 4)

    y = sparse_propagate(x, v, s)
    gx, gv = torch.autograd.grad(y, (x, v), g)
    y_ref = reference_propagate(x, v, s)
    gx_ref, gv_ref = torch.autograd.grad(y_ref, (x, v), g)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(gx, gx_ref)
    torch.testing.assert_close(gv, gv_ref)


def test_edge_grad_kernels_agree():
    s = make_structure(n=300, e=4000, edge_chunk=97)
    g, x = torch.randn(s.n, 5), torch.randn(s.n, 5)
    s.edge_grad_kernel = "sddmm"
    sddmm = _edge_grad(s, g, x)
    s.edge_grad_kernel = "gather"
    torch.testing.assert_close(sddmm, _edge_grad(s, g, x))
    s.edge_grad_kernel = "bogus"
    with pytest.raises(ValueError, match="edge_grad_kernel"):
        _edge_grad(s, g, x)


def test_matches_dense_matrix():
    s = make_structure(n=30, e=200)
    v = torch.randn(s.e)
    dense = torch.zeros(s.n, s.n)
    dense[s.post, s.pre] = v
    x = torch.randn(5, s.n)
    torch.testing.assert_close(propagate(x, v, s), x @ dense.t())


def test_type_pair_parameters_receive_reduced_gradient():
    """values = base * theta[edge_pair]: gradient flows to the small theta vector."""
    s = make_structure()
    rng = np.random.default_rng(1)
    edge_pair = torch.as_tensor(rng.integers(0, 9, s.e))
    base = torch.rand(s.e)
    theta = torch.ones(9, requires_grad=True)
    theta_ref = theta.detach().clone().requires_grad_(True)
    x = torch.randn(s.n, 2)

    sparse_propagate(x, base * theta[edge_pair], s).square().sum().backward()
    reference_propagate(x, base * theta_ref[edge_pair], s).square().sum().backward()
    torch.testing.assert_close(theta.grad, theta_ref.grad)


def test_unrolled_bptt_matches_reference():
    s = make_structure(n=40, e=300)
    v0 = (torch.randn(s.e) * 0.3).requires_grad_(True)
    v1 = v0.detach().clone().requires_grad_(True)
    drive = torch.randn(s.n, 3)

    def run(op, v):
        state = torch.zeros(s.n, 3)
        for _ in range(12):
            state = 0.8 * state + 0.2 * torch.tanh(op(state, v, s) + drive)
        return state.sum()

    run(sparse_propagate, v0).backward()
    run(reference_propagate, v1).backward()
    torch.testing.assert_close(v0.grad, v1.grad)


def saved_numel(fn):
    total = []

    def pack(t):
        total.append(t.numel())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return sum(total)


def test_saves_activation_not_edge_messages():
    """The whole point of the custom op: per-call saved memory is O(N*B), not O(E*B)."""
    s = make_structure(n=50, e=2000)
    b = 8
    x = torch.randn(s.n, b, requires_grad=True)
    v = torch.randn(s.e, requires_grad=True)
    ours = saved_numel(lambda: sparse_propagate(x, v, s))
    ref = saved_numel(lambda: reference_propagate(x, v, s))
    assert ours <= s.n * b + s.e
    assert ref >= s.e * b


def test_rejects_unsorted_or_duplicate_edges():
    with pytest.raises(ValueError, match="sorted"):
        SparseStructure.from_edges([1, 0], [1, 0], 3)
    with pytest.raises(ValueError, match="sorted"):
        SparseStructure.from_edges([0, 0], [1, 1], 3)
    with pytest.raises(ValueError, match="range"):
        SparseStructure.from_edges([0], [5], 3)


def test_shape_validation():
    s = make_structure()
    with pytest.raises(ValueError, match="x must be"):
        sparse_propagate(torch.randn(s.n + 1, 2), torch.randn(s.e), s)
    with pytest.raises(ValueError, match="values must be"):
        sparse_propagate(torch.randn(s.n, 2), torch.randn(s.e + 1), s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    s = make_structure()
    x = torch.randn(s.n, 3)
    v = torch.randn(s.e)
    sc = s.to("cuda")
    torch.testing.assert_close(sparse_propagate(x.cuda(), v.cuda(), sc).cpu(),
                               sparse_propagate(x, v, s), rtol=1e-4, atol=1e-5)
