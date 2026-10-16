"""Memory-lean sparse connectome propagation with a custom autograd (milestone M1).

``y = A(values) @ x`` where ``A`` is the ``N x N`` connectome with rows =
postsynaptic and columns = presynaptic neuron, and ``values`` holds one weight
per edge.

Plain autograd through a gather/scatter implementation saves an ``E``-sized
tensor (``x[pre] * w``) at every call. Unrolled over ``K`` micro-steps x ``T``
frames that is what makes BPTT through the full brain impossible. This
operator saves only ``x`` (``N x B``) per call and recomputes edge products in
the backward pass, in bounded chunks:

* ``grad_x      = A^T @ grad_y``                      (sparse CSR matmul)
* ``grad_values[e] = sum_b grad_y[post_e, b] * x[pre_e, b]``   (SDDMM via
  ``torch.sparse.sampled_addmm``; chunked gathers as a fallback)

``values`` is usually derived from a much smaller parameter vector (per cell
type, or per connected type pair) with ordinary autograd, e.g.
``values = base * theta[edge_pair]``; that gather is computed once per rollout
chunk and shared across steps, so its cost is not per step.

Layout: the operator works on ``(N, B)`` tensors (neurons first) so CSR
matmuls and row gathers need no transposes. :func:`propagate` accepts the
more common ``(B, N)`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import warnings

import numpy as np
import torch
from torch.autograd.function import once_differentiable

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

DEFAULT_EDGE_CHUNK = 1 << 22  # ~4M edges per chunk in the values gradient


@dataclass
class SparseStructure:
    """Fixed sparsity pattern of the connectome on one device.

    Edges must be sorted by ``(post, pre)``, which is the order
    :class:`meta_brain.graph.ConnectomeArrays` stores them in, so the edge
    order *is* CSR order and ``values`` needs no permutation in the forward.
    """

    n: int
    pre: torch.Tensor  # (E,) long, column index in CSR order
    post: torch.Tensor  # (E,) long, row index in CSR order
    crow: torch.Tensor  # (N+1,) row pointer of A
    t_crow: torch.Tensor  # (N+1,) row pointer of A^T
    t_col: torch.Tensor  # (E,) column index (= post) of A^T
    t_perm: torch.Tensor  # (E,) edge index in A^T order
    edge_chunk: int = DEFAULT_EDGE_CHUNK
    edge_grad_kernel: str = "auto"  # "auto" | "sddmm" | "gather"

    @property
    def e(self) -> int:
        return int(self.pre.numel())

    @property
    def device(self) -> torch.device:
        return self.pre.device

    @classmethod
    def from_edges(cls, pre, post, n: int, device: Optional[torch.device | str] = None,
                   index_dtype: torch.dtype = torch.int64,
                   edge_chunk: int = DEFAULT_EDGE_CHUNK,
                   edge_grad_kernel: str = "auto") -> "SparseStructure":
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        if len(pre) > 1:
            key = post * n + pre
            if np.any(np.diff(key) <= 0):
                raise ValueError("Edges must be unique and sorted by (post, pre); use graph.sort_edges.")
        if len(pre) and (pre.min() < 0 or post.min() < 0 or max(pre.max(), post.max()) >= n):
            raise ValueError("Edge index out of range.")

        crow = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(post, minlength=n), out=crow[1:])
        t_perm = np.lexsort((post, pre))
        t_crow = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=t_crow[1:])

        def to(a: np.ndarray, dtype: torch.dtype = index_dtype) -> torch.Tensor:
            return torch.as_tensor(a).to(device=device, dtype=dtype)

        return cls(n=n, pre=to(pre, torch.long), post=to(post, torch.long), crow=to(crow),
                   t_crow=to(t_crow), t_col=to(post[t_perm]), t_perm=to(t_perm, torch.long),
                   edge_chunk=edge_chunk, edge_grad_kernel=edge_grad_kernel)

    def to(self, device: torch.device | str) -> "SparseStructure":
        return SparseStructure(self.n, *(getattr(self, f).to(device) for f in
                                         ("pre", "post", "crow", "t_crow", "t_col", "t_perm")),
                               edge_chunk=self.edge_chunk, edge_grad_kernel=self.edge_grad_kernel)

    def matrix(self, values: torch.Tensor) -> torch.Tensor:
        col = self.pre.to(self.crow.dtype)
        return torch.sparse_csr_tensor(self.crow, col, values, (self.n, self.n))

    def transpose_matrix(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sparse_csr_tensor(self.t_crow, self.t_col, values[self.t_perm], (self.n, self.n))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.pre, self.post, self.crow, self.t_crow, self.t_col, self.t_perm))


def _edge_grad(structure: SparseStructure, grad_y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``sum_b grad_y[post_e, b] * x[pre_e, b]`` for every edge, in CSR (= edge) order.

    This is a sampled dense-dense matmul (``grad_y @ x.T`` read only at the
    connectome's nonzeros), which cuSPARSE computes far faster than gathering
    two ``E x B`` tensors.
    """
    kernel = structure.edge_grad_kernel
    if kernel not in ("auto", "sddmm", "gather"):
        raise ValueError(f"unknown edge_grad_kernel {kernel!r}")
    if kernel != "gather":
        try:
            pattern = structure.matrix(torch.zeros(structure.e, dtype=grad_y.dtype, device=grad_y.device))
            return torch.sparse.sampled_addmm(pattern, grad_y, x.t().contiguous(), beta=0.0).values()
        except (RuntimeError, NotImplementedError):
            if kernel == "sddmm":
                raise
    return _edge_grad_gather(structure, grad_y, x)


def _edge_grad_gather(structure: SparseStructure, grad_y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Fallback: the same edge gradient from chunked row gathers."""
    out = torch.empty(structure.e, dtype=grad_y.dtype, device=grad_y.device)
    step = max(1, int(structure.edge_chunk))
    for s in range(0, structure.e, step):
        e = min(s + step, structure.e)
        out[s:e] = (grad_y.index_select(0, structure.post[s:e]) *
                    x.index_select(0, structure.pre[s:e])).sum(dim=1)
    return out


class _SparsePropagate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, values: torch.Tensor, structure: SparseStructure) -> torch.Tensor:
        ctx.structure = structure
        ctx.save_for_backward(x, values)
        return torch.sparse.mm(structure.matrix(values), x)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: torch.Tensor):
        x, values = ctx.saved_tensors
        structure: SparseStructure = ctx.structure
        grad_y = grad_y.contiguous()
        grad_x = grad_values = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.sparse.mm(structure.transpose_matrix(values), grad_y)
        if ctx.needs_input_grad[1]:
            grad_values = _edge_grad(structure, grad_y, x)
        return grad_x, grad_values, None


def sparse_propagate(x_nb: torch.Tensor, values: torch.Tensor, structure: SparseStructure) -> torch.Tensor:
    """``A(values) @ x`` for ``x`` of shape ``(N, B)``."""
    if x_nb.dim() != 2 or x_nb.shape[0] != structure.n:
        raise ValueError(f"x must be (N={structure.n}, B), got {tuple(x_nb.shape)}")
    if values.shape != (structure.e,):
        raise ValueError(f"values must be (E={structure.e},), got {tuple(values.shape)}")
    return _SparsePropagate.apply(x_nb.contiguous(), values, structure)


def propagate(x_bn: torch.Tensor, values: torch.Tensor, structure: SparseStructure) -> torch.Tensor:
    """Batch-first convenience wrapper: ``(B, N) -> (B, N)``."""
    return sparse_propagate(x_bn.t(), values, structure).t()


def reference_propagate(x_nb: torch.Tensor, values: torch.Tensor, structure: SparseStructure) -> torch.Tensor:
    """Plain-autograd gather/scatter reference used by the tests and the benchmark."""
    msg = x_nb.index_select(0, structure.pre) * values.unsqueeze(1)
    out = torch.zeros_like(x_nb)
    return out.index_add(0, structure.post, msg)


__all__ = [
    "SparseStructure",
    "sparse_propagate",
    "propagate",
    "reference_propagate",
]
