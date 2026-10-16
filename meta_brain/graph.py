"""Vectorized MaleCNS connectome loader with an on-disk cache (milestone M0).

The MaleCNS v1.0 "flat connectome" release ships three files (the ones
``scripts/download_data.py`` fetches):

* ``body-annotations-male-cns-v1.0-minconf-0.5.feather``: one row per body
  (``bodyId``, ``status``, ``type``, ``superclass``, ...);
* ``body-neurotransmitters-male-cns-v1.0.feather``: per-body NT predictions
  (``body``, ``consensus_nt``, ``predicted_nt``);
* ``connectome-weights-male-cns-v1.0-minconf-0.5.feather``: ~152M weighted
  rows (``body_pre``, ``body_post``, ``weight``), one per connected body pair,
  including the many untraced fragments. Keeping ``Traced`` bodies leaves
  ~25.6M edges. Repeated pairs (none in v1.0) would be summed.

The legacy layout (``*meta*.feather`` + ``*simple*edgelist*.feather``) is still
accepted.

The weights table is streamed one Arrow record batch at a time, body ids are
remapped with ``searchsorted``, and duplicate pairs are merged incrementally so
peak memory stays bounded by the number of unique pairs rather than the number
of rows. Every row is accounted for in :class:`LoadReport`.

The resulting :class:`ConnectomeArrays` stores edges sorted by ``(post, pre)``,
which is CSR row order for :mod:`meta_brain.sparse_ops`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOADER_VERSION = 2

# Candidate column names, in priority order. The first one present is used.
META_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "body_id": ("bodyId", "body_id", "body", "root_id"),
    "status": ("status",),
    "type": ("type", "cell_type"),
    "superclass": ("superclass", "super_class"),
    "class": ("class", "cell_class"),
    "nt": ("predictedNt", "consensusNt", "nt_type", "predicted_nt"),
    "nt_confidence": ("predictedNtConfidence", "predictedNtProb", "nt_type_score"),
    "side": ("somaSide", "rootSide", "side"),
}
NT_FILE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "body_id": ("body", "bodyId", "body_id"),
    "consensus": ("consensus_nt", "consensusNt"),
    "predicted": ("predicted_nt", "predictedNt"),
    "confidence": ("predicted_nt_prob", "predicted_nt_confidence", "predictedNtProb",
                   "predictedNtConfidence"),
}
EDGE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "pre": ("body_pre", "bodyId_pre", "pre", "pre_root_id"),
    "post": ("body_post", "bodyId_post", "post", "post_root_id"),
    "weight": ("weight", "syn_count", "count"),
}

FLAT_PATTERNS = {
    "annotations": "body-annotations-*.feather",
    "neurotransmitters": "body-neurotransmitters-*.feather",
    "edges": "connectome-weights-*.feather",
}
LEGACY_PATTERNS = {"annotations": "*meta*.feather", "edges": "*simple*edgelist*.feather"}

DEFAULT_STATUSES: Tuple[str, ...] = ("Traced",)
DEFAULT_MERGE_EVERY = 30_000_000  # rows accumulated before an incremental merge

# Neurotransmitter codes stored per neuron. Signs are a separate, overridable
# mapping so downstream layers (e.g. dopamine as a teaching signal) can decide
# how to treat modulatory transmitters.
NT_CODES: Dict[str, int] = {
    "unknown": 0,
    "acetylcholine": 1,
    "gaba": 2,
    "glutamate": 3,
    "histamine": 4,
    "dopamine": 5,
    "serotonin": 6,
    "octopamine": 7,
    "tyramine": 8,
}
NT_ALIASES: Dict[str, str] = {
    "ach": "acetylcholine",
    "glu": "glutamate",
    "da": "dopamine",
    "5ht": "serotonin",
    "ser": "serotonin",
    "oa": "octopamine",
    "ta": "tyramine",
    "his": "histamine",
}
# Fast-transmission sign by NT code. Glutamate is inhibitory at most fly
# central synapses (GluCl) and histamine is the inhibitory photoreceptor
# transmitter. Modulatory and unknown transmitters default to +1; layer 2
# handles dopamine separately through ``nt_code``.
DEFAULT_NT_SIGN: Dict[str, float] = {
    "unknown": 1.0,
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
    "tyramine": 1.0,
}

SENSORY_SUPERCLASSES = ("ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending", "olsn")
MOTOR_SUPERCLASSES = ("descending_neuron", "vnc_motor", "cb_motor", "dn", "motor")
DESCENDING_SUPERCLASSES = ("descending_neuron", "dn")


@dataclass
class DataFiles:
    annotations: Path
    edges: Path
    neurotransmitters: Optional[Path] = None
    layout: str = "flat"

    def all(self) -> List[Path]:
        return [p for p in (self.annotations, self.neurotransmitters, self.edges) if p is not None]


@dataclass
class LoadReport:
    """Every body and edge row that was read, dropped or merged, so counts can be audited."""

    layout: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    columns_used: Dict[str, str] = field(default_factory=dict)
    statuses: List[str] = field(default_factory=list)
    bodies_annotated: int = 0
    dropped_by_status: int = 0
    n_neurons: int = 0
    nt_source: str = ""
    edge_rows_read: int = 0
    dropped_unknown_pre: int = 0
    dropped_unknown_post: int = 0
    dropped_self_loops: int = 0
    duplicate_rows_merged: int = 0
    dropped_below_min_weight: int = 0
    min_weight: float = 1.0
    edges_retained: int = 0
    synapses_retained: int = 0
    n_cell_types: int = 0
    n_untyped_neurons: int = 0
    n_type_pairs: int = 0
    nt_counts: Dict[str, int] = field(default_factory=dict)
    build_seconds: float = 0.0
    load_seconds: float = 0.0
    from_cache: bool = False

    def accounted_rows(self) -> int:
        return (self.edges_retained + self.dropped_unknown_pre + self.dropped_unknown_post
                + self.dropped_self_loops + self.duplicate_rows_merged + self.dropped_below_min_weight)

    def summary(self) -> str:
        status = f" status in {self.statuses}" if self.statuses else ""
        lines = [
            f"layout={self.layout}  bodies annotated={self.bodies_annotated:,}"
            f"  dropped by{status or ' status'}={self.dropped_by_status:,}",
            f"neurons={self.n_neurons:,}  NT source={self.nt_source}",
            f"edge rows read={self.edge_rows_read:,}",
            f"  dropped unknown pre={self.dropped_unknown_pre:,} post={self.dropped_unknown_post:,}",
            f"  dropped self-loops={self.dropped_self_loops:,}",
            f"  duplicate rows merged={self.duplicate_rows_merged:,}",
            f"  pairs below min weight {self.min_weight:g}={self.dropped_below_min_weight:,}",
            f"edges retained={self.edges_retained:,} (synapses={self.synapses_retained:,})",
            f"cell types={self.n_cell_types:,} (untyped neurons={self.n_untyped_neurons:,}), "
            f"type pairs={self.n_type_pairs:,}",
            f"build time={self.build_seconds:.1f}s"
            + (f", cache load time={self.load_seconds:.1f}s" if self.from_cache else ""),
        ]
        return "\n".join(lines)


@dataclass
class ConnectomeArrays:
    """Canonical sparse connectome. Edges are sorted by ``(post, pre)``."""

    body_id: np.ndarray  # (N,) int64
    pre: np.ndarray  # (E,) int32, neuron index
    post: np.ndarray  # (E,) int32, neuron index
    count: np.ndarray  # (E,) float32, synapse count summed over rows
    nt_code: np.ndarray  # (N,) int8, see NT_CODES
    nt_confidence: np.ndarray  # (N,) float32, NaN when unavailable
    cell_type: np.ndarray  # (N,) int32 index into type_names
    type_names: List[str]
    superclass: np.ndarray  # (N,) int16 index into superclass_names
    superclass_names: List[str]
    side: np.ndarray  # (N,) int8: 0 unknown, 1 left, 2 right, 3 center
    report: LoadReport = field(default_factory=LoadReport)

    @property
    def n(self) -> int:
        return int(len(self.body_id))

    @property
    def e(self) -> int:
        return int(len(self.pre))

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def sign(self, nt_sign: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Per-neuron fast-transmission sign from the NT prediction."""
        table = dict(DEFAULT_NT_SIGN)
        if nt_sign:
            table.update(nt_sign)
        lut = np.ones(len(NT_CODES), dtype=np.float32)
        for name, code in NT_CODES.items():
            lut[code] = table.get(name, 1.0)
        return lut[self.nt_code]

    def superclass_mask(self, names: Sequence[str]) -> np.ndarray:
        wanted = {s.lower() for s in names}
        codes = [i for i, s in enumerate(self.superclass_names) if s in wanted]
        return np.isin(self.superclass, codes)

    def sensory_indices(self) -> np.ndarray:
        return np.flatnonzero(self.superclass_mask(SENSORY_SUPERCLASSES))

    def motor_indices(self) -> np.ndarray:
        return np.flatnonzero(self.superclass_mask(MOTOR_SUPERCLASSES))

    def descending_indices(self) -> np.ndarray:
        return np.flatnonzero(self.superclass_mask(DESCENDING_SUPERCLASSES))

    def type_pair_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """Map every edge to a connected ``(pre_type, post_type)`` pair.

        Returns ``(edge_pair, pair_types)`` where ``edge_pair`` is ``(E,)``
        int32 and ``pair_types`` is ``(P, 2)`` int32 with columns
        ``[pre_type, post_type]``.
        """
        return type_pair_index(self.pre, self.post, self.cell_type, self.n_types)

    def csr_rowptr(self) -> np.ndarray:
        """CSR row pointer with rows = postsynaptic neuron."""
        counts = np.bincount(self.post, minlength=self.n)
        rowptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(counts, out=rowptr[1:])
        return rowptr

    def replace_edges(self, pre: np.ndarray, post: np.ndarray, count: np.ndarray) -> "ConnectomeArrays":
        """Copy with a different edge set (used by the shuffle controls)."""
        pre, post, count = sort_edges(pre, post, count)
        out = ConnectomeArrays(**{**self.__dict__, "pre": pre, "post": post, "count": count})
        out.report = LoadReport(**{**asdict(self.report), "edges_retained": len(pre),
                                   "synapses_retained": int(count.sum())})
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _pick(columns: Sequence[str], candidates: Sequence[str], required: bool, what: str) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise ValueError(f"{what}: none of {list(candidates)} found in columns {list(columns)}")
    return None


def _nt_names(values) -> pd.Series:
    names = pd.Series(values).fillna("").astype(str).str.strip().str.lower()
    return names.map(lambda s: NT_ALIASES.get(s, s))


def _nt_codes(names: pd.Series) -> np.ndarray:
    return names.map(lambda s: NT_CODES.get(s, 0)).to_numpy(dtype=np.int8)


def _normalize_side(values: Optional[pd.Series], n: int) -> np.ndarray:
    if values is None:
        return np.zeros(n, dtype=np.int8)
    s = values.fillna("").astype(str).str.strip().str.lower().str[:1]
    return s.map({"l": 1, "r": 2, "c": 3, "m": 3}).fillna(0).to_numpy(dtype=np.int8)


def cell_type_keys(meta: pd.DataFrame, type_col: Optional[str], superclass_col: Optional[str],
                   class_col: Optional[str]) -> Tuple[np.ndarray, int]:
    """Type key per neuron; untyped neurons fall back to ``untyped:<superclass>:<class>``."""
    n = len(meta)

    def col(name: Optional[str]) -> pd.Series:
        if name is None:
            return pd.Series([""] * n, index=meta.index)
        return meta[name].fillna("").astype(str).str.strip()

    typ, sup, cls = col(type_col), col(superclass_col), col(class_col)
    untyped = (typ == "") | typ.str.lower().isin({"nan", "none", "unknown"})
    keys = typ.where(~untyped, "untyped:" + sup.str.lower() + ":" + cls.str.lower())
    return keys.to_numpy(dtype=object), int(untyped.sum())


def remap_ids(sorted_ids: np.ndarray, order: np.ndarray, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized bodyId -> row index. Returns ``(index, valid_mask)``."""
    if len(sorted_ids) == 0:
        return np.zeros(len(raw), np.int32), np.zeros(len(raw), bool)
    pos = np.searchsorted(sorted_ids, raw)
    pos = np.minimum(pos, len(sorted_ids) - 1)
    valid = sorted_ids[pos] == raw
    return order[pos].astype(np.int32), valid


def sort_edges(pre: np.ndarray, post: np.ndarray, count: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((pre, post))
    return (pre[order].astype(np.int32, copy=False), post[order].astype(np.int32, copy=False),
            count[order].astype(np.float32, copy=False))


def merge_duplicates(pre: np.ndarray, post: np.ndarray, count: np.ndarray, n: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sum the counts of repeated ``(pre, post)`` rows. Returns the merged edges."""
    key = post.astype(np.int64) * n + pre.astype(np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    merged = len(key) - len(uniq)
    if merged == 0:
        return (*sort_edges(pre, post, count), 0)
    summed = np.bincount(inverse, weights=count.astype(np.float64), minlength=len(uniq))
    return ((uniq % n).astype(np.int32), (uniq // n).astype(np.int32),
            summed.astype(np.float32), int(merged))


def _merge_keys(keys: List[np.ndarray], weights: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Merge ``post * n + pre`` keys, summing weights; output is sorted and unique."""
    k = np.concatenate(keys)
    w = np.concatenate(weights)
    uniq, inverse = np.unique(k, return_inverse=True)
    return uniq, np.bincount(inverse, weights=w, minlength=len(uniq))


def type_pair_index(pre: np.ndarray, post: np.ndarray, cell_type: np.ndarray, n_types: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    key = cell_type[pre].astype(np.int64) * n_types + cell_type[post].astype(np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    pair_types = np.stack([uniq // n_types, uniq % n_types], axis=1).astype(np.int32)
    return inverse.astype(np.int32), pair_types


def _feather_columns(path: Path) -> List[str]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        return list(ipc.open_file(pa.memory_map(str(path), "r")).schema.names)
    except pa.ArrowInvalid:
        return list(pd.read_feather(path).columns)


def _iter_edge_batches(path: Path, columns: List[str]) -> Iterator[Dict[str, np.ndarray]]:
    """Yield the edge table one Arrow record batch at a time (bounded memory)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        reader = ipc.open_file(pa.memory_map(str(path), "r"))
    except pa.ArrowInvalid:
        # Feather v1 or a non-IPC file: fall back to a single full read.
        df = pd.read_feather(path, columns=columns)
        yield {c: df[c].to_numpy() for c in columns}
        return
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        yield {c: batch.column(c).to_numpy(zero_copy_only=False) for c in columns}


def find_data_files(data_dir: str | Path) -> DataFiles:
    """Locate the flat-connectome release, falling back to the legacy layout."""
    data_dir = Path(data_dir)

    def first(pattern: str) -> Optional[Path]:
        hits = sorted(data_dir.glob(pattern))
        return hits[0] if hits else None

    ann, edges = first(FLAT_PATTERNS["annotations"]), first(FLAT_PATTERNS["edges"])
    if ann and edges:
        return DataFiles(ann, edges, first(FLAT_PATTERNS["neurotransmitters"]), "flat")
    ann, edges = first(LEGACY_PATTERNS["annotations"]), first(LEGACY_PATTERNS["edges"])
    if ann and edges:
        return DataFiles(ann, edges, None, "legacy")
    raise FileNotFoundError(
        f"No connectome in {data_dir}. Expected {FLAT_PATTERNS['annotations']} + "
        f"{FLAT_PATTERNS['edges']} (+ {FLAT_PATTERNS['neurotransmitters']}); "
        "run scripts/download_data.py.")


def _join_neurotransmitters(nt_path: Path, body_id: np.ndarray, report: LoadReport
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Consensus NT where available and not 'unclear', else the per-body prediction."""
    cols = _feather_columns(nt_path)
    used = {k: _pick(cols, cands, k in ("body_id",), f"neurotransmitters {k}")
            for k, cands in NT_FILE_COLUMNS.items()}
    if used["consensus"] is None and used["predicted"] is None:
        raise ValueError(f"neurotransmitters: no consensus/predicted column in {cols}")
    report.columns_used.update({f"nt.{k}": v for k, v in used.items() if v})
    nt_df = pd.read_feather(nt_path, columns=[v for v in used.values() if v])
    nt_df = nt_df.drop_duplicates(used["body_id"]).set_index(used["body_id"])

    def column(key: str) -> pd.Series:
        if used[key] is None:
            return pd.Series([""] * len(body_id))
        return _nt_names(nt_df[used[key]].reindex(body_id).to_numpy())

    cons, pred = column("consensus"), column("predicted")
    usable = (cons != "") & (cons != "unclear")
    names = cons.where(usable, pred)
    report.nt_source = "consensus_nt, else predicted_nt" if used["consensus"] else "predicted_nt"
    conf = (pd.to_numeric(nt_df[used["confidence"]].reindex(body_id), errors="coerce")
            .to_numpy(dtype=np.float32) if used["confidence"]
            else np.full(len(body_id), np.nan, dtype=np.float32))
    return _nt_codes(names), conf


# --------------------------------------------------------------------------- #
# Loading and caching
# --------------------------------------------------------------------------- #

def build_connectome(files: DataFiles, min_weight: float = 1.0, drop_self_loops: bool = True,
                     statuses: Optional[Sequence[str]] = DEFAULT_STATUSES, verbose: bool = True,
                     merge_every: int = DEFAULT_MERGE_EVERY) -> ConnectomeArrays:
    """Build :class:`ConnectomeArrays` from MaleCNS feather files.

    ``statuses`` keeps only bodies whose ``status`` is listed (``None`` keeps
    all). ``min_weight`` is applied to the *merged* pair weight, i.e. the
    total synapse count between two neurons.
    """
    t0 = time.perf_counter()
    report = LoadReport(layout=files.layout, min_weight=float(min_weight),
                        files={k: str(v) for k, v in asdict(files).items() if isinstance(v, Path)})

    meta = pd.read_feather(files.annotations)
    cols = list(meta.columns)
    used = {k: _pick(cols, cands, k == "body_id", f"annotations {k}") for k, cands in META_COLUMNS.items()}
    report.columns_used.update({f"meta.{k}": v for k, v in used.items() if v})
    report.bodies_annotated = len(meta)
    if statuses and used["status"]:
        keep = meta[used["status"]].astype(str).isin(set(statuses))
        report.dropped_by_status = int((~keep).sum())
        report.statuses = list(statuses)
        meta = meta[keep.to_numpy()].reset_index(drop=True)

    body_id = meta[used["body_id"]].astype(np.int64).to_numpy()
    if len(np.unique(body_id)) != len(body_id):
        raise ValueError("Neuron annotations contain duplicate body ids.")
    n = len(body_id)
    report.n_neurons = n

    keys, n_untyped = cell_type_keys(meta, used["type"], used["superclass"], used["class"])
    type_names, cell_type = np.unique(keys.astype(str), return_inverse=True)
    report.n_cell_types, report.n_untyped_neurons = len(type_names), n_untyped

    sup = (meta[used["superclass"]].fillna("").astype(str).str.strip().str.lower()
           if used["superclass"] else pd.Series([""] * n))
    superclass_names, superclass = np.unique(sup.to_numpy(dtype=str), return_inverse=True)

    if files.neurotransmitters is not None:
        nt_code, nt_conf = _join_neurotransmitters(files.neurotransmitters, body_id, report)
    else:
        nt_code = (_nt_codes(_nt_names(meta[used["nt"]].to_numpy())) if used["nt"]
                   else np.zeros(n, dtype=np.int8))
        nt_conf = (pd.to_numeric(meta[used["nt_confidence"]], errors="coerce").to_numpy(dtype=np.float32)
                   if used["nt_confidence"] else np.full(n, np.nan, dtype=np.float32))
        report.nt_source = f"annotations.{used['nt']}" if used["nt"] else "none"
    inv_nt = {v: k for k, v in NT_CODES.items()}
    report.nt_counts = {inv_nt[int(c)]: int(k) for c, k in zip(*np.unique(nt_code, return_counts=True))}
    side = _normalize_side(meta[used["side"]] if used["side"] else None, n)
    del meta

    order = np.argsort(body_id, kind="stable")
    sorted_ids = body_id[order]

    edge_cols = _feather_columns(files.edges)
    ecol = {k: _pick(edge_cols, cands, k != "weight", f"edges {k}") for k, cands in EDGE_COLUMNS.items()}
    report.columns_used.update({f"edge.{k}": v for k, v in ecol.items() if v})

    uniq_keys = np.zeros(0, np.int64)
    uniq_w = np.zeros(0, np.float64)
    pend_k: List[np.ndarray] = []
    pend_w: List[np.ndarray] = []
    pending = kept_rows = 0
    for batch in _iter_edge_batches(files.edges, [c for c in ecol.values() if c]):
        raw_pre = np.asarray(batch[ecol["pre"]], dtype=np.int64)
        raw_post = np.asarray(batch[ecol["post"]], dtype=np.int64)
        w = (np.asarray(batch[ecol["weight"]], dtype=np.float64) if ecol["weight"]
             else np.ones(len(raw_pre), np.float64))
        report.edge_rows_read += len(w)

        pre_i, ok_pre = remap_ids(sorted_ids, order, raw_pre)
        post_i, ok_post = remap_ids(sorted_ids, order, raw_post)
        report.dropped_unknown_pre += int((~ok_pre).sum())
        report.dropped_unknown_post += int((ok_pre & ~ok_post).sum())
        keep = ok_pre & ok_post
        if drop_self_loops:
            loops = keep & (pre_i == post_i)
            report.dropped_self_loops += int(loops.sum())
            keep &= ~loops

        pend_k.append(post_i[keep].astype(np.int64) * n + pre_i[keep])
        pend_w.append(w[keep])
        pending += int(keep.sum())
        kept_rows += int(keep.sum())
        if pending >= merge_every:
            uniq_keys, uniq_w = _merge_keys([uniq_keys, *pend_k], [uniq_w, *pend_w])
            pend_k, pend_w, pending = [], [], 0
            if verbose:
                print(f"  rows read {report.edge_rows_read:,} -> unique pairs {len(uniq_keys):,}",
                      flush=True)
    uniq_keys, uniq_w = _merge_keys([uniq_keys, *pend_k], [uniq_w, *pend_w])
    del pend_k, pend_w
    report.duplicate_rows_merged = kept_rows - len(uniq_keys)

    strong = uniq_w >= min_weight
    report.dropped_below_min_weight = int((~strong).sum())
    uniq_keys, uniq_w = uniq_keys[strong], uniq_w[strong]
    pre = (uniq_keys % n).astype(np.int32)
    post = (uniq_keys // n).astype(np.int32)
    count = uniq_w.astype(np.float32)
    del uniq_keys
    report.edges_retained = len(pre)
    report.synapses_retained = int(uniq_w.sum())

    graph = ConnectomeArrays(
        body_id=body_id, pre=pre, post=post, count=count, nt_code=nt_code,
        nt_confidence=nt_conf, cell_type=cell_type.astype(np.int32),
        type_names=[str(t) for t in type_names], superclass=superclass.astype(np.int16),
        superclass_names=[str(s) for s in superclass_names], side=side, report=report,
    )
    report.n_type_pairs = int(len(graph.type_pair_index()[1]))
    report.load_seconds = report.build_seconds = time.perf_counter() - t0
    if report.accounted_rows() != report.edge_rows_read:
        raise AssertionError(f"Edge accounting mismatch: {report.accounted_rows()} != {report.edge_rows_read}")
    if verbose:
        print(report.summary())
    return graph


_ARRAY_FIELDS = ("body_id", "pre", "post", "count", "nt_code", "nt_confidence",
                 "cell_type", "superclass", "side")


def _cache_key(files: DataFiles, min_weight: float, drop_self_loops: bool,
               statuses: Optional[Sequence[str]]) -> str:
    parts = [str(LOADER_VERSION), str(min_weight), str(drop_self_loops),
             ",".join(sorted(statuses)) if statuses else "*"]
    for p in files.all():
        st = p.stat()
        parts += [p.name, str(st.st_size), str(int(st.st_mtime))]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def save_cache(graph: ConnectomeArrays, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{k: getattr(graph, k) for k in _ARRAY_FIELDS})
    tmp.replace(path)
    side = {"type_names": graph.type_names, "superclass_names": graph.superclass_names,
            "report": asdict(graph.report)}
    path.with_suffix(".json").write_text(json.dumps(side), encoding="utf-8")


def load_cache(path: Path) -> ConnectomeArrays:
    t0 = time.perf_counter()
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path) as data:
        arrays = {k: data[k] for k in _ARRAY_FIELDS}
    report = LoadReport(**side["report"])
    report.from_cache = True
    report.load_seconds = time.perf_counter() - t0
    return ConnectomeArrays(**arrays, type_names=side["type_names"],
                            superclass_names=side["superclass_names"], report=report)


def load_connectome(data_dir: str | Path = "data/connectome", cache_dir: str | Path | None = None,
                    min_weight: float = 1.0, drop_self_loops: bool = True,
                    statuses: Optional[Sequence[str]] = DEFAULT_STATUSES, use_cache: bool = True,
                    verbose: bool = True) -> ConnectomeArrays:
    """Load the connectome from ``data_dir``, building and caching it on first use.

    The cache is keyed on the source files' names, sizes and mtimes plus the
    loader options, so a new data release or option change rebuilds it.
    """
    data_dir = Path(data_dir)
    files = find_data_files(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / "cache"
    cache = cache_dir / f"connectome_{_cache_key(files, min_weight, drop_self_loops, statuses)}.npz"
    if use_cache and cache.exists() and cache.with_suffix(".json").exists():
        graph = load_cache(cache)
        if verbose:
            print(f"Loaded cached connectome {cache.name}\n{graph.report.summary()}")
        return graph
    graph = build_connectome(files, min_weight, drop_self_loops, statuses, verbose)
    if use_cache:
        save_cache(graph, cache)
    return graph
