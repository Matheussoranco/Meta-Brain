import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from meta_brain.graph import (NT_CODES, build_connectome, find_data_files, load_connectome,
                              merge_duplicates, remap_ids)

ANN = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Rows: repeated pairs, untraced/unknown bodies, a self-loop and a zero-weight pair.
FLAT_EDGES = pd.DataFrame({
    "body_pre":  [10, 10, 20, 30, 99, 40, 50, 10, 20, 40, 60, 10, 30],
    "body_post": [20, 30, 30, 40, 10, 98, 50, 20, 10, 50, 10, 60, 40],
    "weight":    [5,  3,  1,  7,  2,  4,  6,  2,  0,  9,  3,  3,  1],
})


def write_flat(tmp_path, edges=FLAT_EDGES, chunksize=4):
    ann = pd.DataFrame({
        "bodyId": [50, 10, 40, 20, 30, 60],
        "status": ["Traced"] * 5 + ["Orphan"],
        "type": ["T1", "T1", "", "T2", None, "T9"],
        "superclass": ["ol_sensory", "ol_sensory", "descending_neuron", "cb_intrinsic",
                       "cb_intrinsic", "cb_intrinsic"],
        "class": ["photoreceptor", "photoreceptor", "dn", "x", "y", "z"],
        "somaSide": ["L", "R", "L", "R", None, "L"],
    })
    nt = pd.DataFrame({
        "body": [10, 20, 30, 40, 50, 60, 999],
        "consensus_nt": ["gaba", None, "unclear", "glutamate", "acetylcholine", "gaba", "gaba"],
        "predicted_nt": ["gaba", "acetylcholine", "serotonin", "glutamate", "acetylcholine",
                         "gaba", "gaba"],
        "predicted_nt_prob": [0.9, 0.8, 0.4, 0.95, 0.99, 0.7, 0.5],
    })
    ann.to_feather(tmp_path / ANN)
    nt.to_feather(tmp_path / NT)
    feather.write_feather(pa.Table.from_pandas(edges, preserve_index=False), tmp_path / WEIGHTS,
                          chunksize=chunksize)
    return find_data_files(tmp_path)


def edge_dict(g):
    return {(int(g.body_id[a]), int(g.body_id[b])): float(c) for a, b, c in zip(g.pre, g.post, g.count)}


def test_flat_layout_is_found(tmp_path):
    files = write_flat(tmp_path)
    assert files.layout == "flat"
    assert files.annotations.name == ANN and files.edges.name == WEIGHTS and files.neurotransmitters.name == NT


def test_flat_build_accounts_for_every_row(tmp_path):
    g = build_connectome(write_flat(tmp_path), verbose=False)
    r = g.report
    assert r.bodies_annotated == 6 and r.dropped_by_status == 1 and r.n_neurons == 5
    assert r.edge_rows_read == 13
    assert r.dropped_unknown_pre == 2  # 99 and untraced 60
    assert r.dropped_unknown_post == 2  # 98 and untraced 60
    assert r.dropped_self_loops == 1
    assert r.duplicate_rows_merged == 2  # repeated rows of 10->20 and 30->40
    assert r.dropped_below_min_weight == 1  # 20->10 sums to 0
    assert r.edges_retained == 5
    assert r.accounted_rows() == r.edge_rows_read
    assert edge_dict(g) == {(10, 20): 7.0, (10, 30): 3.0, (20, 30): 1.0, (30, 40): 8.0, (40, 50): 9.0}
    assert r.synapses_retained == 28
    key = g.post.astype(np.int64) * g.n + g.pre
    assert np.all(np.diff(key) > 0)
    assert g.csr_rowptr()[-1] == g.e
    assert r.columns_used["edge.pre"] == "body_pre"


def test_min_weight_applies_to_merged_pairs(tmp_path):
    files = write_flat(tmp_path)
    g = build_connectome(files, min_weight=3, verbose=False)
    # 10->20 has rows 5 + 2 and survives; 20->30 (1) and 20->10 (0) do not.
    assert set(edge_dict(g)) == {(10, 20), (10, 30), (30, 40), (40, 50)}
    assert g.report.dropped_below_min_weight == 2
    assert g.report.accounted_rows() == g.report.edge_rows_read


def test_incremental_merge_matches_single_merge(tmp_path):
    files = write_flat(tmp_path, chunksize=2)
    once = build_connectome(files, verbose=False)
    often = build_connectome(files, verbose=False, merge_every=1)
    assert edge_dict(once) == edge_dict(often)
    assert once.report.duplicate_rows_merged == often.report.duplicate_rows_merged


def test_status_filter_can_be_disabled(tmp_path):
    g = build_connectome(write_flat(tmp_path), statuses=None, verbose=False)
    assert g.n == 6 and g.report.dropped_by_status == 0
    assert (60, 10) in edge_dict(g) and (10, 60) in edge_dict(g)


def test_neurotransmitters_join_prefers_consensus(tmp_path):
    g = build_connectome(write_flat(tmp_path), verbose=False)
    by_id = {int(b): i for i, b in enumerate(g.body_id)}
    assert g.nt_code[by_id[10]] == NT_CODES["gaba"]
    assert g.nt_code[by_id[20]] == NT_CODES["acetylcholine"]  # consensus missing -> predicted
    assert g.nt_code[by_id[30]] == NT_CODES["serotonin"]  # consensus 'unclear' -> predicted
    assert g.nt_code[by_id[40]] == NT_CODES["glutamate"]
    assert g.nt_confidence[by_id[50]] == pytest.approx(0.99)
    sign = g.sign()
    assert sign[by_id[50]] == 1.0 and sign[by_id[10]] == -1.0 and sign[by_id[40]] == -1.0
    assert g.sign({"glutamate": 1.0})[by_id[40]] == 1.0
    assert g.report.nt_counts == {"acetylcholine": 2, "gaba": 1, "glutamate": 1, "serotonin": 1}


def test_types_masks_and_pairs(tmp_path):
    g = build_connectome(write_flat(tmp_path), verbose=False)
    by_id = {int(b): i for i, b in enumerate(g.body_id)}
    names = [g.type_names[t] for t in g.cell_type]
    assert names[by_id[10]] == names[by_id[50]] == "T1"
    assert names[by_id[40]] == "untyped:descending_neuron:dn"
    assert names[by_id[30]] == "untyped:cb_intrinsic:y"
    assert g.report.n_untyped_neurons == 2
    assert set(g.body_id[g.sensory_indices()]) == {10, 50}
    assert set(g.body_id[g.descending_indices()]) == {40}
    assert g.side[by_id[50]] == 1 and g.side[by_id[30]] == 0

    edge_pair, pair_types = g.type_pair_index()
    np.testing.assert_array_equal(pair_types[edge_pair, 0], g.cell_type[g.pre])
    np.testing.assert_array_equal(pair_types[edge_pair, 1], g.cell_type[g.post])
    assert len(np.unique(pair_types, axis=0)) == len(pair_types) == g.report.n_type_pairs


def test_cache_roundtrip_and_invalidation(tmp_path):
    write_flat(tmp_path)
    first = load_connectome(tmp_path, verbose=False)
    assert not first.report.from_cache
    second = load_connectome(tmp_path, verbose=False)
    assert second.report.from_cache
    for name in ("body_id", "pre", "post", "count", "cell_type", "nt_code", "superclass", "side"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.type_names == second.type_names
    assert second.report.edges_retained == first.report.edges_retained

    assert not load_connectome(tmp_path, min_weight=5, verbose=False).report.from_cache
    assert not load_connectome(tmp_path, statuses=None, verbose=False).report.from_cache
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 3


def test_legacy_layout_still_loads(tmp_path):
    meta = pd.DataFrame({"bodyId": [1, 2, 3], "superclass": ["ol_sensory", "cb_intrinsic", "dn"],
                         "type": ["A", "B", "C"], "predictedNt": ["acetylcholine", "gaba", "ach"]})
    edges = pd.DataFrame({"bodyId_pre": [1, 2, 2], "bodyId_post": [2, 3, 3], "weight": [4, 1, 2]})
    meta.to_feather(tmp_path / "malecns_v1_0_meta.feather")
    edges.to_feather(tmp_path / "malecns_v1_0_simple_edgelist.feather")
    files = find_data_files(tmp_path)
    assert files.layout == "legacy" and files.neurotransmitters is None
    g = build_connectome(files, verbose=False)
    assert edge_dict(g) == {(1, 2): 4.0, (2, 3): 3.0}
    assert g.report.nt_source == "annotations.predictedNt"
    assert g.nt_code.tolist() == [NT_CODES["acetylcholine"], NT_CODES["gaba"], NT_CODES["acetylcholine"]]


def test_missing_files_and_columns_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        find_data_files(tmp_path)
    bad = pd.DataFrame({"a": [1], "b": [2], "weight": [1]})
    with pytest.raises(ValueError, match="edges pre"):
        build_connectome(write_flat(tmp_path, edges=bad), verbose=False)


def test_remap_and_merge_helpers():
    ids = np.array([30, 10, 20], dtype=np.int64)
    order = np.argsort(ids)
    idx, ok = remap_ids(ids[order], order, np.array([20, 5, 30, 99]))
    assert ok.tolist() == [True, False, True, False]
    assert idx[ok].tolist() == [2, 0]

    pre, post, count, merged = merge_duplicates(np.array([0, 1, 0]), np.array([1, 2, 1]),
                                                np.array([1.0, 2.0, 3.0], np.float32), 3)
    assert merged == 1
    assert list(zip(pre.tolist(), post.tolist(), count.tolist())) == [(0, 1, 4.0), (1, 2, 2.0)]
