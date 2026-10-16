"""The Colab M0/M1 notebook is generated from meta_brain/*.py and must run end to end."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    spec = importlib.util.spec_from_file_location("build_nb", ROOT / "scripts" / "build_colab_m0_m1_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_notebook_is_up_to_date():
    gen = load_generator()
    expected = gen.build(gen.source_digest())
    actual = json.loads(gen.NOTEBOOK.read_text(encoding="utf-8"))
    assert actual == expected, "Run: python scripts/build_colab_m0_m1_notebook.py"


def write_small_release(data_dir: Path, n=600, pairs=6000, seed=0):
    rng = np.random.default_rng(seed)
    body = rng.choice(10**9, n, replace=False).astype(np.int64)
    superclass = rng.choice(["ol_sensory", "cb_intrinsic", "ol_intrinsic", "descending_neuron"], n,
                            p=[0.2, 0.4, 0.3, 0.1])
    pd.DataFrame({
        "bodyId": body,
        "status": np.where(np.arange(n) < n - 40, "Traced", "Orphan"),
        "type": [f"T{t}" for t in rng.integers(0, 8, n)],
        "superclass": superclass,
        "class": "c",
        "somaSide": rng.choice(["L", "R"], n),
    }).to_feather(data_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nts = rng.choice(["acetylcholine", "gaba", "glutamate", "histamine", "dopamine"], n,
                     p=[0.6, 0.15, 0.15, 0.05, 0.05])
    pd.DataFrame({"body": body, "consensus_nt": np.where(rng.random(n) < 0.5, nts, "unclear"),
                  "predicted_nt": nts}).to_feather(data_dir / "body-neurotransmitters-male-cns-v1.0.feather")

    w_out, w_in = rng.pareto(1.5, n) + 1, rng.pareto(1.5, n) + 1
    pre = rng.choice(n, pairs, p=w_out / w_out.sum())
    post = rng.choice(n, pairs, p=w_in / w_in.sum())
    weight = rng.geometric(0.2, pairs)
    # Repeat some pairs on extra rows to exercise duplicate merging.
    split = rng.random(pairs) < 0.3
    pre = np.concatenate([pre, pre[split]])
    post = np.concatenate([post, post[split]])
    weight = np.concatenate([weight, rng.integers(1, 4, split.sum())])
    table = pa.table({"body_pre": body[pre], "body_post": body[post], "weight": weight.astype(np.int32)})
    feather.write_feather(table, data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                          chunksize=1000)


@pytest.mark.filterwarnings("ignore")
def test_notebook_runs_end_to_end_on_a_small_release(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "data", tmp_path / "out"
    data_dir.mkdir()
    write_small_release(data_dir)
    for key, value in {"MB_DATA_DIR": data_dir, "MB_OUT_DIR": out_dir, "MB_OFFLINE": "1",
                       "MB_MIN_NEURONS": "100", "MB_MIN_EDGES": "1000", "MB_DEVICE": "cpu",
                       "MB_BENCH": "tiny", "MB_MICRO_STEPS": "2", "MPLBACKEND": "Agg"}.items():
        monkeypatch.setenv(key, str(value))

    import matplotlib
    matplotlib.use("Agg")
    gen = load_generator()
    namespace = {"__name__": "__main__"}
    for i, src in enumerate(gen.code_cells(gen.build("test"))):
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!"))
        exec(compile(code, f"<notebook cell {i}>", "exec"), namespace)

    results = json.loads((out_dir / "m0_m1_results.json").read_text(encoding="utf-8"))
    assert results["claim_status"] == {"M0": "PASS", "M1": "NOT_EVALUATED_NO_CUDA"}
    m0 = results["criteria"]["m0"]
    assert m0["edge_accounting_closes"] and m0["degree_shuffle_invariants"] and m0["type_shuffle_invariants"]
    load = results["m0_load"]
    assert load["neurons"] == 560 and load["report"]["dropped_by_status"] == 40
    assert load["report"]["duplicate_rows_merged"] > 0
    checks = results["m1_checks"]
    assert checks["gradcheck_cpu"] and checks["full_graph_equivalent"]
    saved = checks["saved_numel_per_call"]
    assert saved["custom"] <= saved["N*B"] + load["edges"] < saved["reference"]
    assert all(r["status"] == "OK" for r in results["m1_benchmark"])
    for name in ("m0_connectome.png", "m0_shuffles.png", "m1_benchmark.png", "m0_shuffles.csv",
                 "m1_benchmark.csv"):
        assert (out_dir / name).stat().st_size > 0
    assert (tmp_path / "out.zip").exists()
