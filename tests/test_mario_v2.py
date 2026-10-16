"""The Mario V2 notebook and script load the real MaleCNS flat-connectome format.

The emulator and CUDA parts cannot run here; these tests cover the data path
(download offline, embedded loader, signed reservoir weights) and check that
every notebook cell at least compiles.
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_colab_notebook import load_generator, write_small_release

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_full_brain_mario_v2.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_sources(nb):
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            out.append("\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!")))
    return out


def test_embedded_loader_is_in_sync():
    nb = notebook()
    assert not load_generator().sync_embedded(copy.deepcopy(nb)), \
        "Run: python scripts/build_colab_m0_m1_notebook.py"
    assert any(src.startswith("# --- embedded from meta_brain/graph.py ---") for src in code_sources(nb))


def test_every_cell_compiles():
    for i, src in enumerate(code_sources(notebook())):
        compile(src, f"<mario v2 cell {i}>", "exec")


@pytest.fixture()
def release(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_small_release(data_dir)
    for key, value in {"MB_DATA_DIR": data_dir, "MB_OUT_DIR": tmp_path / "out", "MB_OFFLINE": "1",
                       "MB_MIN_NEURONS": "100", "MB_MIN_EDGES": "1000"}.items():
        monkeypatch.setenv(key, str(value))
    return data_dir


RUN_MARKER = "make_mario_env('1-1'"


def run_data_cells(nb):
    """Execute the notebook up to the emulator/training cell."""
    namespace = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        if RUN_MARKER in src:
            break
        exec(compile(src, f"<mario v2 cell {i}>", "exec"), namespace)
    return namespace


@pytest.mark.filterwarnings("ignore")
def test_whole_notebook_runs_with_a_fake_emulator(release, monkeypatch, tmp_path):
    """Every cell, including PPO training and the H.264 videos, on CPU with a stand-in for the NES."""
    import matplotlib

    matplotlib.use("Agg")
    from test_mario_probe import FakeSMB

    for key, value in {"MB_DEVICE": "cpu", "MB_MARIO_ENVS": "2", "MB_MARIO_UPDATES": "2",
                       "MB_MARIO_EVAL": "2", "MB_MARIO_MICRO_STEPS": "1", "MB_PROBE_EPISODES": "4",
                       "MB_MARIO_CONDITIONS": "real,zero,clock"}.items():
        monkeypatch.setenv(key, value)
    namespace = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(notebook())):
        if RUN_MARKER in src:
            # Swap the emulator for the fake after the embedded module defined the real factory.
            namespace["make_mario_env"] = lambda *a, **k: (FakeSMB(), ["NOOP", "right", "right+A"])
        exec(compile(src, f"<mario v2 cell {i}>", "exec"), namespace)

    out = tmp_path / "out"
    runs = namespace["RESULTS"]
    assert list(runs) == ["real", "zero", "clock"]
    for condition, results in runs.items():
        assert results["vision"] == condition
        assert results["claim_status"] == "RUN_COMPLETED_BUT_SUCCESS_NOT_DEMONSTRATED"
        assert results["graph"]["neurons"] == 560
        assert results["training"]["agent_steps"] == 2 * 128 * 2
        for name in ("pre.mp4", "post.mp4", "post_sampled.mp4", "curves.png", "results.json"):
            assert (out / condition / f"full_brain_mario_v2_{name}").stat().st_size > 0
    assert runs["real"]["motor_readout_diagnostics"]["motor_features"] == runs["real"]["graph"]["motor"]
    assert runs["zero"]["motor_readout_diagnostics"]["responsive_fraction"] == 0.0
    comparison = namespace["COMPARISON"]
    assert "real_minus_zero_mean_x" in comparison and "real_minus_clock_mean_x" in comparison
    assert (out / "mario_vision_controls.png").stat().st_size > 0
    probe = namespace["PROBE"]
    assert probe["episodes"] == 4 and set(probe["scores"]) == {"goomba_dx", "goomba_near", "pipe_dx", "mario_x"}
    assert (out / "mario_probe.png").stat().st_size > 0 and (out / "mario_probe.json").stat().st_size > 0
    assert (tmp_path / "out.zip").stat().st_size > 0


def test_notebook_data_path_on_real_format(release):
    ns = run_data_cells(notebook())
    info = ns["GRAPH_INFO"]
    assert info["neurons"] == 560  # Traced bodies only
    acct = info["edge_accounting"]
    assert acct["dropped_by_status"] == 40
    assert acct["duplicate_rows_merged"] > 0
    assert acct["columns_used"]["edge.pre"] == "body_pre"
    assert acct["nt_source"].startswith("consensus_nt")
    assert 0 < info["inhibitory_edge_fraction"] < 1
    ew, pre, post = ns["ew"], ns["pre"], ns["post"]
    assert len(ew) == len(pre) == len(post) == info["edges"]
    sign = ns["graph"].sign()
    np.testing.assert_array_equal(np.sign(ew), sign[pre])
    # Incoming energy is normalized to 1 for every neuron with inputs.
    energy = np.bincount(post, ew.astype(np.float64) ** 2, info["neurons"])
    np.testing.assert_allclose(energy[energy > 0], 1.0, rtol=1e-4)


def test_script_loader_matches_notebook(release):
    spec = importlib.util.spec_from_file_location("mario_v2", ROOT / "scripts" / "full_brain_mario_v2.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    graph = script.load_full_connectome(release, min_neurons=100, min_edges=1000)
    ns = run_data_cells(notebook())
    np.testing.assert_array_equal(graph["pre"], ns["pre"])
    np.testing.assert_array_equal(graph["post"], ns["post"])
    np.testing.assert_allclose(graph["weights"], ns["ew"], rtol=1e-6)
    np.testing.assert_array_equal(graph["motor"], ns["motor"])
    assert graph["edge_accounting"]["edges_retained"] == graph["e"]

    with pytest.raises(RuntimeError, match="Full-brain guard"):
        script.load_full_connectome(release, min_neurons=10**6, min_edges=1000)
