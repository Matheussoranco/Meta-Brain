"""Bounded recurrent training, cutoff bootstrapping and persistent Colab resume."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from meta_brain.mario_rl import MarioRunner
from meta_brain.mario_trainable import (FullBrainConfig, RecurrentCollector, TrainableBrain,
                                        generalized_advantages, load_checkpoint, ppo_update,
                                        preflight, run_full_brain, save_checkpoint)
from test_mario_rl import FakeMario, small_graph

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_model(**overrides):
    torch.manual_seed(123)
    cfg = FullBrainConfig(n_envs=2, rollout=8, bptt=4, micro_steps=2, epochs=1,
                          updates=1, max_hours=0, max_steps=12, noop_max=0,
                          validation_episodes=1, final_episodes=1, eval_every=1, checkpoint_every=1,
                          latent=8, hidden=16, **overrides)
    structure, weights = small_graph(n=300)
    model = TrainableBrain(structure, weights, np.arange(60), np.arange(240, 300),
                           np.arange(300) % 8, 3, cfg)
    return model, cfg


def test_gradients_reach_vision_and_internal_dynamics():
    model, cfg = tiny_model()
    diag = preflight(model, cfg)
    assert diag["encoder_gradient_norm"] > 0
    factory = lambda: FakeMario(wall=10_000, die_at=60)
    collector = RecurrentCollector(model, factory, cfg)
    try:
        data = collector.collect()
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        stats = ppo_update(model, opt, data, cfg)
        changed = {k for k, p in model.named_parameters() if not torch.equal(p, before[k])}
        assert stats["grad_norm"] > 0
        assert any(k.startswith("encoder") for k in changed)
        assert {"recurrent", "decay", "input_scale"} <= changed
        # No edge-sized learned parameter or direct visual-to-action head.
        assert all(p.numel() != model.structure.e for p in model.parameters())
    finally:
        collector.close()


def test_time_limit_bootstraps_but_death_does_not():
    rewards = torch.tensor([[1., 1.], [1., 1.]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[0., 10.], [0., 0.]])  # death versus timeout terminal value
    dones = torch.tensor([[True, True], [False, False]])
    adv, ret = generalized_advantages(rewards, values, next_values, dones, .9, .95)
    assert ret[0].tolist() == [1., 10.]
    assert torch.equal(adv, ret)


def test_gymnasium_truncation_is_not_death():
    class TimeLimitMario(FakeMario):
        def step(self, action):
            obs, reward, _, info = super().step(action)
            return obs, reward, False, True, info
    runner = MarioRunner(TimeLimitMario(), frame_skip=4, noop_max=0)
    runner.reset()
    _, _, done, info = runner.step(1)
    assert done and info["truncated"] and not info["died"] and not info["terminated"]


def test_checkpoint_guards_graph_and_configuration(tmp_path):
    model, cfg = tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    save_checkpoint(tmp_path / "latest.pt", model, opt, cfg, "graph-a", 4, [], (.5, 300.))
    with pytest.raises(ValueError, match="graph"):
        load_checkpoint(tmp_path / "latest.pt", model, opt, cfg, "graph-b")
    cfg.vision = "zero"
    with pytest.raises(ValueError, match="vision"):
        load_checkpoint(tmp_path / "latest.pt", model, opt, cfg, "graph-a")


def test_training_restarts_from_saved_update(tmp_path):
    model, cfg = tiny_model()
    factory = lambda: FakeMario(wall=10_000, die_at=60)
    result = run_full_brain(model, factory, cfg, tmp_path, "graph-a", log=lambda _: None)
    assert result["completed_updates"] == 1
    assert (tmp_path / "best.pt").exists()
    model2, cfg2 = tiny_model()
    cfg2.updates = 2
    result2 = run_full_brain(model2, factory, cfg2, tmp_path, "graph-a", log=lambda _: None)
    assert [h["update"] for h in result2["history"]] == [1, 2]
    assert result2["history"][0] == result["history"][0]
    ck = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert ck["update"] == 2 and ck["optimizer"]["state"]
    assert result2["claim_status"] == "RUN_COMPLETED_BUT_SUCCESS_NOT_DEMONSTRATED"


def notebook_generator():
    spec = importlib.util.spec_from_file_location("v3_generator", ROOT / "scripts/build_colab_mario_v3_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v3_notebook_is_generated_and_compiles():
    gen = notebook_generator()
    nb = json.loads(gen.NOTEBOOK.read_text(encoding="utf-8"))
    assert nb == gen.build()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(line for line in cell["source"] if not line.startswith("%"))
            compile(src, "<v3 cell>", "exec")


def test_v3_notebook_runs_all_cells_offline(tmp_path):
    from test_colab_notebook import write_small_release
    data = tmp_path / "data"
    data.mkdir()
    write_small_release(data)
    gen = notebook_generator()
    # A subprocess prevents the notebook's self-contained package installation
    # from replacing modules imported by the rest of the test suite.
    script = tmp_path / "run_notebook.py"
    script.write_text('''import sys, json, torch
torch.set_num_threads(1)
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_mario_rl import FakeMario
nb = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
ns = {'__name__': '__main__'}
for cell in nb['cells']:
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    if src.startswith('# Não reinstala'): continue
    if src.startswith('from meta_brain.colab_mario'):
        import meta_brain.colab_mario as runtime
        import meta_brain.mario_rl as mario_runtime
        runtime.make_mario_env = lambda *a, **kw: (FakeMario(die_at=60), ['NOOP', 'right', 'right+A'])
        mario_runtime.make_mario_env = runtime.make_mario_env
    exec(compile(src, '<v3 notebook cell>', 'exec'), ns)
assert ns['RESULTS']['completed_updates'] == 1
assert ns['GRAPH_INFO']['neurons'] == 560
assert ns['RESULTS']['claim_status'] == 'RUN_COMPLETED_BUT_SUCCESS_NOT_DEMONSTRATED'
''', encoding="utf-8")
    env = dict(os.environ, MB_OFFLINE="1", MB_DEVICE="cpu", MB_DATA_DIR=str(data),
               MB_OUT_DIR=str(tmp_path / "out"), MB_V3_RUNTIME=str(tmp_path / "runtime"),
               MB_MIN_NEURONS="100", MB_MIN_EDGES="1000", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(script), str(gen.NOTEBOOK)], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-6000:]
    assert (tmp_path / "out" / "latest.pt").exists()
