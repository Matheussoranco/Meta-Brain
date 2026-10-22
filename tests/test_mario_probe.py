"""meta_brain.mario_probe with a fake NES that exposes SMB-style RAM."""

import numpy as np
import pytest

from meta_brain.mario_probe import (PIPES_1_1, collect_probe_dataset, plot_probe, ridge_cv, run_probe,
                                    smb_labels, time_features)
from meta_brain.mario_rl import MarioRunner
from test_mario_rl import small_reservoir


class FakeSMB:
    """Mario walks right on action 1 or 2. A Goomba sits at a random level x and is drawn on screen."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.unwrapped = self

    def reset(self):
        self.x, self.t = 40, 0
        self.goomba = int(self.rng.integers(150, 400))
        return self._obs()

    @property
    def ram(self):
        ram = np.zeros(2048, np.uint8)
        ram[0x6D], ram[0x86] = divmod(int(self.x), 256)
        ram[0x0F], ram[0x16] = 1, 0x06
        ram[0x6E], ram[0x87] = divmod(self.goomba, 256)
        ram[0x10], ram[0x17], ram[0x88] = 1, 0x00, 50  # a drawn non-Goomba enemy is ignored
        return ram

    def _obs(self):
        frame = np.zeros((240, 256, 3), np.uint8)
        dx = self.goomba - self.x
        if 0 <= dx < 200:
            col = 40 + int(dx)
            frame[150:200, col:col + 16] = 255  # the Goomba, drawn relative to Mario
        return frame

    def step(self, action):
        self.t += 1
        self.x += 2 if action in (1, 2) else 0
        done = self.t >= 160
        return self._obs(), 1.0, done, {"x_pos": self.x, "flag_get": False}

    def close(self):
        pass


def test_smb_labels_from_ram():
    env = FakeSMB()
    env.reset()
    env.x, env.goomba = 300, 380
    lab = smb_labels(env)
    assert lab["mario_x"] == 300 and lab["goomba_dx"] == 80 and lab["goomba_near"] == 1.0
    assert lab["pipe_dx"] == 448 - 300
    env.goomba = 100  # behind Mario: not "ahead"
    assert np.isnan(smb_labels(env)["goomba_dx"]) and smb_labels(env)["goomba_near"] == 0.0
    env.x = 600  # standing at the 608 pipe
    assert smb_labels(env)["pipe_dx"] == 8
    env.x = 3000  # past every pipe
    assert smb_labels(env)["pipe_dx"] == 256.0
    assert PIPES_1_1 == tuple(sorted(PIPES_1_1))


def test_ridge_cv_recovers_linear_signal_and_auc():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 20))
    y = X[:, 0] * 3 - X[:, 1] + rng.normal(scale=0.1, size=400)
    groups = np.repeat(np.arange(10), 40)
    assert ridge_cv(X, y, groups)["score"] > 0.95
    assert ridge_cv(rng.normal(size=(400, 20)), y, groups)["score"] < 0.1
    labels = (X[:, 0] > 0).astype(float)
    assert ridge_cv(X, labels, groups, binary=True)["score"] > 0.95
    wide = rng.normal(size=(60, 200))  # dual form path
    assert np.isfinite(ridge_cv(wide, wide[:, 0], np.repeat(np.arange(6), 10))["score"])


def test_time_features_one_hot():
    t = time_features(np.array([0, 9, 10, 25]), bin_size=10)
    assert t.shape == (4, 3) and t.sum(1).tolist() == [1, 1, 1, 1] and t[3, 2] == 1


def test_probe_end_to_end(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    res = small_reservoir()
    runner = MarioRunner(FakeSMB(seed=3), frame_skip=2, noop_max=4, stuck_steps=100, max_steps=200, seed=0)
    data = collect_probe_dataset(runner, res, episodes=12, size=(8, 8), seed=0)
    n = len(data["step"])
    assert data["motor_real"].shape == (n, 60) and data["motor_clock"].shape == (n, 60)
    assert data["visual_code"].shape == (n, 128) and len(np.unique(data["episode"])) == 12
    result = run_probe(data)
    scores = result["scores"]
    assert set(scores) == {"goomba_dx", "goomba_near", "pipe_dx", "mario_x"}
    assert scores["goomba_dx"]["visual_code"] > 0.5  # the input carries the Goomba
    assert scores["goomba_dx"]["motor_real"] > scores["goomba_dx"]["motor_clock"]
    assert "goomba_dx" in result["vision_gain_motor_real_minus_clock"]
    plot_probe(result, tmp_path / "probe.png")
    assert (tmp_path / "probe.png").stat().st_size > 0


class FakeGraph:
    def __init__(self, superclass):
        self.superclass = np.asarray(superclass)

    def superclass_mask(self, names):
        return np.isin(self.superclass, list(names))


def test_layer_indices_subsample_and_skip_missing():
    from meta_brain.mario_probe import layer_indices

    g = FakeGraph(["ol_sensory"] * 50 + ["ol_intrinsic"] * 10 + ["cb_intrinsic"] * 5)
    layers = layer_indices(g, ("ol_sensory", "ol_intrinsic", "visual_projection"), max_per_layer=20, seed=0)
    assert list(layers) == ["ol_sensory", "ol_intrinsic"]  # missing superclass skipped, order kept
    assert len(layers["ol_sensory"]) == 20 and set(layers["ol_sensory"]) <= set(range(50))
    assert layers["ol_intrinsic"].tolist() == list(range(50, 60))
    np.testing.assert_array_equal(layers["ol_sensory"], np.sort(layers["ol_sensory"]))


def test_layer_probe_finds_signal_in_driven_layer(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from meta_brain.mario_probe import plot_layer_probe, run_layer_probe

    res = small_reservoir()  # sensory = 0..59 (driven by the frame), motor = 240..299
    runner = MarioRunner(FakeSMB(seed=3), frame_skip=2, noop_max=4, stuck_steps=100, max_steps=200, seed=0)
    layers = {"sensory": np.arange(0, 60), "middle": np.arange(100, 160), "motor": np.arange(240, 300)}
    data = collect_probe_dataset(runner, res, episodes=12, size=(8, 8), seed=0, layers=layers)
    assert data["layer:sensory:real"].shape == (len(data["step"]), 60)
    np.testing.assert_allclose(data["layer:motor:real"], data["motor_real"], atol=1e-6)
    result = run_layer_probe(data, layer_sizes={"sensory": 60})
    assert [r["layer"] for r in result["layers"]] == ["sensory", "middle", "motor"]
    sensory = result["layers"][0]
    assert sensory["neurons_total"] == 60 and sensory["goomba_dx_signal"] > 0.2
    assert result["targets"] == {"goomba_dx": "r2", "goomba_near": "auc"}
    plot_layer_probe(result, tmp_path / "layers.png", ceiling={"goomba_dx": 0.9})
    assert (tmp_path / "layers.png").stat().st_size > 0


def test_sweep_matches_single_config_and_restores_reservoir():
    from meta_brain.mario_probe import collect_sweep_dataset

    layers = {"sensory": np.arange(0, 60), "motor": np.arange(240, 300)}
    res = small_reservoir()
    res.gain, res.input_gain = 1.6, 1.0

    def runner():
        return MarioRunner(FakeSMB(seed=3), frame_skip=2, noop_max=4, stuck_steps=100, max_steps=60, seed=0)

    single = collect_probe_dataset(runner(), res, episodes=3, size=(8, 8), seed=0, layers=layers)
    sweep = collect_sweep_dataset(runner(), res, [(0.5, 4.0), (1.6, 1.0)], layers, episodes=3, size=(8, 8), seed=0)
    assert res.gain == 1.6 and res.input_gain == 1.0  # restored
    for side in ("real", "clock"):
        np.testing.assert_allclose(sweep[f"cfg1:layer:motor:{side}"], single[f"layer:motor:{side}"], atol=1e-5)
    assert not np.allclose(sweep["cfg0:layer:motor:real"], sweep["cfg1:layer:motor:real"])
    np.testing.assert_array_equal(sweep["episode"], single["episode"])
    for name in layers:
        assert sweep["regime"][name]["mean_abs"].shape == (2,)
        assert ((0 <= sweep["regime"][name]["saturated"]) & (sweep["regime"][name]["saturated"] <= 1)).all()


def test_sweep_probe_rows_and_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from meta_brain.mario_probe import (collect_sweep_dataset, plot_sweep_probe, run_sweep_probe,
                                        sweep_configs)

    configs = sweep_configs(gains=(0.5, 1.6), input_gains=(1.0, 4.0))
    assert configs == [(0.5, 1.0), (1.6, 1.0), (0.5, 4.0), (1.6, 4.0)]
    runner = MarioRunner(FakeSMB(seed=3), frame_skip=2, noop_max=4, stuck_steps=100, max_steps=200, seed=0)
    layers = {"sensory": np.arange(0, 60), "middle": np.arange(100, 160), "motor": np.arange(240, 300)}
    data = collect_sweep_dataset(runner, small_reservoir(), configs, layers, episodes=10, size=(8, 8))
    result = run_sweep_probe(data)
    assert len(result["rows"]) == len(configs) * 3 and result["layers"] == ["sensory", "middle", "motor"]
    assert set(result["deepest_layer_with_signal"]) == {f"gain={g:g},input={i:g}" for g, i in configs}
    driven = [r for r in result["rows"] if r["layer"] == "sensory"]
    assert all(r["signal"] > 0.05 for r in driven)  # the driven layer always sees the Goomba
    plot_sweep_probe(result, tmp_path / "sweep.png")
    assert (tmp_path / "sweep.png").stat().st_size > 0


def test_probe_handles_missing_targets():
    data = {k: np.zeros(10, np.float32) for k in ("mario_x", "goomba_near", "pipe_dx")}
    data.update(goomba_dx=np.full(10, np.nan, np.float32), step=np.arange(10), episode=np.zeros(10, int),
                motor_real=np.zeros((10, 3), np.float32), motor_clock=np.zeros((10, 3), np.float32),
                visual_code=np.zeros((10, 4), np.float32))
    result = run_probe(data)
    assert all(np.isnan(v) for v in result["scores"]["goomba_dx"].values())


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_scripted_policy_extremes_still_terminate(prob):
    res = small_reservoir()
    runner = MarioRunner(FakeSMB(), frame_skip=2, noop_max=0, stuck_steps=100, max_steps=50, seed=0)
    data = collect_probe_dataset(runner, res, episodes=2, size=(8, 8), jump_prob=prob)
    assert len(data["step"]) > 0
