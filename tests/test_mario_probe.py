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
