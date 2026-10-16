"""meta_brain.mario_rl with a fake NES environment (the emulator is not needed)."""

import shutil
import subprocess

import numpy as np
import pytest
import torch

from meta_brain.graph import merge_duplicates
from meta_brain.mario_rl import (BatchedRollout, ConnectomeReservoir, MarioRunner, PPOConfig,
                                 ReadoutPolicy, code_dim, encode_frame, evaluate, save_video,
                                 summarize, train_ppo, warmup_normalization)
from meta_brain.sparse_ops import SparseStructure


class FakeMario:
    """Old-gym API. Action 1 moves right; anything else stands still. A wall stops Mario at x=80."""

    def __init__(self, wall=80, die_at=None, flag_at=None):
        self.wall, self.die_at, self.flag_at = wall, die_at, flag_at
        self.resets = 0

    def close(self):
        pass

    def reset(self):
        self.x, self.t = 40, 0
        self.resets += 1
        return self._obs()

    def _obs(self):
        frame = np.zeros((240, 256, 3), np.uint8)
        frame[150:170, int(self.x) % 256] = 255
        return frame

    def step(self, action):
        self.t += 1
        dx = 1 if action == 1 and self.x < self.wall else 0
        self.x += dx
        done = (self.die_at is not None and self.t >= self.die_at)
        flag = self.flag_at is not None and self.x >= self.flag_at
        info = {"x_pos": self.x, "flag_get": flag}
        return self._obs(), float(dx), bool(done or flag), info


def small_graph(n=300, e=3000, seed=0):
    rng = np.random.default_rng(seed)
    pre, post, _, _ = merge_duplicates(rng.integers(0, n, e), rng.integers(0, n, e), np.ones(e, np.float32), n)
    return SparseStructure.from_edges(pre, post, n), torch.as_tensor(rng.normal(0, 0.3, len(pre)).astype(np.float32))


def small_reservoir(micro_steps=2, seed=0):
    s, w = small_graph(seed=seed)
    return ConnectomeReservoir(s, w, sensory=np.arange(0, 60), motor=np.arange(240, 300),
                               input_dim=code_dim((8, 8)), micro_steps=micro_steps, seed=seed)


def test_runner_frame_skip_records_and_sums_reward():
    runner = MarioRunner(FakeMario(), frame_skip=4, noop_max=0, stuck_steps=100, max_steps=100)
    runner.reset(record=True)
    _, reward, done, info = runner.step(1)
    assert reward == 4.0 and not done
    assert len(runner.frames) == 1 + 4  # reset frame + skipped frames
    assert runner.best_x == 44


def test_runner_cuts_stuck_episodes_and_flags_death():
    runner = MarioRunner(FakeMario(wall=50), frame_skip=4, noop_max=0, stuck_steps=3, max_steps=100)
    runner.reset()
    done, steps = False, 0
    while not done:
        _, _, done, info = runner.step(1)
        steps += 1
    assert info["truncated"] and not info["died"]
    assert runner.episode_summary()["max_x"] == 50
    assert steps == 3 + 3  # 10 frames of progress (3 agent steps), then 3 stuck steps

    runner = MarioRunner(FakeMario(die_at=6), frame_skip=4, noop_max=0)
    runner.reset()
    runner.step(1)
    _, _, done, info = runner.step(1)
    assert done and info["died"] and not info["truncated"]


def test_noop_starts_vary_between_episodes():
    env = FakeMario()
    runner = MarioRunner(env, frame_skip=4, noop_max=30, seed=0)
    lengths = set()
    for _ in range(6):
        runner.reset(record=True)
        lengths.add(len(runner.frames))
    assert len(lengths) > 1


def test_encode_frame_crops_hud_and_adds_motion():
    frame = np.zeros((240, 256, 3), np.uint8)
    frame[:32] = 255  # HUD only
    code, small = encode_frame(frame, size=(8, 8))
    assert code.shape == (128,) and np.allclose(small, -1.0)
    moved = frame.copy()
    moved[100:140, 100:140] = 255
    code2, _ = encode_frame(moved, small, size=(8, 8))
    assert np.abs(code2[64:]).sum() > 0  # frame difference sees the change


def test_reservoir_responds_to_input_and_keeps_shapes():
    res = small_reservoir()
    state = res.initial_state(3)
    codes = torch.randn(3, code_dim((8, 8)))
    f1, state = res.step(codes, state)
    assert f1.shape == (3, 60) and state.shape == (300, 3)
    f0, _ = res.step(torch.zeros(3, code_dim((8, 8))), res.initial_state(3))
    assert torch.count_nonzero(f0) == 0
    assert torch.isfinite(f1).all()


def test_ppo_learns_to_move_right():
    torch.manual_seed(0)
    res = small_reservoir()
    runners = [MarioRunner(FakeMario(wall=10_000, die_at=200), frame_skip=2, noop_max=3, seed=i)
               for i in range(4)]
    rollout = BatchedRollout(runners, res, size=(8, 8))
    policy = ReadoutPolicy(res.n_motor, 3, seed=0)
    diag = warmup_normalization(rollout, policy, steps=8, n_actions=3)
    assert 0 < diag["responsive_fraction"] <= 1

    def p_right():
        with torch.no_grad():
            return float(torch.softmax(policy(rollout.features)[0], -1)[:, 1].mean())

    before = p_right()
    history = train_ppo(rollout, policy, PPOConfig(updates=25, rollout=32, lr=3e-3, warmup_steps=0), log=None)
    assert len(history) == 25 and all(np.isfinite(h["policy_loss"]) for h in history)
    assert p_right() > max(0.6, before + 0.2)


def test_evaluate_returns_best_episode_frames():
    res = small_reservoir()
    policy = ReadoutPolicy(res.n_motor, 3)
    runner = MarioRunner(FakeMario(die_at=12), frame_skip=4, noop_max=5, seed=1)
    rows, frames = evaluate(runner, res, policy, episodes=3, greedy=False, size=(8, 8))
    assert len(rows) == 3 and frames
    best = max(rows, key=lambda r: (r["max_x"], r["reward"]))
    s = summarize(rows)
    assert s["max_x_best"] == best["max_x"] and s["death_rate"] == 1.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_is_h264_yuv420p(tmp_path):
    frames = [np.full((240, 256, 3), i * 10, np.uint8) for i in range(20)]
    path = tmp_path / "v.mp4"
    assert save_video(frames, path, fps=60, scale=2) == "h264"
    probe = shutil.which("ffprobe")
    if probe:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name,pix_fmt,width,height", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True).stdout.strip()
        assert out == "h264,512,480,yuv420p"


def test_video_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        save_video([], tmp_path / "x.mp4")


def test_visual_input_modes():
    from meta_brain.mario_rl import VisualInput

    frame_a = np.zeros((240, 256, 3), np.uint8)
    frame_b = np.full((240, 256, 3), 200, np.uint8)
    zero = VisualInput("zero", (8, 8))
    assert not zero(frame_b, None, 3)[0].any()
    clock = VisualInput("clock", (8, 8), seed=1)
    c3a, _ = clock(frame_a, None, 3)
    c3b, _ = clock(frame_b, None, 3)
    c4, _ = clock(frame_a, None, 4)
    np.testing.assert_array_equal(c3a, c3b)  # depends on the step, not on the screen
    assert not np.array_equal(c3a, c4)
    np.testing.assert_array_equal(c3a, VisualInput("clock", (8, 8), seed=1)(None, None, 3)[0])
    real = VisualInput("real", (8, 8))
    assert not np.array_equal(real(frame_a, None, 0)[0], real(frame_b, None, 0)[0])
    with pytest.raises(ValueError, match="vision"):
        VisualInput("blind")


def test_zero_vision_silences_the_reservoir():
    from meta_brain.mario_rl import VisualInput

    res = small_reservoir()
    runners = [MarioRunner(FakeMario(), frame_skip=2, noop_max=2, seed=i) for i in range(2)]
    rollout = BatchedRollout(runners, res, size=(8, 8), vision=VisualInput("zero", (8, 8)))
    policy = ReadoutPolicy(res.n_motor, 3)
    diag = warmup_normalization(rollout, policy, steps=5, n_actions=3)
    assert diag["responsive_fraction"] == 0.0 and diag["mean_abs_activity"] == 0.0


def test_vision_controls_compare(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from dataclasses import replace

    from meta_brain.mario_rl import MarioSettings, compare_vision_conditions, run_experiment

    structure, weights = small_graph()
    base = MarioSettings(n_envs=2, frame_skip=2, micro_steps=1, size=(8, 8), noop_max=3,
                         stuck_steps=20, max_steps=40, eval_episodes=2, seed=0)
    results = {}
    for cond in ("real", "zero", "clock"):
        results[cond] = run_experiment(structure, weights, np.arange(0, 60), np.arange(240, 300),
                                       lambda: FakeMario(die_at=60), tmp_path / cond,
                                       replace(base, vision=cond), PPOConfig(updates=2, rollout=16, warmup_steps=4),
                                       n_actions=3, log=None, prefix="t")
        assert results[cond]["vision"] == cond
        assert len(results[cond]["post_eval_sampled"]["episodes"]) == 2
    assert results["zero"]["motor_readout_diagnostics"]["responsive_fraction"] == 0.0
    summary = compare_vision_conditions(results, tmp_path)
    assert [r["condition"] for r in summary["conditions"]] == ["real", "zero", "clock"]
    assert {"real_minus_zero_mean_x", "real_minus_clock_mean_x"} <= set(summary)
    assert (tmp_path / "mario_vision_controls.png").stat().st_size > 0
    assert (tmp_path / "mario_vision_controls.json").stat().st_size > 0


def test_run_experiment_writes_all_artifacts(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from meta_brain.mario_rl import MarioSettings, run_experiment

    structure, weights = small_graph()
    settings = MarioSettings(n_envs=2, frame_skip=2, micro_steps=1, size=(8, 8), noop_max=3,
                             stuck_steps=20, max_steps=40, eval_episodes=2, seed=0)
    results = run_experiment(structure, weights, np.arange(0, 60), np.arange(240, 300),
                             lambda: FakeMario(die_at=60), tmp_path, settings,
                             PPOConfig(updates=3, rollout=16, warmup_steps=4), {"neurons": 300},
                             n_actions=3, log=None, prefix="t")
    for suffix in ("pre.mp4", "post.mp4", "post_sampled.mp4", "curves.png", "training.csv", "episodes.csv", "readout.pt",
                   "results.json"):
        assert (tmp_path / f"t_{suffix}").stat().st_size > 0
    assert results["training"]["agent_steps"] == 3 * 16 * 2
    assert results["claim_status"] == "RUN_COMPLETED_BUT_SUCCESS_NOT_DEMONSTRATED"
    assert len(results["post_eval"]["episodes"]) == 2
