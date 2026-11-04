"""Real NES smoke test; deliberately small, does not claim full-graph CUDA success."""
import importlib.util

import numpy as np
import pytest
import torch

from meta_brain.mario_rl import make_mario_env
from meta_brain.mario_trainable import FullBrainConfig, TrainableBrain, run_full_brain
from test_mario_rl import small_graph


@pytest.mark.skipif(importlib.util.find_spec("gym_super_mario_bros") is None, reason="emulator not installed")
def test_real_emulator_train_checkpoint_resume_and_video(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = FullBrainConfig(n_envs=2, rollout=8, bptt=4, micro_steps=2, updates=1, epochs=1,
                              max_hours=0, max_steps=12, noop_max=0, eval_every=1, checkpoint_every=1,
                              validation_episodes=1, final_episodes=1, latent=8, hidden=16)
        graph, weights = small_graph()
        env, actions = make_mario_env()
        obs, _ = env.reset(seed=123)
        assert obs.shape == (240, 256, 3)
        assert len(env.step(1)) == 5
        env.close()
        model = TrainableBrain(graph, weights, np.arange(60), np.arange(240, 300),
                               np.arange(300) % 8, len(actions), cfg)
        result = run_full_brain(model, lambda: make_mario_env()[0], cfg, tmp_path, "synthetic-smoke", log=lambda _: None)
        assert result["completed_updates"] == 1
        cfg.updates = 2
        result = run_full_brain(model, lambda: make_mario_env()[0], cfg, tmp_path, "synthetic-smoke", log=lambda _: None)
        assert result["completed_updates"] == 2
        assert (tmp_path / "post_greedy.mp4").stat().st_size > 0
        assert result["post_eval"]["episodes"][0]["truncated"]
        assert not result["post_eval"]["episodes"][0]["died"]
    finally:
        torch.set_num_threads(previous_threads)
