"""Super Mario Bros with a frozen connectome reservoir and a trained motor readout.

Shared by ``scripts/full_brain_mario_v2.py`` and the Mario V2 Colab notebook
(which embeds this file). The scientific setup is unchanged from V2: the
connectome and the visual projection are frozen; only a linear readout of the
motor neurons is trained. What changed is everything around it that kept the
first full run from learning:

* **Frame skip.** One agent step repeats the action for ``frame_skip`` NES
  frames, so a jump can be held long enough to clear a Goomba or a pipe.
* **Episode cut-offs.** An episode ends when Mario makes no progress for
  ``stuck_steps`` agent steps instead of pushing against a pipe until the
  step limit.
* **No-op starts.** A random number of idle frames at reset makes episodes
  differ, so evaluation episodes are not five copies of one trajectory.
* **PPO on the readout.** Batched environments, a value baseline from the
  same motor neurons and clipped updates replace one-episode REINFORCE.
* **Normalized motor features.** Motor activity is standardized with
  statistics from a random-policy warm-up, so the readout sees signals of
  usable scale.
* **Playable videos.** Frames are encoded as H.264 (yuv420p) with ffmpeg; the
  old ``mp4v`` files do not play in browsers.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .sparse_ops import SparseStructure

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def make_mario_env(stage: str = "1-1", actions: str = "right_only"):
    """``gym-super-mario-bros`` stage wrapped in a joypad action space."""
    import gym_super_mario_bros
    from gym_super_mario_bros import actions as action_sets
    from nes_py.wrappers import JoypadSpace

    table = {"right_only": action_sets.RIGHT_ONLY, "simple": action_sets.SIMPLE_MOVEMENT}[actions]
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{stage}-v0"), table)
    return env, ["+".join(a) for a in table]


def _reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step(env, action: int):
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), dict(info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), dict(info)


class MarioRunner:
    """One environment with frame skip, no-op starts, progress cut-off and frame recording."""

    def __init__(self, env, frame_skip: int = 4, noop_max: int = 30, stuck_steps: int = 60,
                 max_steps: int = 1500, seed: int = 0):
        self.env = env
        self.frame_skip = frame_skip
        self.noop_max = noop_max
        self.stuck_steps = stuck_steps
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.record = False
        self.frames: List[np.ndarray] = []
        self.info: Dict = {}
        self.steps = self.since_progress = 0
        self.best_x = 0.0
        self.episode_reward = 0.0

    def _capture(self, obs) -> None:
        if self.record:
            self.frames.append(np.asarray(obs, dtype=np.uint8).copy())

    def reset(self, record: bool = False):
        self.record = record
        self.frames = []
        obs = _reset(self.env)
        self._capture(obs)
        info: Dict = {}
        for _ in range(int(self.rng.integers(0, self.noop_max + 1))):
            obs, _, done, info = _step(self.env, 0)
            self._capture(obs)
            if done:
                obs, info = _reset(self.env), {}
        self.steps = self.since_progress = 0
        self.best_x = float(info.get("x_pos", 0))
        self.episode_reward = 0.0
        self.info = info
        return obs

    def step(self, action: int):
        total, done, info = 0.0, False, {}
        for _ in range(self.frame_skip):
            obs, reward, done, info = _step(self.env, action)
            total += reward
            self._capture(obs)
            if done:
                break
        self.steps += 1
        self.episode_reward += total
        x = float(info.get("x_pos", self.best_x))
        if x > self.best_x + 1:
            self.best_x, self.since_progress = x, 0
        else:
            self.since_progress += 1
        flag = bool(info.get("flag_get", False))
        truncated = not done and (self.since_progress >= self.stuck_steps or self.steps >= self.max_steps)
        info = dict(info, flag_get=flag, died=bool(done and not flag), truncated=truncated)
        self.info = info
        return obs, total, done or truncated, info

    def episode_summary(self) -> Dict:
        i = self.info
        return {"reward": round(self.episode_reward, 2), "steps": self.steps,
                "x_pos": float(i.get("x_pos", self.best_x)), "max_x": self.best_x,
                "flag_get": bool(i.get("flag_get", False)), "died": bool(i.get("died", False)),
                "truncated": bool(i.get("truncated", False))}


# --------------------------------------------------------------------------- #
# Vision and the connectome reservoir
# --------------------------------------------------------------------------- #


def encode_frame(frame, previous: Optional[np.ndarray] = None, size: Tuple[int, int] = (32, 32),
                 crop_top: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Grayscale playfield (HUD cropped), downsampled, plus the change since the last frame."""
    from PIL import Image

    a = np.asarray(frame)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    a = a[crop_top:]
    small = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((size[1], size[0]), Image.BILINEAR),
                       dtype=np.float32) / 127.5 - 1.0
    delta = small if previous is None else small - previous
    return np.concatenate([small.ravel(), delta.ravel()]).astype(np.float32), small


def code_dim(size: Tuple[int, int] = (32, 32)) -> int:
    return 2 * size[0] * size[1]


VISION_MODES = ("real", "zero", "clock")


class VisualInput:
    """What the sensory neurons receive; the blind modes are controls for "does the readout use vision?".

    * ``real``: the game frame (:func:`encode_frame`).
    * ``zero``: nothing. With no drive the reservoir stays silent, so the
      readout can only learn a fixed action distribution.
    * ``clock``: a fixed random code for each agent step since reset, the same
      in every episode and unrelated to the screen. The reservoir is as active
      as with real input and carries elapsed time, so the readout can learn an
      open-loop rhythm of jumps but cannot react to what is on screen.
    """

    def __init__(self, mode: str = "real", size: Tuple[int, int] = (32, 32), seed: int = 0):
        if mode not in VISION_MODES:
            raise ValueError(f"vision must be one of {VISION_MODES}, got {mode!r}")
        self.mode, self.size, self.seed = mode, size, seed
        self.dim = code_dim(size)
        self._clock: Dict[int, np.ndarray] = {}

    def __call__(self, obs, previous: Optional[np.ndarray], step: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.mode == "real":
            return encode_frame(obs, previous, self.size)
        if self.mode == "zero":
            return np.zeros(self.dim, np.float32), None
        code = self._clock.get(step)
        if code is None:
            code = np.random.default_rng([self.seed, step]).uniform(-1, 1, self.dim).astype(np.float32)
            self._clock[step] = code
        return code, None


class ConnectomeReservoir:
    """Frozen recurrent dynamics on the connectome; returns motor-neuron activity.

    ``state <- decay * state + (1 - decay) * tanh(gain * A @ state + drive)``,
    repeated ``micro_steps`` times per agent step, with ``drive`` a fixed
    random projection of the visual code onto the sensory neurons.
    """

    def __init__(self, structure: SparseStructure, weights: torch.Tensor, sensory: Sequence[int],
                 motor: Sequence[int], input_dim: int, micro_steps: int = 2, decay: float = 0.9,
                 gain: float = 1.6, input_gain: float = 1.0, seed: int = 0):
        device = structure.device
        self.n = structure.n
        self.device = device
        self.A = structure.matrix(weights.to(device=device, dtype=torch.float32))
        self.sensory = torch.as_tensor(np.asarray(sensory), dtype=torch.long, device=device)
        self.motor = torch.as_tensor(np.asarray(motor), dtype=torch.long, device=device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.proj = (torch.randn(input_dim, len(self.sensory), generator=gen) / math.sqrt(input_dim)).to(device)
        self.micro_steps, self.decay, self.gain, self.input_gain = micro_steps, decay, gain, input_gain

    @property
    def n_motor(self) -> int:
        return int(self.motor.numel())

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(self.n, batch, device=self.device)

    @torch.no_grad()
    def step(self, codes: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``codes`` is ``(B, input_dim)``, ``state`` is ``(N, B)``; returns ``(B, n_motor)`` features."""
        drive = torch.zeros_like(state)
        drive.index_add_(0, self.sensory, (codes @ self.proj).t() * self.input_gain)
        for _ in range(self.micro_steps):
            msg = torch.sparse.mm(self.A, state)
            state = self.decay * state + (1.0 - self.decay) * torch.tanh(self.gain * msg + drive)
        return state.index_select(0, self.motor).t().contiguous(), state


# --------------------------------------------------------------------------- #
# Readout policy and PPO
# --------------------------------------------------------------------------- #


class ReadoutPolicy(nn.Module):
    """Linear policy and value heads on standardized motor-neuron activity."""

    def __init__(self, n_features: int, n_actions: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.pi = nn.Linear(n_features, n_actions)
        self.v = nn.Linear(n_features, 1)
        with torch.no_grad():
            self.pi.weight.copy_(torch.randn(n_actions, n_features, generator=gen) * 0.01)
            self.pi.bias.zero_()
            self.v.weight.zero_()
            self.v.bias.zero_()

    def fit_normalization(self, features: torch.Tensor) -> Dict[str, float]:
        """Standardize with warm-up statistics; near-constant neurons are scaled, not amplified."""
        std = features.std(dim=0)
        floor = 1e-2 * float(std.mean()) + 1e-6
        self.mean.copy_(features.mean(dim=0))
        self.std.copy_(std.clamp_min(floor))
        return {"motor_features": int(features.shape[1]),
                "responsive_fraction": round(float((std > floor).float().mean()), 4),
                "mean_feature_std": float(std.mean()), "mean_abs_activity": float(features.abs().mean())}

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = (features - self.mean) / self.std
        return self.pi(z), self.v(z).squeeze(-1)


@dataclass
class PPOConfig:
    updates: int = 300
    rollout: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    warmup_steps: int = 64


class BatchedRollout:
    """Steps ``B`` runners through the reservoir in lockstep, resetting finished episodes."""

    def __init__(self, runners: Sequence[MarioRunner], reservoir: ConnectomeReservoir,
                 size: Tuple[int, int] = (32, 32), vision: Optional[VisualInput] = None):
        self.runners, self.reservoir, self.size = list(runners), reservoir, size
        self.vision = vision or VisualInput("real", size)
        self.prev: List[Optional[np.ndarray]] = [None] * len(self.runners)
        self.state = reservoir.initial_state(len(self.runners))
        self.episodes: List[Dict] = []
        codes = [self._encode(i, r.reset()) for i, r in enumerate(self.runners)]
        self.features = self._advance(codes)

    def _encode(self, i: int, obs) -> np.ndarray:
        code, self.prev[i] = self.vision(obs, self.prev[i], self.runners[i].steps)
        return code

    def _advance(self, codes: List[np.ndarray]) -> torch.Tensor:
        batch = torch.as_tensor(np.stack(codes), device=self.reservoir.device)
        features, self.state = self.reservoir.step(batch, self.state)
        return features

    def step(self, actions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.zeros(len(self.runners), np.float32)
        dones = np.zeros(len(self.runners), np.float32)
        codes = []
        for i, (runner, a) in enumerate(zip(self.runners, actions)):
            obs, reward, done, _ = runner.step(int(a))
            rewards[i], dones[i] = reward, float(done)
            if done:
                self.episodes.append(runner.episode_summary())
                obs = runner.reset()
                self.prev[i] = None
                self.state[:, i] = 0.0
            codes.append(self._encode(i, obs))
        self.features = self._advance(codes)
        return rewards, dones


@torch.no_grad()
def warmup_normalization(rollout: BatchedRollout, policy: ReadoutPolicy, steps: int, n_actions: int,
                         seed: int = 0) -> Dict[str, float]:
    """Fit the feature normalization on a random-action rollout."""
    rng = np.random.default_rng(seed)
    feats = [rollout.features]
    for _ in range(steps):
        rollout.step(rng.integers(0, n_actions, len(rollout.runners)))
        feats.append(rollout.features)
    rollout.episodes.clear()
    return policy.fit_normalization(torch.cat(feats).float())


def train_ppo(rollout: BatchedRollout, policy: ReadoutPolicy, cfg: PPOConfig,
              log: Optional[Callable[[str], None]] = print, log_every: int = 10) -> List[Dict]:
    """PPO on the readout heads; the reservoir is frozen, so stored features are reused for every epoch."""
    device = rollout.reservoir.device
    params = list(policy.pi.parameters()) + list(policy.v.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    n_envs = len(rollout.runners)
    history: List[Dict] = []
    for update in range(cfg.updates):
        feats, acts, logps, vals = [], [], [], []
        rews = np.zeros((cfg.rollout, n_envs), np.float32)
        dones = np.zeros((cfg.rollout, n_envs), np.float32)
        first_episode = len(rollout.episodes)
        for t in range(cfg.rollout):
            f = rollout.features
            with torch.no_grad():
                logits, value = policy(f)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            feats.append(f)
            acts.append(a)
            logps.append(dist.log_prob(a))
            vals.append(value)
            rews[t], dones[t] = rollout.step(a.tolist())
        with torch.no_grad():
            last_value = policy(rollout.features)[1].cpu().numpy()

        values = torch.stack(vals).cpu().numpy()
        adv = np.zeros_like(rews)
        gae = np.zeros(n_envs, np.float32)
        for t in reversed(range(cfg.rollout)):
            next_value = last_value if t == cfg.rollout - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rews[t] * cfg.reward_scale + cfg.gamma * next_value * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.lam * nonterminal * gae
            adv[t] = gae
        returns = adv + values

        F_ = torch.cat(feats)
        A_ = torch.cat(acts)
        LP_ = torch.cat(logps)
        ADV = torch.as_tensor(adv.reshape(-1), device=device)
        RET = torch.as_tensor(returns.reshape(-1), device=device)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(A_)
        mb = max(1, n // cfg.minibatches)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        count = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                logits, value = policy(F_[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A_[idx]) - LP_[idx])
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ADV[idx]).mean()
                vloss = (value - RET[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.value_coef * vloss - cfg.entropy * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                stats["policy_loss"] += float(pg.detach())
                stats["value_loss"] += float(vloss.detach())
                stats["entropy"] += float(ent.detach())
                stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                count += 1
        finished = rollout.episodes[first_episode:]
        row = {"update": update, "agent_steps": (update + 1) * cfg.rollout * n_envs,
               "episodes": len(finished),
               "mean_reward": float(np.mean([e["reward"] for e in finished])) if finished else float("nan"),
               "mean_max_x": float(np.mean([e["max_x"] for e in finished])) if finished else float("nan"),
               "best_max_x": float(max([e["max_x"] for e in finished], default=float("nan"))),
               "flags": int(sum(e["flag_get"] for e in finished)),
               **{k: v / max(count, 1) for k, v in stats.items()}}
        history.append(row)
        if log and (update == 0 or (update + 1) % log_every == 0):
            log(f"update {update + 1:4d}/{cfg.updates} steps={row['agent_steps']:,} "
                f"episodes={row['episodes']:3d} mean_x={row['mean_max_x']:.0f} best_x={row['best_max_x']:.0f} "
                f"flags={row['flags']} entropy={row['entropy']:.2f}")
    return history


# --------------------------------------------------------------------------- #
# Evaluation and video
# --------------------------------------------------------------------------- #


def episode_key(row: Dict) -> Tuple:
    """Best episode: a clean flag first, then the furthest progress, then reward."""
    return (int(row["flag_get"] and not row["died"]), row["max_x"], row["reward"])


@torch.no_grad()
def evaluate(runner: MarioRunner, reservoir: ConnectomeReservoir, policy: ReadoutPolicy,
             episodes: int, greedy: bool = True, record: bool = True,
             size: Tuple[int, int] = (32, 32), seed: int = 0,
             vision: Optional[VisualInput] = None) -> Tuple[List[Dict], List[np.ndarray]]:
    """Run ``episodes`` episodes; returns per-episode rows and the frames of the best one."""
    vision = vision or VisualInput("real", size)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows, best_frames, best = [], [], None
    for _ in range(episodes):
        obs = runner.reset(record=record)
        prev, state, done = None, reservoir.initial_state(1), False
        while not done:
            code, prev = vision(obs, prev, runner.steps)
            features, state = reservoir.step(torch.as_tensor(code[None], device=reservoir.device), state)
            logits, _ = policy(features)
            if greedy:
                action = int(logits.argmax(-1))
            else:
                action = int(torch.multinomial(torch.softmax(logits.cpu(), -1), 1, generator=gen))
            obs, _, done, _ = runner.step(action)
        row = runner.episode_summary()
        rows.append(row)
        if record and (best is None or episode_key(row) > episode_key(best)):
            best, best_frames = row, runner.frames
    return rows, best_frames


def summarize(rows: List[Dict]) -> Dict:
    return {"episodes": rows,
            "reward_mean": float(np.mean([r["reward"] for r in rows])),
            "max_x_mean": float(np.mean([r["max_x"] for r in rows])),
            "max_x_best": float(max(r["max_x"] for r in rows)),
            "success_rate": float(np.mean([r["flag_get"] and not r["died"] for r in rows])),
            "death_rate": float(np.mean([r["died"] for r in rows])),
            "all_success_no_death": bool(all(r["flag_get"] and not r["died"] for r in rows))}


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: int = 60, scale: int = 2) -> str:
    """Write an H.264/yuv420p MP4 that browsers and Colab can play; returns the codec used.

    Falls back to OpenCV's ``mp4v`` (which most browsers cannot play) only when
    no ``ffmpeg`` binary is available.
    """
    if not frames:
        raise ValueError("no frames to write")
    path = Path(path)
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-vf", f"scale={w * scale}:{h * scale}:flags=neighbor",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                a = np.asarray(frame, dtype=np.uint8)
                if a.ndim == 2:
                    a = np.repeat(a[..., None], 3, axis=-1)
                proc.stdin.write(np.ascontiguousarray(a[..., :3]).tobytes())
        finally:
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[:300]}")
        return "h264"
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * scale, h * scale))
    try:
        for frame in frames:
            a = np.asarray(frame, dtype=np.uint8)
            if a.ndim == 2:
                a = np.repeat(a[..., None], 3, axis=-1)
            a = cv2.resize(a[..., :3], (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            writer.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return "mp4v"


def config_dict(cfg: PPOConfig) -> Dict:
    return asdict(cfg)


# --------------------------------------------------------------------------- #
# Whole experiment
# --------------------------------------------------------------------------- #


@dataclass
class MarioSettings:
    n_envs: int = 8
    frame_skip: int = 4
    micro_steps: int = 2
    size: Tuple[int, int] = (32, 32)
    noop_max: int = 30
    stuck_steps: int = 60
    max_steps: int = 1500
    eval_episodes: int = 10
    video_fps: int = 60
    seed: int = 123
    vision: str = "real"  # "real" | "zero" | "clock", see VisualInput


def _plot_history(history: List[Dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [h["agent_steps"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    mean_x = np.array([h["mean_max_x"] for h in history], dtype=float)
    best_x = np.array([h["best_max_x"] for h in history], dtype=float)
    axes[0].plot(steps, mean_x, alpha=0.4, label="mean max x per update")
    if len(mean_x) >= 5 and np.isfinite(mean_x).any():
        filled = np.where(np.isfinite(mean_x), mean_x, np.nanmean(mean_x))
        axes[0].plot(steps[4:], np.convolve(filled, np.ones(5) / 5, mode="valid"), label="5-update average")
    axes[0].plot(steps, best_x, ".", ms=3, alpha=0.5, label="best episode")
    axes[0].set_title("Progress in World 1-1 (x position)")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [h["mean_reward"] for h in history])
    axes[1].set_title("Mean episode reward")
    axes[2].plot(steps, [h["entropy"] for h in history], label="policy entropy")
    axes[2].plot(steps, [h["clip_frac"] for h in history], label="PPO clip fraction")
    axes[2].set_title("Readout optimization")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("agent steps")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(structure: SparseStructure, weights: torch.Tensor, sensory: Sequence[int],
                   motor: Sequence[int], env_factory: Callable[[], object], out_dir: str | Path,
                   settings: Optional[MarioSettings] = None, ppo: Optional[PPOConfig] = None,
                   graph_info: Optional[Dict] = None, n_actions: int = 5,
                   log: Optional[Callable[[str], None]] = print, prefix: str = "full_brain_mario_v2") -> Dict:
    """Pre-training evaluation, PPO on the readout, post-training evaluation and all artifacts."""
    import json
    import time

    import pandas as pd

    settings = settings or MarioSettings()
    ppo = ppo or PPOConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(settings.seed)
    started = time.time()
    reservoir = ConnectomeReservoir(structure, weights, sensory, motor, code_dim(settings.size),
                                    micro_steps=settings.micro_steps, seed=settings.seed)

    def runner(i: int) -> MarioRunner:
        return MarioRunner(env_factory(), settings.frame_skip, settings.noop_max, settings.stuck_steps,
                           settings.max_steps, seed=settings.seed + i)

    vision = VisualInput(settings.vision, settings.size, settings.seed)
    rollout = BatchedRollout([runner(i) for i in range(settings.n_envs)], reservoir, settings.size, vision)
    policy = ReadoutPolicy(reservoir.n_motor, n_actions, seed=settings.seed).to(reservoir.device)
    diagnostics = warmup_normalization(rollout, policy, ppo.warmup_steps, n_actions, seed=settings.seed)
    if log:
        log(f"motor readout: {diagnostics}")

    eval_runner = runner(10_000)
    pre_rows, pre_frames = evaluate(eval_runner, reservoir, policy, settings.eval_episodes,
                                    size=settings.size, seed=settings.seed, vision=vision)
    pre = summarize(pre_rows)
    if log:
        log(f"pre-training: mean max x={pre['max_x_mean']:.0f} best={pre['max_x_best']:.0f}")

    t_train = time.time()
    history = train_ppo(rollout, policy, ppo, log=log)
    train_minutes = (time.time() - t_train) / 60

    post_rows, post_frames = evaluate(eval_runner, reservoir, policy, settings.eval_episodes,
                                      size=settings.size, seed=settings.seed + 1, vision=vision)
    post = summarize(post_rows)
    # Greedy evaluation turns a blind policy into one fixed action, so every condition is
    # also evaluated with sampled actions; the comparison between conditions uses both.
    sampled_rows, sampled_frames = evaluate(eval_runner, reservoir, policy, settings.eval_episodes,
                                            greedy=False, size=settings.size, seed=settings.seed + 2,
                                            vision=vision)
    post_sampled = summarize(sampled_rows)
    if log:
        log(f"post-training [{settings.vision}]: greedy mean max x={post['max_x_mean']:.0f} "
            f"best={post['max_x_best']:.0f} | sampled mean max x={post_sampled['max_x_mean']:.0f} "
            f"best={post_sampled['max_x_best']:.0f} | flags={sum(r['flag_get'] for r in post_rows)}"
            f"/{len(post_rows)}")

    codec = save_video(pre_frames, out / f"{prefix}_pre.mp4", settings.video_fps)
    save_video(post_frames, out / f"{prefix}_post.mp4", settings.video_fps)
    # The trained policy is stochastic; its best behavior only shows up with sampled actions.
    save_video(sampled_frames, out / f"{prefix}_post_sampled.mp4", settings.video_fps)
    _plot_history(history, out / f"{prefix}_curves.png")
    pd.DataFrame(history).to_csv(out / f"{prefix}_training.csv", index=False)
    pd.DataFrame(rollout.episodes).to_csv(out / f"{prefix}_episodes.csv", index=False)
    torch.save({"policy": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                "settings": asdict(settings), "ppo": asdict(ppo)}, out / f"{prefix}_readout.pt")
    steps = history[-1]["agent_steps"] if history else 0
    results = {
        "experiment": "Meta-Brain full-connectome Mario V2 (PPO readout)",
        "graph": graph_info or {},
        "trainable": "linear readout of motor neurons (policy + value); connectome and visual projection frozen",
        "vision": settings.vision,
        "observation": {"real": f"{settings.size[0]}x{settings.size[1]} grayscale playfield + frame difference",
                        "zero": "none (blind control: zero sensory drive)",
                        "clock": "blind control: fixed random code per step since reset"}[settings.vision],
        "settings": asdict(settings),
        "ppo": asdict(ppo),
        "motor_readout_diagnostics": diagnostics,
        "training": {"agent_steps": steps, "nes_frames": steps * settings.frame_skip,
                     "episodes": len(rollout.episodes), "minutes": round(train_minutes, 2),
                     "agent_steps_per_second": round(steps / max(train_minutes * 60, 1e-9), 1)},
        "pre_eval": pre,
        "post_eval": post,
        "post_eval_sampled": post_sampled,
        "history": history,
        "video_codec": codec,
        "videos": {"pre": f"{prefix}_pre.mp4 (best greedy episode before training)",
                   "post": f"{prefix}_post.mp4 (best greedy episode after training)",
                   "post_sampled": f"{prefix}_post_sampled.mp4 (best sampled-action episode after training)"},
        "runtime_minutes": round((time.time() - started) / 60, 2),
        "claim_status": ("DEMONSTRATED_SUCCESS" if post["all_success_no_death"]
                         else "RUN_COMPLETED_BUT_SUCCESS_NOT_DEMONSTRATED"),
    }
    (out / f"{prefix}_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return results


def compare_vision_conditions(results: Dict[str, Dict], out_dir: str | Path,
                              name: str = "mario_vision_controls") -> Dict:
    """Table, plot and JSON comparing runs that differ only in ``settings.vision``.

    Reads ``post_eval`` (greedy), ``post_eval_sampled`` and ``history`` from
    each :func:`run_experiment` result. This is one seed per condition, so it
    can show a large effect but cannot settle a small one.
    """
    import json

    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond, r in results.items():
        hist = r.get("history", [])
        tail = [h["mean_max_x"] for h in hist[-max(1, len(hist) // 10):] if np.isfinite(h["mean_max_x"])]
        rows.append({
            "condition": cond,
            "pre_greedy_mean_x": r["pre_eval"]["max_x_mean"],
            "post_greedy_mean_x": r["post_eval"]["max_x_mean"],
            "post_greedy_best_x": r["post_eval"]["max_x_best"],
            "post_sampled_mean_x": r["post_eval_sampled"]["max_x_mean"],
            "post_sampled_best_x": r["post_eval_sampled"]["max_x_best"],
            "post_death_rate": r["post_eval"]["death_rate"],
            "train_final_mean_x": float(np.mean(tail)) if tail else float("nan"),
            "flags": int(sum(e["flag_get"] for e in r["post_eval"]["episodes"])),
            "responsive_motor_fraction": r["motor_readout_diagnostics"]["responsive_fraction"],
        })

    def best_of(row):
        return max(row["post_greedy_mean_x"], row["post_sampled_mean_x"])

    summary = {"conditions": rows, "single_seed": True}
    real = next((r for r in rows if r["condition"] == "real"), None)
    if real is not None:
        for row in rows:
            if row["condition"] != "real":
                summary[f"real_minus_{row['condition']}_mean_x"] = round(best_of(real) - best_of(row), 1)
                summary[f"real_minus_{row['condition']}_train_x"] = round(
                    real["train_final_mean_x"] - row["train_final_mean_x"], 1)

    colors = {"real": "#4878d0", "zero": "#d65f5f", "clock": "#ee854a"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for cond, r in results.items():
        hist = r.get("history", [])
        if not hist:
            continue
        steps = np.array([h["agent_steps"] for h in hist])
        x = np.array([h["mean_max_x"] for h in hist], dtype=float)
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        k = min(10, len(x))
        axes[0].plot(steps[k - 1:], np.convolve(x, np.ones(k) / k, mode="valid"),
                     color=colors.get(cond), label=cond)
    axes[0].set_title("Training: mean max x per update (10-update average)")
    axes[0].set_xlabel("agent steps")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    idx = np.arange(len(rows))
    axes[1].bar(idx - 0.2, [r["post_greedy_mean_x"] for r in rows], 0.4, label="greedy",
                color=[colors.get(r["condition"], "gray") for r in rows])
    axes[1].bar(idx + 0.2, [r["post_sampled_mean_x"] for r in rows], 0.4, label="sampled",
                color=[colors.get(r["condition"], "gray") for r in rows], alpha=0.5)
    axes[1].set_xticks(idx, [r["condition"] for r in rows])
    axes[1].set_title("After training: mean max x over evaluation episodes")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary
