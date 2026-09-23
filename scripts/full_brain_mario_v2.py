"""Full-connectome Mario experiment for Meta-Brain V2.

This experiment is deliberately conservative about its claim:

* it refuses to run when the loaded graph is not the full MaleCNS graph;
* the connectome is a fixed, sparse recurrent reservoir;
* only the motor readout is trained with reward-modulated policy gradient;
* a run is called a success only when the post-training evaluation reaches
  the level without a death in the requested evaluation episodes.

The script is intended to be called from the Colab notebook
``notebooks/colab_full_brain_mario_v2.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_data_files(data_dir: Path) -> Tuple[Path, Path]:
    meta = sorted(data_dir.glob("*meta*.feather"))
    edges = sorted(data_dir.glob("*simple*edgelist*.feather"))
    if not meta:
        meta = sorted(data_dir.glob("*.feather"))
    if not edges:
        raise FileNotFoundError(
            f"No simple edgelist found in {data_dir}. "
            "Expected *simple*edgelist*.feather."
        )
    if not meta:
        raise FileNotFoundError(
            f"No neuron metadata found in {data_dir}. Expected *meta*.feather."
        )
    return meta[0], edges[0]


def load_full_connectome(data_dir: Path, min_neurons: int, min_edges: int) -> Dict[str, Any]:
    meta_path, edge_path = find_data_files(data_dir)
    print(f"Loading metadata: {meta_path}")
    meta = pd.read_feather(meta_path)
    print(f"Loading full edge list: {edge_path}")
    edge_df = pd.read_feather(edge_path)

    required_meta = {"bodyId", "superclass"}
    required_edge = {"bodyId_pre", "bodyId_post", "weight"}
    if not required_meta.issubset(meta.columns):
        raise ValueError(f"Metadata is missing columns: {required_meta - set(meta.columns)}")
    if not required_edge.issubset(edge_df.columns):
        raise ValueError(f"Edges are missing columns: {required_edge - set(edge_df.columns)}")

    neuron_ids = meta["bodyId"].astype(np.int64).to_numpy()
    if len(np.unique(neuron_ids)) != len(neuron_ids):
        raise ValueError("Neuron metadata contains duplicate bodyId values.")
    if len(neuron_ids) < min_neurons or len(edge_df) < min_edges:
        raise RuntimeError(
            "Full-brain guard failed: loaded data has "
            f"{len(neuron_ids):,} neurons and {len(edge_df):,} edges, but the "
            f"experiment requires at least {min_neurons:,} neurons and {min_edges:,} edges."
        )

    id_to_idx = {int(nid): i for i, nid in enumerate(neuron_ids.tolist())}
    pre_raw = edge_df["bodyId_pre"].astype(np.int64).to_numpy()
    post_raw = edge_df["bodyId_post"].astype(np.int64).to_numpy()
    keep = np.fromiter(
        ((int(a) in id_to_idx and int(b) in id_to_idx) for a, b in zip(pre_raw, post_raw)),
        dtype=bool,
        count=len(edge_df),
    )
    pre = np.fromiter((id_to_idx[int(x)] for x in pre_raw[keep]), dtype=np.int64)
    post = np.fromiter((id_to_idx[int(x)] for x in post_raw[keep]), dtype=np.int64)
    weights = edge_df.loc[keep, "weight"].astype(np.float32).to_numpy()
    weights = np.log1p(np.maximum(weights, 0.0)).astype(np.float32)

    nt = meta.get("predictedNt", pd.Series([""] * len(meta))).fillna("").astype(str).str.lower()
    signed = np.ones(len(meta), dtype=np.float32)
    signed[nt.str.contains("gaba|glutamate|glu", regex=True).to_numpy()] = -1.0
    # Normalize incoming connection energy so the full graph is numerically stable.
    indeg_sq = np.zeros(len(meta), dtype=np.float32)
    np.add.at(indeg_sq, post, weights * weights)
    weights = weights / np.sqrt(np.maximum(indeg_sq[post], 1e-6))
    weights *= signed[pre]

    superclasses = meta["superclass"].fillna("").astype(str).str.lower()
    sensory_mask = superclasses.isin(
        {"ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending", "olsn"}
    ).to_numpy()
    motor_mask = superclasses.isin(
        {"descending_neuron", "vnc_motor", "cb_motor", "dn", "motor"}
    ).to_numpy()
    sensory = np.flatnonzero(sensory_mask).astype(np.int64)
    motor = np.flatnonzero(motor_mask).astype(np.int64)
    if len(sensory) == 0 or len(motor) == 0:
        raise ValueError("Could not identify sensory and motor neurons from metadata.")

    print(
        f"FULL CONNECTOME VERIFIED: neurons={len(meta):,}, edges={len(pre):,}, "
        f"sensory={len(sensory):,}, motor={len(motor):,}"
    )
    return {
        "meta_path": str(meta_path),
        "edge_path": str(edge_path),
        "n": int(len(meta)),
        "e": int(len(pre)),
        "pre": pre,
        "post": post,
        "weights": weights,
        "sensory": sensory,
        "motor": motor,
    }


class FullConnectomeReservoir(nn.Module):
    """A fixed sparse full-connectome reservoir plus trainable motor readout."""

    def __init__(self, graph: Dict[str, Any], input_dim: int, n_actions: int, seed: int):
        super().__init__()
        self.n = graph["n"]
        self.pre = torch.as_tensor(graph["pre"], dtype=torch.long)
        self.post = torch.as_tensor(graph["post"], dtype=torch.long)
        self.edge_weight = torch.as_tensor(graph["weights"], dtype=torch.float32)
        self.sensory = torch.as_tensor(graph["sensory"], dtype=torch.long)
        self.motor = torch.as_tensor(graph["motor"], dtype=torch.long)
        self.n_actions = n_actions
        self.input_gain = 1.0
        self.decay = 0.90
        self.syn_scale = 1.6
        self.device_name = "cpu"

        gen = torch.Generator(device="cpu").manual_seed(seed)
        # This projection is fixed. It only turns a compact visual code into
        # currents on all identified sensory neurons; it is not trained.
        proj = torch.randn(input_dim, len(self.sensory), generator=gen) / math.sqrt(input_dim)
        self.register_buffer("input_projection", proj)
        self.W_out = nn.Parameter(torch.randn(n_actions, len(self.motor), generator=gen) * 0.03)
        self.b_out = nn.Parameter(torch.zeros(n_actions))

    def to_device(self, device: torch.device) -> "FullConnectomeReservoir":
        self.to(device)
        self.pre = self.pre.to(device)
        self.post = self.post.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.sensory = self.sensory.to(device)
        self.motor = self.motor.to(device)
        self.device_name = str(device)
        return self

    def step(self, visual_code: torch.Tensor, state: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance one game decision; gradients never traverse the 6M-edge graph."""
        if visual_code.ndim == 1:
            visual_code = visual_code.unsqueeze(0)
        with torch.no_grad():
            batch = visual_code.shape[0]
            if state is None:
                state = torch.zeros(batch, self.n, device=visual_code.device)
            drive = torch.zeros(batch, self.n, device=visual_code.device)
            sensory_drive = visual_code @ self.input_projection
            drive.index_add_(1, self.sensory, sensory_drive * self.input_gain)
            msg = torch.zeros_like(state)
            source = state[:, self.pre] * self.edge_weight.unsqueeze(0)
            msg.index_add_(1, self.post, source)
            state = self.decay * state + (1.0 - self.decay) * torch.tanh(
                self.syn_scale * msg + drive
            )
        motor_activity = state[:, self.motor]
        logits = motor_activity @ self.W_out.t() + self.b_out
        return logits, state


def make_env():
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import RIGHT_ONLY
    except Exception as exc:
        raise RuntimeError(
            "Mario environment unavailable. In Colab run the installation cell "
            "and restart the runtime if nes_py imports fail."
        ) from exc
    # The maintained package exposes the NES Super Mario Bros levels through
    # this Gym-style API. RIGHT_ONLY keeps the first experiment tractable.
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    return JoypadSpace(env, RIGHT_ONLY), len(RIGHT_ONLY)


def reset_env(env, seed: int | None = None):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action: int):
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), dict(info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), dict(info)


def compact_visual_code(frame: np.ndarray, previous: np.ndarray | None, side: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    img = Image.fromarray(arr.astype(np.uint8)).resize((side, side))
    current = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    delta = current if previous is None else current - previous
    code = np.concatenate([current.reshape(-1), delta.reshape(-1)]).astype(np.float32)
    return code, current


def info_success(info: Dict[str, Any], done: bool) -> bool:
    return bool(info.get("flag_get", False) or info.get("level_complete", False))


def rollout(
    env,
    model: FullConnectomeReservoir,
    device: torch.device,
    max_steps: int,
    greedy: bool,
    seed: int,
    record: bool = False,
) -> Dict[str, Any]:
    obs = reset_env(env, seed=seed)
    state = None
    prev = None
    frames: List[np.ndarray] = []
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    total_reward = 0.0
    info: Dict[str, Any] = {}
    success = False
    for _ in range(max_steps):
        if record:
            frames.append(np.asarray(obs).copy())
        code, prev = compact_visual_code(obs, prev)
        visual = torch.from_numpy(code).to(device)
        logits, state = model.step(visual, state)
        dist = torch.distributions.Categorical(logits=logits)
        action = int(logits.argmax(-1).item()) if greedy else int(dist.sample().item())
        if not greedy:
            log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
            entropies.append(dist.entropy())
        obs, reward, done, info = step_env(env, action)
        rewards.append(reward)
        total_reward += reward
        success = info_success(info, done)
        if done:
            break
    died = bool(done and not success)
    return {
        "reward": float(total_reward),
        "steps": len(rewards),
        "success": success,
        "died": died,
        "x_pos": float(info.get("x_pos", np.nan)),
        "flag_get": bool(info.get("flag_get", False)),
        "log_probs": log_probs,
        "entropies": entropies,
        "rewards": rewards,
        "frames": frames,
    }


def discounted_returns(rewards: Sequence[float], gamma: float) -> torch.Tensor:
    out = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        out.append(running)
    return torch.tensor(list(reversed(out)), dtype=torch.float32)


def train_policy(
    env,
    model: FullConnectomeReservoir,
    device: torch.device,
    iterations: int,
    max_steps: int,
    gamma: float,
    lr: float,
    seed: int,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam([model.W_out, model.b_out], lr=lr)
    history: List[Dict[str, float]] = []
    for it in range(iterations):
        episode = rollout(env, model, device, max_steps, greedy=False, seed=seed + it)
        if not episode["log_probs"]:
            raise RuntimeError("Rollout produced no policy-gradient samples.")
        returns = discounted_returns(episode["rewards"], gamma).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        lp = torch.stack(episode["log_probs"])
        ent = torch.stack(episode["entropies"])
        loss = -(lp * returns).mean() - 0.01 * ent.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.W_out, model.b_out], 1.0)
        optimizer.step()
        row = {
            "iteration": float(it),
            "reward": float(episode["reward"]),
            "loss": float(loss.detach().cpu()),
            "steps": float(episode["steps"]),
            "success": float(episode["success"]),
            "x_pos": float(episode["x_pos"]),
        }
        history.append(row)
        if it == 0 or (it + 1) % 10 == 0:
            print(
                f"iter={it:04d} reward={row['reward']:+.1f} "
                f"steps={int(row['steps'])} x={row['x_pos']:.0f}"
            )
    return history


def evaluate(env, model, device, episodes: int, max_steps: int, seed: int, record: bool) -> Dict[str, Any]:
    rows = []
    all_frames: List[np.ndarray] = []
    for i in range(episodes):
        row = rollout(env, model, device, max_steps, greedy=True, seed=seed + i, record=record)
        rows.append({k: row[k] for k in ("reward", "steps", "success", "died", "x_pos", "flag_get")})
        if record and not all_frames:
            all_frames = row["frames"]
    return {
        "episodes": rows,
        "reward_mean": float(np.mean([r["reward"] for r in rows])),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "death_rate": float(np.mean([r["died"] for r in rows])),
        "all_success_no_death": bool(all(r["success"] and not r["died"] for r in rows)),
        "frames": all_frames,
    }


def save_video(frames: Sequence[np.ndarray], path: Path, fps: int = 30) -> None:
    import cv2

    if not frames:
        raise ValueError("No frames were recorded for video.")
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            writer.write(cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_artifacts(out_dir: Path, history, pre, post, model, graph, config) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "full_brain_mario_v2_training.csv", index=False)
    with torch.no_grad():
        torch.save(
            {
                "W_out": model.W_out.detach().cpu(),
                "b_out": model.b_out.detach().cpu(),
                "n_neurons": graph["n"],
                "n_edges": graph["e"],
                "motor_count": len(graph["motor"]),
                "config": config,
            },
            out_dir / "full_brain_mario_v2_readout.pt",
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if history:
        h = hist_df
        axes[0].plot(h["iteration"], h["reward"], alpha=0.35, label="episode")
        axes[0].plot(h["iteration"], h["reward"].rolling(20, min_periods=1).mean(), label="rolling mean")
        axes[1].plot(h["iteration"], h["x_pos"], color="darkgreen")
        axes[2].plot(h["iteration"], h["loss"], color="darkorange")
    axes[0].set_title("Mario reward during RL")
    axes[1].set_title("Mario x progress")
    axes[2].set_title("Readout policy loss")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "full_brain_mario_v2_curves.png", dpi=160)
    plt.close(fig)

    serial = {
        "experiment": "Meta-Brain full-connectome Mario V2",
        "full_connectome_guard": True,
        "graph": {
            "neurons": graph["n"],
            "edges": graph["e"],
            "sensory": len(graph["sensory"]),
            "motor": len(graph["motor"]),
            "metadata_file": graph["meta_path"],
            "edge_file": graph["edge_path"],
        },
        "config": config,
        "pre_eval": {k: v for k, v in pre.items() if k != "frames"},
        "post_eval": {k: v for k, v in post.items() if k != "frames"},
        "claim_status": "DEMONSTRATED_SUCCESS" if post["all_success_no_death"] else "RUN_COMPLETED_BUT_SUCCESS_NOT_DEMONSTRATED",
    }
    (out_dir / "full_brain_mario_v2_results.json").write_text(
        json.dumps(serial, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
        encoding="utf-8",
    )
    if pre["frames"]:
        save_video(pre["frames"], out_dir / "full_brain_mario_v2_pre.mp4")
    if post["frames"]:
        save_video(post["frames"], out_dir / "full_brain_mario_v2_post.mp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/connectome")
    parser.add_argument("--out-dir", default="results/full_brain_mario_v2")
    parser.add_argument("--iterations", type=int, default=int(os.getenv("MB_MARIO_ITERS", "300")))
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-neurons", type=int, default=100000)
    parser.add_argument("--min-edges", type=int, default=5000000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Full-brain Mario V2 requires CUDA; refusing a misleading CPU run.")
    seed_all(args.seed)
    device = torch.device("cuda")
    started = time.time()
    graph = load_full_connectome(Path(args.data_dir), args.min_neurons, args.min_edges)
    env, n_actions = make_env()
    model = FullConnectomeReservoir(graph, input_dim=512, n_actions=n_actions, seed=args.seed).to_device(device)
    config = {
        "iterations": args.iterations,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "device": str(device),
        "actions": n_actions,
        "observation": "16x16 grayscale frame plus frame difference",
        "trainable": "motor readout only; connectome and visual projection frozen",
    }
    print("Evaluating pre-training policy...")
    pre = evaluate(env, model, device, args.eval_episodes, args.max_steps, args.seed + 10000, record=True)
    print("Training readout with reward-modulated policy gradient...")
    history = train_policy(env, model, device, args.iterations, args.max_steps, 0.99, 2e-3, args.seed + 20000)
    print("Evaluating post-training policy...")
    post = evaluate(env, model, device, args.eval_episodes, args.max_steps, args.seed + 30000, record=True)
    config["runtime_minutes"] = round((time.time() - started) / 60.0, 2)
    save_artifacts(Path(args.out_dir), history, pre, post, model, graph, config)
    print(json.dumps({"pre": {k: pre[k] for k in ("reward_mean", "success_rate", "death_rate")},
                      "post": {k: post[k] for k in ("reward_mean", "success_rate", "death_rate", "all_success_no_death")},
                      "out_dir": args.out_dir}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
