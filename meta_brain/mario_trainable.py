"""Full-connectome recurrent PPO, with bounded BPTT and restartable Colab runs.

V2 remains the frozen-reservoir baseline. V3 trains a visual encoder, cell-type
input/recurrent gains, biases and time constants, and nonlinear motor heads.
The signed edges and topology stay fixed; there is no image-to-action bypass.
Resume restores training, not the emulator's exact in-flight trajectories.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .graph import ConnectomeArrays
from .mario_rl import MarioRunner, VisualInput, episode_key, save_video, summarize
from .sparse_ops import SparseStructure, sparse_propagate


@dataclass
class FullBrainConfig:
    seed: int = 123
    vision: str = "real"
    n_envs: int = 4
    rollout: int = 128
    bptt: int = 16
    micro_steps: int = 4
    updates: int = 10_000
    epochs: int = 3
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.02
    value_coef: float = 0.5
    reward_scale: float = 0.1
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    frame_skip: int = 4
    stuck_steps: int = 90
    max_steps: int = 2_000
    noop_max: int = 30
    checkpoint_every: int = 10
    eval_every: int = 50
    validation_episodes: int = 3
    final_episodes: int = 20
    max_hours: float = 3.0
    latent: int = 64
    hidden: int = 128

    def validate(self):
        for key in ("n_envs", "rollout", "bptt", "micro_steps", "updates", "epochs", "frame_skip",
                    "checkpoint_every", "eval_every", "validation_episodes", "final_episodes"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.rollout % self.bptt:
            raise ValueError("rollout must be divisible by bptt")
        if self.vision not in ("real", "zero", "clock"):
            raise ValueError("vision must be real, zero or clock")


def graph_fingerprint(graph: ConnectomeArrays) -> str:
    """Content identity; a same-size but shuffled graph must not load a checkpoint."""
    h = hashlib.sha256()
    for a in (graph.body_id, graph.pre, graph.post, graph.count, graph.nt_code,
              graph.cell_type, graph.superclass):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(memoryview(a).cast("B"))
    h.update(json.dumps([graph.type_names, graph.superclass_names]).encode())
    return h.hexdigest()


def reservoir_weights(graph: ConnectomeArrays) -> np.ndarray:
    w = np.log1p(graph.count).astype(np.float32)
    energy = np.bincount(graph.post, weights=w * w, minlength=graph.n)
    w /= np.sqrt(np.maximum(energy[graph.post], 1e-6)).astype(np.float32)
    return w * graph.sign()[graph.pre]


class TrainableBrain(nn.Module):
    """All visual information must traverse the sparse biological graph.

    Cell-type parameters modulate neuron dynamics rather than creating E-sized
    trainable weights. The N x B activations dominate the BPTT tape, not E x B.
    The fixed latent-to-sensory basis is compact (64 x sensory, not 2048 x sensory).
    It is not claimed to be a biological retinotopic map.
    """
    def __init__(self, structure: SparseStructure, weights, sensory, motor, cell_type,
                 n_actions: int, cfg: FullBrainConfig):
        super().__init__()
        cfg.validate()
        self.structure, self.cfg, self.n_actions = structure, cfg, n_actions
        device = structure.device
        # Constant graph tensors are rebuilt from fingerprinted data on resume.
        for name, a, dtype in (("weights", weights, torch.float32), ("sensory", sensory, torch.long),
                               ("motor", motor, torch.long), ("types", cell_type, torch.long)):
            self.register_buffer(name, torch.as_tensor(a, dtype=dtype, device=device), persistent=False)
        if not len(self.sensory) or not len(self.motor):
            raise ValueError("The full graph needs visual sensory and motor neurons")
        n_types = int(self.types.max()) + 1
        gen = torch.Generator().manual_seed(cfg.seed)
        proj = torch.randn(cfg.latent, len(self.sensory), generator=gen) / math.sqrt(cfg.latent)
        self.register_buffer("projection", proj.to(device), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(32 * 8 * 8, cfg.latent), nn.Tanh())
        # 0 < recurrence < .9; .2 < decay < .95; .25 < input gain < 4.
        self.recurrent = nn.Parameter(torch.full((n_types,), math.log(0.5 / 0.4)))
        self.decay = nn.Parameter(torch.full((n_types,), math.log(0.5 / 0.25)))  # initial .7
        self.input_scale = nn.Parameter(torch.full((n_types,), math.log(0.75 / 3.0)))
        self.bias = nn.Parameter(torch.zeros(n_types))
        self.motor_norm = nn.LayerNorm(len(self.motor), eps=1e-8)
        self.head = nn.Sequential(nn.Linear(len(self.motor), cfg.hidden), nn.Tanh())
        self.pi, self.value = nn.Linear(cfg.hidden, n_actions), nn.Linear(cfg.hidden, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        self.to(device)

    @property
    def device(self):
        return self.weights.device

    def initial_state(self, batch):
        return torch.zeros(self.structure.n, batch, device=self.device)

    def forward(self, images, state):
        code = self.encoder(images)
        if self.cfg.vision == "zero":
            code = code * 0  # blind control; learned intrinsic dynamics may still form a clock
        scale = .25 + 3.75 * torch.sigmoid(self.input_scale[self.types[self.sensory]])
        drive = torch.zeros_like(state).index_add(0, self.sensory, (code @ self.projection).t() * scale[:, None])
        gain = .9 * torch.sigmoid(self.recurrent[self.types])[:, None]
        decay = .2 + .75 * torch.sigmoid(self.decay[self.types])[:, None]
        bias = .1 * torch.tanh(self.bias[self.types])[:, None]
        for _ in range(self.cfg.micro_steps):
            message = sparse_propagate(state, self.weights, self.structure)
            state = decay * state + (1 - decay) * torch.tanh(gain * message + drive + bias)
        motor = state.index_select(0, self.motor).t()
        h = self.head(self.motor_norm(motor * 10))
        return self.pi(h), self.value(h).squeeze(-1), state


class RecurrentCollector:
    def __init__(self, model, factory, cfg, seed_offset=0):
        self.model, self.cfg = model, cfg
        self.runners = [MarioRunner(factory(), cfg.frame_skip, cfg.noop_max, cfg.stuck_steps,
                                    cfg.max_steps, cfg.seed + seed_offset + i) for i in range(cfg.n_envs)]
        self.vision = VisualInput(cfg.vision, seed=cfg.seed)
        self.prev = [None] * cfg.n_envs
        self.images = np.stack([self.encode(i, r.reset()) for i, r in enumerate(self.runners)])
        self.state = model.initial_state(cfg.n_envs)
        self.reset_mask = np.zeros(cfg.n_envs, np.float32)

    def encode(self, i, obs):
        code, self.prev[i] = self.vision(obs, self.prev[i], self.runners[i].steps)
        return code.reshape(2, 32, 32)

    def close(self):
        for r in self.runners:
            r.env.close()

    @torch.no_grad()
    def collect(self):
        cfg, device = self.cfg, self.model.device
        data = {k: [] for k in ("images", "resets", "actions", "logp", "values", "rewards", "dones", "bootstrap")}
        boundaries, episodes = [], []
        for t in range(cfg.rollout):
            if t % cfg.bptt == 0:
                boundaries.append(self.state.detach().clone())
            images = torch.as_tensor(self.images, device=device)
            masks = torch.as_tensor(self.reset_mask, device=device)
            state = self.state * (1 - masks)[None, :]
            logits, values, next_state = self.model(images, state)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            next_images, rewards, dones, terminated = [], [], [], []
            terminal_images, cutoffs = [], []
            for i, (runner, action) in enumerate(zip(self.runners, actions.tolist())):
                obs, reward, done, info = runner.step(action)
                encoded = self.encode(i, obs)
                terminal_images.append(encoded)
                rewards.append(reward * cfg.reward_scale)
                dones.append(done)
                terminated.append(bool(info.get("terminated", done and not info["truncated"])))
                if done:
                    episodes.append(runner.episode_summary())
                    self.prev[i] = None
                    encoded = self.encode(i, runner.reset())
                next_images.append(encoded)
                cutoffs.append(bool(done and not terminated[-1]))
            # Reuse the next step's value for continuing transitions; evaluate only
            # time-limit/stuck terminal observations separately (never the reset frame).
            bootstrap = torch.zeros(cfg.n_envs, device=device)
            ids = np.flatnonzero(cutoffs)
            if len(ids):
                _, v, _ = self.model(torch.as_tensor(np.stack([terminal_images[i] for i in ids]), device=device),
                                     next_state[:, ids])
                bootstrap[torch.as_tensor(ids, device=device)] = v
            for key, value in (("images", images), ("resets", masks), ("actions", actions),
                               ("logp", dist.log_prob(actions)), ("values", values),
                               ("rewards", torch.tensor(rewards, device=device)),
                               ("dones", torch.tensor(dones, device=device)), ("bootstrap", bootstrap)):
                data[key].append(value)
            self.images = np.stack(next_images)
            self.state, self.reset_mask = next_state, np.asarray(dones, np.float32)
        _, last, _ = self.model(torch.as_tensor(self.images, device=device),
                                self.state * (1 - torch.as_tensor(self.reset_mask, device=device))[None, :])
        data = {k: torch.stack(v) for k, v in data.items()}
        nxt = torch.cat([data["values"][1:], last[None]], dim=0)
        nxt = torch.where(data["dones"], data["bootstrap"], nxt)
        data["advantages"], data["returns"] = generalized_advantages(
            data["rewards"], data["values"], nxt, data["dones"], cfg.gamma, cfg.gae_lambda)
        data["states"], data["episodes"] = boundaries, episodes
        return data


def generalized_advantages(rewards, values, next_values, dones, gamma, lam):
    """next_values is zero for deaths, terminal value for time limits, next value otherwise."""
    adv, running = torch.zeros_like(values), torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        running = delta + gamma * lam * (~dones[t]).float() * running
        adv[t] = running
    return adv, adv + values


def ppo_update(model, optimizer, data, cfg):
    """Shuffle contiguous sequences, never timesteps; differentiate through the brain.

    Boundary states are detached rollout states (truncated recurrent PPO), and
    within-sequence episode resets also cut the recurrent gradient.
    """
    adv = data["advantages"]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    stats = []
    for _ in range(cfg.epochs):
        for block in torch.randperm(cfg.rollout // cfg.bptt).tolist():
            start = block * cfg.bptt
            state = data["states"][block].detach()
            logits, vals = [], []
            for t in range(start, start + cfg.bptt):
                state = state * (1 - data["resets"][t])[None, :]
                pi, v, state = model(data["images"][t], state)
                logits.append(pi)
                vals.append(v)
            sl = slice(start, start + cfg.bptt)
            dist = torch.distributions.Categorical(logits=torch.stack(logits))
            logp = dist.log_prob(data["actions"][sl])
            logratio = logp - data["logp"][sl]
            ratio = logratio.exp()
            pg = -torch.minimum(ratio * adv[sl], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[sl]).mean()
            v = torch.stack(vals)
            vclip = data["values"][sl] + (v - data["values"][sl]).clamp(-cfg.clip, cfg.clip)
            vloss = .5 * torch.maximum((v - data["returns"][sl]).square(),
                                      (vclip - data["returns"][sl]).square()).mean()
            ent = dist.entropy().mean()
            loss = pg + cfg.value_coef * vloss - cfg.entropy * ent
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss; last complete checkpoint is preserved")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            kl = float(((ratio - 1) - logratio).mean().detach())
            # Stop before applying an update already outside the KL trust region.
            if kl > cfg.target_kl:
                optimizer.zero_grad(set_to_none=True)
                return _mean_stats(stats)
            optimizer.step()
            stats.append(dict(loss=float(loss.detach()), entropy=float(ent.detach()), kl=kl, grad_norm=float(norm)))
    return _mean_stats(stats)


def _mean_stats(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ("loss", "entropy", "kl", "grad_norm")} if rows else {}


@torch.no_grad()
def evaluate_brain(model, factory, cfg, episodes, seed, greedy=True, record=False, vision_mode=None):
    """Fresh seeded starts for matched before/after and held-out final evaluation."""
    runner = MarioRunner(factory(), cfg.frame_skip, cfg.noop_max, cfg.stuck_steps, cfg.max_steps,
                         seed, record_stride=cfg.frame_skip)
    vision = VisualInput(vision_mode or cfg.vision, seed=cfg.seed)
    generator = torch.Generator().manual_seed(seed)
    rows, best, frames = [], None, []
    try:
        for _ in range(episodes):
            obs, prev = runner.reset(record=record), None
            state, done = model.initial_state(1), False
            while not done:
                code, prev = vision(obs, prev, runner.steps)
                image = torch.as_tensor(code.reshape(1, 2, 32, 32), device=model.device)
                logits, _, state = model(image, state)
                action = (int(logits.argmax(-1)) if greedy else
                          int(torch.multinomial(logits.softmax(-1).cpu(), 1, generator=generator)))
                obs, _, done, _ = runner.step(action)
            row = runner.episode_summary()
            rows.append(row)
            if best is None or episode_key(row) > episode_key(best):
                best, frames = row, runner.frames
    finally:
        runner.env.close()
    return summarize(rows), frames


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path, model, optimizer, cfg, graph_id, update, history, best_key):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    state = {"format": 3, "config": asdict(cfg), "graph_id": graph_id, "update": update,
             "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "history": history, "best_key": list(best_key),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else [],
             "resume_semantics": "weights_optimizer_counters_rng; fresh emulator episodes"}
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer, cfg, graph_id):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != 3 or ck["graph_id"] != graph_id:
        raise ValueError("Checkpoint belongs to another graph or format; choose a new run folder")
    mutable = {"updates", "max_hours", "checkpoint_every", "eval_every", "validation_episodes", "final_episodes"}
    for k, v in asdict(cfg).items():
        if k not in mutable and ck["config"].get(k) != v:
            raise ValueError(f"Resume config mismatch: {k}; use the original config or a new run folder")
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for k, value in state.items():
                if torch.is_tensor(value):
                    state[k] = value.to(model.device)
    torch.set_rng_state(ck["torch_rng"])
    if model.device.type == "cuda" and ck["cuda_rng"]:
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck


def preflight(model, cfg):
    """Actual forward/backward on the loaded full graph, before a long run."""
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    state = model.initial_state(cfg.n_envs)
    loss = torch.zeros((), device=model.device)
    for _ in range(cfg.bptt):
        logits, v, state = model(torch.randn(cfg.n_envs, 2, 32, 32, device=model.device), state)
        loss = loss + logits.square().mean() + v.square().mean()
    loss.backward()
    grads = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
    if not all(math.isfinite(g) for g in grads.values()):
        raise FloatingPointError("Preflight produced a nonfinite gradient")
    if not any(g > 0 for k, g in grads.items() if k.startswith("encoder")):
        raise RuntimeError("No visual gradient reaches the action head through this graph/BPTT window")
    # Include Adam's two moment buffers in the measured budget, without changing weights.
    probe_optimizer = torch.optim.Adam(model.parameters(), lr=0)
    probe_optimizer.step()
    model.zero_grad(set_to_none=True)
    return {"trainable_parameters": sum(p.numel() for p in model.parameters()),
            "encoder_gradient_norm": sum(g for k, g in grads.items() if k.startswith("encoder")),
            "peak_gpu_gib": (torch.cuda.max_memory_allocated(model.device) / 2**30
                             if model.device.type == "cuda" else None),
            "n_envs": cfg.n_envs, "bptt": cfg.bptt, "micro_steps": cfg.micro_steps}


def run_full_brain(model, factory: Callable, cfg, out_dir, graph_id, resume=True, log=print):
    """Persistent last/best checkpoints, bounded session time, independent final evaluation."""
    cfg.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    latest, best_path = out / "latest.pt", out / "best.pt"
    history, first, best_key = [], 0, (-1.0, -1.0)
    if latest.exists() and not resume:
        raise FileExistsError("Run already exists; resume it or choose another folder")
    if resume and latest.exists():
        ck = load_checkpoint(latest, model, optimizer, cfg, graph_id)
        first, history, best_key = ck["update"], ck["history"], tuple(ck["best_key"])
        log(f"Resuming update {first}; emulator episodes restart.")
    elif not (out / "pre_eval.json").exists():
        before, _ = evaluate_brain(model, factory, cfg, cfg.validation_episodes, cfg.seed + 100_000)
        atomic_json(out / "pre_eval.json", before)
    # An update=0 checkpoint makes the first session recoverable too.
    save_checkpoint(latest, model, optimizer, cfg, graph_id, first, history, best_key)
    collector = RecurrentCollector(model, factory, cfg, seed_offset=first * cfg.n_envs)
    started, completed, stop_reason = time.monotonic(), first, "update_target"
    try:
        for update in range(first, cfg.updates):
            if cfg.max_hours > 0 and time.monotonic() - started >= cfg.max_hours * 3600:
                stop_reason = "session_time_budget"
                break
            t0 = time.monotonic()
            data = collector.collect()
            stats = ppo_update(model, optimizer, data, cfg)
            episodes = data.pop("episodes")
            del data
            row = {"update": update + 1, "agent_steps": (update + 1) * cfg.rollout * cfg.n_envs,
                   "episodes": len(episodes), "flags": sum(e["flag_get"] for e in episodes),
                   "mean_max_x": float(np.mean([e["max_x"] for e in episodes])) if episodes else None,
                   "best_max_x": max((e["max_x"] for e in episodes), default=None),
                   "seconds": round(time.monotonic() - t0, 3), **stats}
            history.append(row)
            completed = update + 1
            if completed % cfg.checkpoint_every == 0:
                save_checkpoint(latest, model, optimizer, cfg, graph_id, completed, history, best_key)
                atomic_json(out / "progress.json", {"update": completed, "latest": row})
            if completed % cfg.eval_every == 0:
                result, _ = evaluate_brain(model, factory, cfg, cfg.validation_episodes, cfg.seed + 100_000)
                key = (result["success_rate"], result["max_x_mean"])
                if key > best_key:
                    best_key = key
                    save_checkpoint(best_path, model, optimizer, cfg, graph_id, completed, history, best_key)
                    atomic_json(out / "best_validation.json", result)
                # Persist best_key in latest even if eval/checkpoint periods differ.
                save_checkpoint(latest, model, optimizer, cfg, graph_id, completed, history, best_key)
            if completed == 1 or completed % cfg.checkpoint_every == 0:
                log(f"update {completed}/{cfg.updates} steps={row['agent_steps']:,} "
                    f"mean_x={row['mean_max_x']} flags={row['flags']} entropy={row.get('entropy', 0):.3f}")
        save_checkpoint(latest, model, optimizer, cfg, graph_id, completed, history, best_key)
    except KeyboardInterrupt:
        log("Interrupted. Resume from the last complete checkpoint; partial updates were not saved.")
        raise
    finally:
        collector.close()
    # Final RNG seeds differ from validation. Finite no-op offsets may overlap;
    # this is evaluation on the same stage, not held-out-level generalization.
    selected = best_path if best_path.exists() else latest
    selected_ck = load_checkpoint(selected, model, None, cfg, graph_id)
    greedy, frames = evaluate_brain(model, factory, cfg, cfg.final_episodes, cfg.seed + 200_000, record=True)
    sampled, sampled_frames = evaluate_brain(model, factory, cfg, cfg.final_episodes,
                                            cfg.seed + 200_000, greedy=False, record=True)
    codec = save_video(frames, out / "post_greedy.mp4", fps=max(1, round(60 / cfg.frame_skip)))
    save_video(sampled_frames, out / "post_sampled.mp4", fps=max(1, round(60 / cfg.frame_skip)))
    results = {"experiment": "Meta-Brain V3 trainable full-connectome PPO", "graph_id": graph_id,
               "config": asdict(cfg), "completed_updates": completed, "selected_update": selected_ck["update"],
               "stop_reason": stop_reason, "post_eval": greedy, "post_eval_sampled": sampled,
               "selected_checkpoint": selected.name, "video_codec": codec,
               "trainable": "CNN vision; cell-type input/recurrent gains, decay, bias; nonlinear motor heads",
               "fixed": "signed synapse counts, topology and compact sensory projection basis; no visual bypass",
               "claim_status": ("DEMONSTRATED_SUCCESS" if greedy["all_success_no_death"]
                                else "RUN_COMPLETED_BUT_SUCCESS_NOT_DEMONSTRATED"),
               "resume_semantics": "training resumes; in-flight emulator trajectories restart",
               "history": history}
    atomic_json(out / "results.json", results)
    return results
