"""Linear probe: is the game's visual layout decodable from motor-neuron activity?

Runs before (and independently of) any RL. A scripted policy walks right with
random jumps through World 1-1. At every agent step the same frame sequence
drives the frozen connectome reservoir twice, as two batch columns:

* ``motor_real``: motor activity with the real visual input;
* ``motor_clock``: motor activity with the ``clock`` input (a code that
  depends only on the step since reset, see :class:`VisualInput`), the
  blind control.

Ground truth comes from the emulator's RAM (``smb_labels``): Mario's level x,
the distance to the nearest Goomba ahead and the distance to the next pipe.
Ridge regressions, cross-validated by episode, decode each target from

* ``visual_code``: the encoded frame itself (the ceiling: is it in the input?);
* ``motor_real`` and ``motor_clock`` (does vision reach the motor neurons?);
* ``time``: one-hot bins of the step since reset (what elapsed time alone explains).

:func:`run_layer_probe` repeats the real-vs-clock comparison for each layer of
the visual pathway (:data:`VISUAL_PATHWAY`) to locate where the screen stops
being decodable.

If ``motor_real`` decodes the Goomba or pipe distance no better than
``motor_clock``, visual information is not reaching the readout, and RL on the
readout cannot use it. Since 1-1 is deterministic, distance to the next pipe is
partly predictable from time; the Goomba distance and the ``motor_clock``
control are what separate seeing from counting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .mario_rl import ConnectomeReservoir, MarioRunner, VisualInput

# World 1-1 pipe left edges in level pixels. 608, 736 and 912 match where agents
# get stuck (x = 594, 722, 898: the pipe edge minus Mario's width).
PIPES_1_1 = (448, 608, 736, 912, 2608, 2864)
GOOMBA = 0x06
MAX_DX = 256.0  # one screen ahead
NEAR_DX = 96.0


def smb_labels(env, pipes: Sequence[int] = PIPES_1_1) -> Dict[str, float]:
    """Mario x, nearest Goomba ahead and next pipe, from Super Mario Bros RAM.

    RAM map (NES SMB): Mario x = 0x6D page * 256 + 0x86; enemy slots 0-4 have
    drawn flag 0x0F+i, type 0x16+i (0x06 = Goomba), page 0x6E+i and x 0x87+i.
    """
    ram = np.asarray(getattr(env, "unwrapped", env).ram)
    mario_x = float(int(ram[0x6D]) * 256 + int(ram[0x86]))
    goomba = np.nan
    for i in range(5):
        if ram[0x0F + i] and ram[0x16 + i] == GOOMBA:
            dx = float(int(ram[0x6E + i]) * 256 + int(ram[0x87 + i])) - mario_x
            if -16 <= dx <= MAX_DX and (np.isnan(goomba) or dx < goomba):
                goomba = dx
    ahead = [p - mario_x for p in pipes if p + 32 > mario_x]
    pipe = min(max(ahead[0], 0.0), MAX_DX) if ahead else MAX_DX
    return {"mario_x": mario_x, "goomba_dx": goomba, "goomba_near": float(goomba <= NEAR_DX),
            "pipe_dx": float(pipe)}


class ScriptedWalker:
    """Walk right; now and then hold a jump for 2-6 steps, then release so the next jump can start."""

    def __init__(self, seed: int = 0, jump_prob: float = 0.2, right: int = 1, jump: int = 2):
        self.rng = np.random.default_rng(seed)
        self.jump_prob, self.right, self.jump = jump_prob, right, jump
        self.reset()

    def reset(self) -> None:
        self.hold = self.cooldown = 0

    def act(self) -> int:
        if self.hold > 0:
            self.hold -= 1
            self.cooldown = 1 if self.hold == 0 else 0
            return self.jump
        if self.cooldown == 0 and self.rng.random() < self.jump_prob:
            self.hold = int(self.rng.integers(2, 7)) - 1
            return self.jump
        self.cooldown = 0
        return self.right


@torch.no_grad()
def collect_probe_dataset(runner: MarioRunner, reservoir: ConnectomeReservoir, episodes: int = 30,
                          size=(32, 32), seed: int = 0, jump_prob: float = 0.2,
                          right: int = 1, jump: int = 2,
                          layers: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Scripted run through the level; returns features and RAM labels per agent step.

    ``layers`` maps a name to neuron indices (see :func:`layer_indices`); their
    activity is recorded as ``layer:<name>:real`` and ``layer:<name>:clock``.
    """
    walker = ScriptedWalker(seed, jump_prob, right, jump)
    real, clock = VisualInput("real", size, seed), VisualInput("clock", size, seed)
    layer_idx = {name: torch.as_tensor(np.asarray(idx), dtype=torch.long, device=reservoir.device)
                 for name, idx in (layers or {}).items()}
    cols: Dict[str, List] = {k: [] for k in ("motor_real", "motor_clock", "visual_code", "step", "episode",
                                             "mario_x", "goomba_dx", "goomba_near", "pipe_dx")}
    for name in layer_idx:
        cols[f"layer:{name}:real"], cols[f"layer:{name}:clock"] = [], []
    for ep in range(episodes):
        obs = runner.reset()
        prev, state, done = None, reservoir.initial_state(2), False
        walker.reset()
        while not done:
            code_r, prev = real(obs, prev, runner.steps)
            code_c, _ = clock(obs, None, runner.steps)
            feats, state = reservoir.step(torch.as_tensor(np.stack([code_r, code_c]), device=reservoir.device),
                                          state)
            feats = feats.float().cpu().numpy()
            labels = smb_labels(runner.env)
            cols["motor_real"].append(feats[0])
            cols["motor_clock"].append(feats[1])
            for name, idx in layer_idx.items():
                act = state.index_select(0, idx).float().cpu().numpy()
                cols[f"layer:{name}:real"].append(act[:, 0])
                cols[f"layer:{name}:clock"].append(act[:, 1])
            cols["visual_code"].append(code_r)
            cols["step"].append(runner.steps)
            cols["episode"].append(ep)
            for k, v in labels.items():
                cols[k].append(v)
            obs, _, done, _ = runner.step(walker.act())
    return {k: np.asarray(v, dtype=np.float32 if k not in ("step", "episode") else np.int64)
            for k, v in cols.items()}


def time_features(steps: np.ndarray, bin_size: int = 10) -> np.ndarray:
    """One-hot bins of the step since reset: what elapsed time alone can explain."""
    bins = (np.asarray(steps) // bin_size).astype(np.int64)
    out = np.zeros((len(bins), int(bins.max()) + 1), np.float32)
    out[np.arange(len(bins)), bins] = 1.0
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels > 0.5, labels <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def ridge_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, binary: bool = False,
             lambdas: Sequence[float] = (1e-2, 1e-1, 1, 10, 100, 1000), folds: int = 5) -> Dict[str, float]:
    """Ridge regression with group (episode) K-fold; returns the best out-of-fold score over ``lambdas``.

    The score is R^2 for continuous targets and ROC AUC for binary ones. The same
    procedure is applied to every feature set, so comparisons between them are fair.
    """
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % folds))
    fold = np.array([fold_of[g] for g in groups])
    preds = {lam: np.zeros_like(y) for lam in lambdas}
    for f in range(min(folds, len(uniq))):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        if Xtr.shape[1] <= Xtr.shape[0]:
            s, V = np.linalg.eigh(Xtr.T @ Xtr)
            proj = V.T @ (Xtr.T @ (y[tr] - ym))
            for lam in lambdas:
                preds[lam][te] = Xte @ (V @ (proj / (s + lam * len(Xtr)))) + ym
        else:  # dual form when there are more features than samples
            s, U = np.linalg.eigh(Xtr @ Xtr.T)
            proj = U.T @ (y[tr] - ym)
            for lam in lambdas:
                alpha = U @ (proj / (s + lam * len(Xtr)))
                preds[lam][te] = Xte @ (Xtr.T @ alpha) + ym
    best = {"score": -np.inf, "lambda": None}
    for lam, p in preds.items():
        if binary:
            score = _auc(p, y)
        else:
            score = 1.0 - float(((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        if np.isfinite(score) and score > best["score"]:
            best = {"score": float(score), "lambda": lam}
    best["metric"] = "auc" if binary else "r2"
    best["n"] = int(len(y))
    return best


TARGETS = {"goomba_dx": False, "goomba_near": True, "pipe_dx": False, "mario_x": False}
FEATURE_SETS = ("visual_code", "motor_real", "motor_clock", "time")


def run_probe(data: Dict[str, np.ndarray], targets: Optional[Dict[str, bool]] = None) -> Dict:
    """Decode every target from every feature set; ``goomba_dx`` uses only steps with a Goomba ahead."""
    targets = targets or TARGETS
    features = {name: data[name] for name in FEATURE_SETS if name != "time"}
    features["time"] = time_features(data["step"])
    table: Dict[str, Dict[str, Dict]] = {}
    for target, binary in targets.items():
        y = data[target]
        mask = np.isfinite(y)
        if mask.sum() < 20 or len(np.unique(data["episode"][mask])) < 2:
            table[target] = {name: {"score": float("nan"), "n": int(mask.sum())} for name in features}
            continue
        table[target] = {name: ridge_cv(X[mask], y[mask], data["episode"][mask], binary)
                         for name, X in features.items()}
    gains = {t: round(r["motor_real"]["score"] - r["motor_clock"]["score"], 4)
             for t, r in table.items() if np.isfinite(r["motor_real"]["score"])}
    return {
        "samples": int(len(data["step"])),
        "episodes": int(len(np.unique(data["episode"]))),
        "goomba_visible_fraction": round(float(np.isfinite(data["goomba_dx"]).mean()), 4),
        "max_mario_x": float(np.nanmax(data["mario_x"])),
        "scores": {t: {name: round(v["score"], 4) for name, v in r.items()} for t, r in table.items()},
        "details": table,
        "vision_gain_motor_real_minus_clock": gains,
    }


def plot_probe(result: Dict, path) -> None:
    import matplotlib.pyplot as plt

    targets = list(result["scores"])
    names = list(FEATURE_SETS)
    colors = {"visual_code": "#4c4c4c", "motor_real": "#4878d0", "motor_clock": "#ee854a", "time": "#956cb4"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        vals = [result["scores"][t].get(name, np.nan) for t in targets]
        ax.bar(np.arange(len(targets)) + (j - (len(names) - 1) / 2) * width, vals, width,
               label=name, color=colors[name])
    ax.set_xticks(np.arange(len(targets)),
                  [f"{t}\n({'AUC' if TARGETS.get(t) else 'R²'})" for t in targets])
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.0)
    ax.set_title("What can a linear readout decode? (cross-validated by episode)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Visual pathway from the photoreceptors to the brain's output, in order.
VISUAL_PATHWAY = ("ol_sensory", "ol_intrinsic", "visual_projection", "cb_intrinsic", "descending_neuron")


def layer_indices(graph, names: Sequence[str] = VISUAL_PATHWAY, max_per_layer: int = 2000,
                  seed: int = 0) -> Dict[str, np.ndarray]:
    """Neuron indices per superclass, randomly subsampled to at most ``max_per_layer``.

    Subsampling keeps every layer's probe the same size as the motor probe
    (about 2,000 features), so layers are compared at equal decoder capacity.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name in names:
        idx = np.flatnonzero(graph.superclass_mask([name]))
        if len(idx) == 0:
            continue
        if len(idx) > max_per_layer:
            idx = np.sort(rng.choice(idx, max_per_layer, replace=False))
        out[name] = idx
    return out


def run_layer_probe(data: Dict[str, np.ndarray], targets: Optional[Dict[str, bool]] = None,
                    layer_sizes: Optional[Dict[str, int]] = None) -> Dict:
    """Decode each target from each recorded layer, with real input and with the blind clock.

    Layers appear in the order they were recorded (the visual pathway order).
    ``signal`` is real minus clock: what the layer carries about the screen
    beyond elapsed time. Where it drops to about zero is where the visual signal
    is lost.
    """
    targets = targets or {"goomba_dx": False, "goomba_near": True}
    names = [k.split(":")[1] for k in data if k.startswith("layer:") and k.endswith(":real")]
    rows = []
    for name in names:
        row = {"layer": name, "neurons_probed": int(data[f"layer:{name}:real"].shape[1])}
        if layer_sizes and name in layer_sizes:
            row["neurons_total"] = int(layer_sizes[name])
        for target, binary in targets.items():
            y = data[target]
            mask = np.isfinite(y)
            if mask.sum() < 20 or len(np.unique(data["episode"][mask])) < 2:
                continue
            groups = data["episode"][mask]
            real = ridge_cv(data[f"layer:{name}:real"][mask], y[mask], groups, binary)["score"]
            clock = ridge_cv(data[f"layer:{name}:clock"][mask], y[mask], groups, binary)["score"]
            row[f"{target}_real"] = round(real, 4)
            row[f"{target}_clock"] = round(clock, 4)
            row[f"{target}_signal"] = round(real - clock, 4)
        rows.append(row)
    return {"layers": rows, "targets": {t: ("auc" if b else "r2") for t, b in targets.items()}}


def plot_layer_probe(result: Dict, path, ceiling: Optional[Dict[str, float]] = None) -> None:
    """Real vs blind decoding along the pathway, one panel per target."""
    import matplotlib.pyplot as plt

    rows = result["layers"]
    targets = list(result["targets"])
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 4.5), squeeze=False)
    x = np.arange(len(rows))
    labels = [r["layer"].replace("_", "\n") for r in rows]
    for ax, target in zip(axes[0], targets):
        real = [r.get(f"{target}_real", np.nan) for r in rows]
        clock = [r.get(f"{target}_clock", np.nan) for r in rows]
        ax.plot(x, real, "o-", color="#4878d0", label="real screen")
        ax.plot(x, clock, "s--", color="#ee854a", label="blind clock")
        ax.fill_between(x, clock, real, color="#4878d0", alpha=0.12)
        if ceiling and target in ceiling:
            ax.axhline(ceiling[target], color="#4c4c4c", lw=1, ls=":", label="encoded frame (ceiling)")
        ax.set_xticks(x, labels, fontsize=8)
        metric = result["targets"][target].upper()
        ax.set_title(f"{target} ({metric}) along the visual pathway")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Dynamics sweep: does the screen get past the optic lobe at lower gain?
# --------------------------------------------------------------------------- #

def sweep_configs(gains: Sequence[float] = (0.5, 0.9, 1.2, 1.6),
                  input_gains: Sequence[float] = (1.0, 4.0)) -> List[Tuple[float, float]]:
    return [(float(g), float(ig)) for ig in input_gains for g in gains]


def config_name(config: Tuple[float, float]) -> str:
    return f"gain={config[0]:g},input={config[1]:g}"


@torch.no_grad()
def collect_sweep_dataset(runner: MarioRunner, reservoir: ConnectomeReservoir,
                          configs: Sequence[Tuple[float, float]], layers: Dict[str, np.ndarray],
                          episodes: int = 30, size=(32, 32), seed: int = 0, jump_prob: float = 0.2,
                          right: int = 1, jump: int = 2, saturation: float = 0.9) -> Dict:
    """One scripted run, many reservoirs: every ``(gain, input_gain)`` config sees the same frames.

    Each config gets two batch columns (real screen, blind clock); ``gain`` and
    ``input_gain`` are set per column, so a single sparse matmul advances every
    config at once. Also records each layer's mean ``|activity|`` and the
    fraction of saturated units (``|state| > saturation``) with the real screen.
    """
    walker = ScriptedWalker(seed, jump_prob, right, jump)
    real, clock = VisualInput("real", size, seed), VisualInput("clock", size, seed)
    dev = reservoir.device
    n_cols = 2 * len(configs)
    gains = torch.tensor([c[0] for c in configs for _ in (0, 1)], device=dev)
    input_gains = torch.tensor([c[1] for c in configs for _ in (0, 1)], device=dev)
    layer_idx = {name: torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev) for name, idx in layers.items()}
    cols: Dict[str, List] = {k: [] for k in ("step", "episode", "mario_x", "goomba_dx", "goomba_near", "pipe_dx")}
    for k in range(len(configs)):
        for name in layer_idx:
            cols[f"cfg{k}:layer:{name}:real"], cols[f"cfg{k}:layer:{name}:clock"] = [], []
    abs_sum = {name: torch.zeros(n_cols, device=dev) for name in layer_idx}
    sat_sum = {name: torch.zeros(n_cols, device=dev) for name in layer_idx}
    steps = 0
    saved = (reservoir.gain, reservoir.input_gain)
    reservoir.gain, reservoir.input_gain = gains, input_gains
    try:
        for ep in range(episodes):
            obs = runner.reset()
            prev, state, done = None, reservoir.initial_state(n_cols), False
            walker.reset()
            while not done:
                code_r, prev = real(obs, prev, runner.steps)
                code_c, _ = clock(obs, None, runner.steps)
                codes = np.stack([code_r, code_c] * len(configs))
                _, state = reservoir.step(torch.as_tensor(codes, device=dev), state)
                for name, idx in layer_idx.items():
                    act = state.index_select(0, idx)
                    abs_sum[name] += act.abs().mean(0)
                    sat_sum[name] += (act.abs() > saturation).float().mean(0)
                    act = act.float().cpu().numpy()
                    for k in range(len(configs)):
                        cols[f"cfg{k}:layer:{name}:real"].append(act[:, 2 * k])
                        cols[f"cfg{k}:layer:{name}:clock"].append(act[:, 2 * k + 1])
                for key, v in smb_labels(runner.env).items():
                    cols[key].append(v)
                cols["step"].append(runner.steps)
                cols["episode"].append(ep)
                steps += 1
                obs, _, done, _ = runner.step(walker.act())
    finally:
        reservoir.gain, reservoir.input_gain = saved
    data = {k: np.asarray(v, dtype=np.int64 if k in ("step", "episode") else np.float32) for k, v in cols.items()}
    data["configs"] = [tuple(c) for c in configs]
    data["regime"] = {name: {"mean_abs": (abs_sum[name] / max(steps, 1)).cpu().numpy()[0::2],
                             "saturated": (sat_sum[name] / max(steps, 1)).cpu().numpy()[0::2]}
                      for name in layer_idx}
    return data


def run_sweep_probe(data: Dict, target: str = "goomba_dx", binary: bool = False) -> Dict:
    """Real-minus-clock decoding of ``target`` for every config and layer."""
    names = [k.split(":")[2] for k in data if k.startswith("cfg0:layer:") and k.endswith(":real")]
    y = data[target]
    mask = np.isfinite(y)
    groups = data["episode"][mask]
    rows = []
    for k, config in enumerate(data["configs"]):
        for j, name in enumerate(names):
            real = ridge_cv(data[f"cfg{k}:layer:{name}:real"][mask], y[mask], groups, binary)["score"]
            clock = ridge_cv(data[f"cfg{k}:layer:{name}:clock"][mask], y[mask], groups, binary)["score"]
            rows.append({"config": config_name(config), "gain": config[0], "input_gain": config[1],
                         "layer": name, "real": round(real, 4), "clock": round(clock, 4),
                         "signal": round(real - clock, 4),
                         "mean_abs_activity": round(float(data["regime"][name]["mean_abs"][k]), 4),
                         "saturated_fraction": round(float(data["regime"][name]["saturated"][k]), 4)})
    deepest = {}
    for config in {r["config"] for r in rows}:
        mine = [r for r in rows if r["config"] == config]
        past = [r["layer"] for r in mine if r["signal"] > 0.05]
        deepest[config] = past[-1] if past else None
    return {"target": target, "metric": "auc" if binary else "r2", "layers": names, "rows": rows,
            "deepest_layer_with_signal": deepest,
            "signal_threshold": 0.05,
            "samples": int(mask.sum())}


def plot_sweep_probe(result: Dict, path) -> None:
    """Signal (real minus clock) along the pathway, one line per config; plus saturation."""
    import matplotlib.pyplot as plt

    layers = result["layers"]
    configs = list(dict.fromkeys(r["config"] for r in result["rows"]))
    gains = sorted({r["gain"] for r in result["rows"]})
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    x = np.arange(len(layers))
    for config in configs:
        mine = {r["layer"]: r for r in result["rows"] if r["config"] == config}
        g, ig = mine[layers[0]]["gain"], mine[layers[0]]["input_gain"]
        color = cmap(gains.index(g) / max(len(gains) - 1, 1))
        style = "-" if ig == min(r["input_gain"] for r in result["rows"]) else "--"
        axes[0].plot(x, [mine[n]["signal"] for n in layers], style, marker="o", color=color, label=config)
        axes[1].plot(x, [mine[n]["saturated_fraction"] for n in layers], style, marker="o", color=color)
    axes[0].axhline(result["signal_threshold"], color="gray", lw=0.8, ls=":")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title(f"Screen signal along the pathway ({result['target']}, {result['metric'].upper()} real − clock)")
    axes[1].set_title("Fraction of saturated units (|state| > 0.9), real screen")
    for ax in axes:
        ax.set_xticks(x, [n.replace("_", "\n") for n in layers], fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
