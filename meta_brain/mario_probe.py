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

If ``motor_real`` decodes the Goomba or pipe distance no better than
``motor_clock``, visual information is not reaching the readout, and RL on the
readout cannot use it. Since 1-1 is deterministic, distance to the next pipe is
partly predictable from time; the Goomba distance and the ``motor_clock``
control are what separate seeing from counting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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


@torch.no_grad()
def collect_probe_dataset(runner: MarioRunner, reservoir: ConnectomeReservoir, episodes: int = 30,
                          size=(32, 32), seed: int = 0, jump_prob: float = 0.2,
                          right: int = 1, jump: int = 2) -> Dict[str, np.ndarray]:
    """Scripted run through the level; returns features and RAM labels per agent step."""
    rng = np.random.default_rng(seed)
    real, clock = VisualInput("real", size, seed), VisualInput("clock", size, seed)
    cols: Dict[str, List] = {k: [] for k in ("motor_real", "motor_clock", "visual_code", "step", "episode",
                                             "mario_x", "goomba_dx", "goomba_near", "pipe_dx")}
    for ep in range(episodes):
        obs = runner.reset()
        prev, state, done = None, reservoir.initial_state(2), False
        hold = cooldown = 0
        while not done:
            code_r, prev = real(obs, prev, runner.steps)
            code_c, _ = clock(obs, None, runner.steps)
            feats, state = reservoir.step(torch.as_tensor(np.stack([code_r, code_c]), device=reservoir.device),
                                          state)
            feats = feats.float().cpu().numpy()
            labels = smb_labels(runner.env)
            cols["motor_real"].append(feats[0])
            cols["motor_clock"].append(feats[1])
            cols["visual_code"].append(code_r)
            cols["step"].append(runner.steps)
            cols["episode"].append(ep)
            for k, v in labels.items():
                cols[k].append(v)
            # Walk right; now and then hold a jump for 2-6 steps, then release so the next jump can start.
            if hold > 0:
                hold -= 1
                action = jump
                cooldown = 1 if hold == 0 else 0
            elif cooldown == 0 and rng.random() < jump_prob:
                hold = int(rng.integers(2, 7)) - 1
                action = jump
            else:
                cooldown = 0
                action = right
            obs, _, done, _ = runner.step(action)
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
