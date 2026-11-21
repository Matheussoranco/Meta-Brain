# Evidence Pack — Meta-Brain `#2` (`META-CI`)

- **Pack ID:** `evd-meta-2-offline-ci-20261120`
- **Template version:** 1.0
- **Generated at (UTC):** `2026-11-21T00:14:00Z`
- **Verification status (local offline suite):** `PASS`
- **Review & merge readiness:** `PENDING_REVIEW` (awaiting independent review by Claude Code and final integration assessment by Engineering Lead; GitHub Actions CI run will verify on Linux Python 3.13 upon PR creation)
- **Task mode:** `github`
- **Task reference:** `Matheussoranco/Meta-Brain#2` (trace `META-CI`)
- **Repository:** `Matheussoranco/Meta-Brain`
- **Branch:** `sail/meta-offline-ci`
- **Base revision:** `ee8792b`
- **Current revision:** recorded in PR description and issue `#2` comment
- **Working-tree state:** clean after commit
- **Producer:** `Gemini 4 Neon (High) — Senior Software & Product Engineer`

## 1. Executive result

Declared missing offline notebook smoke-test dependencies (`requests`, `ipython`) in CI (`.github/workflows/ci.yml`), `requirements.txt`, and `setup.py`, and diagnosed and resolved the PPO learning regression in `meta_brain/mario_rl.py` while **retaining the exact original behavioral threshold** (`assert p_right() > max(0.6, before + 0.2)`) in `tests/test_mario_rl.py::test_ppo_learns_to_move_right`.

Root-cause diagnosis on synthetic fixtures (no external dataset or weight downloads) identified two coupled defects in V2 `mario_rl.py`:
1. **Unclamped standardized readout features (`ReadoutPolicy.forward`)**: After an 8-step warm-up from zero reservoir state (`decay=0.9`, `micro_steps=2`), steady-state motor activity drifted to `|z|_mean = 7.64`, `|z|_max = 73.44` by step 50–80 while freshly-reset episodes dropped back to `|z|_mean = 0.30` (a 25× distribution jump).
2. **Unguarded PPO updates (`train_ppo`)**: Unlike V3 `meta_brain/mario_trainable.py` (which uses `target_kl = 0.03` and clipped value loss `vclip`), V2 `train_ppo` had neither KL trust-region early stopping nor value clipping. At episode death cutoffs (`die_at=200`), unclipped value errors and normalized negative advantages drove `lr=3e-3` Adam updates outside the trust region, collapsing `p_right` on **5 of 10 seeds** locally (`mean=0.597, min=0.166`) and on `seed=0` in CI run `44837965248` (`torch 2.15.0`, observed `0.411384 < 0.6`).

Adding standard `[-10.0, 10.0]` feature clamping in `ReadoutPolicy.forward` plus `target_kl=0.03` early stopping and clipped value loss in `train_ppo` (matching `mario_trainable.py`) raised multi-seed pass rate from **5/10** to **20/20** (`terminal p_right mean=0.995, min=0.916`; 32-step rollout `mean=0.999, min=0.973`). All embedded notebooks (`v2`, `v3`) were re-synced via the repository's build scripts, and the full local test suite passes (`94 passed, 2 skipped`).

## 2. Exact dependency versions (CI run `44837965248` vs local Python 3.13.5)

| Package | CI Run `44837965248` (Ubuntu Python `3.13.18`) | Local Workstation (Windows Python `3.13.5`) |
| --- | --- | --- |
| `numpy` | `2.5.5` | `2.2.6` |
| `scipy` | `1.18.1` | `1.16.3` |
| `pandas` | `3.1.1` | `2.3.3` |
| `pyarrow` | `26.0.1` | `22.0.0` |
| `pyyaml` | `6.0.4` | `6.0.3` |
| `torch` | `2.15.0` | `2.9.0+cpu` |
| `networkx` | `3.7` | `3.6.1` |
| `matplotlib` | `3.11.2` | `3.10.7` |
| `pillow` | `13.0.0` | `11.3.0` |
| `pytest` | `9.2.0` | `8.4.2` |
| `gymnasium` | `1.4.0` | `1.2.2` |
| `gym-super-mario-bros` | `9.1.0` | not installed locally (`test_mario_real.py` skipped locally; runs in CI) |
| `nes-py` | `9.0.1` | not installed locally |
| `requests` | **missing** (caused `test_colab_notebook.py` failure) | `2.32.5` |
| `ipython` | **missing** (caused `test_mario_v2.py` and `test_mario_trainable.py` failures) | `9.17.1` |

## 3. PPO Diagnosis & Per-Seed Evidence

### 3.1 Feature drift after 8-step warm-up (`seed=0`, baseline `ee8792b`)

| Step `t` after warm-up | `env_steps` | `\|z\|_mean` | `\|z\|_max` |
| ---: | --- | ---: | ---: |
| 0 | `[8, 8, 8, 8]` | 1.45 | 2.18 |
| 5 | `[13, 13, 13, 13]` | 3.24 | 11.41 |
| 10 | `[18, 18, 18, 18]` | 4.78 | 28.59 |
| 25 | `[33, 33, 33, 33]` | 7.11 | 63.75 |
| 50 | `[58, 58, 58, 58]` | 7.64 | 72.74 |
| 80 | `[88, 88, 88, 88]` | 7.64 | 73.44 |
| 95 (post-reset) | `[4, 3, 4, 4]` | 0.30 | 1.18 |

### 3.2 Ablation study (10 fixed seeds `0..9`, `updates=25, rollout=32`)

| Configuration | Terminal `p_right` Pass (`> max(0.6, before+0.2)`) | Terminal mean / min | 32-step rollout Pass | Rollout mean / min |
| --- | --- | --- | --- | --- |
| Baseline (`ee8792b`: `gain=0.5`, `lr=3e-3`, `die_at=200`) | **5/10** | 0.597 / 0.166 | 8/10 | 0.780 / 0.217 |
| Prior `gain=1.6` (`lr=3e-3`, `die_at=200`) | 7/10 | 0.719 / 0.000 | 8/10 | 0.822 / 0.145 |
| Baseline + `die_at=None` (no death cutoff) | 10/10 | 1.000 / 0.999 | 10/10 | 1.000 / 0.999 |
| Baseline + `clamp_z=10` only | 8/10 | 0.793 / 0.053 | 9/10 | 0.907 / 0.293 |
| Baseline + `target_kl=0.03` only | 9/10 | 0.855 / 0.000 | 9/10 | 0.897 / 0.016 |
| **Combined (`clamp_z=10` + `target_kl=0.03` + `vclip`)** | **20/20** | **0.995 / 0.916** | **20/20** | **0.999 / 0.973** |

### 3.3 Per-seed comparison (`seeds 0..14` baseline vs `seeds 0..19` fixed)

| Seed | Initial `before` | Baseline (`ee8792b`) terminal `p_right` | Baseline pass? | Fixed (`sail/meta-offline-ci`) terminal `p_right` | Fixed 32-step `p_right` | Fixed pass? |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | 0.343 | 0.842 (CI `torch 2.15`: **0.411**) | True (False CI) | 1.000 | 1.000 | **True** |
| 1 | 0.326 | 0.401 | **False** | 1.000 | 1.000 | **True** |
| 2 | 0.291 | 0.166 | **False** | 1.000 | 1.000 | **True** |
| 3 | 0.301 | 0.931 | True | 0.916 | 0.973 | **True** |
| 4 | 0.325 | 0.958 | True | 0.993 | 0.999 | **True** |
| 5 | 0.331 | 0.189 | **False** | 1.000 | 1.000 | **True** |
| 6 | 0.331 | 0.276 | **False** | 0.999 | 1.000 | **True** |
| 7 | 0.360 | 0.994 | True | 1.000 | 1.000 | **True** |
| 8 | 0.318 | 0.997 | True | 0.989 | 0.999 | **True** |
| 9 | 0.315 | 0.220 | **False** | 1.000 | 1.000 | **True** |
| 10 | 0.294 | 0.999 | True | 0.998 | 1.000 | **True** |
| 11 | 0.283 | 0.667 | True | 0.999 | 1.000 | **True** |
| 12 | 0.346 | 0.384 | **False** | 1.000 | 1.000 | **True** |
| 13 | 0.324 | 0.096 | **False** | 1.000 | 1.000 | **True** |
| 14 | 0.326 | 0.844 | True | 1.000 | 1.000 | **True** |
| 15–19 | 0.291–0.370 | — | — | 0.999–1.000 | 1.000 | **True (5/5)** |

## 4. Change inventory

- `.github/workflows/ci.yml`: added `requests` and `ipython` to the CI dependency install step.
- `requirements.txt` & `setup.py`: declared `ipython>=8.0.0` and `requests>=2.31.0`.
- `meta_brain/mario_rl.py`:
  - `ReadoutPolicy.forward`: clamp standardized features `z` to `[-10.0, 10.0]`.
  - `PPOConfig`: added `target_kl: float = 0.03` (aligned with `meta_brain/mario_trainable.py`).
  - `train_ppo`: added KL trust-region early stopping (`approx_kl > cfg.target_kl`), clipped value loss (`vclip`), and `kl` logging in `history`.
- `notebooks/colab_full_brain_mario_v2.ipynb`, `notebooks/colab_full_brain_mario_v3.ipynb`: re-synced embedded `meta_brain/mario_rl.py` using `scripts/build_colab_m0_m1_notebook.py` and `scripts/build_colab_mario_v3_notebook.py`.
- `tests/test_mario_rl.py`: kept `test_ppo_learns_to_move_right` untouched; added `test_readout_policy_clamps_standardized_outliers` and 4-seed parametrized regression test `test_ppo_learns_to_move_right_across_seeds` (`[1, 2, 5, 6]`).
- `docs/evidence/meta-2/{EVIDENCE.md, INTERACTION_REPORT.md, events.jsonl}`: evidence pack and interaction audit trail.
- **Intentionally untouched:** `AGENTS.md` (owned by PR `#1`), real-NES emulator pins, and all behavioral test thresholds.

## 5. Verification commands

| Command | Exit code | Result |
| --- | ---: | --- |
| `python scripts/build_colab_m0_m1_notebook.py` | 0 | `wrote notebooks/colab_m0_m1_infrastructure.ipynb`, `synced embedded modules in notebooks/colab_full_brain_mario_v2.ipynb` |
| `python scripts/build_colab_mario_v3_notebook.py` | 0 | `Generated notebooks/colab_full_brain_mario_v3.ipynb` |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q --tb=short` | 0 | `94 passed, 2 skipped in 59.63s` (skipped: `test_mario_real.py` NES emulator not installed on Windows host, `test_video_is_h264_yuv420p` ffmpeg check; both run in GitHub Actions CI) |
| Multi-seed 20-seed PPO diagnostic (`seeds 0..19`) | 0 | `20/20 pass` (`term mean=0.995, min=0.916`; `roll32 mean=0.999, min=0.973`) |

## 6. Risks and limitations

- Local Windows host does not have `gym-super-mario-bros` / `nes-py` installed (`test_mario_real.py` skipped locally); GitHub Actions CI on Ubuntu Python 3.13 will execute all 95+ tests including real NES smoke test, `requests`, `IPython`, and `torch 2.15.0` upon PR creation.
- No external MaleCNS dataset or pretrained weights were downloaded; all validation used existing synthetic fixtures.

<!-- evidence-pack:Matheussoranco/Meta-Brain#2:sail/meta-offline-ci -->
