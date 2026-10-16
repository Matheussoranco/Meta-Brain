"""Generate ``notebooks/colab_m0_m1_infrastructure.ipynb`` and sync embedded code.

The notebook is self-contained (no repository clone): the library code of
``meta_brain/graph.py``, ``shuffle.py``, ``sparse_ops.py`` and
``benchmark.py`` is embedded verbatim into code cells, minus the package-relative
imports. Regenerate after changing any of them:

    python scripts/build_colab_m0_m1_notebook.py

It also refreshes, in place, every ``# --- embedded from meta_brain/<name>.py ---``
cell of the hand-written notebooks in ``SYNCED_NOTEBOOKS`` (the Mario V2 notebook).
``tests/test_colab_notebook.py`` fails when any committed notebook is stale.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_m0_m1_infrastructure.ipynb"
MODULES = ("graph", "shuffle", "sparse_ops", "benchmark")
# Hand-written notebooks whose embedded-module cells are refreshed in place.
SYNCED_NOTEBOOKS = (ROOT / "notebooks" / "colab_full_brain_mario_v2.ipynb",)

_STRIP = re.compile(r"^[ \t]*from (__future__|\.\w*) import [^\n]*\n", re.MULTILINE)
_EMBED_HEADER = re.compile(r"# --- embedded from meta_brain/(\w+)\.py ---")


def module_source(name: str) -> str:
    src = (ROOT / "meta_brain" / f"{name}.py").read_text(encoding="utf-8")
    return f"# --- embedded from meta_brain/{name}.py ---\n" + _STRIP.sub("", src).strip() + "\n"


INTRO = """# Meta-Brain M0/M1: conectoma completo, controles e propagação esparsa treinável

Este notebook é **autocontido**: não clona o repositório. O código de `meta_brain/graph.py`, `shuffle.py`, `sparse_ops.py` e `benchmark.py` está embutido nas células da seção 2. O arquivo é gerado por `scripts/build_colab_m0_m1_notebook.py`, então não o edite à mão.

O que ele faz:

1. Baixa os três arquivos oficiais do MaleCNS v1.0 (anotações, neurotransmissores e pesos, ~1,2 GB) ou os reutiliza do Google Drive.
2. **M0, carga do grafo:** monta o conectoma de neurônios `Traced` (descartando as linhas que ligam a fragmentos não rastreados), faz cache em disco e contabiliza cada linha lida.
3. **M0, controles:** gera os três nulos (preserva grau, preserva tipo, aleatório) e verifica as invariantes de cada um.
4. **M1, correção:** roda `gradcheck` e compara o operador customizado com a referência por autograd no grafo completo.
5. **M1, benchmark:** mede BPTT pelo cérebro inteiro (K micro-passos por frame, checkpoint por frame) e compara tempo e memória com a referência.

Ao final, salva gráficos PNG, um CSV com o benchmark e `m0_m1_results.json` em `/content/meta_brain_m0_m1`, além de um `.zip` com tudo.

**Critérios** (`claim_status` no JSON):
- **M0:** todas as linhas de aresta contabilizadas, guard de escala, cache carregando em menos de 2 min e controles exatos (resíduo < 0,1%).
- **M1:** `gradcheck` e equivalência com a referência; `K=8`, `B=8` e janela de 16 frames cabem em 16 GB de GPU.

Sem GPU CUDA, o M1 fica `NOT_EVALUATED_NO_CUDA`. O critério não é avaliado em CPU."""

INSTALL = """# Colab already ships torch, numpy, pandas and matplotlib.
# No -U: upgrading pyarrow/requests breaks packages Colab pins (google-colab, cudf).
!pip -q install pyarrow psutil requests"""

CONFIG = r'''import gc, json, os, platform, shutil, sys, time, zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _env(name, default, cast=str):
    value = os.environ.get(name)
    return default if value in (None, "") else cast(value)


IN_COLAB = "google.colab" in sys.modules
SOURCE_DIGEST = "__SOURCE_DIGEST__"  # sha1 of the embedded meta_brain/*.py sources

# --- Parameters ---------------------------------------------------------------
# True: cache the 1.2 GB download and the graph cache on Google Drive (asks for Drive access).
USE_DRIVE = False  # @param {type:"boolean"}
STATUSES = ("Traced",)  # bodies kept from the annotations table
MIN_WEIGHT = 1.0  # minimum synapses per neuron pair
SHUFFLE_SEED = 0
MIN_NEURONS, MIN_EDGES = 100_000, 5_000_000  # full-brain guard
M0_CACHE_LOAD_TARGET_S = 120.0  # M0: the cached graph loads in under 2 minutes
M0_SHUFFLE_RESIDUAL_MAX = 1e-3  # M0: at most 0.1% of edges merged/dropped by a control
M1_MEMORY_TARGET_GIB = 16.0  # M1: K=8, B=8, 16-frame BPTT window fits in 16 GB
MICRO_STEPS = 8
BENCH = "full"  # "full" on a GPU; "tiny" is a smoke test

# Environment overrides (used by the repository's notebook test).
DATA_DIR = Path(_env("MB_DATA_DIR", "/content/malecns_v1_0"))
OUT_DIR = Path(_env("MB_OUT_DIR", "/content/meta_brain_m0_m1"))
OFFLINE = _env("MB_OFFLINE", "0") == "1"
MIN_NEURONS = _env("MB_MIN_NEURONS", MIN_NEURONS, int)
MIN_EDGES = _env("MB_MIN_EDGES", MIN_EDGES, int)
BENCH = _env("MB_BENCH", BENCH)
MICRO_STEPS = _env("MB_MICRO_STEPS", MICRO_STEPS, int)
DEVICE = torch.device(_env("MB_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
USE_DRIVE = USE_DRIVE and IN_COLAB and not OFFLINE
if DEVICE.type != "cuda" and BENCH == "full":
    print("No CUDA GPU: the M1 benchmark falls back to a tiny CPU grid and M1 is not evaluated.")
    BENCH = "tiny"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"

try:
    import psutil
    RAM_GIB = round(psutil.virtual_memory().total / 2**30, 1)
except ImportError:
    RAM_GIB = None
GPU_NAME = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None
GPU_GIB = (round(torch.cuda.get_device_properties(DEVICE).total_memory / 2**30, 1)
           if DEVICE.type == "cuda" else None)

RESULTS = {
    "experiment": "Meta-Brain M0/M1 infrastructure",
    "source_digest": SOURCE_DIGEST,
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": str(DEVICE), "gpu": GPU_NAME, "gpu_gib": GPU_GIB, "ram_gib": RAM_GIB,
                    "colab": IN_COLAB},
    "config": {"statuses": list(STATUSES), "min_weight": MIN_WEIGHT, "micro_steps": MICRO_STEPS,
               "bench": BENCH, "min_neurons": MIN_NEURONS, "min_edges": MIN_EDGES},
}
print(json.dumps(RESULTS["environment"], indent=2))'''

DRIVE = r'''# Optional: keep the 1.2 GB download and the graph cache on Google Drive between sessions.
DRIVE_DATA = None
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_DATA = Path("/content/drive/MyDrive/meta_brain/malecns_v1_0")
        DRIVE_DATA.mkdir(parents=True, exist_ok=True)
        CACHE_DIR = DRIVE_DATA.parent / "graph_cache"
        print("Drive cache:", DRIVE_DATA, "| graph cache:", CACHE_DIR)
    except Exception as exc:  # Drive is a convenience, never a requirement
        print("Drive not mounted:", type(exc).__name__, str(exc)[:120])
print("data:", DATA_DIR, "| graph cache:", CACHE_DIR, "| outputs:", OUT_DIR)'''

DATA_MD = """## 1. Dados oficiais do MaleCNS v1.0

São os três arquivos da *flat connectome* publicados em [male-cns.janelia.org/download](https://male-cns.janelia.org/download/). O de pesos tem ~152 milhões de linhas, uma por par de corpos conectados. A maioria envolve fragmentos não rastreados; com só os neurônios `Traced` sobram ~25,6 milhões de arestas."""

DOWNLOAD = r'''import requests

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]


def fetch(name):
    path = DATA_DIR / name
    if path.exists() and path.stat().st_size > 1000:
        return f"reusing {name} ({path.stat().st_size / 1e9:.2f} GB)"
    if DRIVE_DATA is not None and (DRIVE_DATA / name).exists():
        shutil.copy2(DRIVE_DATA / name, path)
        return f"copied {name} from Drive"
    tmp = path.with_suffix(".part")
    with requests.get(f"{BASE}/{name}", stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, last = 0, 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                if done - last > 200e6:
                    print(f"  {name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
                    last = done
    tmp.replace(path)
    if DRIVE_DATA is not None:
        shutil.copy2(path, DRIVE_DATA / name)
    return f"downloaded {name} ({path.stat().st_size / 1e9:.2f} GB)"


if OFFLINE:
    print("offline: using files already in", DATA_DIR)
else:
    t0 = time.time()
    for name in FILES:
        print(fetch(name), flush=True)
    print(f"data ready in {time.time() - t0:.0f}s")
print(sorted(p.name for p in DATA_DIR.glob("*.feather")))'''

LIB_MD = """## 2. Biblioteca embutida

As quatro células abaixo são cópias de `meta_brain/*.py` cujo hash está em `SOURCE_DIGEST`, sem os imports relativos do pacote. Não há nada para configurar nelas."""

M0_LOAD_MD = """## 3. M0: carga, cache e auditoria das arestas

A primeira execução monta o grafo a partir dos arquivos e grava o cache. Em seguida o notebook recarrega do cache para medir o tempo do critério. O relatório precisa fechar a conta: lidas = mantidas + descartadas + mescladas."""

M0_LOAD = r'''t0 = time.perf_counter()
graph = load_connectome(DATA_DIR, CACHE_DIR, min_weight=MIN_WEIGHT, statuses=STATUSES)
first_call_s = time.perf_counter() - t0
built_this_run = not graph.report.from_cache

t0 = time.perf_counter()
graph = load_connectome(DATA_DIR, CACHE_DIR, min_weight=MIN_WEIGHT, statuses=STATUSES, verbose=False)
cache_load_s = time.perf_counter() - t0

r = graph.report
if graph.n < MIN_NEURONS or graph.e < MIN_EDGES:
    raise RuntimeError(f"Full-brain guard failed: {graph.n:,} neurons, {graph.e:,} edges "
                       f"(need {MIN_NEURONS:,} / {MIN_EDGES:,}).")
if r.accounted_rows() != r.edge_rows_read:
    raise AssertionError("edge accounting does not close")

sign = graph.sign()
in_deg = np.bincount(graph.post, minlength=graph.n)
out_deg = np.bincount(graph.pre, minlength=graph.n)
RESULTS["m0_load"] = {
    "report": asdict(r),
    "built_this_run": built_this_run,
    "first_call_seconds": round(first_call_s, 2),
    "cache_load_seconds": round(cache_load_s, 2),
    "neurons": graph.n,
    "edges": graph.e,
    "sensory": int(len(graph.sensory_indices())),
    "motor": int(len(graph.motor_indices())),
    "descending": int(len(graph.descending_indices())),
    "inhibitory_neuron_fraction": round(float((sign < 0).mean()), 4),
    "inhibitory_edge_fraction": round(float((sign[graph.pre] < 0).mean()), 4),
    "mean_in_degree": round(float(in_deg.mean()), 2),
    "max_in_degree": int(in_deg.max()),
    "max_out_degree": int(out_deg.max()),
    "isolated_neurons": int(((in_deg == 0) & (out_deg == 0)).sum()),
}
print(f"\ncache load: {cache_load_s:.1f}s | first call: {first_call_s:.1f}s "
      f"({'built' if built_this_run else 'cached'})")
print(json.dumps({k: v for k, v in RESULTS["m0_load"].items() if k != "report"}, indent=2))'''

M0_PLOTS = r'''import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(17, 9))

ax = axes[0, 0]
acct = {
    "retained": r.edges_retained,
    "duplicate rows\nmerged": r.duplicate_rows_merged,
    "unknown pre": r.dropped_unknown_pre,
    "unknown post": r.dropped_unknown_post,
    "self-loops": r.dropped_self_loops,
    f"pair < {r.min_weight:g}": r.dropped_below_min_weight,
}
ax.barh(list(acct)[::-1], np.maximum(list(acct.values())[::-1], 1), color="#4878d0")
ax.set_xscale("log")
ax.set_title(f"Edge rows read: {r.edge_rows_read:,}")
for i, v in enumerate(list(acct.values())[::-1]):
    ax.text(max(v, 1), i, f" {v:,}", va="center", fontsize=8)

ax = axes[0, 1]
for deg, label in ((in_deg, "in-degree"), (out_deg, "out-degree")):
    vals, counts = np.unique(deg[deg > 0], return_counts=True)
    ax.loglog(vals, counts, ".", alpha=0.5, label=label)
ax.set_title("Degree distribution")
ax.set_xlabel("degree")
ax.set_ylabel("neurons")
ax.legend()

ax = axes[0, 2]
bins = np.logspace(0, np.log10(max(graph.count.max(), 10)), 50)
ax.hist(graph.count, bins=bins, color="#6acc64")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Synapses per neuron pair")
ax.set_xlabel("synapses")

ax = axes[1, 0]
nt = pd.Series(r.nt_counts).sort_values()
colors = ["#d65f5f" if DEFAULT_NT_SIGN.get(k, 1) < 0 else "#4878d0" for k in nt.index]
ax.barh(nt.index, nt.values, color=colors)
ax.set_xscale("log")
ax.set_title("Neurotransmitter (red = inhibitory sign)")

ax = axes[1, 1]
sc = pd.Series(graph.superclass_names)[graph.superclass].value_counts().head(12)[::-1]
ax.barh(sc.index, sc.values, color="#956cb4")
ax.set_xscale("log")
ax.set_title("Neurons per superclass")

ax = axes[1, 2]
type_sizes = np.bincount(graph.cell_type)
ax.hist(type_sizes, bins=np.logspace(0, np.log10(type_sizes.max() + 1), 40), color="#ee854a")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(f"Neurons per cell type ({graph.n_types:,} types, {r.n_type_pairs:,} pairs)")

fig.suptitle(f"MaleCNS v1.0 ({', '.join(STATUSES)}): {graph.n:,} neurons, {graph.e:,} edges")
fig.tight_layout()
fig.savefig(OUT_DIR / "m0_connectome.png", dpi=150)
plt.show()'''

SHUFFLE_MD = """## 4. M0: controles nulos

Cada controle reembaralha só o que precisa:

| Controle | Mantém | Destrói |
|---|---|---|
| `degree_preserving` | graus de entrada e saída e força de saída de cada neurônio | identidade dos parceiros, tabela tipo a tipo |
| `type_preserving` | tabela de conectividade tipo a tipo e graus por bloco (trocas de arestas dentro do bloco) | fiação individual (ex.: retinotopia) |
| `random_sparse` | N, E e o multiconjunto de pesos | todo o resto |

A reciprocidade e a sobreposição com o grafo original mostram quanto de estrutura cada controle remove."""

SHUFFLE = r'''def edge_key(pre, post, n):
    return post.astype(np.int64) * n + pre.astype(np.int64)


def fraction_present(keys, sorted_ref):
    i = np.minimum(np.searchsorted(sorted_ref, keys), len(sorted_ref) - 1)
    return float((sorted_ref[i] == keys).mean())


def type_table(pre, post, count):
    ep, pairs = type_pair_index(pre, post, graph.cell_type, graph.n_types)
    key = pairs[:, 0].astype(np.int64) * graph.n_types + pairs[:, 1]
    return key, np.bincount(ep), np.round(np.bincount(ep, count.astype(np.float64)), 3)


ref_keys = edge_key(graph.pre, graph.post, graph.n)  # CSR order: already sorted
ref_table = type_table(graph.pre, graph.post, graph.count)
ref_in, ref_out = in_deg, out_deg
ref_strength = np.bincount(graph.pre, graph.count.astype(np.float64), graph.n)


def describe(name, pre, post, count, report=None, seconds=0.0):
    keys = edge_key(pre, post, graph.n)
    table = type_table(pre, post, count)
    row = {
        "control": name,
        "edges": int(len(pre)),
        "seconds": round(seconds, 1),
        "reciprocity": round(fraction_present(edge_key(post, pre, graph.n), np.sort(keys)), 4),
        "overlap_with_original": round(fraction_present(keys, ref_keys), 4),
        "in_out_degree_preserved": bool(np.array_equal(np.bincount(post, minlength=graph.n), ref_in)
                                        and np.array_equal(np.bincount(pre, minlength=graph.n), ref_out)),
        "out_strength_preserved": bool(np.allclose(np.bincount(pre, count.astype(np.float64), graph.n),
                                                   ref_strength)),
        "type_table_preserved": bool(len(table[0]) == len(ref_table[0])
                                     and all(np.array_equal(a, b) for a, b in zip(table, ref_table))),
    }
    if report is not None:
        row.update(repair_rounds=report.repair_rounds, swap_sweeps=report.swap_sweeps,
                   swaps_accepted=report.swaps_accepted,
                   residual_self_loops_dropped=report.residual_self_loops_dropped,
                   residual_duplicates_merged=report.residual_duplicates_merged,
                   residual_fraction=round((report.residual_self_loops_dropped
                                            + report.residual_duplicates_merged) / report.edges_in, 6))
    return row


rows = [describe("original", graph.pre, graph.post, graph.count)]
controls = {
    "degree_preserving": lambda: degree_preserving_shuffle(graph.pre, graph.post, graph.count, graph.n,
                                                           seed=SHUFFLE_SEED),
    "type_preserving": lambda: type_preserving_shuffle(graph.pre, graph.post, graph.count, graph.n,
                                                       graph.cell_type, seed=SHUFFLE_SEED),
    "random_sparse": lambda: random_sparse_graph(graph.n, graph.e, graph.count, seed=SHUFFLE_SEED),
}
degree_samples = {}
for name, make in controls.items():
    t0 = time.perf_counter()
    (p, q, c), rep = make()
    rows.append(describe(name, p, q, c, rep, time.perf_counter() - t0))
    degree_samples[name] = np.bincount(q, minlength=graph.n)
    print(rows[-1], flush=True)
    del p, q, c
    gc.collect()

shuffles = pd.DataFrame(rows).set_index("control")
shuffles.to_csv(OUT_DIR / "m0_shuffles.csv")
RESULTS["m0_shuffles"] = shuffles.reset_index().to_dict(orient="records")
shuffles'''

SHUFFLE_PLOTS = r'''fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
names = list(shuffles.index)
palette = ["#4c4c4c", "#4878d0", "#6acc64", "#d65f5f"]

axes[0].bar(names, shuffles["reciprocity"], color=palette)
axes[0].set_title("Reciprocity (fraction of edges with a reverse edge)")
axes[1].bar(names, shuffles["overlap_with_original"], color=palette)
axes[1].set_title("Edges shared with the original graph")
for ax in axes[:2]:
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(ax.containers[0].datavalues):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

ax = axes[2]
rng = np.random.default_rng(0)
idx = rng.choice(graph.n, min(20_000, graph.n), replace=False)
for name, color in (("random_sparse", "#d65f5f"), ("degree_preserving", "#4878d0")):
    ax.scatter(ref_in[idx] + 1, degree_samples[name][idx] + 1, s=3, alpha=0.4, color=color, label=name)
lim = ref_in.max() + 2
ax.plot([1, lim], [1, lim], "k--", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("original in-degree + 1")
ax.set_ylabel("control in-degree + 1")
ax.set_title("In-degree: kept by the shuffle, not by a random graph")
ax.legend(markerscale=4)

fig.tight_layout()
fig.savefig(OUT_DIR / "m0_shuffles.png", dpi=150)
plt.show()
shuffles[["in_out_degree_preserved", "out_strength_preserved", "type_table_preserved"]]'''

M1_CHECK_MD = """## 5. M1: correção do `SparsePropagate`

1. `gradcheck` em float64 num grafo pequeno, em CPU e, se houver, em CUDA.
2. No **grafo completo**, saída e gradientes do operador customizado contra a referência por autograd (gather/scatter).
3. Memória guardada por chamada: o operador customizado guarda só `x` (N×B) e `values` (E). A referência guarda um tensor E×B."""

M1_CHECK = r'''checks = {}


def small_structure(device, n=40, e=300, seed=0):
    rng = np.random.default_rng(seed)
    p, q, _, _ = merge_duplicates(rng.integers(0, n, e), rng.integers(0, n, e), np.ones(e, np.float32), n)
    return SparseStructure.from_edges(p, q, n, device=device, edge_chunk=37)


for dev in ["cpu"] + (["cuda"] if DEVICE.type == "cuda" else []):
    s = small_structure(dev)
    x = torch.randn(s.n, 3, dtype=torch.float64, device=dev, requires_grad=True)
    v = torch.randn(s.e, dtype=torch.float64, device=dev, requires_grad=True)
    try:
        checks[f"gradcheck_{dev}"] = bool(torch.autograd.gradcheck(lambda a, b: sparse_propagate(a, b, s), (x, v)))
    except Exception as exc:
        checks[f"gradcheck_{dev}"] = False
        checks[f"gradcheck_{dev}_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

readout = graph.descending_indices()
if len(readout) == 0:
    readout = graph.motor_indices()
edge_pair, pair_types = graph.type_pair_index()
tensors = prepare_tensors(graph.n, graph.pre, graph.post, graph.count, sign, graph.cell_type,
                          graph.n_types, readout, DEVICE, edge_pair=edge_pair, n_pairs=len(pair_types))
S = tensors["structure"]

torch.manual_seed(0)
xb = torch.randn(S.n, 2, device=DEVICE, requires_grad=True)
vb = tensors["base"].clone().requires_grad_(True)
gy = torch.randn(S.n, 2, device=DEVICE)
outs = {}
for name, fn in (("custom", sparse_propagate), ("reference", reference_propagate)):
    y = fn(xb, vb, S)
    gx, gv = torch.autograd.grad(y, (xb, vb), gy)
    outs[name] = (y.detach(), gx, gv)
    del y


def rel_err(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


checks["full_graph_rel_err"] = {k: rel_err(outs["custom"][i], outs["reference"][i])
                                for i, k in enumerate(("y", "grad_x", "grad_values"))}
checks["full_graph_equivalent"] = all(e < 1e-4 for e in checks["full_graph_rel_err"].values())
del outs, gx, gv


def saved_numel(fn):
    sizes = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: sizes.append(t.numel()) or t, lambda t: t):
        fn()
    return sum(sizes)


xs = torch.randn(S.n, 8, device=DEVICE, requires_grad=True)
checks["saved_numel_per_call"] = {
    "custom": saved_numel(lambda: sparse_propagate(xs, vb, S)),
    "reference": saved_numel(lambda: reference_propagate(xs, vb, S)),
    "N*B": S.n * 8,
    "E*B": S.e * 8,
}
del xs, xb, vb, gy
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
RESULTS["m1_checks"] = checks
print(json.dumps(checks, indent=2))'''

M1_BENCH_MD = """## 6. M1: benchmark de BPTT pelo cérebro inteiro

A carga de trabalho é a da camada 1: dinâmica de taxa com vazamento, `tau` e `v0` por tipo, ganho por par de tipos, `K` micro-passos por frame e checkpoint por frame. Três grupos:

- **compare:** operador customizado contra a referência na mesma configuração.
- **batch_scaling:** janela de 16 frames com B de 1 a 16. A linha **B=8** é o critério do M1.
- **no_checkpoint:** quanto a memória cresce com o número de frames sem checkpoint.

Configurações que estouram a memória aparecem como `OOM`, o que também é um resultado."""

M1_BENCH = r'''if BENCH == "full":
    GRID = [
        ("compare", dict(op="custom", batch=8, frames=4, use_checkpoint=True)),
        ("compare", dict(op="reference", batch=8, frames=4, use_checkpoint=True)),
        *[("batch_scaling", dict(op="custom", batch=b, frames=16, use_checkpoint=True)) for b in (1, 4, 8, 16)],
        *[("no_checkpoint", dict(op="custom", batch=8, frames=f, use_checkpoint=False)) for f in (1, 2, 4)],
        ("no_checkpoint", dict(op="reference", batch=8, frames=2, use_checkpoint=False)),
    ]
    CRITERION = dict(group="batch_scaling", op="custom", batch=8, frames=16)
else:
    GRID = [
        ("compare", dict(op="custom", batch=2, frames=2, use_checkpoint=True)),
        ("compare", dict(op="reference", batch=2, frames=2, use_checkpoint=True)),
        *[("batch_scaling", dict(op="custom", batch=b, frames=2, use_checkpoint=True)) for b in (1, 2)],
        *[("no_checkpoint", dict(op="custom", batch=2, frames=f, use_checkpoint=False)) for f in (1, 2)],
    ]
    CRITERION = dict(group="batch_scaling", op="custom", batch=2, frames=2)

bench_rows = []
for group, cfg in GRID:
    t0 = time.perf_counter()
    row = bptt_workload(**tensors, micro_steps=MICRO_STEPS, iters=2, warmup=1, **cfg)
    row.update(group=group, wall_s=round(time.perf_counter() - t0, 1))
    bench_rows.append(row)
    print({k: row.get(k) for k in ("group", "op", "batch", "frames", "checkpoint", "status",
                                   "ms_per_micro_step", "peak_memory_gib", "wall_s")}, flush=True)

bench = pd.DataFrame(bench_rows)
bench.to_csv(OUT_DIR / "m1_benchmark.csv", index=False)
RESULTS["m1_benchmark"] = bench_rows
cols = ["group", "op", "batch", "frames", "checkpoint", "status", "forward_s", "backward_s",
        "ms_per_micro_step", "peak_memory_gib"]
bench[[c for c in cols if c in bench]]'''

M1_PLOTS = r'''ok = bench[bench["status"] == "OK"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
colors = {"custom": "#4878d0", "reference": "#d65f5f"}

cmp_rows = bench[bench["group"] == "compare"]
for ax, metric, title in ((axes[0], "ms_per_micro_step", "Forward + backward per micro-step (ms)"),
                          (axes[1], "peak_memory_gib", "Peak memory (GiB)")):
    vals = cmp_rows.get(metric, pd.Series(0.0, index=cmp_rows.index)).fillna(0).to_numpy()
    ax.bar(cmp_rows["op"], vals, color=[colors[o] for o in cmp_rows["op"]])
    for i, (v, st) in enumerate(zip(vals, cmp_rows["status"])):
        ax.text(i, v, f"{v:.2f}" if st == "OK" else st, ha="center", va="bottom")
    first = cmp_rows.iloc[0]
    ax.set_title(f"{title}\nB={first['batch']}, {first['frames']} frames, K={MICRO_STEPS}")

ax = axes[2]
bs = ok[(ok["group"] == "batch_scaling")]
ax.plot(bs["batch"], bs["peak_memory_gib"], "o-", color=colors["custom"], label="custom + checkpoint")
ax.axhline(M1_MEMORY_TARGET_GIB, color="k", ls="--", lw=1, label=f"M1 target {M1_MEMORY_TARGET_GIB:g} GiB")
if GPU_GIB:
    ax.axhline(GPU_GIB, color="gray", ls=":", lw=1, label=f"{GPU_NAME} {GPU_GIB} GiB")
ax.set_xlabel("batch")
ax.set_ylabel("peak GiB")
ax.set_title(f"Memory vs batch ({CRITERION['frames']}-frame window)")
ax.legend(fontsize=8)

ax = axes[3]
nc = bench[bench["group"] == "no_checkpoint"]
for op, sub in nc.groupby("op"):
    good = sub[sub["status"] == "OK"]
    ax.plot(good["frames"], good["peak_memory_gib"], "o-", color=colors[op], label=f"{op}, no checkpoint")
    for _, bad in sub[sub["status"] != "OK"].iterrows():
        y = bad.get("peak_memory_gib")
        ax.annotate(bad["status"], (bad["frames"], y if pd.notna(y) else 0), color=colors[op])
ax.set_xlabel("frames in the BPTT window")
ax.set_ylabel("peak GiB")
ax.set_title("Memory growth without checkpointing")
ax.legend(fontsize=8)

memory_note = ("" if DEVICE.type == "cuda"
               else "\nCPU: memory is the process-wide peak, so it does not compare configurations")
fig.suptitle(f"M1 on {GPU_NAME or 'CPU'}: {S.n:,} neurons, {S.e:,} edges, "
             f"{tensors['n_pairs']:,} type-pair parameters{memory_note}")
fig.tight_layout()
fig.savefig(OUT_DIR / "m1_benchmark.png", dpi=150)
plt.show()'''

FINAL_MD = """## 7. Critérios e artefatos

`m0_m1_results.json` é o registro da evidência. `claim_status` só marca `PASS` quando o critério foi medido e atendido."""

FINAL = r'''def find_row(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    return sel.iloc[0].to_dict() if len(sel) else None


controls_only = shuffles.drop(index="original")
m0 = {
    "edge_accounting_closes": r.accounted_rows() == r.edge_rows_read,
    "full_brain_guard": graph.n >= MIN_NEURONS and graph.e >= MIN_EDGES,
    "cache_load_seconds": RESULTS["m0_load"]["cache_load_seconds"],
    "cache_load_under_target": RESULTS["m0_load"]["cache_load_seconds"] < M0_CACHE_LOAD_TARGET_S,
    "degree_shuffle_invariants": bool(shuffles.loc["degree_preserving", "in_out_degree_preserved"]
                                      and shuffles.loc["degree_preserving", "out_strength_preserved"]),
    "type_shuffle_invariants": bool(shuffles.loc["type_preserving", "type_table_preserved"]),
    "max_residual_fraction": float(controls_only["residual_fraction"].max()),
}
m0["controls_exact_enough"] = m0["max_residual_fraction"] <= M0_SHUFFLE_RESIDUAL_MAX
m0_pass = all(v for k, v in m0.items() if isinstance(v, bool))

crit = find_row(bench, **CRITERION)
m1 = {
    "gradcheck": all(v for k, v in checks.items() if k.startswith("gradcheck_") and isinstance(v, bool)),
    "full_graph_equivalent": checks["full_graph_equivalent"],
    "criterion_config": CRITERION | {"micro_steps": MICRO_STEPS},
    "criterion_status": crit and crit["status"],
    "criterion_peak_gib": crit and crit.get("peak_memory_gib"),
    "criterion_ms_per_micro_step": crit and crit.get("ms_per_micro_step"),
}
cmp_ok = bench[(bench["group"] == "compare") & (bench["status"] == "OK")].set_index("op")
if {"custom", "reference"} <= set(cmp_ok.index):
    m1["speedup_vs_reference"] = round(cmp_ok.loc["reference", "ms_per_micro_step"]
                                       / cmp_ok.loc["custom", "ms_per_micro_step"], 2)
    m1["memory_ratio_vs_reference"] = round(cmp_ok.loc["reference", "peak_memory_gib"]
                                            / cmp_ok.loc["custom", "peak_memory_gib"], 2)
m1_fits = bool(crit and crit["status"] == "OK" and crit["peak_memory_gib"] < M1_MEMORY_TARGET_GIB)
if DEVICE.type != "cuda" or BENCH != "full":
    m1_status = "NOT_EVALUATED_NO_CUDA" if DEVICE.type != "cuda" else "NOT_EVALUATED_TINY_GRID"
else:
    m1_status = "PASS" if (m1["gradcheck"] and m1["full_graph_equivalent"] and m1_fits) else "FAIL"

RESULTS["criteria"] = {"m0": m0, "m1": m1}
RESULTS["claim_status"] = {"M0": "PASS" if m0_pass else "FAIL", "M1": m1_status}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (Path, torch.device)):
        return str(o)
    raise TypeError(type(o).__name__)


(OUT_DIR / "m0_m1_results.json").write_text(json.dumps(RESULTS, indent=2, default=_jsonable), encoding="utf-8")
archive = OUT_DIR.parent / f"{OUT_DIR.name}.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT_DIR.iterdir()):
        z.write(p, p.name)
if DRIVE_DATA is not None:
    shutil.copy2(archive, DRIVE_DATA.parent / archive.name)

print(json.dumps(RESULTS["claim_status"], indent=2))
print(json.dumps(RESULTS["criteria"], indent=2, default=_jsonable))
print("\noutputs:", *sorted(p.name for p in OUT_DIR.iterdir()), archive.name, sep="\n  ")
try:
    from IPython.display import Image, display
    for name in ("m0_connectome.png", "m0_shuffles.png", "m1_benchmark.png"):
        display(Image(filename=str(OUT_DIR / name)))
except ImportError:
    pass'''


def source_digest() -> str:
    h = hashlib.sha1()
    for name in MODULES:
        h.update((ROOT / "meta_brain" / f"{name}.py").read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def cells(digest: str):
    yield "markdown", INTRO
    yield "code", INSTALL
    yield "code", CONFIG.replace("__SOURCE_DIGEST__", digest)
    yield "code", DRIVE
    yield "markdown", DATA_MD
    yield "code", DOWNLOAD
    yield "markdown", LIB_MD
    for name in MODULES:
        yield "code", module_source(name)
    yield "markdown", M0_LOAD_MD
    yield "code", M0_LOAD
    yield "code", M0_PLOTS
    yield "markdown", SHUFFLE_MD
    yield "code", SHUFFLE
    yield "code", SHUFFLE_PLOTS
    yield "markdown", M1_CHECK_MD
    yield "code", M1_CHECK
    yield "markdown", M1_BENCH_MD
    yield "code", M1_BENCH
    yield "code", M1_PLOTS
    yield "markdown", FINAL_MD
    yield "code", FINAL


def build(digest: str) -> dict:
    out = []
    for i, (kind, src) in enumerate(cells(digest)):
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def code_cells(nb: dict) -> list:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def sync_embedded(nb: dict) -> bool:
    """Refresh every ``# --- embedded from meta_brain/<name>.py ---`` cell in place."""
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        match = _EMBED_HEADER.match("".join(cell["source"]))
        if match:
            lines = module_source(match.group(1)).splitlines(keepends=True)
            if cell["source"] != lines:
                cell["source"] = lines
                changed = True
    return changed


def write_notebook(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    digest = source_digest()
    nb = build(digest)
    write_notebook(NOTEBOOK, nb)
    print(f"wrote {NOTEBOOK.relative_to(ROOT)} ({len(nb['cells'])} cells, source digest {digest})")
    for path in SYNCED_NOTEBOOKS:
        other = json.loads(path.read_text(encoding="utf-8"))
        if sync_embedded(other):
            write_notebook(path, other)
            print(f"synced embedded modules in {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
