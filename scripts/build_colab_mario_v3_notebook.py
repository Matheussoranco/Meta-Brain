"""Generate a self-contained Colab Pro notebook from the tested package sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_full_brain_mario_v3.ipynb"
MODULES = ("graph", "sparse_ops", "mario_rl", "mario_trainable", "colab_mario")

INTRO = """# Meta-Brain V3 — conectoma completo treinável no Colab Pro

Escolha **Ambiente de execução → Alterar tipo de ambiente → GPU**. Ative **RAM alta** se disponível. Execute as células em ordem. Não é necessário clonar o GitHub: o código está embutido neste notebook.

Esta versão treina a visão, ganhos/constantes por tipo de neurônio e a decisão motora com PPO recorrente. Todas as ações passam pelo conectoma completo `Traced`; não há atalho da imagem para a política. A topologia e os sinais sinápticos permanecem fixos. A V2 continua sendo a referência com reservatório congelado.

**Persistência:** autorize o Drive na célula 2. Dados/cache são reutilizados; `latest.pt` permite continuar o treinamento em outra sessão. Use o mesmo `RUN_NAME`, `VISION` e `SEED` e execute tudo novamente. A retomada restaura pesos, otimizador e contadores; os episódios em andamento recomeçam. Não é reprodução bit a bit de uma sessão ininterrupta.

**Limites:** Colab Pro não garante uma GPU específica nem duração fixa. O orçamento abaixo limita a sessão de treinamento e salva antes da avaliação final; download, preparação e avaliação levam tempo adicional. Nenhum resultado de vitória está pré-calculado. O sucesso só é marcado se todos os episódios finais determinísticos alcançarem a bandeira sem morrer.

Fonte: [limites oficiais do Colab](https://research.google.com/colaboratory/faq.html)."""

INSTALL = """# Não reinstala PyTorch nem faz downgrade de NumPy do Colab.
# Pacotes modernos usam Gymnasium e funcionam com NumPy 2.
%pip -q install gym-super-mario-bros==9.1.0 nes-py==9.0.1 'gymnasium>=1.0,<2' pyarrow requests
import shutil, subprocess
if shutil.which('ffmpeg') is None:
    subprocess.run(['apt-get', '-qq', 'update'], check=True)
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], check=True)
"""

CONFIG = """import os, sys, json, gc
from pathlib import Path
import torch

USE_DRIVE = True  # @param {type:"boolean"}
RUN_NAME = 'mario_v3'  # @param {type:"string"}
VISION = 'real'  # @param ["real", "zero", "clock"]
SEED = 123  # @param {type:"integer"}
TARGET_UPDATES = 10000  # @param {type:"integer"}
SESSION_HOURS = 3.0  # @param {type:"number"}
FINAL_EPISODES = 20  # @param {type:"integer"}
N_ENVS = 4  # @param {type:"integer"}
BPTT = 16  # @param {type:"integer"}
MICRO_STEPS = 4  # @param {type:"integer"}

# Overrides for automated offline smoke tests only.
OFFLINE = os.environ.get('MB_OFFLINE') == '1'
DEVICE = os.environ.get('MB_DEVICE', 'cuda')
if DEVICE != 'cuda' and not OFFLINE:
    raise RuntimeError('O experimento completo exige GPU CUDA. Selecione GPU no menu do Colab.')
if DEVICE == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('GPU não está ativa. Altere o tipo de ambiente e execute novamente.')
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in ('.', '..'):
    raise ValueError('RUN_NAME deve ser apenas um nome de pasta, sem barras.')
if not USE_DRIVE and not OFFLINE:
    print('USE_DRIVE=False: resultados só durarão enquanto esta sessão existir.')
ROOT = Path('/content/meta_brain_v3')
if USE_DRIVE and not OFFLINE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/Meta-Brain')
RUN_DIR = Path(os.environ.get('MB_OUT_DIR', str(ROOT / 'runs' / f'{RUN_NAME}_{VISION}_seed{SEED}')))
DATA_DIR = Path(os.environ.get('MB_DATA_DIR', '/content/malecns_v1_0'))
CACHE_DIR = ROOT / 'graph_cache'
PERSISTENT_DATA = ROOT / 'data' if USE_DRIVE and not OFFLINE else None
if OFFLINE:
    CACHE_DIR = DATA_DIR / 'cache'
RUN_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = Path(os.environ.get('MB_V3_RUNTIME', '/content/meta_brain_v3_runtime'))
print('Resultados e checkpoints:', RUN_DIR)
"""

PREPARE = """from meta_brain.colab_mario import fetch_data, prepare_brain, runtime_info
from meta_brain.mario_trainable import FullBrainConfig, atomic_json, run_full_brain
from meta_brain.mario_rl import make_mario_env

cfg = FullBrainConfig(seed=SEED, vision=VISION, updates=TARGET_UPDATES, max_hours=SESSION_HOURS,
                      final_episodes=FINAL_EPISODES, n_envs=N_ENVS, bptt=BPTT, micro_steps=MICRO_STEPS)
if OFFLINE:
    cfg.n_envs = int(os.environ.get('MB_MARIO_ENVS', 2))
    cfg.rollout, cfg.bptt = 8, 4
    cfg.micro_steps = int(os.environ.get('MB_MARIO_MICRO_STEPS', 2))
    cfg.updates = int(os.environ.get('MB_MARIO_UPDATES', 1))
    cfg.epochs, cfg.max_steps, cfg.noop_max = 1, 12, 0
    cfg.eval_every = cfg.checkpoint_every = 1
    cfg.validation_episodes = cfg.final_episodes = 1
    cfg.max_hours = 0
cfg.validate()
if not OFFLINE:
    # Fail fast on the emulator before downloading 1.2 GB of connectome data.
    probe, _ = make_mario_env()
    try:
        observation, info = probe.reset(seed=SEED)
        probe.step(1)
        print('Emulador real OK:', observation.shape)
    finally:
        probe.close()
    fetch_data(DATA_DIR, PERSISTENT_DATA)
    atomic_json(RUN_DIR / 'runtime.json', runtime_info(torch.device(DEVICE)))
MIN_NEURONS = int(os.environ.get('MB_MIN_NEURONS', 100000))
MIN_EDGES = int(os.environ.get('MB_MIN_EDGES', 5000000))
model, cfg, GRAPH_INFO = prepare_brain(DATA_DIR, CACHE_DIR, RUN_DIR, cfg, DEVICE,
                                      MIN_NEURONS, MIN_EDGES, auto_memory=True)
print('Configuração efetiva:', cfg)
print('Grafo completo:', GRAPH_INFO['neurons'], 'neurônios;', GRAPH_INFO['edges'], 'arestas')
"""

TRAIN = """# Salva latest.pt a cada 10 updates e best.pt nas avaliações de validação.
# Interromper/perder a sessão mantém o último checkpoint completo no Drive.
RESULTS = run_full_brain(model, lambda: make_mario_env()[0], cfg, RUN_DIR, GRAPH_INFO['graph_id'])
RESULTS['graph'] = GRAPH_INFO
atomic_json(RUN_DIR / 'results.json', RESULTS)
print('Resultado:', RESULTS['claim_status'])
print('Vitórias (determinístico):', RESULTS['post_eval']['success_rate'])
print('Vitórias (amostrado):', RESULTS['post_eval_sampled']['success_rate'])
"""

DISPLAY = """import matplotlib.pyplot as plt
from IPython.display import Video, display
history = RESULTS['history']
if history:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r['agent_steps'] for r in history], [r['mean_max_x'] for r in history])
    axes[0].set_title('Progresso médio nos episódios encerrados')
    axes[0].set_xlabel('Decisões acumuladas'); axes[0].set_ylabel('Maior posição x')
    axes[1].plot([r['agent_steps'] for r in history], [r.get('entropy') for r in history])
    axes[1].set_title('Exploração da política'); axes[1].set_xlabel('Decisões acumuladas')
    fig.tight_layout(); fig.savefig(RUN_DIR / 'training.png', dpi=140); plt.show()
for name in ('post_greedy.mp4', 'post_sampled.mp4'):
    print(name, '— melhor episódio, não a média')
    display(Video(str(RUN_DIR / name), embed=True))
print('Arquivos:', *sorted(p.name for p in RUN_DIR.iterdir()), sep='\\n  ')
"""


def build():
    files = {f"meta_brain/{m}.py": (ROOT / "meta_brain" / f"{m}.py").read_text(encoding="utf-8")
             for m in MODULES}
    files["meta_brain/__init__.py"] = '"""Self-contained Meta-Brain Colab runtime."""\n'
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    source = ("# Exact package sources; regenerated, not downloaded from a mutable branch.\n"
              f"SOURCE_DIGEST = {digest!r}\nSOURCE_FILES = {files!r}\n"
              "for relative, content in SOURCE_FILES.items():\n"
              "    path = RUNTIME_DIR / relative\n"
              "    path.parent.mkdir(parents=True, exist_ok=True)\n"
              "    path.write_text(content, encoding='utf-8')\n"
              "sys.path.insert(0, str(RUNTIME_DIR))\n"
              "# Re-running all cells uses the newly embedded code, even after edits.\n"
              "for name in list(sys.modules):\n"
              "    if name == 'meta_brain' or name.startswith('meta_brain.'):\n"
              "        del sys.modules[name]\n"
              "print('Código embutido:', SOURCE_DIGEST[:12])\n")
    parts = [("markdown", INTRO), ("markdown", "## 1. Instalação"), ("code", INSTALL),
             ("markdown", "## 2. Drive e parâmetros"), ("code", CONFIG),
             ("markdown", "## 3. Código autocontido"), ("code", source),
             ("markdown", "## 4. Dados, auditoria e teste de memória\n\nO pré-teste calcula gradientes no grafo completo. Se faltar memória, reduz o lote; a escala do conectoma é preservada. Em uma retomada, conserva a configuração original. O primeiro carregamento exige mais RAM que os seguintes."),
             ("code", PREPARE), ("markdown", "## 5. Treinar ou continuar\n\nPara continuar na próxima sessão, mantenha o nome e execute todas as células. Aumente `TARGET_UPDATES` para estender o treino. A avaliação escolhe o melhor checkpoint em inícios de validação e testa em outros inícios. Os controles `zero` e `clock` usam pastas e treinamentos independentes; compare o mesmo número de updates, lote e sementes."),
             ("code", TRAIN), ("markdown", "## 6. Resultados e vídeos"), ("code", DISPLAY),
             ("markdown", "## Interpretação e retomada\n\n`latest.pt`: continuar o treino. `best.pt`: melhor checkpoint de validação. `preflight.json`: escala real, memória e gradiente visual. `results.json`: avaliações finais completas. Os vídeos mostram apenas o melhor episódio. Uma taxa de sucesso zero é uma execução válida sem vitória demonstrada.\n\nApós desconexão, selecione GPU novamente e execute tudo com os mesmos parâmetros. Não apague a pasta do Drive. Caso a GPU da nova sessão seja menor e a configuração salva não caiba, obtenha uma GPU com mais memória; não altere silenciosamente a configuração de um experimento em andamento.\n\nA V3 é uma nova hipótese de arquitetura. Compare vários seeds e controles antes de atribuir resultados à topologia biológica. Não equivale a uma reprodução biológica completa, nem a um treino independente de cada sinapse.")]
    cells = []
    for i, (kind, text) in enumerate(parts):
        cell = {"cell_type": kind, "id": f"v3-{i:02d}", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "meta_brain_source_sha256": digest}}


def main():
    NOTEBOOK.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Generated {NOTEBOOK}")


if __name__ == "__main__":
    main()
