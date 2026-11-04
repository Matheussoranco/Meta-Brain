"""Shared Colab/CLI preparation; no full dataset or GPU is needed to import it."""
from __future__ import annotations

import gc
import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .graph import load_connectome
from .mario_rl import make_mario_env
from .mario_trainable import (FullBrainConfig, TrainableBrain, atomic_json, graph_fingerprint,
                              preflight, reservoir_weights, run_full_brain)
from .sparse_ops import SparseStructure


DATA_FILES = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)
DATA_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"


def fetch_data(data_dir, persistent_dir=None, log=print):
    """Stream each download, check length and Feather readability before atomic promotion."""
    import requests
    import pyarrow.feather as feather

    local = Path(data_dir)
    local.mkdir(parents=True, exist_ok=True)
    persistent = Path(persistent_dir) if persistent_dir is not None else None
    if persistent is not None:
        persistent.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        path = local / name
        if path.exists():
            # memory_map avoids loading the ~1 GB file just to verify its schema.
            feather.read_table(path, columns=[], memory_map=True)
            continue
        tmp = path.with_name(path.name + ".part")
        if persistent is not None and (persistent / name).exists():
            log(f"Copying {name} from Drive")
            shutil.copyfile(persistent / name, tmp)
        else:
            log(f"Downloading {name}")
            with requests.get(f"{DATA_BASE}/{name}", stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("content-length", 0))
                written = 0
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(8 << 20):
                        written += f.write(chunk)
                if expected and written != expected:
                    raise IOError(f"Incomplete download: {name}: {written}/{expected} bytes")
        feather.read_table(tmp, columns=[], memory_map=True)
        tmp.replace(path)
        if persistent is not None and not (persistent / name).exists():
            dest_tmp = persistent / (name + ".part")
            shutil.copyfile(path, dest_tmp)
            dest_tmp.replace(persistent / name)


def runtime_info(device):
    import importlib.metadata
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_gib": torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else None,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pyarrow", "gymnasium", "gym-super-mario-bros", "nes-py")}}


def prepare_brain(data_dir, cache_dir, out_dir, cfg, device="cuda", min_neurons=100_000,
                  min_edges=5_000_000, auto_memory=True, log=print):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab, then run all cells again")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for playable H.264 videos; run the installation cell")
    # The real emulator is tested BEFORE the expensive graph build.
    env, actions = make_mario_env()
    try:
        env.reset()
        env.step(1)
    finally:
        env.close()
    graph = load_connectome(data_dir, cache_dir=cache_dir)
    if graph.n < min_neurons or graph.e < min_edges:
        raise RuntimeError(f"Full-brain guard failed: {graph.n:,} neurons and {graph.e:,} edges")
    visual = np.flatnonzero(graph.superclass_mask(("ol_sensory", "olsn")))
    motor = graph.motor_indices()
    if not len(visual):
        raise ValueError("No optic sensory cells found; refusing to inject images into nonvisual senses")
    graph_id = graph_fingerprint(graph)
    info = {"neurons": graph.n, "edges": graph.e, "visual_sensory": len(visual), "motor": len(motor),
            "graph_id": graph_id, "load_report": asdict(graph.report)}
    log(graph.report.summary())
    # Existing runs keep their original batch, BPTT and architecture across GPU changes.
    latest = Path(out_dir) / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        hours, updates, final_episodes = cfg.max_hours, cfg.updates, cfg.final_episodes
        requested_vision, requested_seed = cfg.vision, cfg.seed
        cfg = FullBrainConfig(**saved["config"])
        if cfg.vision != requested_vision or cfg.seed != requested_seed:
            raise ValueError("Run folder contains another vision/seed; choose a new run name")
        cfg.max_hours, cfg.updates, cfg.final_episodes = hours, updates, final_episodes
        auto_memory = False
    weights = reservoir_weights(graph)
    structure = SparseStructure.from_edges(graph.pre, graph.post, graph.n, device=device, index_dtype=torch.int32)
    types = graph.cell_type.copy()
    del graph
    gc.collect()
    while True:
        torch.manual_seed(cfg.seed)
        model = None
        try:
            model = TrainableBrain(structure, weights, visual, motor, types, len(actions), cfg)
            check = preflight(model, cfg)
            break
        except torch.cuda.OutOfMemoryError:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if not auto_memory or (cfg.n_envs == 1 and cfg.bptt <= 4):
                raise RuntimeError("Full graph/BPTT does not fit this GPU. Use more VRAM, or a new run with smaller batch/BPTT") from None
            if cfg.n_envs > 1:
                cfg.n_envs = max(1, cfg.n_envs // 2)
            else:
                cfg.bptt = max(4, cfg.bptt // 2)
            log(f"Retrying preflight with {cfg.n_envs} environments and BPTT={cfg.bptt}")
    atomic_json(Path(out_dir) / "preflight.json", {"graph": info, "memory_and_gradients": check, "config": asdict(cfg)})
    log(f"Preflight OK: {json.dumps(check)}")
    return model, cfg, info


def launch(data_dir, cache_dir, out_dir, cfg=None, device="cuda", min_neurons=100_000,
           min_edges=5_000_000, auto_memory=True, log=print):
    cfg = cfg or FullBrainConfig()
    model, cfg, info = prepare_brain(data_dir, cache_dir, out_dir, cfg, device, min_neurons, min_edges, auto_memory, log)
    try:
        results = run_full_brain(model, lambda: make_mario_env()[0], cfg, out_dir, info["graph_id"], log=log)
        results["graph"] = info
        atomic_json(Path(out_dir) / "results.json", results)
        return results
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
