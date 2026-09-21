# Meta-Brain: Metalearning on Drosophila Connectome for Mario Bros

> **I.S.A.A.C. Project** — Reducing the distance between intention and execution.

An experimental framework for metalearning on the *Drosophila melanogaster* male central nervous system connectome (MaleCNS v1.0), applying the fly's actual synaptic wiring to learn to play Super Mario Bros.

## Overview

This project implements a novel approach to biologically-inspired AI:

1. **Connectome as Architecture**: Uses the complete *Drosophila* male CNS connectome (166,700 neurons, ~6M synapses) as the structural scaffold for a spiking neural network
2. **Metalearning for Adaptation**: Implements MAML, Reptile, and plasticity-based learning to rapidly adapt the fixed connectome structure to the Mario task
3. **Sensorimotor Grounding**: Maps Mario's visual/game state to the fly's sensory neurons (optic lobe, olfactory, auditory, mechanosensory) and reads out from motor pathways (descending neurons, VNC motor circuits)

## Scientific Foundation

Based on **Berg et al. (2026)** — *Sexual dimorphism in the complete Drosophila male central nervous system connectome* (Cell, 189, 5504–5526)

Key connectome features leveraged:
- **Complete sensorimotor pathways**: From sensory neurons → interneurons → descending neurons → VNC motor neurons
- **Dimorphic hotspots**: Circuits like LoVP92 → AOTU008 → DNg13 that integrate visual information for sex-specific behaviors
- **Recurrent loops**: Auditory feedback circuits (song detection ↔ production) as models for sensorimotor integration
- **Neuromodulatory systems**: Dopaminergic (reward), octopaminergic (arousal), serotonergic (behavioral state) for plasticity gating

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MARIO ENVIRONMENT                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Visual     │  │ Proprio-    │  │ Reward/     │             │
│  │  (frame)    │  │ ception     │  │ Punishment  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SENSORY ENCODING                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ OLSN         │ │ Mechanosen-  │ │ DAN/OAN      │            │
│  │ (optic lobe) │ │ sory (GRN)   │ │ (reward)     │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│              DROSOPHILA CONNECTOME SNN                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Sensory → Interneurons (MB, CX, LH) → Descending → Motor │  │
│  │  166K neurons, 6M synapses, STDP + neuromodulation       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MOTOR DECODING                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ DNg13        │ │ DNg31        │ │ Wing         │            │
│  │ (turning)    │ │ (walking)    │ │ premotor     │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MARIO ACTIONS                                │
│  [RIGHT, JUMP, RIGHT+JUMP, LEFT, ...]                          │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone and enter directory
cd C:/Users/mathe/Documents/Code/Meta-Brain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Connectome Data

Download the MaleCNS v1.0 data from Janelia:

```bash
# Option 1: Local files (recommended for large dataset)
# Download from https://male-cns.janelia.org/download/
# Place in data/connectome/

# Option 2: Programmatic access via neuPrint (requires token)
# Get token from https://neuprint.janelia.org (Account → Token)
export NEUPRINT_TOKEN="your_token_here"
```

See `data/connectome/README.md` for detailed download instructions.

## Usage

### Quick Start: Random Baseline
```bash
meta-brain --mode baseline
```

### Training with STDP Plasticity (Recommended)
```bash
meta-brain --mode plasticity
```

### Training with MAML (Gradient-based Metalearning)
```bash
meta-brain --mode maml
```

### Resume from Checkpoint
```bash
meta-brain --mode plasticity --checkpoint checkpoint_latest.pt
```

## Configuration

All parameters in `config/config.yaml`:

- **Connectome**: Dataset version, neuPrint settings, neuron type selections
- **Simulator**: Brian2 (spiking) or rate-coded, timestep, neuron/synapse parameters
- **Environment**: Mario action space, frame preprocessing, reward shaping
- **Metalearning**: Algorithm (MAML/Reptile), learning rates, inner steps, meta-batch size
- **Training**: Curriculum stages, logging, checkpointing, device selection

## Project Structure

```
Meta-Brain/
├── config/
│   └── config.yaml              # All hyperparameters
├── connectome/
│   └── loader.py                # Connectome loading & graph construction
├── simulator/
│   └── snn.py                   # Brian2 SNN & rate-coded SNN
├── environment/
│   └── mario_env.py             # Mario Bros wrapper & encoding
├── metalearning/
│   └── algorithms.py            # MAML, Reptile, ANIL, differentiable SNN
├── training/
│   └── train.py                 # Main training pipeline
├── data/
│   └── connectome/              # Connectome data files (gitignored)
├── checkpoints/                 # Model checkpoints (gitignored)
├── logs/                        # Training logs (gitignored)
├── results/                     # Evaluation results (gitignored)
├── notebooks/                   # Jupyter notebooks for analysis
├── scripts/                     # Utility scripts
├── tests/                       # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Key Concepts

### Connectome-to-SNN Mapping

| Connectome Element | SNN Implementation |
|---|---|
| Neuron | AdEx/Izhikevich model (Brian2) or rate unit |
| Synapse | Conductance-based, Dale's law enforced |
| Neurotransmitter | ACh=exc, GABA/Glu=inh, DA/OA/5HT=modulatory |
| STDP | Pair-based, neuromodulator-gated |
| Sensory neurons | Poisson input encoding game state |
| Motor neurons | Rate readout → discrete action mapping |

### Metalearning Tasks

1. **Basic Movement**: Move right, survive
2. **Enemy Avoidance**: Jump over Goombas, avoid pits
3. **Jumping**: Platform navigation
4. **Level Completion**: Reach flag pole

Each task has a custom reward function. Metalearning optimizes *initial synaptic weights* for fast within-episode adaptation via STDP.

### Curriculum Learning

Progressive task difficulty:
```
Stage 1 (1K ep): Random exploration → collect initial data
Stage 2 (5K ep): Simple navigation → move right
Stage 3 (10K ep): Enemy avoidance → jump timing
Stage 4 (20K ep): Level completion → flag pole
Stage 5 (50K ep): Speedrun → optimize time
```

## Research Directions

This framework enables investigating:

- **Structure-function relationships**: How does connectome topology constrain learnable behaviors?
- **Dimorphism & specialization**: Do male-specific circuits (pC1, LoVP92 hotspots) confer advantages for specific task types?
- **Neuromodulatory gating**: Can dopamine/octopamine signals from the connectome implement credit assignment?
- **Metalearning in biological networks**: Does MAML on connectome structure discover plasticity rules that match biology?
- **Sensorimotor integration**: How do recurrent loops (auditory feedback, visual-motor) support closed-loop control?

## Expected Challenges

| Challenge | Mitigation |
|---|---|
| 166K neurons too large for full simulation | Sensorimotor subgraph extraction (3-5K neurons) |
| Brian2 on Windows/CUDA issues | Rate-coded differentiable SNN as alternative |
| Sparse rewards in Mario | Reward shaping, curriculum, intrinsic motivation |
| Sim-to-real gap (connectome vs behavior) | Focus on *circuit motifs* not whole-brain simulation |
| Credit assignment in deep SNN | Neuromodulatory STDP, surrogate gradients |

## Citation

If you use this framework, please cite:

```bibtex
@article{berg2026malecns,
  title={Sexual dimorphism in the complete Drosophila male central nervous system connectome},
  author={Berg, Stuart and Beckett, Isabella R and Costa, Marta and Schlegel, Philipp and others},
  journal={Cell},
  volume={189},
  pages={5504--5526},
  year={2026},
  doi={10.1016/j.cell.2026.08.015}
}
```

```bibtex
@software{meta-brain,
  title={Meta-Brain: Metalearning on Drosophila Connectome for Mario Bros},
  author={Soranço, Matheus},
  year={2026},
  url={https://github.com/matheuss/meta-brain}
}
```

## License

MIT License — see LICENSE file.

## Acknowledgments

- **Janelia FlyEM Team** for the MaleCNS connectome
- **FlyWire Consortium** for female brain connectome (comparison)
- **NeuPrint** team for programmatic access
- **Nous Research** for Hermes Agent infrastructure

---

*Built with I.S.A.A.C. — Intelligent Synthetic Autonomous Agent Companion*