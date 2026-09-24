# Meta-Brain

## Full-connectome-inspired reinforcement learning with the *Drosophila melanogaster* connectome

Meta-Brain is an experimental framework for using the male *Drosophila melanogaster* central nervous system connectome as a sparse recurrent computational substrate for embodied learning.

The project maps visual game observations into sensory neurons, propagates activity through a connectome-derived network, and decodes motor activity into discrete actions. The current demonstration uses Super Mario Bros. and reinforcement learning.

This project does not claim to reproduce the human brain, consciousness, or general intelligence. It tests a narrower and measurable question:

> Can a digitally instantiated biological connectome support task-specific adaptive behavior?

## Current status

The project has two experimental tracks:

1. **Corridor control**
   - Sensorimotor connectome subgraph.
   - 40,000 neurons.
   - 1,199,959 synaptic edges.
   - Closed-loop reinforcement learning.
   - Reward improved from `-2.08` before training to `7.13` after training.
   - Post-training success rate: `25%`.

2. **Full-connectome Mario control**
   - MaleCNS metadata loaded with `211,577` neurons.
   - Full-scale sparse recurrent execution on CUDA.
   - Visual input from game frames.
   - Motor readout trained with policy-gradient reinforcement learning.
   - Some training episodes reached approximately `x=1,800` in World 1-1.
   - Final deterministic evaluation reached approximately `x=314` on average.
   - Final evaluation did not yet reach the flag: `success_rate=0`, `flag_get=false`, and all five evaluation episodes ended in death.

The Mario result should therefore be described as successful partial behavioral learning, not yet as robust level completion.

## Demonstration artifacts

The repository can include the following experiment artifacts:

- [Pre-training Mario video](assets/full_brain_mario_v2_pre.mp4)
- [Post-training Mario video](assets/full_brain_mario_v2_post.mp4)
- [Training curves](assets/full_brain_mario_v2_curves.png)
- [Training history](assets/full_brain_mario_v2_training.csv)
- [Evaluation results](assets/full_brain_mario_v2_results.json)
- [Motor readout checkpoint](assets/full_brain_mario_v2_readout.pt)
- [Self-contained Colab notebook](notebooks/colab_full_brain_mario_v2.ipynb)

The videos show the best recorded evaluation episode, selected by:

1. successful completion without death;
2. greatest horizontal progress;
3. highest reward.

## System architecture

```text
+---------------------------+
| Super Mario Bros.         |
| RGB game frames           |
+-------------+-------------+
              |
              v
+---------------------------+
| Visual encoding           |
| 16x16 grayscale image    |
| + frame difference        |
| 512-dimensional code      |
+-------------+-------------+
              |
              v
+---------------------------+
| Connectome input mapping  |
| Fixed projection into     |
| identified sensory cells |
+-------------+-------------+
              |
              v
+---------------------------+
| Sparse recurrent brain    |
| Connectome-derived graph  |
| Leaky nonlinear dynamics  |
| CUDA sparse propagation   |
+-------------+-------------+
              |
              v
+---------------------------+
| Motor readout             |
| Descending / motor cells  |
| Trainable policy logits   |
+-------------+-------------+
              |
              v
+---------------------------+
| Discrete Mario actions    |
| RIGHT_ONLY action space   |
+---------------------------+
```

## Connectome representation

The MaleCNS data is represented as:

- neuron metadata;
- presynaptic neuron indices;
- postsynaptic neuron indices;
- weighted synaptic connections;
- sensory neuron index set;
- motor neuron index set.

The model keeps the connectome sparse. It does not create a dense matrix with shape `N x N`.

For each decision step:

```text
sensory drive -> sparse recurrent propagation -> motor activity -> action logits
```

Synaptic weights are log-transformed and normalized by incoming connection energy to keep the full graph numerically stable.

The connectome structure and sensory projection are frozen during the Mario experiment. The trainable component is the motor readout.

## Learning procedure

The Mario experiment uses episodic policy-gradient reinforcement learning.

For each episode:

1. Reset the game.
2. Encode the current frame and frame difference.
3. Propagate activity through the sparse connectome.
4. Sample an action from the motor readout policy.
5. Receive the game reward.
6. Compute discounted returns.
7. Update the motor readout using policy gradients.

The current experiment is therefore best described as:

> Frozen connectome reservoir + trainable motor readout + visual reinforcement learning.

It is not yet a full MAML experiment. The repository contains MAML and related metalearning components, but the Mario result reported here is driven primarily by policy-gradient optimization of the readout.

## Reproducing the self-contained Colab experiment

Open:

```text
notebooks/colab_full_brain_mario_v2.ipynb
```

The notebook:

1. Installs the required packages.
2. Downloads the MaleCNS v1.0 files.
3. Verifies the loaded graph scale.
4. Loads the full sparse network.
5. Creates the Gym-style Mario environment.
6. Runs pre-training evaluation.
7. Trains the motor readout with reinforcement learning.
8. Runs post-training evaluation.
9. Generates videos, plots, CSV metrics, JSON results and a checkpoint.

The experiment requires:

- CUDA-enabled GPU;
- sufficient GPU memory;
- MaleCNS v1.0 data;
- `gym-super-mario-bros`;
- the required game ROM/environment assets;
- several gigabytes of storage for the connectome files.

## Installation

```bash
git clone https://github.com/Matheussoranco/Meta-Brain.git
cd Meta-Brain

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Project structure

```text
Meta-Brain/
├── config/
│   └── config.yaml
├── connectome/
│   └── loader.py
├── simulator/
│   └── snn.py
├── environment/
│   └── mario_env.py
├── metalearning/
│   └── algorithms.py
├── training/
│   └── train.py
├── notebooks/
│   ├── colab_sizable_brain_test.ipynb
│   └── colab_full_brain_mario_v2.ipynb
├── scripts/
│   ├── download_data.py
│   └── full_brain_mario_v2.py
├── data/
│   └── connectome/
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Research questions

Meta-Brain is intended as an experimental platform for studying:

- whether connectome topology constrains learnable behavior;
- how biological graph structure compares with shuffled or artificial graphs;
- whether sensorimotor pathways support visual control;
- how neuromodulatory plasticity can be incorporated into connectome-based models;
- whether full-connectome execution provides measurable benefits over sensorimotor subgraphs;
- how biologically inspired architectures compare with conventional neural networks.

## Required next experiments

The current Mario result is promising but not conclusive. The next validation steps are:

- fix and verify the reported retained edge count;
- run multiple random seeds;
- compare against a random-readout baseline;
- compare against a shuffled-connectome control;
- compare full-connectome and sensorimotor-subgraph variants;
- evaluate on held-out levels;
- report confidence intervals over episode success;
- train until stable flag completion;
- publish all configuration files and checkpoints.

## Scientific limitations

This project should not be interpreted as:

- a simulation of the human brain;
- a complete biological simulation of every neuron and synapse;
- evidence of consciousness;
- evidence of human-level general intelligence;
- proof that biological and digital intelligence are identical.

The scientific claim is narrower:

> A large biological connectome can be represented digitally as a sparse recurrent network and can support measurable task-specific adaptive behavior.

## Citation

If you use this project or its experiments, please cite the repository and the underlying connectome work.

```bibtex
@software{metabrain2026,
  title        = {Meta-Brain: Full-connectome-inspired reinforcement learning with the Drosophila connectome},
  author       = {Soranço, Matheus},
  year         = {2026},
  url          = {https://github.com/Matheussoranco/Meta-Brain}
}
```

```bibtex
@article{berg2026malecns,
  title   = {Sexual dimorphism in the complete Drosophila male central nervous system connectome},
  journal = {Cell},
  year    = {2026},
  doi     = {10.1016/j.cell.2026.08.015}
}
```

## License

MIT License.

## Acknowledgements

- Janelia FlyEM Team
- MaleCNS connectome project
- neuPrint and FlyWire communities
- Researchers developing Gym, reinforcement learning and spiking neural network tooling

---

Built with I.S.A.A.C. — Intelligent Synthetic Autonomous Agent Companion.
