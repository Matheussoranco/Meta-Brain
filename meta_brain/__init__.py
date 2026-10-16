"""
Meta-Brain: Metalearning on Drosophila Connectome for Mario Bros
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "Matheus Soranço"

# The legacy modules live at the repo root level, not inside meta_brain/. They
# are imported lazily so that ``import meta_brain.graph`` does not pull in
# Brian2, Gymnasium or the Mario environment.
_LAZY = {
    # Connectome
    'ConnectomeLoader': 'connectome.loader',
    'create_default_loader': 'connectome.loader',
    'NeuronMetadata': 'connectome.loader',
    'ConnectomeGraph': 'connectome.loader',
    # Simulator
    'DrosophilaSNN': 'simulator.snn',
    'RateCodedSNN': 'simulator.snn',
    'create_snn': 'simulator.snn',
    # Environment
    'MarioEnv': 'environment.mario_env',
    'create_mario_env': 'environment.mario_env',
    # Metalearning
    'MAML': 'metalearning.algorithms',
    'Reptile': 'metalearning.algorithms',
    'ANIL': 'metalearning.algorithms',
    'SNNMetalearner': 'metalearning.algorithms',
    'DifferentiableSNN': 'metalearning.algorithms',
    'MetalearningConfig': 'metalearning.algorithms',
    'create_differentiable_snn_from_connectome': 'metalearning.algorithms',
    'create_metalearner': 'metalearning.algorithms',
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'meta_brain' has no attribute {name!r}")
    return getattr(import_module(module), name)


__all__ = list(_LAZY)
