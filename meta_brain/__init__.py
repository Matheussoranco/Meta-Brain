"""
Meta-Brain: Metalearning on Drosophila Connectome for Mario Bros
"""

__version__ = "0.1.0"
__author__ = "Matheus Soranço"

# The modules live at the repo root level, not inside meta_brain/
# Import them directly (they should be on sys.path when installed via pip install -e .)
try:
    from connectome.loader import ConnectomeLoader, create_default_loader, NeuronMetadata, ConnectomeGraph
    from simulator.snn import DrosophilaSNN, RateCodedSNN, create_snn
    from environment.mario_env import MarioEnv, create_mario_env
    from metalearning.algorithms import (MAML, Reptile, ANIL, SNNMetalearner,
                                          DifferentiableSNN, MetalearningConfig,
                                          create_differentiable_snn_from_connectome,
                                          create_metalearner)
except ImportError:
    # Fallback: try with meta_brain prefix if used as package
    try:
        from meta_brain.connectome.loader import ConnectomeLoader, create_default_loader, NeuronMetadata, ConnectomeGraph
        from meta_brain.simulator.snn import DrosophilaSNN, RateCodedSNN, create_snn
        from meta_brain.environment.mario_env import MarioEnv, create_mario_env
        from meta_brain.metalearning.algorithms import (MAML, Reptile, ANIL, SNNMetalearner,
                                                          DifferentiableSNN, MetalearningConfig,
                                                          create_differentiable_snn_from_connectome,
                                                          create_metalearner)
    except ImportError:
        # Modules not available - will be available after `pip install -e .`
        pass

__all__ = [
    # Connectome
    'ConnectomeLoader',
    'create_default_loader',
    'NeuronMetadata',
    'ConnectomeGraph',
    # Simulator
    'DrosophilaSNN',
    'RateCodedSNN',
    'create_snn',
    # Environment
    'MarioEnv',
    'create_mario_env',
    # Metalearning
    'MAML',
    'Reptile',
    'ANIL',
    'SNNMetalearner',
    'DifferentiableSNN',
    'MetalearningConfig',
    'create_differentiable_snn_from_connectome',
    'create_metalearner',
]