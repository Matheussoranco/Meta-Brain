"""
Meta-Brain: Metalearning on Drosophila Connectome for Mario Bros
"""

__version__ = "0.1.0"
__author__ = "Matheus Soranço"

from .connectome.loader import ConnectomeLoader, create_default_loader, NeuronMetadata, ConnectomeGraph
from .simulator.snn import DrosophilaSNN, RateCodedSNN, create_snn
from .environment.mario_env import MarioEnv, create_mario_env
from .metalearning.algorithms import (MAML, Reptile, ANIL, SNNMetalearner, 
                                       DifferentiableSNN, MetalearningConfig,
                                       create_differentiable_snn_from_connectome,
                                       create_metalearner)

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