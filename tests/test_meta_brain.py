"""
Tests for Meta-Brain components.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectome.loader import NeuronMetadata, ConnectomeGraph
from simulator.snn import NeuronParams, Neurotransmitter, create_snn
from metalearning.algorithms import MetalearningConfig, DifferentiableSNN


class TestConnectome:
    """Test connectome data structures."""
    
    def test_neuron_metadata(self):
        """Test NeuronMetadata creation."""
        neuron = NeuronMetadata(
            body_id=12345,
            type="DNp05",
            instance="left",
            superclass="DN",
            class_="descending",
            subtype="p05",
            hemisphere="L",
            soma_side="L",
            root_side="L",
            hemilineage="DM4",
            flow="4",
            neurotransmitter="acetylcholine",
            flywire_type="DNp05",
            manc_type="DNp05",
            hemibrain_type="DNp05",
        )
        
        assert neuron.body_id == 12345
        assert neuron.type == "DNp05"
        assert neuron.neurotransmitter == "acetylcholine"
    
    def test_connectome_graph(self):
        """Test ConnectomeGraph construction."""
        neurons = {
            1: NeuronMetadata(body_id=1, type="A", instance="left", superclass="sensory",
                             neurotransmitter="acetylcholine",
                             class_="", subtype="", hemisphere="",
                             soma_side="", root_side="", hemilineage="",
                             flow="", flywire_type="", manc_type="", hemibrain_type=""),
            2: NeuronMetadata(body_id=2, type="B", instance="right", superclass="interneuron",
                             neurotransmitter="GABA",
                             class_="", subtype="", hemisphere="",
                             soma_side="", root_side="", hemilineage="",
                             flow="", flywire_type="", manc_type="", hemibrain_type=""),
            3: NeuronMetadata(body_id=3, type="C", instance="left", superclass="motor",
                             neurotransmitter="acetylcholine",
                             class_="", subtype="", hemisphere="",
                             soma_side="", root_side="", hemilineage="",
                             flow="", flywire_type="", manc_type="", hemibrain_type=""),
        }
        
        edges = [(1, 2, 5.0), (2, 3, 3.0), (1, 3, 1.0)]
        
        graph = ConnectomeGraph(neurons=neurons, edges=edges)
        
        assert len(graph.neurons) == 3
        assert len(graph.edges) == 3
        
        # Test adjacency
        inputs_2 = graph.get_inputs(2)
        assert len(inputs_2) == 1
        assert inputs_2[0] == (1, 5.0)
        
        outputs_1 = graph.get_outputs(1)
        assert len(outputs_1) == 2


class TestSimulator:
    """Test SNN simulator components."""
    
    def test_neuron_params(self):
        """Test NeuronParams."""
        params = NeuronParams(
            id=1,
            type="DN",
            superclass="motor",
            neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        )
        
        assert params.id == 1
        assert params.neurotransmitter == Neurotransmitter.ACETYLCHOLINE
    
    def test_neurotransmitter_enum(self):
        """Test Neurotransmitter enum."""
        assert Neurotransmitter.ACETYLCHOLINE.value == "acetylcholine"
        assert Neurotransmitter.GABA.value == "GABA"
        assert Neurotransmitter.DOPAMINE.value == "dopamine"


class TestMetalearning:
    """Test metalearning components."""
    
    def test_metalearning_config(self):
        """Test MetalearningConfig."""
        config = MetalearningConfig(
            algorithm="MAML",
            meta_lr=0.001,
            inner_lr=0.01,
            inner_steps=5,
        )
        
        assert config.algorithm == "MAML"
        assert config.meta_lr == 0.001
        assert config.inner_steps == 5
    
    def test_differentiable_snn_creation(self):
        """Test DifferentiableSNN creation."""
        n_neurons = 100
        n_inputs = 10
        n_outputs = 5
        
        # Random connectivity
        connectivity = torch.rand(n_neurons, n_neurons)
        connectivity = (connectivity > 0.9).float()  # Sparse
        neuron_types = torch.zeros(n_neurons)
        neuron_types[80:] = 1  # Last 20 inhibitory
        
        snn = DifferentiableSNN(
            n_neurons=n_neurons,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            connectivity=connectivity,
            neuron_types=neuron_types,
        )
        
        # Test forward pass
        T, batch = 20, 4
        inputs = torch.randn(T, batch, n_inputs)
        
        outputs, state = snn(inputs)
        
        assert outputs.shape == (T, batch, n_outputs)
        assert 'v' in state
        assert 'spikes' in state


class TestIntegration:
    """Integration tests."""
    
    def test_config_loading(self):
        """Test config.yaml loads correctly."""
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'connectome' in config
        assert 'simulator' in config
        assert 'environment' in config
        assert 'metalearning' in config
        assert 'training' in config
    
    def test_imports(self):
        """Test all main modules can be imported."""
        # Test direct imports from modules instead of package
        from connectome.loader import ConnectomeLoader
        from simulator.snn import DrosophilaSNN, RateCodedSNN
        from metalearning.algorithms import MAML, Reptile, DifferentiableSNN
        
        # Just verify they're accessible
        assert ConnectomeLoader is not None
        assert DrosophilaSNN is not None
        assert RateCodedSNN is not None
        assert MAML is not None
        assert Reptile is not None
        assert DifferentiableSNN is not None
        
        # Skip environment imports (requires gymnasium, nes-py)
        # from environment.mario_env import MarioEnv
        # assert MarioEnv is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])