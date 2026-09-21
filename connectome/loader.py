"""
Connectome data loading and processing for Drosophila Male CNS.
Interfaces with neuPrint and local data files.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import yaml


@dataclass
class NeuronMetadata:
    """Metadata for a single neuron from the connectome."""
    body_id: int
    type: str
    instance: str
    superclass: str
    class_: str
    subtype: str
    hemisphere: str
    soma_side: str
    root_side: str
    hemilineage: str
    flow: str
    neurotransmitter: str
    flywire_type: str
    manc_type: str
    hemibrain_type: str
    fru_expression: bool = False
    dsx_expression: bool = False
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Synapse:
    """Single synaptic connection."""
    pre_id: int
    post_id: int
    x: float
    y: float
    z: float
    weight: float = 1.0
    roi: str = ""


@dataclass
class ConnectomeGraph:
    """Graph representation of the connectome."""
    neurons: Dict[int, NeuronMetadata]
    edges: List[Tuple[int, int, float]]  # (pre, post, weight)
    adjacency: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    reverse_adjacency: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    
    def __post_init__(self):
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build adjacency lists for efficient traversal."""
        self.adjacency = {nid: [] for nid in self.neurons}
        self.reverse_adjacency = {nid: [] for nid in self.neurons}
        for pre, post, weight in self.edges:
            self.adjacency[pre].append((post, weight))
            self.reverse_adjacency[post].append((pre, weight))
    
    def get_inputs(self, neuron_id: int) -> List[Tuple[int, float]]:
        """Get all presynaptic partners."""
        return self.reverse_adjacency.get(neuron_id, [])
    
    def get_outputs(self, neuron_id: int) -> List[Tuple[int, float]]:
        """Get all postsynaptic partners."""
        return self.adjacency.get(neuron_id, [])
    
    def get_neurons_by_type(self, neuron_type: str) -> List[int]:
        """Get all neuron IDs matching a type."""
        return [nid for nid, meta in self.neurons.items() if meta.type == neuron_type]
    
    def get_neurons_by_superclass(self, superclass: str) -> List[int]:
        """Get all neuron IDs matching a superclass."""
        return [nid for nid, meta in self.neurons.items() if meta.superclass == superclass]
    
    def get_sensory_neurons(self) -> List[int]:
        """Get all sensory neuron IDs."""
        sensory_superclasses = ["OLSN", "ORN", "AN", "GRN", "TN"]
        ids = []
        for sc in sensory_superclasses:
            ids.extend(self.get_neurons_by_superclass(sc))
        return ids
    
    def get_motor_neurons(self) -> List[int]:
        """Get all motor neuron IDs."""
        motor_superclasses = ["MN", "DN"]
        ids = []
        for sc in motor_superclasses:
            ids.extend(self.get_neurons_by_superclass(sc))
        return ids


class ConnectomeLoader:
    """Load and process Drosophila connectome data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.connectome_config = self.config['connectome']
        self.data_path = Path(self.connectome_config['download_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self._graph: Optional[ConnectomeGraph] = None
        self._neuron_index: Dict[str, int] = {}  # type -> index
    
    def load_from_neuprint(self, token: str, dataset: str = None) -> ConnectomeGraph:
        """Load connectome from neuPrint API."""
        try:
            from neuprint import Client, fetch_neurons, fetch_synapses
        except ImportError:
            raise ImportError("neuprint-python not installed. Run: pip install neuprint-python")
        
        dataset = dataset or self.connectome_config['dataset']
        client = Client(self.connectome_config['neuprint_url'], dataset=dataset, token=token)
        
        # Fetch all neurons
        print("Fetching neurons from neuPrint...")
        neurons_df, _ = fetch_neurons(client, "*")
        
        # Fetch synapses (this is large, consider chunking)
        print("Fetching synapses from neuPrint...")
        synapses_df = fetch_synapses(client, "*")
        
        return self._build_graph(neurons_df, synapses_df)
    
    def load_from_files(self, 
                        meta_path: str = None,
                        edgelist_path: str = None,
                        synapses_path: str = None) -> ConnectomeGraph:
        """Load connectome from local Feather/Parquet files."""
        meta_path = meta_path or self.data_path / "malecns_meta.feather"
        edgelist_path = edgelist_path or self.data_path / "malecns_simple_edgelist.feather"
        
        print(f"Loading metadata from {meta_path}...")
        meta_df = pd.read_feather(meta_path)
        
        print(f"Loading edgelist from {edgelist_path}...")
        edge_df = pd.read_feather(edgelist_path)
        
        return self._build_graph(meta_df, edge_df)
    
    def _build_graph(self, meta_df: pd.DataFrame, edge_df: pd.DataFrame) -> ConnectomeGraph:
        """Build ConnectomeGraph from dataframes."""
        print("Building connectome graph...")
        
        # Build neuron metadata
        neurons = {}
        for _, row in meta_df.iterrows():
            nid = int(row['bodyId'])
            neurons[nid] = NeuronMetadata(
                body_id=nid,
                type=row.get('type', ''),
                instance=row.get('instance', ''),
                superclass=row.get('superclass', ''),
                class_=row.get('class', ''),
                subtype=row.get('subtype', ''),
                hemisphere=row.get('somaSide', ''),
                soma_side=row.get('somaSide', ''),
                root_side=row.get('rootSide', ''),
                hemilineage=row.get('itoleeHl', ''),
                flow=row.get('flow', ''),
                neurotransmitter=row.get('predictedNt', ''),
                flywire_type=row.get('flywireType', ''),
                manc_type=row.get('mancType', ''),
                hemibrain_type=row.get('hemibrainType', ''),
                fru_expression=bool(row.get('fru', 0)),
                dsx_expression=bool(row.get('dsx', 0)),
                x=row.get('x', 0.0),
                y=row.get('y', 0.0),
                z=row.get('z', 0.0),
            )
        
        # Build edges
        edges = []
        for _, row in edge_df.iterrows():
            pre = int(row['bodyId_pre'])
            post = int(row['bodyId_post'])
            weight = float(row.get('weight', 1.0))
            edges.append((pre, post, weight))
        
        graph = ConnectomeGraph(neurons=neurons, edges=edges)
        self._graph = graph
        
        print(f"Loaded {len(neurons)} neurons and {len(edges)} connections")
        return graph
    
    def get_subgraph(self, 
                     neuron_ids: List[int],
                     include_presynaptic: int = 1,
                     include_postsynaptic: int = 1) -> ConnectomeGraph:
        """Extract a subgraph around specified neurons."""
        if self._graph is None:
            raise ValueError("No graph loaded. Call load_from_files() or load_from_neuprint() first.")
        
        # Expand to include neighbors
        expanded_ids = set(neuron_ids)
        
        for _ in range(include_presynaptic):
            new_ids = set()
            for nid in expanded_ids:
                for pre, _ in self._graph.get_inputs(nid):
                    new_ids.add(pre)
            expanded_ids.update(new_ids)
        
        for _ in range(include_postsynaptic):
            new_ids = set()
            for nid in expanded_ids:
                for post, _ in self._graph.get_outputs(nid):
                    new_ids.add(post)
            expanded_ids.update(new_ids)
        
        # Filter neurons and edges
        sub_neurons = {nid: self._graph.neurons[nid] for nid in expanded_ids if nid in self._graph.neurons}
        sub_edges = [(pre, post, w) for pre, post, w in self._graph.edges 
                     if pre in expanded_ids and post in expanded_ids]
        
        return ConnectomeGraph(neurons=sub_neurons, edges=sub_edges)
    
    def get_sensorimotor_subgraph(self, 
                                  sensory_types: List[str] = None,
                                  motor_types: List[str] = None,
                                  max_interneuron_layers: int = 3) -> ConnectomeGraph:
        """Extract sensorimotor pathway subgraph."""
        if self._graph is None:
            raise ValueError("No graph loaded.")
        
        sensory_types = sensory_types or self.connectome_config['sensory_types']
        motor_types = motor_types or self.connectome_config['motor_types']
        
        # Get sensory and motor neuron IDs
        sensory_ids = []
        for st in sensory_types:
            sensory_ids.extend(self._graph.get_neurons_by_superclass(st))
        
        motor_ids = []
        for mt in motor_types:
            motor_ids.extend(self._graph.get_neurons_by_superclass(mt))
        
        print(f"Found {len(sensory_ids)} sensory and {len(motor_ids)} motor neurons")
        
        # Start with sensory and motor neurons
        all_ids = set(sensory_ids + motor_ids)
        
        # Expand through interneurons up to max_interneuron_layers
        current_layer = set(sensory_ids)
        for layer in range(max_interneuron_layers):
            next_layer = set()
            for nid in current_layer:
                for post, _ in self._graph.get_outputs(nid):
                    if post not in all_ids:
                        next_layer.add(post)
                        all_ids.add(post)
            current_layer = next_layer
            if not current_layer:
                break
            print(f"Layer {layer+1}: added {len(next_layer)} interneurons")
        
        # Also trace backwards from motor neurons
        current_layer = set(motor_ids)
        for layer in range(max_interneuron_layers):
            next_layer = set()
            for nid in current_layer:
                for pre, _ in self._graph.get_inputs(nid):
                    if pre not in all_ids:
                        next_layer.add(pre)
                        all_ids.add(pre)
            current_layer = next_layer
            if not current_layer:
                break
        
        # Filter
        sub_neurons = {nid: self._graph.neurons[nid] for nid in all_ids if nid in self._graph.neurons}
        sub_edges = [(pre, post, w) for pre, post, w in self._graph.edges 
                     if pre in all_ids and post in all_ids]
        
        print(f"Sensorimotor subgraph: {len(sub_neurons)} neurons, {len(sub_edges)} connections")
        return ConnectomeGraph(neurons=sub_neurons, edges=sub_edges)
    
    def get_dimorphic_hotspot_subgraph(self) -> ConnectomeGraph:
        """Extract the dimorphic hotspot circuit from the paper."""
        if self._graph is None:
            raise ValueError("No graph loaded.")
        
        hotspot_types = self.connectome_config['dimorphic_hotspot_types']
        hotspot_ids = []
        for ht in hotspot_types:
            hotspot_ids.extend(self._graph.get_neurons_by_type(ht))
        
        print(f"Found {len(hotspot_ids)} dimorphic hotspot neurons")
        
        # Expand 2 hops in each direction
        return self.get_subgraph(hotspot_ids, include_presynaptic=2, include_postsynaptic=2)
    
    def export_for_simulation(self, graph: ConnectomeGraph, output_path: str):
        """Export graph in format suitable for SNN simulation."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Neuron properties
        neuron_data = []
        for nid, meta in graph.neurons.items():
            neuron_data.append({
                'id': nid,
                'type': meta.type,
                'superclass': meta.superclass,
                'neurotransmitter': meta.neurotransmitter,
                'x': meta.x,
                'y': meta.y,
                'z': meta.z,
                'fru': meta.fru_expression,
                'dsx': meta.dsx_expression,
            })
        
        pd.DataFrame(neuron_data).to_feather(output_path / "neurons.feather")
        
        # Edges
        edge_data = []
        for pre, post, weight in graph.edges:
            edge_data.append({
                'pre': pre,
                'post': post,
                'weight': weight,
            })
        
        pd.DataFrame(edge_data).to_feather(output_path / "edges.feather")
        
        # Adjacency for fast lookup
        adj_data = {}
        for nid in graph.neurons:
            adj_data[nid] = {
                'inputs': graph.get_inputs(nid),
                'outputs': graph.get_outputs(nid),
            }
        
        with open(output_path / "adjacency.json", 'w') as f:
            json.dump(adj_data, f)
        
        print(f"Exported simulation data to {output_path}")


def create_default_loader(config_path: str = "config/config.yaml") -> ConnectomeLoader:
    """Factory function to create a ConnectomeLoader with default config."""
    return ConnectomeLoader(config_path)