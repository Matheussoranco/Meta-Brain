"""
Spiking Neural Network Simulator for Drosophila Connectome.
Uses Brian2 for efficient GPU-accelerated simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path


class Neurotransmitter(Enum):
    """Neurotransmitter types with their effects."""
    ACETYLCHOLINE = "acetylcholine"  # Excitatory
    GABA = "GABA"                    # Inhibitory
    GLUTAMATE = "glutamate"          # Inhibitory in Drosophila
    DOPAMINE = "dopamine"            # Modulatory
    OCTOPAMINE = "octopamine"        # Modulatory
    SEROTONIN = "serotonin"          # Modulatory
    UNKNOWN = "unknown"


@dataclass
class NeuronParams:
    """Parameters for a single neuron."""
    id: int
    type: str
    superclass: str
    neurotransmitter: Neurotransmitter
    # Membrane properties
    Cm: float = 100.0       # pF
    gL: float = 5.0         # nS
    EL: float = -65.0       # mV
    VT: float = -50.0       # mV
    Vreset: float = -65.0   # mV
    tau_ref: float = 2.0    # ms
    # Synaptic properties
    tau_syn_e: float = 5.0  # ms
    tau_syn_i: float = 10.0 # ms
    # Position
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # Gene expression
    fru: bool = False
    dsx: bool = False


@dataclass
class SynapseParams:
    """Parameters for a synaptic connection."""
    pre: int
    post: int
    weight: float
    delay: float = 1.0      # ms
    neurotransmitter: Neurotransmitter = Neurotransmitter.ACETYLCHOLINE


class DrosophilaSNN:
    """
    Spiking Neural Network implementation of Drosophila connectome.
    Uses Brian2 for simulation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sim_config = self.config['simulator']
        self.dt = self.sim_config['dt']
        
        # Brian2 objects (initialized in build())
        self.network = None
        self.neuron_group = None
        self.synapses = None
        self.spike_monitors = {}
        self.state_monitors = {}
        
        # Neuron and synapse data
        self.neuron_params: Dict[int, NeuronParams] = {}
        self.synapse_params: List[SynapseParams] = []
        self.id_to_index: Dict[int, int] = {}
        self.index_to_id: Dict[int, int] = {}
        
        # Input/output neuron indices
        self.sensory_indices: List[int] = []
        self.motor_indices: List[int] = []
        self.interneuron_indices: List[int] = []
        
        # Plasticity
        self.stdp_enabled = self.sim_config['plasticity']['stdp_enabled']
        
    def build_from_connectome(self, 
                              neuron_data: List[Dict],
                              edge_data: List[Dict],
                              sensory_ids: List[int],
                              motor_ids: List[int]):
        """Build the SNN from connectome data."""
        print("Building SNN from connectome data...")
        
        # Import Brian2 here to avoid import issues
        try:
            import brian2 as b2
        except ImportError:
            raise ImportError("Brian2 not installed. Run: pip install brian2")
        
        # Create neuron index mapping
        all_ids = [n['id'] for n in neuron_data]
        self.id_to_index = {nid: i for i, nid in enumerate(all_ids)}
        self.index_to_id = {i: nid for i, nid in enumerate(all_ids)}
        n_neurons = len(all_ids)
        
        # Parse neuron parameters
        for n in neuron_data:
            nid = n['id']
            nt_str = n.get('neurotransmitter', 'unknown').lower()
            try:
                nt = Neurotransmitter(nt_str)
            except ValueError:
                nt = Neurotransmitter.UNKNOWN
            
            self.neuron_params[nid] = NeuronParams(
                id=nid,
                type=n['type'],
                superclass=n['superclass'],
                neurotransmitter=nt,
                x=n.get('x', 0),
                y=n.get('y', 0),
                z=n.get('z', 0),
                fru=n.get('fru', False),
                dsx=n.get('dsx', False),
            )
        
        # Parse synapse parameters
        for e in edge_data:
            pre = e['pre']
            post = e['post']
            weight = e['weight']
            
            if pre in self.id_to_index and post in self.id_to_index:
                pre_nt = self.neuron_params[pre].neurotransmitter
                self.synapse_params.append(SynapseParams(
                    pre=pre,
                    post=post,
                    weight=weight,
                    neurotransmitter=pre_nt,
                ))
        
        print(f"Parsed {n_neurons} neurons and {len(self.synapse_params)} synapses")
        
        # Identify sensory and motor indices
        self.sensory_indices = [self.id_to_index[nid] for nid in sensory_ids if nid in self.id_to_index]
        self.motor_indices = [self.id_to_index[nid] for nid in motor_ids if nid in self.id_to_index]
        self.interneuron_indices = [i for i in range(n_neurons) 
                                     if i not in self.sensory_indices and i not in self.motor_indices]
        
        print(f"Sensory: {len(self.sensory_indices)}, Motor: {len(self.motor_indices)}, "
              f"Interneurons: {len(self.interneuron_indices)}")
        
        # Build Brian2 network
        self._build_brian_network(n_neurons)
    
    def _build_brian_network(self, n_neurons: int):
        """Construct the Brian2 network."""
        import brian2 as b2
        
        b2.start_scope()
        b2.defaultclock.dt = self.dt * b2.ms
        
        # Neuron equations (AdEx model for biological realism)
        eqs = '''
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + I_syn_e + I_syn_i + I_ext) / Cm : volt (unless refractory)
        dI_syn_e/dt = -I_syn_e / tau_syn_e : amp
        dI_syn_i/dt = -I_syn_i / tau_syn_i : amp
        I_ext : amp
        Cm : farad
        gL : siemens
        EL : volt
        VT : volt
        Vreset : volt
        DeltaT : volt
        tau_syn_e : second
        tau_syn_i : second
        tau_ref : second
        '''
        
        # Create neuron group
        self.neuron_group = b2.NeuronGroup(n_neurons, eqs, 
                                           threshold='v > VT',
                                           reset='v = Vreset',
                                           refractory='tau_ref',
                                           method='euler')
        
        # Set parameters for each neuron
        for nid, params in self.neuron_params.items():
            idx = self.id_to_index[nid]
            self.neuron_group.Cm[idx] = params.Cm * b2.pF
            self.neuron_group.gL[idx] = params.gL * b2.nS
            self.neuron_group.EL[idx] = params.EL * b2.mV
            self.neuron_group.VT[idx] = params.VT * b2.mV
            self.neuron_group.Vreset[idx] = params.Vreset * b2.mV
            self.neuron_group.DeltaT[idx] = 2.0 * b2.mV  # AdEx sharpness
            self.neuron_group.tau_syn_e[idx] = params.tau_syn_e * b2.ms
            self.neuron_group.tau_syn_i[idx] = params.tau_syn_i * b2.ms
            self.neuron_group.tau_ref[idx] = params.tau_ref * b2.ms
            self.neuron_group.v[idx] = params.EL * b2.mV
            self.neuron_group.I_ext[idx] = 0 * b2.pA
        
        # Create synapses
        self.synapses = b2.Synapses(self.neuron_group, self.neuron_group,
                                    model='''w : siemens
                                             tau_syn : second
                                             is_excitatory : boolean
                                             Ee : volt
                                             Ei : volt''',
                                    on_pre='''
                                    I_syn_e_post += w * is_excitatory * (Ee - v_post)
                                    I_syn_i_post += w * (1 - is_excitatory) * (Ei - v_post)
                                    ''',
                                    delay=1.0 * b2.ms)

        # Synapse parameters
        Ee = 0 * b2.mV    # Excitatory reversal (ACh)
        Ei = -70 * b2.mV  # Inhibitory reversal (GABA/Glu)
        weight_scale = self.sim_config['synapse_weight_scale'] * b2.nS

        # Connect synapses
        pre_indices = [self.id_to_index[s.pre] for s in self.synapse_params]
        post_indices = [self.id_to_index[s.post] for s in self.synapse_params]
        weights = [s.weight * weight_scale for s in self.synapse_params]
        is_exc = [s.neurotransmitter in [Neurotransmitter.ACETYLCHOLINE] for s in self.synapse_params]

        self.synapses.connect(i=pre_indices, j=post_indices)
        self.synapses.w = weights
        self.synapses.is_excitatory = is_exc
        self.synapses.Ee = Ee
        self.synapses.Ei = Ei
        self.synapses.tau_syn = self.sim_config['synaptic_time_constant'] * b2.ms

        print(f"Created {len(self.synapses)} synaptic connections")
        
        # Add STDP if enabled
        if self.stdp_enabled:
            self._add_stdp()
        
        # Create monitors
        self.spike_monitors['all'] = b2.SpikeMonitor(self.neuron_group)
        self.state_monitors['v'] = b2.StateMonitor(self.neuron_group, 'v', record=self.motor_indices[:10])
        
        # Create network
        self.network = b2.Network(self.neuron_group, self.synapses, 
                                  self.spike_monitors['all'],
                                  self.state_monitors['v'])
        
        print("Brian2 network built successfully")
    
    def _add_stdp(self):
            """Add STDP plasticity to synapses."""
            import brian2 as b2

            stdp_config = self.sim_config['plasticity']

            # Rewrite the synapse model with STDP included
            # We need to recreate the synapses with the extended model
            stdp_eqs = '''
            dApre/dt = -Apre / tau_plus : 1 (event-driven)
            dApost/dt = -Apost / tau_minus : 1 (event-driven)
            '''

            stdp_on_pre = '''
            Apre += a_plus
            w = clip(w + Apost, 0*siemens, wmax)
            '''

            stdp_on_post = '''
            Apost += a_minus
            w = clip(w + Apre, 0*siemens, wmax)
            '''

            # Create new synapses with STDP
            self.synapses = b2.Synapses(self.neuron_group, self.neuron_group,
                                        model='''w : siemens
                                                 tau_syn : second
                                                 is_excitatory : boolean
                                                 Ee : volt
                                                 Ei : volt
                                                 Apre : 1
                                                 Apost : 1
                                                 tau_plus : second
                                                 tau_minus : second
                                                 a_plus : 1
                                                 a_minus : 1
                                                 wmax : siemens''' + stdp_eqs,
                                        on_pre='''
                                        I_syn_e_post += w * is_excitatory * (Ee - v_post)
                                        I_syn_i_post += w * (1 - is_excitatory) * (Ei - v_post)
                                        ''' + stdp_on_pre,
                                        on_post=stdp_on_post,
                                        delay=1.0 * b2.ms)

            # Reconnect
            pre_indices = [self.id_to_index[s.pre] for s in self.synapse_params]
            post_indices = [self.id_to_index[s.post] for s in self.synapse_params]
            weights = [s.weight * weight_scale for s in self.synapse_params]
            is_exc = [s.neurotransmitter in [Neurotransmitter.ACETYLCHOLINE] for s in self.synapse_params]

            self.synapses.connect(i=pre_indices, j=post_indices)
            self.synapses.w = weights
            self.synapses.is_excitatory = is_exc
            self.synapses.Ee = Ee
            self.synapses.Ei = Ei
            self.synapses.tau_syn = self.sim_config['synaptic_time_constant'] * b2.ms

            # Initialize STDP variables
            self.synapses.Apre = 0
            self.synapses.Apost = 0
            self.synapses.tau_plus = stdp_config['stdp_tau_plus'] * b2.ms
            self.synapses.tau_minus = stdp_config['stdp_tau_minus'] * b2.ms
            self.synapses.a_plus = stdp_config['stdp_a_plus']
            self.synapses.a_minus = -stdp_config['stdp_a_minus']
            self.synapses.wmax = 10 * b2.nS
    
    def inject_sensory_input(self, sensory_rates: Dict[int, float], duration: float):
        """
        Inject sensory input as Poisson spike trains.
        
        Args:
            sensory_rates: Dict mapping neuron_id -> firing rate (Hz)
            duration: Simulation duration (ms)
        """
        import brian2 as b2
        
        # Create Poisson input for each sensory neuron
        for nid, rate in sensory_rates.items():
            if nid not in self.id_to_index:
                continue
            idx = self.id_to_index[nid]
            
            # Create PoissonGroup for this sensory neuron
            pg = b2.PoissonGroup(1, rates=rate * b2.Hz)
            syn = b2.Synapses(pg, self.neuron_group, on_pre='v += 0.5*mV')
            syn.connect(j=idx)
            
            # Add to network temporarily
            self.network.add(pg, syn)
            
            # Store for cleanup
            if not hasattr(self, '_input_sources'):
                self._input_sources = []
            self._input_sources.append((pg, syn))
        
        # Run simulation
        self.network.run(duration * b2.ms)
        
        # Clean up input sources
        for pg, syn in self._input_sources:
            self.network.remove(pg, syn)
        self._input_sources = []
    
    def run(self, duration: float, external_current: Dict[int, float] = None):
        """Run the simulation for specified duration."""
        import brian2 as b2
        
        if external_current:
            for nid, current in external_current.items():
                if nid in self.id_to_index:
                    idx = self.id_to_index[nid]
                    self.neuron_group.I_ext[idx] = current * b2.pA
        
        self.network.run(duration * b2.ms)
        
        if external_current:
            for nid in external_current:
                if nid in self.id_to_index:
                    idx = self.id_to_index[nid]
                    self.neuron_group.I_ext[idx] = 0 * b2.pA
    
    def get_motor_output(self, time_window: float = None) -> Dict[int, float]:
        """
        Get motor neuron firing rates.
        
        Returns:
            Dict mapping motor neuron_id -> firing rate (Hz)
        """
        if time_window is None:
            time_window = self.sim_config['simulation_time']
        
        spike_mon = self.spike_monitors['all']
        rates = {}
        
        for midx in self.motor_indices:
            nid = self.index_to_id[midx]
            spikes = spike_mon.t[spike_mon.i == midx]
            if len(spikes) > 0:
                # Filter to time window
                recent_spikes = spikes[spikes > (spike_mon.t[-1] - time_window * b2.ms)]
                rates[nid] = len(recent_spikes) / (time_window / 1000.0)  # Hz
            else:
                rates[nid] = 0.0
        
        return rates
    
    def get_spike_trains(self, neuron_ids: List[int] = None) -> Dict[int, np.ndarray]:
        """Get spike times for specified neurons."""
        spike_mon = self.spike_monitors['all']
        
        if neuron_ids is None:
            neuron_ids = list(self.id_to_index.keys())
        
        trains = {}
        for nid in neuron_ids:
            if nid in self.id_to_index:
                idx = self.id_to_index[nid]
                spikes = spike_mon.t[spike_mon.i == idx]
                trains[nid] = np.array(spikes / b2.ms)
        
        return trains
    
    def set_plasticity(self, enabled: bool):
        """Enable or disable STDP plasticity."""
        self.stdp_enabled = enabled
        if hasattr(self.synapses, 'Apre'):
            # STDP is always active in Brian2 once added
            # We can effectively disable by setting learning rates to 0
            if not enabled:
                self.synapses.a_plus = 0
                self.synapses.a_minus = 0
            else:
                stdp_config = self.sim_config['plasticity']
                self.synapses.a_plus = stdp_config['stdp_a_plus']
                self.synapses.a_minus = -stdp_config['stdp_a_minus']
    
    def save_weights(self, path: str):
        """Save synaptic weights."""
        import brian2 as b2
        weights = np.array(self.synapses.w / b2.nS)
        pre = np.array(self.synapses.i)
        post = np.array(self.synapses.j)
        np.savez(path, weights=weights, pre=pre, post=post)
    
    def load_weights(self, path: str):
        """Load synaptic weights."""
        import brian2 as b2
        data = np.load(path)
        self.synapses.w = data['weights'] * b2.nS


class RateCodedSNN(DrosophilaSNN):
    """
    Rate-coded version for faster simulation.
    Uses population rate coding instead of individual spikes.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path)
        self.use_rate_coding = True
        self.population_size = 10  # neurons per population
    
    def build_from_connectome(self, 
                              neuron_data: List[Dict],
                              edge_data: List[Dict],
                              sensory_ids: List[int],
                              motor_ids: List[int]):
        """Build rate-coded network."""
        # Group neurons by type into populations
        from collections import defaultdict
        
        type_to_ids = defaultdict(list)
        for n in neuron_data:
            type_to_ids[n['type']].append(n['id'])
        
        # Create populations
        self.populations = {}
        self.pop_id_to_neurons = {}
        pop_idx = 0
        
        for ntype, ids in type_to_ids.items():
            self.populations[ntype] = pop_idx
            self.pop_id_to_neurons[pop_idx] = ids
            pop_idx += 1
        
        n_pops = len(self.populations)
        print(f"Created {n_pops} populations from {len(neuron_data)} neurons")
        
        # Build population-level connectivity
        pop_edges = defaultdict(float)
        for e in edge_data:
            pre_type = None
            post_type = None
            for n in neuron_data:
                if n['id'] == e['pre']:
                    pre_type = n['type']
                if n['id'] == e['post']:
                    post_type = n['type']
            
            if pre_type and post_type and pre_type in self.populations and post_type in self.populations:
                pop_edges[(self.populations[pre_type], self.populations[post_type])] += e['weight']
        
        # Build rate model
        self._build_rate_model(n_pops, pop_edges)
        
        # Map sensory/motor
        self.sensory_pops = set()
        self.motor_pops = set()
        for nid in sensory_ids:
            for n in neuron_data:
                if n['id'] == nid and n['type'] in self.populations:
                    self.sensory_pops.add(self.populations[n['type']])
        for nid in motor_ids:
            for n in neuron_data:
                if n['id'] == nid and n['type'] in self.populations:
                    self.motor_pops.add(self.populations[n['type']])
    
    def _build_rate_model(self, n_pops: int, pop_edges: Dict[Tuple[int, int], float]):
        """Build Wilson-Cowan rate model."""
        import torch
        import torch.nn as nn
        
        self.n_pops = n_pops
        self.tau = 20.0  # ms
        
        # Weight matrix
        W = np.zeros((n_pops, n_pops))
        for (pre, post), weight in pop_edges.items():
            W[post, pre] = weight  # post x pre
        
        self.W = torch.tensor(W, dtype=torch.float32)
        self.rates = torch.zeros(n_pops)
        self.external_input = torch.zeros(n_pops)
        
        # Activation function
        self.activation = nn.Softplus()
    
    def step(self, dt: float = 1.0):
        """Single simulation step."""
        import torch
        
        # Wilson-Cowan dynamics: tau * dr/dt = -r + f(W*r + I_ext)
        dr = (-self.rates + self.activation(self.W @ self.rates + self.external_input)) / self.tau
        self.rates += dr * dt
        self.rates = torch.clamp(self.rates, 0, 200)  # Max 200 Hz
    
    def run(self, duration: float, sensory_input: Dict[int, float] = None):
        """Run rate model."""
        if sensory_input:
            for pop_idx, rate in sensory_input.items():
                if pop_idx < self.n_pops:
                    self.external_input[pop_idx] = rate
        
        steps = int(duration / 1.0)  # 1ms steps
        for _ in range(steps):
            self.step(1.0)
        
        if sensory_input:
            self.external_input.zero_()
    
    def get_motor_output(self) -> Dict[int, float]:
        """Get motor population rates."""
        output = {}
        for pop_idx in self.motor_pops:
            output[pop_idx] = self.rates[pop_idx].item()
        return output
    
    def set_sensory_input(self, sensory_rates: Dict[int, float]):
        """Set sensory input rates."""
        for pop_idx, rate in sensory_rates.items():
            if pop_idx < self.n_pops:
                self.external_input[pop_idx] = rate


def create_snn(config_path: str = "config/config.yaml", 
               rate_coded: bool = False) -> DrosophilaSNN:
    """Factory function to create SNN."""
    if rate_coded:
        return RateCodedSNN(config_path)
    return DrosophilaSNN(config_path)