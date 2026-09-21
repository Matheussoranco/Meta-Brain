"""
Metalearning algorithms for connectome-based agents.
Implements MAML, Reptile, and related algorithms for SNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import yaml
from pathlib import Path


@dataclass
class MetalearningConfig:
    """Configuration for metalearning algorithms."""
    algorithm: str = "MAML"
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    inner_steps: int = 5
    meta_batch_size: int = 4
    num_tasks: int = 100
    first_order: bool = False  # FO-MAML approximation
    
    # SNN-specific
    spike_coding: str = "rate"
    simulation_time: float = 100.0
    decision_interval: float = 50.0


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning for spiking neural networks.
    Adapts synaptic weights for fast task adaptation.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: MetalearningConfig = None):
        super().__init__()
        self.model = model
        self.config = config or MetalearningConfig()
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.meta_lr
        )
        
        # Inner optimizer (per-task)
        self.inner_optimizer_class = torch.optim.SGD
        self.inner_lr = self.config.inner_lr
        self.inner_steps = self.config.inner_steps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def adapt(self, 
              loss_fn: Callable,
              train_data: Tuple[torch.Tensor, torch.Tensor],
              steps: int = None) -> nn.Module:
        """
        Perform inner-loop adaptation.
        Returns adapted model (with fast weights).
        """
        steps = steps or self.inner_steps
        
        # Clone model parameters for fast adaptation
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for step in range(steps):
            # Forward pass with fast weights
            x, y = train_data
            logits = self._forward_with_weights(x, fast_weights)
            loss = loss_fn(logits, y)
            
            # Compute gradients wrt fast weights
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                        create_graph=not self.config.first_order)
            
            # Update fast weights
            for (name, param), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = param - self.inner_lr * grad
        
        # Create adapted model
        adapted_model = deepcopy(self.model)
        adapted_model.load_state_dict(fast_weights)
        return adapted_model
    
    def _forward_with_weights(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass using specific weights."""
        # This is a simplified version - in practice, use higher-order functions
        # or functional API for proper fast weight adaptation
        return self.model(x)  # Placeholder
    
    def meta_update(self, 
                    tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                    loss_fn: Callable) -> Dict[str, float]:
        """
        Perform meta-update across tasks.
        
        Each task: (train_x, train_y, test_x, test_y)
        """
        meta_loss = 0.0
        task_losses = []
        
        for train_x, train_y, test_x, test_y in tasks:
            # Inner loop adaptation
            adapted_model = self.adapt(loss_fn, (train_x, train_y))
            
            # Evaluate on test data
            test_logits = adapted_model(test_x)
            test_loss = loss_fn(test_logits, test_y)
            
            meta_loss += test_loss
            task_losses.append(test_loss.item())
        
        meta_loss /= len(tasks)
        
        # Meta-gradient step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'task_losses': task_losses,
            'mean_task_loss': np.mean(task_losses),
        }


class Reptile(nn.Module):
    """
    Reptile meta-learning algorithm.
    Simpler than MAML, moves initialization toward task solutions.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: MetalearningConfig = None):
        super().__init__()
        self.model = model
        self.config = config or MetalearningConfig()
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.meta_lr
        )
        
        self.inner_optimizer_class = torch.optim.SGD
        self.inner_lr = self.config.inner_lr
        self.inner_steps = self.config.inner_steps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def adapt(self, 
              loss_fn: Callable,
              train_data: Tuple[torch.Tensor, torch.Tensor],
              steps: int = None) -> Dict[str, torch.Tensor]:
        """
        Perform inner-loop adaptation and return final weights.
        """
        steps = steps or self.inner_steps
        
        # Create inner optimizer
        inner_model = deepcopy(self.model)
        inner_opt = self.inner_optimizer_class(inner_model.parameters(), lr=self.inner_lr)
        
        for step in range(steps):
            x, y = train_data
            logits = inner_model(x)
            loss = loss_fn(logits, y)
            
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
        
        # Return adapted weights
        return {name: param.clone() for name, param in inner_model.named_parameters()}
    
    def meta_update(self, 
                    tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                    loss_fn: Callable) -> Dict[str, float]:
        """
        Reptile meta-update: move init toward task solutions.
        """
        # Store initial weights
        init_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        adapted_weights_list = []
        
        for train_x, train_y, test_x, test_y in tasks:
            adapted_weights = self.adapt(loss_fn, (train_x, train_y))
            adapted_weights_list.append(adapted_weights)
        
        # Compute meta-gradient: move init toward average of adapted weights
        meta_grads = {}
        for name in init_weights:
            adapted_stack = torch.stack([aw[name] for aw in adapted_weights_list])
            mean_adapted = adapted_stack.mean(dim=0)
            meta_grads[name] = init_weights[name] - mean_adapted
        
        # Apply meta-gradient
        self.meta_optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name in meta_grads:
                param.grad = meta_grads[name]
        
        self.meta_optimizer.step()
        
        return {'meta_loss': 0.0}  # Reptile doesn't have explicit meta-loss


class ANIL(nn.Module):
    """
    Almost No Inner Loop (ANIL).
    Only adapts the last layer (head), keeping body fixed.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 head_names: List[str],
                 config: MetalearningConfig = None):
        super().__init__()
        self.model = model
        self.head_names = head_names
        self.config = config or MetalearningConfig()
        
        # Separate parameters
        self.body_params = []
        self.head_params = []
        
        for name, param in model.named_parameters():
            if any(hn in name for hn in head_names):
                self.head_params.append((name, param))
            else:
                self.body_params.append((name, param))
        
        # Meta-optimizer for body
        self.meta_optimizer = torch.optim.Adam(
            [p for _, p in self.body_params],
            lr=self.config.meta_lr
        )
        
        # Inner optimizer for head
        self.inner_lr = self.config.inner_lr
        self.inner_steps = self.config.inner_steps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def adapt(self, 
              loss_fn: Callable,
              train_data: Tuple[torch.Tensor, torch.Tensor],
              steps: int = None) -> Dict[str, torch.Tensor]:
        """Adapt only head parameters."""
        steps = steps or self.inner_steps
        
        # Clone head params
        fast_head = {name: param.clone() for name, param in self.head_params}
        body_state = {name: param.clone() for name, param in self.body_params}
        
        for step in range(steps):
            x, y = train_data
            
            # Forward with fast head + fixed body
            # This requires functional forward - simplified here
            logits = self.model(x)  # Placeholder
            loss = loss_fn(logits, y)
            
            grads = torch.autograd.grad(loss, fast_head.values(),
                                        create_graph=not self.config.first_order)
            
            for (name, _), grad in zip(self.head_params, grads):
                fast_head[name] = fast_head[name] - self.inner_lr * grad
        
        # Return all weights (body fixed, head adapted)
        all_weights = {**body_state, **fast_head}
        return all_weights
    
    def meta_update(self, tasks, loss_fn):
        """Meta-update on body parameters."""
        # Similar to MAML but only body params get meta-gradients
        pass


class SNNMetalearner:
    """
    High-level metalearner for connectome-based SNNs.
    Bridges between metalearning algorithms and SNN simulation.
    """
    
    def __init__(self, 
                 snn,  # DrosophilaSNN or RateCodedSNN
                 config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.metalearning_config = MetalearningConfig(**self.config['metalearning'])
        self.snn = snn
        
        # Task distribution
        self.tasks = []
        self.task_descriptions = []
        
        # Training history
        self.history = {
            'meta_loss': [],
            'task_losses': [],
            'adaptation_performance': [],
        }
    
    def create_task(self, 
                    name: str,
                    sensory_mapping: Callable,
                    motor_decoding: Callable,
                    reward_fn: Callable,
                    env_config: Dict = None) -> Dict:
        """Create a metalearning task."""
        task = {
            'name': name,
            'sensory_mapping': sensory_mapping,
            'motor_decoding': motor_decoding,
            'reward_fn': reward_fn,
            'env_config': env_config or {},
        }
        self.tasks.append(task)
        self.task_descriptions.append(name)
        return task
    
    def generate_task_batch(self, batch_size: int = None) -> List[Dict]:
        """Generate a batch of tasks for meta-training."""
        batch_size = batch_size or self.metalearning_config.meta_batch_size
        return np.random.choice(self.tasks, size=batch_size, replace=True).tolist()
    
    def evaluate_task(self, 
                      task: Dict,
                      snn,
                      num_episodes: int = 3) -> float:
        """
        Evaluate SNN on a task.
        Returns average reward.
        """
        total_reward = 0.0
        
        for ep in range(num_episodes):
            # Run episode
            episode_reward = self._run_episode(task, snn)
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def _run_episode(self, task: Dict, snn) -> float:
        """Run a single episode."""
        # This is a placeholder - actual implementation depends on SNN type
        # For rate-coded SNN:
        if hasattr(snn, 'run'):
            # Setup environment
            env_config = task['env_config']
            # ... run simulation ...
            pass
        
        return 0.0  # Placeholder
    
    def meta_train(self, 
                   num_iterations: int = 1000,
                   eval_interval: int = 100) -> Dict:
        """
        Main meta-training loop.
        """
        print(f"Starting meta-training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Sample task batch
            task_batch = self.generate_task_batch()
            
            # For each task, adapt and evaluate
            task_losses = []
            
            for task in task_batch:
                # Inner loop: adapt SNN to task
                # This involves modifying synaptic weights
                adapted_snn = self._inner_adapt(task, self.snn)
                
                # Evaluate adapted SNN
                loss = -self.evaluate_task(task, adapted_snn)  # Negative reward = loss
                task_losses.append(loss)
            
            # Meta-update: update initialization (synaptic weights)
            meta_loss = np.mean(task_losses)
            self._meta_update(task_batch, task_losses)
            
            # Log
            self.history['meta_loss'].append(meta_loss)
            self.history['task_losses'].append(task_losses)
            
            if iteration % eval_interval == 0:
                print(f"Iteration {iteration}: meta_loss={meta_loss:.4f}, "
                      f"mean_task_loss={np.mean(task_losses):.4f}")
                
                # Evaluate on held-out tasks
                eval_perf = self._evaluate_heldout()
                self.history['adaptation_performance'].append(eval_perf)
        
        return self.history
    
    def _inner_adapt(self, task: Dict, snn) -> Any:
        """
        Inner loop adaptation for a task.
        Modifies synaptic weights based on task experience.
        """
        # This is where the actual SNN weight adaptation happens
        # Could use:
        # - STDP with neuromodulatory signals
        # - Gradient-based adaptation (if using differentiable SNN)
        # - Evolutionary strategies
        # - Policy gradient on SNN parameters
        
        # For now, return original SNN
        return snn
    
    def _meta_update(self, task_batch: List[Dict], task_losses: List[float]):
        """
        Meta-update: modify initial synaptic weights for better adaptation.
        """
        # This would update the SNN's initial connectivity
        # For rate-coded SNN, can use gradient-based meta-learning
        # For spiking SNN, might use evolutionary meta-learning
        pass
    
    def _evaluate_heldout(self) -> float:
        """Evaluate on held-out tasks."""
        return 0.0


# Differentiable SNN wrapper for gradient-based metalearning
class DifferentiableSNN(nn.Module):
    """
    Differentiable spiking neural network using surrogate gradients.
    Compatible with PyTorch autograd for MAML/Reptile.
    """
    
    def __init__(self, 
                 n_neurons: int,
                 n_inputs: int,
                 n_outputs: int,
                 connectivity: torch.Tensor,  # (n_neurons, n_neurons) adjacency
                 neuron_types: torch.Tensor,  # 0=excitatory, 1=inhibitory
                 dt: float = 1.0,
                 tau_mem: float = 20.0,
                 tau_syn: float = 5.0,
                 threshold: float = 1.0,
                 reset: float = 0.0,
                 surrogate_scale: float = 10.0):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.reset = reset
        self.surrogate_scale = surrogate_scale
        
        # Fixed connectivity (from connectome)
        self.register_buffer('connectivity', connectivity.float())
        self.register_buffer('neuron_types', neuron_types.float())
        
        # Learnable parameters
        # Input weights
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_inputs) * 0.1)
        
        # Synaptic weights (modulate connectome structure)
        self.W_syn = nn.Parameter(torch.ones_like(connectivity) * 0.5)
        
        # Output readout weights
        self.W_out = nn.Parameter(torch.randn(n_outputs, n_neurons) * 0.1)
        
        # Neuron parameters
        self.v_th = nn.Parameter(torch.ones(n_neurons) * threshold)
        self.tau_mem_param = nn.Parameter(torch.ones(n_neurons) * tau_mem)
        self.tau_syn_param = nn.Parameter(torch.ones(n_neurons) * tau_syn)
        
    def surrogate_spike(self, v: torch.Tensor) -> torch.Tensor:
        """Surrogate gradient for spike function."""
        # Fast sigmoid surrogate
        return torch.sigmoid(self.surrogate_scale * (v - self.v_th))
    
    def forward(self, 
                inputs: torch.Tensor,  # (T, batch, n_inputs)
                init_state: Dict = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through time.
        
        Returns:
            outputs: (T, batch, n_outputs)
            final_state: dict with membrane potentials, synaptic currents
        """
        T, batch, _ = inputs.shape
        device = inputs.device
        
        # State variables
        if init_state is None:
            v = torch.zeros(batch, self.n_neurons, device=device)
            i_syn = torch.zeros(batch, self.n_neurons, device=device)
            spikes = torch.zeros(batch, self.n_neurons, device=device)
        else:
            v = init_state['v']
            i_syn = init_state['i_syn']
            spikes = init_state['spikes']
        
        # Simulation
        outputs = []
        spike_trains = []
        
        for t in range(T):
            # Input current
            i_in = inputs[t] @ self.W_in.t()  # (batch, n_neurons)
            
            # Recurrent current
            # Excitatory connections
            exc_mask = (self.neuron_types == 0).float()
            inh_mask = (self.neuron_types == 1).float()
            
            W_eff = self.W_syn * self.connectivity
            W_exc = W_eff * exc_mask.unsqueeze(0)
            W_inh = W_eff * inh_mask.unsqueeze(0)
            
            i_rec_exc = spikes @ W_exc.t()
            i_rec_inh = spikes @ W_inh.t()
            
            # Total synaptic current
            i_syn = i_syn + (i_in + i_rec_exc - i_rec_inh - i_syn) * (self.dt / self.tau_syn_param)
            
            # Membrane potential
            v = v + (i_syn - v) * (self.dt / self.tau_mem_param)
            
            # Spike generation
            spikes = self.surrogate_spike(v)
            
            # Reset (soft reset for differentiability)
            v = v * (1 - spikes) + self.reset * spikes
            
            # Output
            out = spikes @ self.W_out.t()
            outputs.append(out)
            spike_trains.append(spikes)
        
        outputs = torch.stack(outputs)  # (T, batch, n_outputs)
        spike_trains = torch.stack(spike_trains)  # (T, batch, n_neurons)
        
        final_state = {
            'v': v,
            'i_syn': i_syn,
            'spikes': spikes,
        }
        
        return outputs, final_state
    
    def get_firing_rates(self, spike_trains: torch.Tensor, window: int = None) -> torch.Tensor:
        """Compute firing rates from spike trains."""
        if window is None:
            window = spike_trains.shape[0]
        rates = spike_trains[-window:].mean(dim=0) / (self.dt / 1000.0)  # Hz
        return rates


def create_differentiable_snn_from_connectome(graph, 
                                               sensory_ids: List[int],
                                               motor_ids: List[int],
                                               config_path: str = "config/config.yaml") -> DifferentiableSNN:
    """Create differentiable SNN from connectome graph."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sim_config = config['simulator']
    
    # Map neuron IDs to indices
    all_ids = list(graph.neurons.keys())
    id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
    n_neurons = len(all_ids)
    
    # Build connectivity matrix
    connectivity = torch.zeros(n_neurons, n_neurons)
    neuron_types = torch.zeros(n_neurons)  # 0=exc, 1=inh
    
    for nid, meta in graph.neurons.items():
        idx = id_to_idx[nid]
        # Set neuron type based on neurotransmitter
        nt = meta.neurotransmitter.lower()
        if nt in ['gaba', 'glutamate']:
            neuron_types[idx] = 1  # inhibitory
    
    for pre, post, weight in graph.edges:
        if pre in id_to_idx and post in id_to_idx:
            connectivity[id_to_idx[post], id_to_idx[pre]] = weight
    
    # Normalize connectivity
    connectivity = connectivity / (connectivity.max() + 1e-8)
    
    # Sensory and motor indices
    sensory_idx = [id_to_idx[nid] for nid in sensory_ids if nid in id_to_idx]
    motor_idx = [id_to_idx[nid] for nid in motor_ids if nid in id_to_idx]
    
    snn = DifferentiableSNN(
        n_neurons=n_neurons,
        n_inputs=len(sensory_idx),
        n_outputs=len(motor_idx),
        connectivity=connectivity,
        neuron_types=neuron_types,
        dt=sim_config['dt'],
        tau_mem=sim_config['membrane_time_constant'],
        tau_syn=sim_config['synaptic_time_constant'],
        threshold=sim_config['default_threshold'] * -1,  # Convert to positive
        reset=sim_config['default_reset'] * -1,
    )
    
    # Store mapping for encoding/decoding
    snn.sensory_idx = sensory_idx
    snn.motor_idx = motor_idx
    snn.id_to_idx = id_to_idx
    snn.idx_to_id = {i: nid for nid, i in id_to_idx.items()}
    
    return snn


def create_metalearner(snn, config_path: str = "config/config.yaml") -> SNNMetalearner:
    """Factory function to create metalearner."""
    return SNNMetalearner(snn, config_path)