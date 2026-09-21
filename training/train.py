"""
Main training pipeline for Meta-Brain.
Orchestrates connectome loading, SNN simulation, environment, and metalearning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yaml
import logging
from pathlib import Path
import time
from tqdm import tqdm

# Local imports
from connectome.loader import ConnectomeLoader, create_default_loader
from simulator.snn import create_snn, DrosophilaSNN, RateCodedSNN
from environment.mario_env import create_mario_env, create_mario_vec_env, MarioEnv
from metalearning.algorithms import (MetalearningConfig, MAML, Reptile, 
                                       SNNMetalearner, DifferentiableSNN,
                                       create_differentiable_snn_from_connectome,
                                       create_metalearner)


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    iteration: int = 0
    meta_loss_history: List[float] = None
    task_performance: Dict[str, List[float]] = None
    snn_state: Dict = None
    optimizer_state: Dict = None
    best_performance: float = -float('inf')
    
    def __post_init__(self):
        if self.meta_loss_history is None:
            self.meta_loss_history = []
        if self.task_performance is None:
            self.task_performance = {}


class MetaBrainTrainer:
    """
    Main training orchestrator for Meta-Brain.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.training_config = self.config['training']
        self.paths = self.config['paths']
        
        # Setup logging
        self._setup_logging()
        
        # Components (initialized in setup())
        self.loader: Optional[ConnectomeLoader] = None
        self.connectome_graph = None
        self.snn = None
        self.env = None
        self.metalearner = None
        self.diff_snn = None
        self.maml = None
        
        # Training state
        self.state = TrainingState()
        self.device = torch.device(self.training_config['device'] 
                                    if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        for path_key in ['data', 'checkpoints', 'logs', 'results', 'cache']:
            Path(self.paths[path_key]).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config['logging']
        log_level = getattr(logging, log_config['level'])
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.paths['logs']) / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MetaBrain')
    
    def setup_connectome(self, use_local_files: bool = True) -> bool:
        """Load and prepare connectome data."""
        self.logger.info("Setting up connectome...")
        
        self.loader = create_default_loader(self.config_path)
        
        try:
            if use_local_files:
                # Try loading from local files
                self.connectome_graph = self.loader.load_from_files()
            else:
                # Would need neuPrint token
                token = self.config['connectome'].get('token')
                if token:
                    self.connectome_graph = self.loader.load_from_neuprint(token)
                else:
                    self.logger.warning("No neuPrint token, trying local files")
                    self.connectome_graph = self.loader.load_from_files()
            
            self.logger.info(f"Loaded connectome: {len(self.connectome_graph.neurons)} neurons, "
                           f"{len(self.connectome_graph.edges)} connections")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load connectome: {e}")
            return False
    
    def setup_sensorimotor_subgraph(self):
        """Extract sensorimotor subgraph for simulation."""
        self.logger.info("Extracting sensorimotor subgraph...")
        
        self.connectome_graph = self.loader.get_sensorimotor_subgraph(
            max_interneuron_layers=3
        )
        
        self.logger.info(f"Sensorimotor subgraph: {len(self.connectome_graph.neurons)} neurons, "
                        f"{len(self.connectome_graph.edges)} connections")
    
    def setup_snn(self, rate_coded: bool = True):
        """Initialize SNN simulator."""
        self.logger.info(f"Setting up {'rate-coded' if rate_coded else 'spiking'} SNN...")
        
        # Prepare data for SNN
        neuron_data = []
        for nid, meta in self.connectome_graph.neurons.items():
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
        
        edge_data = []
        for pre, post, weight in self.connectome_graph.edges:
            edge_data.append({'pre': pre, 'post': post, 'weight': weight})
        
        # Get sensory and motor neuron IDs
        sensory_ids = self.connectome_graph.get_sensory_neurons()
        motor_ids = self.connectome_graph.get_motor_neurons()
        
        self.logger.info(f"Sensory neurons: {len(sensory_ids)}, Motor neurons: {len(motor_ids)}")
        
        # Create SNN
        self.snn = create_snn(self.config_path, rate_coded=rate_coded)
        self.snn.build_from_connectome(neuron_data, edge_data, sensory_ids, motor_ids)
        
        # Also create differentiable SNN for gradient-based metalearning
        if rate_coded:
            self.diff_snn = create_differentiable_snn_from_connectome(
                self.connectome_graph, sensory_ids, motor_ids, self.config_path
            ).to(self.device)
            
            # Create MAML wrapper
            metalearning_config = MetalearningConfig(**self.config['metalearning'])
            self.maml = MAML(self.diff_snn, metalearning_config)
            self.logger.info("Created differentiable SNN and MAML")
    
    def setup_environment(self):
        """Initialize Mario environment."""
        self.logger.info("Setting up Mario environment...")
        
        self.env = create_mario_env(self.config_path)
        self.logger.info("Mario environment ready")
    
    def setup_metalearning(self):
        """Initialize metalearning system."""
        self.logger.info("Setting up metalearning...")
        
        self.metalearner = create_metalearner(self.snn, self.config_path)
        
        # Define tasks
        self._define_tasks()
        
        self.logger.info(f"Defined {len(self.metalearner.tasks)} tasks")
    
    def _define_tasks(self):
        """Define metalearning tasks for Mario."""
        
        # Task 1: Basic movement
        def sensory_mapping_basic(obs, info):
            return self.env.encode_for_connectome(obs, info)
        
        def motor_decoding_basic(motor_rates):
            return self.env.decode_motor_output(motor_rates)
        
        def reward_basic(info):
            # Reward for moving right, staying alive
            x_pos = info.get('x_pos', 0)
            return x_pos * 0.01 + info.get('reward', 0)
        
        self.metalearner.create_task(
            name="basic_movement",
            sensory_mapping=sensory_mapping_basic,
            motor_decoding=motor_decoding_basic,
            reward_fn=reward_basic,
            env_config={'level': '1-1'}
        )
        
        # Task 2: Enemy avoidance
        def reward_avoidance(info):
            base = info.get('x_pos', 0) * 0.01
            # Penalty for dying
            if info.get('life_lost', False):
                base -= 10
            return base
        
        self.metalearner.create_task(
            name="enemy_avoidance",
            sensory_mapping=sensory_mapping_basic,
            motor_decoding=motor_decoding_basic,
            reward_fn=reward_avoidance,
            env_config={'level': '1-1'}
        )
        
        # Task 3: Jumping
        def reward_jumping(info):
            base = info.get('x_pos', 0) * 0.01
            # Bonus for jumping (y velocity)
            if info.get('velocity_y', 0) > 0:
                base += 0.1
            return base
        
        self.metalearner.create_task(
            name="jumping",
            sensory_mapping=sensory_mapping_basic,
            motor_decoding=motor_decoding_basic,
            reward_fn=reward_jumping,
            env_config={'level': '1-1'}
        )
        
        # Task 4: Level completion
        def reward_completion(info):
            base = info.get('x_pos', 0) * 0.01
            if info.get('flag_get', False):
                base += 100
            return base
        
        self.metalearner.create_task(
            name="level_completion",
            sensory_mapping=sensory_mapping_basic,
            motor_decoding=motor_decoding_basic,
            reward_fn=reward_completion,
            env_config={'level': '1-1'}
        )
    
    def run_random_baseline(self, num_episodes: int = 10) -> Dict:
        """Run random action baseline."""
        self.logger.info(f"Running random baseline for {num_episodes} episodes...")
        
        total_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = 0
            ep_len = 0
            
            while not done:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_len += 1
            
            total_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            self.logger.info(f"Episode {ep}: reward={ep_reward:.2f}, length={ep_len}")
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_length': np.mean(episode_lengths),
        }
    
    def run_snn_episode(self, snn, task: Dict, max_steps: int = 1000) -> Tuple[float, Dict]:
        """Run a single episode with SNN control."""
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # For rate-coded SNN
        if isinstance(snn, RateCodedSNN):
            while not done and step < max_steps:
                # Encode observation
                encoded = task['sensory_mapping'](obs, info)
                
                # Set sensory input
                visual_input = encoded['visual']
                # Map to sensory populations
                sensory_rates = {}
                for i, pop_idx in enumerate(snn.sensory_pops):
                    if i < len(visual_input):
                        sensory_rates[pop_idx] = visual_input[i] * 100  # Scale to Hz
                
                snn.set_sensory_input(sensory_rates)
                
                # Run SNN for decision interval
                snn.run(self.config['metalearning']['decision_interval'])
                
                # Get motor output
                motor_rates = snn.get_motor_output()
                
                # Decode action
                action = task['motor_decoding'](motor_rates)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
        
        # For spiking SNN (Brian2)
        elif isinstance(snn, DrosophilaSNN):
            while not done and step < max_steps:
                encoded = task['sensory_mapping'](obs, info)
                
                # Convert to sensory rates
                sensory_rates = {}
                visual = encoded['visual']
                sensory_ids = snn.sensory_indices
                for i, nid in enumerate(sensory_ids):
                    if i < len(visual):
                        sensory_rates[nid] = visual[i] * 100
                
                # Inject sensory input
                snn.inject_sensory_input(sensory_rates, 
                                        self.config['metalearning']['decision_interval'])
                
                # Get motor output
                motor_rates = snn.get_motor_output()
                
                # Decode action
                action = task['motor_decoding'](motor_rates)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
        
        return total_reward, info
    
    def evaluate_snn(self, snn, num_episodes: int = 5) -> Dict:
        """Evaluate SNN on all tasks."""
        results = {}
        
        for task in self.metalearner.tasks:
            task_rewards = []
            for _ in range(num_episodes):
                reward, _ = self.run_snn_episode(snn, task)
                task_rewards.append(reward)
            
            results[task['name']] = {
                'mean_reward': np.mean(task_rewards),
                'std_reward': np.std(task_rewards),
            }
            
            # Track for curriculum
            if task['name'] not in self.state.task_performance:
                self.state.task_performance[task['name']] = []
            self.state.task_performance[task['name']].append(np.mean(task_rewards))
        
        return results
    
    def train_rate_snn_maml(self, num_iterations: int = 1000):
        """Train rate-coded SNN with MAML."""
        self.logger.info("Starting rate-coded SNN MAML training...")
        
        if self.diff_snn is None or self.maml is None:
            self.logger.error("Differentiable SNN or MAML not initialized")
            return
        
        # Meta-training loop
        for iteration in tqdm(range(num_iterations), desc="Meta-training"):
            self.state.iteration = iteration
            
            # Sample tasks
            task_batch = self.metalearner.generate_task_batch()
            
            # Prepare task data for MAML
            tasks_data = []
            for task in task_batch:
                # Generate train/test data for this task
                train_x, train_y, test_x, test_y = self._generate_task_data(task)
                tasks_data.append((train_x, train_y, test_x, test_y))
            
            # MAML meta-update
            loss_fn = nn.MSELoss()
            metrics = self.maml.meta_update(tasks_data, loss_fn)
            
            self.state.meta_loss_history.append(metrics['meta_loss'])
            
            # Log
            if iteration % self.training_config['log_interval'] == 0:
                self.logger.info(f"Iter {iteration}: meta_loss={metrics['meta_loss']:.4f}")
            
            # Evaluate
            if iteration % self.training_config['eval_interval'] == 0:
                eval_results = self.evaluate_snn(self.snn)
                self.logger.info(f"Evaluation: {eval_results}")
                
                # Checkpoint
                self.save_checkpoint(f"checkpoint_iter_{iteration}.pt")
            
            # Save periodic checkpoint
            if iteration % self.training_config['checkpoint_interval'] == 0:
                self.save_checkpoint(f"checkpoint_latest.pt")
        
        self.logger.info("Training complete")
        self.save_checkpoint("checkpoint_final.pt")
    
    def _generate_task_data(self, task: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test data for a task by running episodes."""
        # This is a simplified version - in practice, would collect trajectories
        # and create supervised learning targets
        
        n_samples = 32
        n_inputs = len(self.diff_snn.sensory_idx)
        n_outputs = len(self.diff_snn.motor_idx)
        T = 10  # time steps
        
        # Random data for now - replace with actual trajectory collection
        train_x = torch.randn(n_samples, T, n_inputs).to(self.device)
        train_y = torch.randn(n_samples, T, n_outputs).to(self.device)
        test_x = torch.randn(n_samples // 2, T, n_inputs).to(self.device)
        test_y = torch.randn(n_samples // 2, T, n_outputs).to(self.device)
        
        return train_x, train_y, test_x, test_y
    
    def train_with_plasticity(self, num_episodes: int = 1000):
        """Train SNN using STDP plasticity and neuromodulation."""
        self.logger.info("Starting plasticity-based training...")
        
        # Enable plasticity
        self.snn.set_plasticity(True)
        
        for episode in tqdm(range(num_episodes), desc="Plasticity training"):
            # Select task (curriculum)
            task = self._select_curriculum_task(episode)
            
            # Run episode
            reward, info = self.run_snn_episode(self.snn, task)
            
            # Apply neuromodulatory signals based on reward
            self._apply_neuromodulation(reward, info)
            
            # Log
            if episode % self.training_config['log_interval'] == 0:
                self.logger.info(f"Episode {episode}: reward={reward:.2f}")
            
            # Evaluate
            if episode % self.training_config['eval_interval'] == 0:
                eval_results = self.evaluate_snn(self.snn)
                self.logger.info(f"Evaluation: {eval_results}")
        
        # Disable plasticity after training
        self.snn.set_plasticity(False)
        self.save_checkpoint("plasticity_final.pt")
    
    def _select_curriculum_task(self, episode: int) -> Dict:
        """Select task based on curriculum."""
        curriculum = self.config['training']['curriculum']
        if not curriculum['enabled']:
            return np.random.choice(self.metalearner.tasks)
        
        total = 0
        for stage in curriculum['stages']:
            total += stage['episodes']
            if episode < total:
                # Find task matching stage
                for task in self.metalearner.tasks:
                    if stage['name'] in task['name']:
                        return task
        
        return self.metalearner.tasks[-1]
    
    def _apply_neuromodulation(self, reward: float, info: Dict):
        """Apply dopamine/octopamine signals to modulate STDP."""
        # This would modify STDP parameters based on reward
        # Positive reward -> enhance LTP (dopamine)
        # Negative reward -> enhance LTD (punishment)
        pass
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = Path(self.paths['checkpoints']) / filename
        
        checkpoint = {
            'iteration': self.state.iteration,
            'meta_loss_history': self.state.meta_loss_history,
            'task_performance': self.state.task_performance,
            'config': self.config,
        }
        
        if self.diff_snn is not None:
            checkpoint['diff_snn_state'] = self.diff_snn.state_dict()
        
        if self.maml is not None:
            checkpoint['maml_optimizer'] = self.maml.meta_optimizer.state_dict()
        
        if hasattr(self.snn, 'save_weights'):
            weights_path = path.with_suffix('.weights.npz')
            self.snn.save_weights(str(weights_path))
            checkpoint['weights_path'] = str(weights_path)
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        path = Path(self.paths['checkpoints']) / filename
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.state.iteration = checkpoint['iteration']
        self.state.meta_loss_history = checkpoint['meta_loss_history']
        self.state.task_performance = checkpoint['task_performance']
        
        if self.diff_snn is not None and 'diff_snn_state' in checkpoint:
            self.diff_snn.load_state_dict(checkpoint['diff_snn_state'])
        
        if self.maml is not None and 'maml_optimizer' in checkpoint:
            self.maml.meta_optimizer.load_state_dict(checkpoint['maml_optimizer'])
        
        if 'weights_path' in checkpoint and hasattr(self.snn, 'load_weights'):
            self.snn.load_weights(checkpoint['weights_path'])
        
        self.logger.info(f"Loaded checkpoint from {path}")
    
    def run_full_pipeline(self, mode: str = "plasticity"):
        """Run the complete training pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("META-BRAIN TRAINING PIPELINE")
        self.logger.info("=" * 60)
        
        # Setup
        if not self.setup_connectome():
            self.logger.error("Connectome setup failed")
            return
        
        self.setup_sensorimotor_subgraph()
        self.setup_snn(rate_coded=(mode == "maml"))
        self.setup_environment()
        self.setup_metalearning()
        
        # Random baseline
        baseline = self.run_random_baseline(10)
        self.logger.info(f"Random baseline: {baseline}")
        
        # Initial evaluation
        initial_eval = self.evaluate_snn(self.snn)
        self.logger.info(f"Initial evaluation: {initial_eval}")
        
        # Training
        if mode == "maml":
            self.train_rate_snn_maml(self.training_config['max_iterations'])
        elif mode == "plasticity":
            self.train_with_plasticity(5000)
        else:
            self.logger.error(f"Unknown mode: {mode}")
        
        # Final evaluation
        final_eval = self.evaluate_snn(self.snn)
        self.logger.info(f"Final evaluation: {final_eval}")
        
        # Save results
        self.save_results({
            'baseline': baseline,
            'initial': initial_eval,
            'final': final_eval,
            'history': self.state.meta_loss_history,
        })
        
        self.env.close()
        self.logger.info("Pipeline complete")
    
    def save_results(self, results: Dict):
        """Save training results."""
        import json
        path = Path(self.paths['results']) / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results = convert(results)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to {path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Brain: Connectome-based Metalearning for Mario Bros")
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['maml', 'plasticity', 'baseline'], default='plasticity',
                        help='Training mode')
    parser.add_argument('--checkpoint', help='Checkpoint to resume from')
    parser.add_argument('--local-files', action='store_true', help='Use local connectome files')
    args = parser.parse_args()
    
    trainer = MetaBrainTrainer(args.config)
    
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    if args.mode == 'baseline':
        trainer.setup_connectome(args.local_files)
        trainer.setup_environment()
        trainer.run_random_baseline(20)
    else:
        trainer.run_full_pipeline(mode=args.mode)


if __name__ == '__main__':
    main()