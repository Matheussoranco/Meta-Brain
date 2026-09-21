"""
Mario Bros environment wrapper for connectome-based agents.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class MarioState:
    """Processed Mario state for neural input."""
    # Visual features (downsampled frame)
    frame: np.ndarray  # (C, H, W)
    # Game state
    x_pos: float
    y_pos: float
    velocity_x: float
    velocity_y: float
    on_ground: bool
    lives: int
    coins: int
    score: int
    time: int
    world: int
    level: int
    # Enemy positions (relative)
    enemies: np.ndarray  # (N, 2) - relative x, y
    # Platform info
    platforms: np.ndarray  # (M, 4) - x, y, w, h


class MarioEnv:
    """
    Mario Bros environment with connectome-friendly interface.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env_config = self.config['environment']
        
        # Import gym-super-mario-bros
        try:
            import gym_super_mario_bros
            from nes_py.wrappers import JoypadSpace
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
        except ImportError:
            raise ImportError("gym-super-mario-bros not installed. Run: pip install gym-super-mario-bros nes-py")
        
        # Create base environment
        self.env = gym_super_mario_bros.make(self.env_config['name'])
        
        # Action space wrapper
        actions = self.env_config['actions']
        self.env = JoypadSpace(self.env, actions)
        
        # Wrappers for preprocessing
        self._setup_wrappers()
        
        # Action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        
        # Frame stack
        self.frame_stack = self.env_config['frame_stack']
        self.frames = []
        
    def _setup_wrappers(self):
        """Setup preprocessing wrappers."""
        from gymnasium.wrappers import (
            FrameStackObservation,
            ResizeObservation,
            GrayScaleObservation,
            TransformObservation,
        )
        
        # Resize
        self.env = ResizeObservation(self.env, self.env_config['resize'])
        
        # Grayscale
        if self.env_config['grayscale']:
            self.env = GrayScaleObservation(self.env, keep_dim=True)
        
        # Frame stack
        self.env = FrameStackObservation(self.env, self.frame_stack)
        
        # Reward scaling
        self.env = TransformObservation(
            self.env,
            lambda obs: obs.astype(np.float32) / 255.0
        )
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        
        self.episode_count += 1
        self.step_count = 0
        self.total_reward = 0.0
        self.frames = [obs] * self.frame_stack
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Scale reward
        reward *= self.env_config['reward_scale']
        
        self.step_count += 1
        self.total_reward += reward
        
        # Check max episode steps
        if self.step_count >= self.env_config['max_episode_steps']:
            truncated = True
        
        return obs, reward, terminated, truncated, info
    
    def get_state(self, info: Dict) -> MarioState:
        """Extract structured state from info dict."""
        return MarioState(
            frame=np.array([]),  # Filled by caller
            x_pos=info.get('x_pos', 0),
            y_pos=info.get('y_pos', 0),
            velocity_x=info.get('velocity_x', 0),
            velocity_y=info.get('velocity_y', 0),
            on_ground=info.get('on_ground', False),
            lives=info.get('lives', 2),
            coins=info.get('coins', 0),
            score=info.get('score', 0),
            time=info.get('time', 400),
            world=info.get('world', 1),
            level=info.get('level', 1),
            enemies=np.array([]),
            platforms=np.array([]),
        )
    
    def encode_for_connectome(self, obs: np.ndarray, info: Dict) -> Dict[str, np.ndarray]:
        """
        Encode observation for connectome input.
        Maps game state to sensory neuron activation patterns.
        """
        state = self.get_state(info)
        state.frame = obs
        
        # Visual input -> optic lobe sensory neurons (OLSN)
        visual_input = self._encode_visual(obs)
        
        # Position/velocity -> proprioceptive/mechanosensory
        proprio_input = self._encode_proprioception(state)
        
        # Enemy detection -> visual projection neurons
        enemy_input = self._encode_enemies(state)
        
        # Reward/punishment -> dopaminergic/octopaminergic
        reward_signal = self._encode_reward(info)
        
        return {
            'visual': visual_input,      # For OLSN neurons
            'proprioception': proprio_input,  # For mechanosensory
            'enemies': enemy_input,      # For VPNs
            'reward': reward_signal,     # For DAN/OAN
            'raw_state': state,
        }
    
    def _encode_visual(self, obs: np.ndarray) -> np.ndarray:
        """Encode visual frame for optic lobe input."""
        # obs shape: (frame_stack, H, W) or (frame_stack, H, W, 1)
        if obs.ndim == 4:
            obs = obs.squeeze(-1)
        
        # Downsample further for connectome input
        # Use max pooling to preserve important features
        from skimage.measure import block_reduce
        try:
            visual = block_reduce(obs[-1], block_size=(4, 4), func=np.max)
        except ImportError:
            # Simple downsampling
            visual = obs[-1, ::4, ::4]
        
        # Normalize
        visual = visual.astype(np.float32) / 255.0
        return visual.flatten()
    
    def _encode_proprioception(self, state: MarioState) -> np.ndarray:
        """Encode proprioceptive state."""
        return np.array([
            state.x_pos / 3000.0,      # Normalized position
            state.y_pos / 240.0,
            state.velocity_x / 50.0,   # Normalized velocity
            state.velocity_y / 50.0,
            float(state.on_ground),
            state.lives / 5.0,
            state.coins / 100.0,
            state.score / 10000.0,
            state.time / 400.0,
        ], dtype=np.float32)
    
    def _encode_enemies(self, state: MarioState) -> np.ndarray:
        """Encode enemy positions relative to Mario."""
        # Placeholder - would extract from RAM or frame analysis
        # For now, return zeros
        return np.zeros(10, dtype=np.float32)
    
    def _encode_reward(self, info: Dict) -> np.ndarray:
        """Encode reward signals for neuromodulatory systems."""
        reward = info.get('reward', 0)
        return np.array([
            max(reward, 0),      # Positive reward -> dopamine
            max(-reward, 0),     # Negative reward -> punishment
        ], dtype=np.float32)
    
    def decode_motor_output(self, motor_rates: Dict[int, float]) -> int:
        """
        Decode motor neuron firing rates to game action.
        
        Maps descending neuron (DN) activity to discrete actions.
        """
        # Simple decoding: map specific DN types to actions
        # This should be learned through metalearning
        action_values = np.zeros(self.action_space.n)
        
        # Example mapping (to be learned)
        # DN types for different behaviors:
        # DNg13 -> turning
        # DNg31 -> walking
        # DNg04 -> jumping
        
        # For now, simple heuristic
        total_rate = sum(motor_rates.values())
        if total_rate > 0:
            # Use relative rates to choose action
            rates_array = np.array(list(motor_rates.values()))
            action_idx = np.argmax(rates_array) % self.action_space.n
            return action_idx
        
        return 0  # NOOP
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def render(self):
        """Render environment."""
        return self.env.render()


class MarioVecEnv:
    """Vectorized Mario environments for parallel training."""
    
    def __init__(self, num_envs: int, config_path: str = "config/config.yaml"):
        self.num_envs = num_envs
        self.envs = [MarioEnv(config_path) for _ in range(num_envs)]
        self.config_path = config_path
    
    def reset(self, seeds: List[int] = None) -> List[Tuple[np.ndarray, Dict]]:
        if seeds is None:
            seeds = [None] * self.num_envs
        return [env.reset(seed) for env, seed in zip(self.envs, seeds)]
    
    def step(self, actions: List[int]) -> List[Tuple[np.ndarray, float, bool, bool, Dict]]:
        return [env.step(action) for env, action in zip(self.envs, actions)]
    
    def encode_batch(self, observations: List[np.ndarray], infos: List[Dict]) -> List[Dict]:
        return [env.encode_for_connectome(obs, info) 
                for env, obs, info in zip(self.envs, observations, infos)]
    
    def decode_batch(self, motor_outputs: List[Dict[int, float]]) -> List[int]:
        return [env.decode_motor_output(motor) 
                for env, motor in zip(self.envs, motor_outputs)]
    
    def close(self):
        for env in self.envs:
            env.close()


def create_mario_env(config_path: str = "config/config.yaml") -> MarioEnv:
    """Factory function to create Mario environment."""
    return MarioEnv(config_path)


def create_mario_vec_env(num_envs: int, config_path: str = "config/config.yaml") -> MarioVecEnv:
    """Factory function to create vectorized Mario environment."""
    return MarioVecEnv(num_envs, config_path)