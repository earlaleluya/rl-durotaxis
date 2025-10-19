#!/usr/bin/env python3
"""
Practical Training Loop for Hybrid Actor-Critic with Multi-Component Rewards

A streamlined, runnable training implementation that demonstrates:
1. Component-specific value learning with your reward system
2. Action masking for topology constraints
3. Efficient experience collection and policy updates
4. Progressive learning with reward component weighting

This is designed to run immediately and show real training progress.
"""

import sys
import os
import json
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Any
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class TrajectoryBuffer:
    """Buffer to store multiple episodes of trajectories for batch training"""
    
    def __init__(self):
        self.episodes = []
        self.current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'final_values': None,
            'terminated': False,
            'success': False
        }
    
    def start_episode(self):
        """Start a new episode in the buffer"""
        self.current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'old_values': [],  # Store old values for PPO value clipping
            'log_probs': [],
            'final_values': None,
            'terminated': False,
            'success': False
        }
    
    def add_step(self, state, action, reward, value, log_prob, old_value=None):
        """Add a step to the current episode"""
        self.current_episode['states'].append(state)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['values'].append(value)
        self.current_episode['old_values'].append(old_value if old_value is not None else value)
        self.current_episode['log_probs'].append(log_prob)
    
    def finish_episode(self, final_values, terminated, success):
        """Finish the current episode and add it to the buffer"""
        self.current_episode['final_values'] = final_values
        self.current_episode['terminated'] = terminated
        self.current_episode['success'] = success
        self.episodes.append(self.current_episode)
    
    def get_batch_data(self):
        """Get all trajectories as batch data"""
        if not self.episodes:
            return None
            
        # Concatenate all episodes into single lists
        all_states = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_old_values = []
        all_log_probs = []
        all_returns = []
        all_advantages = []
        
        for episode in self.episodes:
            all_states.extend(episode['states'])
            all_actions.extend(episode['actions'])
            all_rewards.extend(episode['rewards'])
            all_values.extend(episode['values'])
            all_old_values.extend(episode['old_values'])
            all_log_probs.extend(episode['log_probs'])
            all_returns.extend(episode['returns'])
            all_advantages.extend(episode['advantages'])
        
        return {
            'states': all_states,
            'actions': all_actions,
            'rewards': all_rewards,
            'values': all_values,
            'old_values': all_old_values,
            'log_probs': all_log_probs,
            'returns': all_returns,
            'advantages': all_advantages
        }
    
    def create_minibatches(self, minibatch_size: int):
        """Create random minibatches from the buffer data"""
        batch_data = self.get_batch_data()
        if not batch_data:
            return []
            
        total_steps = len(batch_data['states'])
        indices = list(range(total_steps))
        random.shuffle(indices)
        
        minibatches = []
        for i in range(0, total_steps, minibatch_size):
            end_idx = min(i + minibatch_size, total_steps)
            batch_indices = indices[i:end_idx]
            
            minibatch = {}
            for key, data in batch_data.items():
                minibatch[key] = [data[idx] for idx in batch_indices]
            
            minibatches.append(minibatch)
        
        return minibatches
    
    def compute_returns_and_advantages_for_all_episodes(self, gamma: float, gae_lambda: float):
        """Compute returns and advantages for all episodes in the buffer"""
        for episode in self.episodes:
            rewards = episode['rewards']
            values = episode['values']
            final_values = episode['final_values']
            terminated = episode['terminated']

            # Extract tensor from dict if needed
            processed_values = []
            for v in values:
                if isinstance(v, dict):
                    # Try 'total_value' first, fallback to 'total_reward', else error
                    if 'total_value' in v:
                        processed_values.append(v['total_value'])
                    elif 'total_reward' in v:
                        processed_values.append(v['total_reward'])
                    else:
                        raise ValueError(f"Value dict missing 'total_value' and 'total_reward': {v}")
                else:
                    processed_values.append(v)

            # Convert values to tensors for easier computation
            values_tensor = torch.stack(processed_values) if processed_values else torch.tensor([])
            final_value = None
            if isinstance(final_values, dict):
                if 'total_value' in final_values:
                    final_value = final_values['total_value']
                elif 'total_reward' in final_values:
                    final_value = final_values['total_reward']
                else:
                    final_value = torch.tensor(0.0)
            else:
                final_value = final_values if final_values is not None else torch.tensor(0.0)

            # Compute GAE advantages and returns
            returns = []
            advantages = []
            gae = 0

            # Work backwards through the episode
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = final_value if not terminated else torch.tensor(0.0)
                else:
                    next_value = values_tensor[t + 1]

                # Get reward components
                reward_dict = rewards[t]
                reward = reward_dict.get('total_reward', 0.0)

                # TD error
                delta = reward + gamma * next_value - values_tensor[t]

                # GAE calculation
                gae = delta + gamma * gae_lambda * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values_tensor[t])

            # Store computed values
            episode['returns'] = returns
            episode['advantages'] = advantages
    
    def clear(self):
        """Clear all episodes from the buffer"""
        self.episodes = []
        self.current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'old_values': [],
            'log_probs': [],
            'final_values': None,
            'terminated': False,
            'success': False
        }
    
    def __len__(self):
        """Return the number of episodes in the buffer"""
        return len(self.episodes)
    
    def get_episode_stats(self):
        """Get statistics about episodes in the buffer"""
        if not self.episodes:
            return {}
            
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in self.episodes:
            total_reward = sum(r.get('total_reward', 0.0) for r in episode['rewards'])
            episode_rewards.append(total_reward)
            episode_lengths.append(len(episode['rewards']))
            if episode['success']:
                success_count += 1
        
        return {
            'num_episodes': len(self.episodes),
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'avg_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'success_rate': success_count / len(self.episodes) if self.episodes else 0.0,
            'total_steps': sum(episode_lengths)
        }
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


from durotaxis_env import DurotaxisEnv
from state import TopologyState  
from encoder import GraphInputEncoder
from actor_critic import HybridActorCritic
from config_loader import ConfigLoader


def moving_average(data, window=20):
    """Calculate moving average for smoother trend tracking"""
    if len(data) < window:
        return np.mean(data) if data else 0.0
    return np.mean(data[-window:])


def exponential_moving_average(data, alpha=0.1):
    """Calculate exponential moving average for responsive smoothing"""
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    
    ema = data[0]
    for value in data[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema


def robust_moving_average(data, window=20, outlier_threshold=2.0):
    """Calculate moving average with outlier removal for robust smoothing"""
    if len(data) < window:
        recent_data = data
    else:
        recent_data = data[-window:]
    
    if not recent_data:
        return 0.0
    
    # Remove outliers using z-score
    if len(recent_data) > 3:
        mean = np.mean(recent_data)
        std = np.std(recent_data)
        if std > 0:
            z_scores = np.abs((np.array(recent_data) - mean) / std)
            filtered_data = [recent_data[i] for i in range(len(recent_data)) if z_scores[i] < outlier_threshold]
            if filtered_data:
                return np.mean(filtered_data)
    
    return np.mean(recent_data)


class DurotaxisTrainer:
    """Streamlined trainer for hybrid actor-critic with component rewards"""
    
    def __init__(self, 
                 config_path: str = "config.yaml",
                 **overrides):
        """
        Initialize trainer with configuration from YAML file
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        **overrides
            Parameter overrides for any configuration values
        """
        
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        # Get trainer configuration with overrides
        config = self.config_loader.get_trainer_config()
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        
        # Apply configuration
        self.total_episodes = config.get('total_episodes', 1000)
        self.max_steps = config.get('max_steps', 200)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = config.get('save_dir', "./training_results")
        self.entropy_bonus_coeff = config.get('entropy_bonus_coeff', 0.05)  # Increased for stronger regularization
        self.moving_avg_window = config.get('moving_avg_window', 20)
        self.log_every = config.get('log_every', 50)
        self.progress_print_every = config.get('progress_print_every', 5)
        self.checkpoint_every = config.get('checkpoint_every', None)
        self.substrate_type = config.get('substrate_type', 'random')
        
        # Environment setup parameters (load from environment section)
        env_config = self.config_loader.get_environment_config()
        self.substrate_size = tuple(env_config.get('substrate_size', [200, 200]))
        self.init_num_nodes = env_config.get('init_num_nodes', 1)
        self.max_critical_nodes = env_config.get('max_critical_nodes', 50)
        self.threshold_critical_nodes = env_config.get('threshold_critical_nodes', 200)
        self.delta_time = env_config.get('delta_time', 3)
        self.delta_intensity = env_config.get('delta_intensity', 2.50)
        
        # Encoder configuration parameters
        self.encoder_hidden_dim = config.get('encoder_hidden_dim', 128)
        self.encoder_output_dim = config.get('encoder_output_dim', 64)
        self.encoder_num_layers = config.get('encoder_num_layers', 4)
        
        # Random substrate parameter ranges
        self.linear_m_range = tuple(config.get('linear_m_range', [0.01, 0.1]))
        self.linear_b_range = tuple(config.get('linear_b_range', [0.5, 2.0]))
        self.exponential_m_range = tuple(config.get('exponential_m_range', [0.01, 0.05]))
        self.exponential_b_range = tuple(config.get('exponential_b_range', [0.5, 1.5]))
        
        # Component weights and policy weights from config
        self.component_weights = config.get('component_weights', {
            'total_reward': 1.0,
            'graph_reward': 0.4,
            'spawn_reward': 0.3,
            'delete_reward': 0.2,
            'edge_reward': 0.2,
            'total_node_reward': 0.3
        })
        
        self.policy_loss_weights = config.get('policy_loss_weights', {
            'discrete_weight': 0.7,
            'continuous_weight': 0.3,
            'entropy_weight': 0.01
        })
        
        # Load clip_epsilon from algorithm section
        algorithm_config = self.config_loader.get_algorithm_config()
        self.policy_loss_weights['clip_epsilon'] = algorithm_config.get('clip_epsilon', 0.2)
        
        # Adaptive weighting parameters
        self.weight_update_momentum = config.get('weight_momentum', 0.9)
        self.normalize_weights_every = config.get('normalize_weights_every', 10)
        self.enable_adaptive_scaling = config.get('enable_adaptive_scaling', True)
        self.scaling_warmup_episodes = config.get('scaling_warmup_episodes', 50)
        
        # Reward normalization configuration
        reward_norm_config = config.get('reward_normalization', {})
        self.enable_per_episode_norm = reward_norm_config.get('enable_per_episode_norm', True)
        self.enable_cross_episode_scaling = reward_norm_config.get('enable_cross_episode_scaling', True)
        self.min_episode_length = reward_norm_config.get('min_episode_length', 5)
        self.normalization_method = reward_norm_config.get('normalization_method', 'adaptive')
        
        # Enhanced learnable component weighting configuration
        advantage_config = config.get('advantage_weighting', {})
        self.enable_learnable_weights = advantage_config.get('enable_learnable_weights', True)
        self.enable_attention_weighting = advantage_config.get('enable_attention_weighting', True)
        self.weight_learning_rate = advantage_config.get('weight_learning_rate', 0.01)
        self.weight_regularization = advantage_config.get('weight_regularization', 0.001)
        
        # Enhanced entropy regularization configuration
        entropy_config = config.get('entropy_regularization', {})
        self.enable_adaptive_entropy = entropy_config.get('enable_adaptive_entropy', True)
        self.entropy_coeff_start = entropy_config.get('entropy_coeff_start', 0.2)  # Start higher
        self.entropy_coeff_end = entropy_config.get('entropy_coeff_end', 0.01)    # End higher
        self.entropy_decay_episodes = entropy_config.get('entropy_decay_episodes', 800)  # Decay slower
        self.discrete_entropy_weight = entropy_config.get('discrete_entropy_weight', 1.2) # Stronger for discrete
        self.continuous_entropy_weight = entropy_config.get('continuous_entropy_weight', 0.7) # Stronger for continuous
        self.min_entropy_threshold = entropy_config.get('min_entropy_threshold', 0.15) # Higher threshold
        
        # Batch training configuration
        self.rollout_batch_size = config.get('rollout_batch_size', 10)
        self.update_epochs = config.get('update_epochs', 4)
        self.minibatch_size = config.get('minibatch_size', 64)
        
        # Value clipping configuration
        self.enable_value_clipping = config.get('enable_value_clipping', True)
        self.value_clip_epsilon = config.get('value_clip_epsilon', 0.2)
        
        # Adaptive gradient scaling configuration
        gradient_config = config.get('gradient_scaling', {})
        self.enable_gradient_scaling = gradient_config.get('enable_adaptive_scaling', True)
        self.gradient_norm_target = gradient_config.get('gradient_norm_target', 1.0)
        self.scaling_momentum = gradient_config.get('scaling_momentum', 0.9)
        self.min_scaling_factor = gradient_config.get('min_scaling_factor', 0.1)
        self.max_scaling_factor = gradient_config.get('max_scaling_factor', 10.0)
        self.gradient_warmup_steps = gradient_config.get('warmup_steps', 100)
        
        # Initialize gradient scaling state
        self.gradient_step_count = 0
        self.discrete_grad_norm_ema = None
        self.continuous_grad_norm_ema = None
        self.adaptive_discrete_weight = self.policy_loss_weights['discrete_weight']
        self.adaptive_continuous_weight = self.policy_loss_weights['continuous_weight']
        
        # Empty graph handling configuration
        empty_graph_config = config.get('empty_graph_handling', {})
        self.enable_graceful_recovery = empty_graph_config.get('enable_graceful_recovery', True)
        self.recovery_num_nodes = empty_graph_config.get('recovery_num_nodes', 1)
        self.log_recoveries = empty_graph_config.get('log_recoveries', True)
        self.empty_graph_recovery_count = 0  # Track recovery statistics
        
        # NEW: Curriculum learning configuration
        experimental_config = self.config_loader.config.get('experimental', {})
        curriculum_config = experimental_config.get('curriculum_learning', {})
        # Also allow curriculum definition under top-level trainer section (common in config.yaml)
        trainer_curriculum = self.config_loader.config.get('trainer', {}).get('curriculum_learning', {})
        # Prefer trainer section if present, otherwise fall back to experimental
        if trainer_curriculum:
            curriculum_config = trainer_curriculum

        self.enable_curriculum = curriculum_config.get('enable_curriculum', False)

        # Keep some legacy fields (not required when using explicit stage definitions)
        self.phase_1_episodes = curriculum_config.get('phase_1_episodes', 300)
        self.phase_2_episodes = curriculum_config.get('phase_2_episodes', 600)
        self.phase_1_config = curriculum_config.get('phase_1_config', {})
        self.phase_2_config = curriculum_config.get('phase_2_config', {})
        self.phase_3_config = curriculum_config.get('phase_3_config', {})

        # Build scaled curriculum stages that map to trainer.total_episodes if enabled
        self.curriculum_stages = []  # list of dicts: {'name','start','end','config'}
        if self.enable_curriculum:
            try:
                # Ask ConfigLoader for normalized curriculum config if available
                normalized_curr = self.config_loader.get_curriculum_config()
                # Merge normalized settings into curriculum_config (normalized takes precedence)
                if normalized_curr:
                    # copy normalized stages/overlap into curriculum_config variable used below
                    curriculum_config = {**curriculum_config, **normalized_curr}

                self._build_scaled_curriculum(curriculum_config)
            except Exception:
                # Non-fatal: leave curriculum_stages empty and fallback to legacy behavior
                self.curriculum_stages = []
        
        # NEW: Success criteria configuration
        success_config = experimental_config.get('success_criteria', {})
        self.enable_multiple_criteria = success_config.get('enable_multiple_criteria', False)
        self.survival_success_steps = success_config.get('survival_success_steps', 10)
        self.reward_success_threshold = success_config.get('reward_success_threshold', -20)
        self.growth_success_nodes = success_config.get('growth_success_nodes', 2)
        self.exploration_success_steps = success_config.get('exploration_success_steps', 15)
        
        # Initialize trajectory buffer for batch training
        self.trajectory_buffer = TrajectoryBuffer()
        
        # Current entropy coefficient (will be adapted during training)
        self.current_entropy_coeff = self.entropy_coeff_start
        
        # Create run directory with automatic numbering
        self.run_dir = self.create_run_directory(self.save_dir)
        print(f"📁 Created run directory: {self.run_dir} (Run #{self.run_number})")
        
        # Get environment configuration
        env_config = self.config_loader.get_environment_config()
        
        # Environment setup (propagate empty-graph recovery preferences)
        env_overrides = dict(overrides)
        env_overrides.setdefault('empty_graph_recovery_enabled', self.enable_graceful_recovery)
        env_overrides.setdefault('empty_graph_recovery_nodes', self.recovery_num_nodes)

        self.env = DurotaxisEnv(
            config_path=config_path,
            **env_overrides  # Allow overrides for environment parameters too
        )
        
        # Initialize substrate based on type
        self._initialize_substrate()
        
        # Initialize the state extractor with the environment's topology
        self.state_extractor = TopologyState()
        
        # Test environment setup
        test_obs, test_info = self.env.reset()
        self.state_extractor.set_topology(self.env.topology)
        print(f"   Environment initialized: {self.env.topology.graph.num_nodes()} nodes")
        
        self.weight_updates_count = 0                     # Track number of updates for normalization
        
        # Component configuration - match your environment's components
        self.component_names = [
            'total_reward',
            'graph_reward', 
            'spawn_reward',
            'delete_reward',
            'edge_reward',
            'total_node_reward'
        ]
        
        # Create network
        encoder_config = self.config_loader.get_encoder_config()
        actor_critic_config = self.config_loader.get_actor_critic_config()
        
        self.encoder = GraphInputEncoder(
            hidden_dim=encoder_config.get('hidden_dim', 128),
            out_dim=encoder_config.get('out_dim', 64),
            num_layers=encoder_config.get('num_layers', 4)
        ).to(self.device)
        
        self.network = HybridActorCritic(
            encoder=self.encoder,
            hidden_dim=actor_critic_config.get('hidden_dim', 128),
            value_components=actor_critic_config.get('value_components', self.component_names),
            dropout_rate=actor_critic_config.get('dropout_rate', 0.1)
        ).to(self.device)
        
        # Enhanced Learnable Component Weighting System
        self._initialize_learnable_weights()
        
        # Optimizer (includes learnable weights)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.weight_optimizer = optim.Adam(self.learnable_component_weights.parameters(), lr=self.weight_learning_rate)
        
        # Learning rate scheduler for long runs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        
        # Training tracking
        self.episode_rewards = defaultdict(list)
        self.losses = defaultdict(list)
        # Per-episode loss tracking (one value per episode: average total loss applied to episodes in a batch)
        self.episode_losses = []
        self.smoothed_rewards = []  # Track moving average of total rewards
        self.smoothed_losses = []   # Track moving average of losses
        self.best_total_reward = float('-inf')
        
        # Enhanced detailed node logging per step
        self.detailed_node_logs = []  # Store detailed node data per step per episode
        
        # Load detailed logging configuration from config file
        detailed_logging_config = config.get('detailed_logging', {})
        self.enable_detailed_logging = detailed_logging_config.get('enable_detailed_logging', True)
        self.save_detailed_logs_to_json = detailed_logging_config.get('save_detailed_logs_to_json', True)
        self.single_file_mode = detailed_logging_config.get('single_file_mode', True)
        self.log_node_positions = detailed_logging_config.get('log_node_positions', True)
        self.log_spawn_parameters = detailed_logging_config.get('log_spawn_parameters', True)
        self.log_persistent_ids = detailed_logging_config.get('log_persistent_ids', True)
        self.log_connectivity = detailed_logging_config.get('log_connectivity', True)
        self.log_substrate_values = detailed_logging_config.get('log_substrate_values', True)
        self.compress_logs = detailed_logging_config.get('compress_logs', False)
        self.max_log_file_size_mb = detailed_logging_config.get('max_log_file_size_mb', 50)
        
        # Initialize single file logging variables
        self.detailed_logs_file_path = None
        self.all_episodes_data = {
            'training_metadata': {
                'created_timestamp': time.time(),
                'total_episodes_logged': 0,
                'configuration': {}
            },
            'episodes': []
        }
        
        # Spawn parameter tracking for each episode
        self.current_episode_spawn_params = {
            'gamma': [],
            'alpha': [],
            'noise': [],
            'theta': []
        }
        
        # Reward component tracking for each episode
        self.current_episode_rewards = {
            'graph_reward': [],
            'spawn_reward': [],
            'delete_reward': [],
            'edge_reward': [],
            'total_node_reward': [],
            'total_reward': []
        }

        # Per-step node counts for current episode (used to compute min/mean/max/std per episode)
        self.current_episode_node_counts = []
        
        # Summary attributes for consolidated one-line logging
        self.current_spawn_summary = {'count': 0}
        self.current_reward_summary = []
        
        # Adaptive reward scaling
        self.component_running_stats = {
            component: {
                'mean': 0.0,
                'var': 1.0,
                'count': 0,
                'raw_rewards': deque(maxlen=1000)  # Keep recent rewards for statistics
            } for component in self.component_names
        }
        self.enable_adaptive_scaling = True
        self.scaling_warmup_episodes = 50  # Episodes before scaling kicks in
        
        print(f"🚀 DurotaxisTrainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"   Learnable weight parameters: {sum(p.numel() for p in self.learnable_component_weights.parameters()):,}")
        print(f"   Component names: {self.component_names}")
        print(f"   Substrate type: {self.substrate_type}")
        if self.substrate_type == 'random':
            print(f"   Substrate updates: Every episode with random type and parameters")
        else:
            print(f"   Substrate config: Fixed {self.substrate_type} substrate")
        if self.checkpoint_every is not None:
            print(f"   Checkpoint frequency: every {self.checkpoint_every} episodes")
        else:
            print(f"   Checkpoint frequency: disabled (only best + final models saved)")
        print(f"   Progress print frequency: every {self.progress_print_every} episodes")
        print(f"   Detailed node logging: {'enabled' if self.enable_detailed_logging else 'disabled'}")
        if self.enable_detailed_logging:
            print(f"   └─ Save to JSON: {'enabled' if self.save_detailed_logs_to_json else 'disabled'}")
            print(f"   └─ File mode: {'single file' if self.single_file_mode else 'separate files'}")
            print(f"   └─ Log positions: {'yes' if self.log_node_positions else 'no'}")
            print(f"   └─ Log spawn params: {'yes' if self.log_spawn_parameters else 'no'}")
            print(f"   └─ Log persistent IDs: {'yes' if self.log_persistent_ids else 'no'}")
            print(f"   └─ Log connectivity: {'yes' if self.log_connectivity else 'no'}")
            print(f"   └─ Log substrate values: {'yes' if self.log_substrate_values else 'no'}")
            print(f"   └─ Compression: {'enabled' if self.compress_logs else 'disabled'}")
            print(f"   └─ Max file size: {self.max_log_file_size_mb} MB")
    
    def _initialize_learnable_weights(self):
        """Initialize the enhanced learnable component weighting system"""
        import torch.nn as nn
        
        num_components = len(self.component_names)
        
        # Tier 1: Learnable component weights (your original idea)
        if self.enable_learnable_weights:
            # Create a simple module to hold parameters
            class LearnableWeights(nn.Module):
                def __init__(self, num_components, enable_attention):
                    super().__init__()
                    self.base_weights = nn.Parameter(torch.ones(num_components))
                    if enable_attention:
                        self.attention_weights = nn.Linear(num_components, num_components)
                    else:
                        self.attention_weights = None
                        
            self.learnable_component_weights = LearnableWeights(num_components, self.enable_attention_weighting)
        else:
            # Fallback to fixed weights
            class FixedWeights(nn.Module):
                def __init__(self, initial_weights):
                    super().__init__()
                    self.base_weights = nn.Parameter(torch.tensor(initial_weights), requires_grad=False)
                    self.attention_weights = None
                    
            initial_weights = [self.component_weights[comp] for comp in self.component_names]
            self.learnable_component_weights = FixedWeights(initial_weights)
        
        # Component name to index mapping for efficient lookups
        self.component_to_idx = {comp: idx for idx, comp in enumerate(self.component_names)}
        
        # Move to device
        self.learnable_component_weights = self.learnable_component_weights.to(self.device)
    
    def create_action_mask(self, state_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Create action mask to prevent invalid topology operations"""
        num_nodes = state_dict.get('num_nodes', 0)
        if num_nodes == 0:
            # Return None for empty graphs - graceful recovery will handle this
            return None
        
        # Basic action masking rules
        mask = torch.ones(num_nodes, 2, dtype=torch.bool, device=self.device)  # [spawn, delete]
        
        # Don't delete if too few nodes (prevent disconnection)
        if num_nodes <= 2:
            mask[:, 1] = False  # No deletion allowed
        
        # Don't spawn from nodes with many connections (prevent overcrowding)
        # This is a simplified heuristic - you can make it more sophisticated
        if hasattr(self.env.topology.graph, 'ndata') and 'degree' in self.env.topology.graph.ndata:
            degrees = self.env.topology.graph.ndata['degree']
            high_degree = degrees >= 4
            mask[high_degree, 0] = False  # No spawning from high-degree nodes
        
        return mask
    
    def update_component_stats(self, rewards: List[Dict]):
        """Update running statistics for adaptive reward scaling"""
        if not self.enable_adaptive_scaling:
            return
            
        # Aggregate rewards by component across the episode
        episode_component_totals = defaultdict(float)
        for reward_dict in rewards:
            for component, reward in reward_dict.items():
                if component in self.component_names:
                    episode_component_totals[component] += reward
        
        # Update running statistics for each component
        for component in self.component_names:
            stats = self.component_running_stats[component]
            episode_reward = episode_component_totals.get(component, 0.0)
            
            # Add to recent rewards buffer
            stats['raw_rewards'].append(episode_reward)
            
            # Update running mean and variance using Welford's online algorithm
            stats['count'] += 1
            delta = episode_reward - stats['mean']
            stats['mean'] += delta / stats['count']
            delta2 = episode_reward - stats['mean']
            stats['var'] += delta * delta2
            
            # Compute standard deviation (avoid division by zero)
            if stats['count'] > 1:
                std = (stats['var'] / (stats['count'] - 1)) ** 0.5
                stats['std'] = max(std, 1e-6)  # Minimum std to avoid division by zero
            else:
                stats['std'] = 1.0
    
    def get_component_scaling_factors(self) -> Dict[str, float]:
        """
        Tier 2 Scaling: Cross-episode adaptive scaling factors
        
        Computes scaling factors based on historical component statistics
        to maintain balance across different episodes and training phases.
        """
        scaling_factors = {}
        
        # Check if cross-episode scaling is enabled
        if not self.enable_cross_episode_scaling or not self.enable_adaptive_scaling:
            return {component: 1.0 for component in self.component_names}
        
        # Collect recent standard deviations
        component_stds = {}
        for component in self.component_names:
            stats = self.component_running_stats[component]
            if len(stats['raw_rewards']) >= 10:  # Need enough samples
                recent_rewards = list(stats['raw_rewards'])
                component_stds[component] = max(np.std(recent_rewards), 1e-6)
            else:
                component_stds[component] = 1.0
        
        # Find reference scale (median std to be robust to outliers)
        if component_stds:
            reference_std = np.median(list(component_stds.values()))
            
            # Compute scaling factors to normalize to reference scale
            for component in self.component_names:
                component_std = component_stds[component]
                # Scale factor to bring component to reference scale
                scaling_factors[component] = reference_std / component_std
                
                # Clip extreme scaling factors
                scaling_factors[component] = np.clip(scaling_factors[component], 0.1, 10.0)
        else:
            scaling_factors = {component: 1.0 for component in self.component_names}
        
        return scaling_factors
    
    def normalize_episode_rewards(self, rewards: List[Dict]) -> Dict[str, List[float]]:
        """
        Tier 1 Normalization: Per-episode reward component normalization
        
        Normalizes each reward component within the current episode to prevent
        high-magnitude components from dominating the learning signal.
        
        Args:
            rewards: List of reward dictionaries for the episode
            
        Returns:
            Dictionary mapping component names to normalized reward lists
        """
        normalized_rewards = {}
        
        # Skip normalization if disabled or episode too short
        if not self.enable_per_episode_norm or len(rewards) < self.min_episode_length:
            for component in self.component_names:
                normalized_rewards[component] = [r.get(component, 0.0) for r in rewards]
            return normalized_rewards
        
        for component in self.component_names:
            # Extract rewards for this component
            rewards_np = np.array([r.get(component, 0.0) for r in rewards])
            
            # Skip normalization if all rewards are zero or very small variance
            if len(rewards_np) == 0 or np.abs(rewards_np).max() < 1e-8:
                normalized_rewards[component] = rewards_np.tolist()
                continue
            
            # Apply selected normalization method
            if self.normalization_method == 'zscore':
                normalized_rewards[component] = self._zscore_normalize(rewards_np)
            elif self.normalization_method == 'minmax':
                normalized_rewards[component] = self._minmax_normalize(rewards_np)
            else:  # 'adaptive' method
                normalized_rewards[component] = self._adaptive_normalize(rewards_np)
        
        return normalized_rewards
    
    def _zscore_normalize(self, rewards_np: np.ndarray) -> List[float]:
        """Standard z-score normalization"""
        mean = rewards_np.mean()
        std = rewards_np.std()
        if std < 1e-8:
            return np.zeros_like(rewards_np).tolist()
        return ((rewards_np - mean) / std).tolist()
    
    def _minmax_normalize(self, rewards_np: np.ndarray) -> List[float]:
        """Min-max normalization to [0, 1] range"""
        reward_min, reward_max = rewards_np.min(), rewards_np.max()
        reward_range = reward_max - reward_min
        if reward_range < 1e-8:
            return np.zeros_like(rewards_np).tolist()
        return ((rewards_np - reward_min) / reward_range).tolist()
    
    def _adaptive_normalize(self, rewards_np: np.ndarray) -> List[float]:
        """Adaptive normalization that chooses best method based on data characteristics"""
        mean = rewards_np.mean()
        std = rewards_np.std()
        
        if std < 1e-8:  # Near-constant rewards
            # Use min-max normalization for constant/near-constant rewards
            reward_range = rewards_np.max() - rewards_np.min()
            if reward_range > 1e-8:
                return ((rewards_np - rewards_np.min()) / reward_range - 0.5).tolist()
            else:
                return np.zeros_like(rewards_np).tolist()
        else:
            # Use z-score normalization for variable rewards
            return ((rewards_np - mean) / std).tolist()
    
    def collate_graph_batch(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate multiple graph states into a single batch for efficient processing"""
        if not states:
            return {}
        
        # Handle empty states
        valid_states = [s for s in states if s.get('num_nodes', 0) > 0]
        if not valid_states:
            return {
                'node_features': torch.empty(0, states[0]['node_features'].shape[-1], device=self.device),
                'graph_features': torch.stack([s['graph_features'] for s in states]),
                'edge_attr': torch.empty(0, states[0].get('edge_attr', torch.empty(0, 3)).shape[-1], device=self.device),
                'edge_index': torch.empty(2, 0, dtype=torch.long, device=self.device),
                'batch': torch.empty(0, dtype=torch.long, device=self.device),
                'num_nodes': 0,
                'batch_size': len(states),
                'node_counts': [0] * len(states)
            }
        
        # Collect features
        all_node_features = []
        all_edge_features = []
        all_edge_indices = []
        batch_indices = []
        node_counts = []
        
        node_offset = 0
        
        for batch_idx, state in enumerate(states):
            num_nodes = state.get('num_nodes', 0)
            node_counts.append(num_nodes)
            
            if num_nodes > 0:
                # Node features
                node_feats = state['node_features']
                all_node_features.append(node_feats)
                
                # Batch indices for nodes
                batch_indices.extend([batch_idx] * num_nodes)
                
                # Edge features and indices
                if 'edge_attr' in state and 'edge_index' in state:
                    edge_attr = state['edge_attr']
                    edge_index = state['edge_index']
                    
                    # Convert edge_index from tuple to tensor if needed
                    if isinstance(edge_index, tuple):
                        src, dst = edge_index
                        # Ensure src/dst tensors are on the trainer device
                        if isinstance(src, torch.Tensor):
                            src = src.to(self.device)
                        if isinstance(dst, torch.Tensor):
                            dst = dst.to(self.device)
                        edge_index = torch.stack([src, dst], dim=0).to(self.device)
                    else:
                        # If already a tensor, move to device
                        edge_index = edge_index.to(self.device)

                    # Ensure edge_attr is on the correct device
                    if isinstance(edge_attr, torch.Tensor):
                        edge_attr = edge_attr.to(self.device)
                    
                    if edge_index.shape[1] > 0:  # Has edges
                        # Adjust edge indices for batching
                        adjusted_edge_index = edge_index + node_offset
                        all_edge_indices.append(adjusted_edge_index)
                        all_edge_features.append(edge_attr)
                
                node_offset += num_nodes
        
        # Concatenate all features
        if all_node_features:
            batched_node_features = torch.cat(all_node_features, dim=0)
            batched_batch = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        else:
            batched_node_features = torch.empty(0, states[0]['node_features'].shape[-1], device=self.device)
            batched_batch = torch.empty(0, dtype=torch.long, device=self.device)
        
        if all_edge_features and all_edge_indices:
            batched_edge_features = torch.cat(all_edge_features, dim=0)
            batched_edge_index = torch.cat(all_edge_indices, dim=1)
        else:
            # Handle case with no edges
            edge_dim = states[0].get('edge_attr', torch.empty(0, 3)).shape[-1]
            batched_edge_features = torch.empty(0, edge_dim, device=self.device)
            batched_edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        # Graph-level features (one per graph in batch)
        graph_features = torch.stack([s['graph_features'] for s in states])
        
        return {
            'node_features': batched_node_features,
            'graph_features': graph_features,
            'edge_attr': batched_edge_features,
            'edge_index': batched_edge_index,
            'batch': batched_batch,
            'num_nodes': batched_node_features.shape[0],
            'batch_size': len(states),
            'node_counts': node_counts
        }
    
    def unbatch_network_output(self, batched_output: Dict[str, torch.Tensor], 
                              node_counts: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Split batched network output back into individual samples"""
        if not batched_output or sum(node_counts) == 0:
            # Handle empty batch case
            empty_output = {
                'discrete_actions': torch.empty(0, dtype=torch.long),
                'continuous_actions': torch.empty(0, 4),
                'discrete_log_probs': torch.empty(0),
                'continuous_log_probs': torch.empty(0),
                'total_log_probs': torch.empty(0),
                'value_predictions': {k: torch.tensor(0.0) for k in self.component_names}
            }
            return [empty_output] * len(node_counts)
        
        # Split node-level outputs
        individual_outputs = []
        node_start = 0
        
        for i, num_nodes in enumerate(node_counts):
            if num_nodes == 0:
                # Empty graph
                output = {
                    'discrete_actions': torch.empty(0, dtype=torch.long, device=self.device),
                    'continuous_actions': torch.empty(0, 4, device=self.device),
                    'discrete_log_probs': torch.empty(0, device=self.device),
                    'continuous_log_probs': torch.empty(0, device=self.device),
                    'total_log_probs': torch.empty(0, device=self.device),
                    'value_predictions': {k: torch.tensor(0.0, device=self.device) for k in self.component_names}
                }
            else:
                node_end = node_start + num_nodes
                
                # Extract node-level outputs for this graph
                output = {}
                
                # Node-level tensors (discrete/continuous actions and log probs)
                for key in ['discrete_actions', 'continuous_actions', 'discrete_log_probs', 'continuous_log_probs', 'total_log_probs']:
                    if key in batched_output and batched_output[key].shape[0] > 0:
                        output[key] = batched_output[key][node_start:node_end]
                    else:
                        # Handle missing keys
                        if 'actions' in key:
                            if 'discrete' in key:
                                output[key] = torch.empty(0, dtype=torch.long, device=self.device)
                            else:
                                output[key] = torch.empty(0, 4, device=self.device)
                        else:
                            output[key] = torch.empty(0, device=self.device)
                
                # Graph-level value predictions (one per graph)
                if 'value_predictions' in batched_output:
                    output['value_predictions'] = {}
                    for component, values in batched_output['value_predictions'].items():
                        if values.shape[0] > i:
                            output['value_predictions'][component] = values[i]
                        else:
                            output['value_predictions'][component] = torch.tensor(0.0, device=self.device)
                else:
                    output['value_predictions'] = {k: torch.tensor(0.0, device=self.device) for k in self.component_names}
                
                node_start = node_end
            
            individual_outputs.append(output)
        
        return individual_outputs
    
    def update_adaptive_component_weights(self, advantages: Dict[str, torch.Tensor]) -> None:
        """Update component weights based on advantage magnitude (adaptive attention)"""
        # Update weights based on mean absolute advantage
        for comp in self.component_names:
            if comp in advantages:
                mean_abs_adv = torch.mean(torch.abs(advantages[comp]))
                # Exponential moving average update
                old_weight = self.component_weights[comp]
                self.component_weights[comp] = (
                    self.weight_update_momentum * old_weight + 
                    (1 - self.weight_update_momentum) * mean_abs_adv.item()
                )
        
        self.weight_updates_count += 1
        
        # Periodic normalization to prevent weight drift
        if self.weight_updates_count % self.normalize_weights_every == 0:
            self.normalize_component_weights()
    
    def compute_enhanced_advantage_weights(self, advantages: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Enhanced learnable advantage weighting system
        
        Combines three approaches:
        1. Learnable base weights (trainable parameters)
        2. Attention-based dynamic weighting (context-dependent)  
        3. Adaptive scaling for stability
        
        Args:
            advantages: Dictionary of advantages per component
            
        Returns:
            Final combined weighted advantages
        """
        if not self.enable_learnable_weights:
            # Fallback to traditional weighting
            return self._compute_traditional_weighted_advantages(advantages)
        
        device = next(iter(advantages.values())).device
        batch_size = next(iter(advantages.values())).shape[0]
        
        # Step 1: Extract advantages in component order
        advantage_tensor = torch.stack([
            advantages[comp] for comp in self.component_names
        ], dim=1)  # Shape: [batch_size, num_components]
        
        # Step 2: Learnable base weights (Tier 1)
        base_weights = torch.softmax(self.learnable_component_weights.base_weights, dim=0)
        
        # Step 3: Attention-based dynamic weighting (Tier 2)
        if self.enable_attention_weighting and self.learnable_component_weights.attention_weights is not None:
            # Compute attention weights based on advantage magnitudes
            advantage_magnitudes = torch.abs(advantage_tensor).mean(dim=0)  # [num_components]
            attention_logits = self.learnable_component_weights.attention_weights(advantage_magnitudes)
            attention_weights = torch.softmax(attention_logits, dim=0)
            
            # Combine base weights with attention
            final_weights = base_weights * attention_weights
            final_weights = final_weights / final_weights.sum()  # Renormalize
        else:
            final_weights = base_weights
        
        # Step 4: Apply weights to advantages
        weighted_advantages = (advantage_tensor * final_weights.unsqueeze(0)).sum(dim=1)
        
        # Step 5: Normalize final advantages
        if len(weighted_advantages) > 1:
            weighted_advantages = (weighted_advantages - weighted_advantages.mean()) / (weighted_advantages.std() + 1e-8)
        
        return weighted_advantages
    
    def _compute_traditional_weighted_advantages(self, advantages: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fallback to traditional component weighting method"""
        total_advantages = torch.zeros(len(next(iter(advantages.values()))), device=self.device)
        for component, adv in advantages.items():
            weight = self.component_weights.get(component, 1.0)
            total_advantages += weight * adv
        
        # Normalize advantages
        if len(total_advantages) > 1:
            total_advantages = (total_advantages - total_advantages.mean()) / (total_advantages.std() + 1e-8)
        
        return total_advantages
    
    def update_learnable_weights(self, advantages: Dict[str, torch.Tensor], policy_loss: torch.Tensor):
        """Update learnable component weights based on policy performance"""
        if not self.enable_learnable_weights:
            return
        
        # Compute weight regularization loss
        weight_reg_loss = self.weight_regularization * torch.norm(self.learnable_component_weights.base_weights)
        
        # Total loss for weight optimization
        total_weight_loss = policy_loss + weight_reg_loss
        
        # Update learnable weights
        self.weight_optimizer.zero_grad()
        total_weight_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.learnable_component_weights.parameters(), 0.5)
        self.weight_optimizer.step()
    
    def compute_adaptive_entropy_coefficient(self, episode: int) -> float:
        """
        Adaptive entropy coefficient scheduling
        
        Starts high for exploration, gradually decays to allow exploitation.
        Critical for preventing premature policy collapse in hybrid action spaces.
        """
        if not self.enable_adaptive_entropy:
            return self.entropy_coeff_start
        
        # Linear decay from start to end coefficient
        if episode < self.entropy_decay_episodes:
            progress = episode / self.entropy_decay_episodes
            current_coeff = self.entropy_coeff_start * (1 - progress) + self.entropy_coeff_end * progress
        else:
            current_coeff = self.entropy_coeff_end
        
        # Update current coefficient for monitoring
        self.current_entropy_coeff = current_coeff
        return current_coeff

    def compute_adaptive_gradient_scaling(self, discrete_loss: torch.Tensor, continuous_loss: torch.Tensor) -> Tuple[float, float]:
        """
        Compute adaptive gradient scaling weights to balance discrete and continuous learning
        
        Args:
            discrete_loss: Policy loss for discrete actions
            continuous_loss: Policy loss for continuous actions
            
        Returns:
            Tuple of (adaptive_discrete_weight, adaptive_continuous_weight)
        """
        if not self.enable_gradient_scaling:
            return self.policy_loss_weights['discrete_weight'], self.policy_loss_weights['continuous_weight']
        
        # Compute gradients for each component separately (without applying them)
        discrete_gradients = []
        continuous_gradients = []
        
        # Get gradients for discrete loss
        if discrete_loss.requires_grad:
            discrete_grads = torch.autograd.grad(
                discrete_loss, 
                [p for p in self.network.parameters() if p.requires_grad], 
                retain_graph=True, 
                create_graph=False,
                allow_unused=True
            )
            discrete_gradients = [g for g in discrete_grads if g is not None]
        
        # Get gradients for continuous loss  
        if continuous_loss.requires_grad:
            continuous_grads = torch.autograd.grad(
                continuous_loss,
                [p for p in self.network.parameters() if p.requires_grad], 
                retain_graph=True, 
                create_graph=False,
                allow_unused=True
            )
            continuous_gradients = [g for g in continuous_grads if g is not None]
        
        # Compute gradient norms
        discrete_grad_norm = 0.0
        if discrete_gradients:
            discrete_grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in discrete_gradients)).item()
        
        continuous_grad_norm = 0.0
        if continuous_gradients:
            continuous_grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in continuous_gradients)).item()
        
        # Update EMA of gradient norms
        if self.discrete_grad_norm_ema is None:
            self.discrete_grad_norm_ema = discrete_grad_norm
            self.continuous_grad_norm_ema = continuous_grad_norm
        else:
            self.discrete_grad_norm_ema = (self.scaling_momentum * self.discrete_grad_norm_ema + 
                                         (1 - self.scaling_momentum) * discrete_grad_norm)
            self.continuous_grad_norm_ema = (self.scaling_momentum * self.continuous_grad_norm_ema + 
                                           (1 - self.scaling_momentum) * continuous_grad_norm)
        
        self.gradient_step_count += 1
        
        # During warmup, use original weights
        if self.gradient_step_count < self.gradient_warmup_steps:
            return self.policy_loss_weights['discrete_weight'], self.policy_loss_weights['continuous_weight']
        
        # Compute adaptive scaling factors
        if self.discrete_grad_norm_ema > 0 and self.continuous_grad_norm_ema > 0:
            # Target: both components should have similar gradient norms
            discrete_scale = self.gradient_norm_target / self.discrete_grad_norm_ema
            continuous_scale = self.gradient_norm_target / self.continuous_grad_norm_ema
            
            # Clamp scaling factors to prevent extreme values
            discrete_scale = max(self.min_scaling_factor, min(self.max_scaling_factor, discrete_scale))
            continuous_scale = max(self.min_scaling_factor, min(self.max_scaling_factor, continuous_scale))
            
            # Apply scaling to original weights
            base_discrete = self.policy_loss_weights['discrete_weight']
            base_continuous = self.policy_loss_weights['continuous_weight']
            
            scaled_discrete = base_discrete * discrete_scale
            scaled_continuous = base_continuous * continuous_scale
            
            # Normalize to maintain relative importance while balancing gradients
            total_weight = scaled_discrete + scaled_continuous
            if total_weight > 0:
                self.adaptive_discrete_weight = scaled_discrete / total_weight
                self.adaptive_continuous_weight = scaled_continuous / total_weight
        
        return self.adaptive_discrete_weight, self.adaptive_continuous_weight
    
    def compute_enhanced_entropy_loss(self, eval_output: Dict[str, torch.Tensor], episode: int) -> Dict[str, torch.Tensor]:
        """
        Enhanced entropy regularization for hybrid action spaces
        
        Implements your core idea with improvements:
        1. Adaptive entropy scheduling (high→low over training)
        2. Separate handling for discrete vs continuous actions  
        3. Minimum entropy protection against policy collapse
        4. Action space specific weighting
        
        Args:
            eval_output: Network evaluation output containing entropy information
            episode: Current episode number for adaptive scheduling
            
        Returns:
            Dictionary of entropy losses for monitoring and optimization
        """
        entropy_losses = {}
        
        # Get adaptive entropy coefficient
        entropy_coeff = self.compute_adaptive_entropy_coefficient(episode)
        
        if 'entropy' in eval_output:
            # Combined entropy (your original idea enhanced)
            total_entropy = eval_output['entropy'].mean()
            
            # Minimum entropy protection (prevent complete collapse)
            if total_entropy < self.min_entropy_threshold:
                # Boost entropy if too low
                entropy_penalty = (self.min_entropy_threshold - total_entropy) * 2.0
                entropy_losses['entropy_penalty'] = entropy_penalty
            
            # Main entropy regularization (encourage exploration)
            entropy_losses['total'] = -entropy_coeff * total_entropy
            
        else:
            # Separate discrete and continuous entropy handling
            total_entropy_loss = torch.tensor(0.0, device=self.device)
            
            if 'discrete_entropy' in eval_output:
                discrete_entropy = eval_output['discrete_entropy'].mean()
                
                # Discrete actions need strong entropy (topology decisions are critical)
                discrete_loss = -entropy_coeff * self.discrete_entropy_weight * discrete_entropy
                entropy_losses['discrete'] = discrete_loss
                total_entropy_loss += discrete_loss
                
                # Monitor discrete entropy collapse
                if discrete_entropy < self.min_entropy_threshold:
                    discrete_penalty = (self.min_entropy_threshold - discrete_entropy) * 3.0
                    entropy_losses['discrete_penalty'] = discrete_penalty
                    total_entropy_loss += discrete_penalty
            
            if 'continuous_entropy' in eval_output:
                continuous_entropy = eval_output['continuous_entropy'].mean()
                
                # Continuous actions can be more focused (parameter fine-tuning)
                continuous_loss = -entropy_coeff * self.continuous_entropy_weight * continuous_entropy
                entropy_losses['continuous'] = continuous_loss
                total_entropy_loss += continuous_loss
                
                # Monitor continuous entropy collapse
                if continuous_entropy < self.min_entropy_threshold * 0.5:  # Lower threshold for continuous
                    continuous_penalty = (self.min_entropy_threshold * 0.5 - continuous_entropy) * 1.5
                    entropy_losses['continuous_penalty'] = continuous_penalty
                    total_entropy_loss += continuous_penalty
            
            # Combined loss for backward compatibility
            if total_entropy_loss.item() != 0:
                entropy_losses['total'] = total_entropy_loss
        
        return entropy_losses
    
    def normalize_component_weights(self) -> None:
        """Normalize component weights to sum to 1 for balanced contribution"""
        total_weight = sum(self.component_weights.values())
        if total_weight > 0:
            for comp in self.component_weights:
                self.component_weights[comp] /= total_weight
    
    def get_component_weight_insights(self) -> Dict[str, float]:
        """Get insights about current component weight distribution"""
        weights = list(self.component_weights.values())
        return {
            'max_weight': max(weights),
            'min_weight': min(weights),
            'weight_range': max(weights) - min(weights),
            'weight_std': np.std(weights),
            'dominant_component': max(self.component_weights.items(), key=lambda x: x[1])[0],
            'total_sum': sum(weights)
        }
    
    def get_training_statistics(self, window: int = 20) -> Dict[str, float]:
        """Get comprehensive training statistics with moving averages"""
        stats = {}
        
        # Reward statistics
        stats['total_reward_ma'] = moving_average(self.episode_rewards['total_reward'], window)
        for comp in self.component_names:
            stats[f'{comp}_ma'] = moving_average(self.episode_rewards[comp], window)
        
        # Loss statistics  
        for loss_name, loss_history in self.losses.items():
            stats[f'{loss_name}_ma'] = moving_average(loss_history, window)
        
        # Component weight insights
        weight_insights = self.get_component_weight_insights()
        stats.update(weight_insights)
        
        # Training progress metrics
        if self.episode_rewards['total_reward']:
            stats['episodes_completed'] = len(self.episode_rewards['total_reward'])
            stats['best_episode_reward'] = max(self.episode_rewards['total_reward'])
            stats['recent_improvement'] = (
                moving_average(self.episode_rewards['total_reward'], window//2) - 
                moving_average(self.episode_rewards['total_reward'][:len(self.episode_rewards['total_reward'])//2], window)
                if len(self.episode_rewards['total_reward']) > window else 0
            )
        
        return stats
    
    def create_run_directory(self, base_dir: str) -> str:
        """Create a new run directory with automatic numbering (run0001, run0002, etc.)"""
        import glob
        import re
        
        # Ensure base directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Find existing run directories
        existing_runs = glob.glob(os.path.join(base_dir, "run[0-9][0-9][0-9][0-9]"))
        
        # Extract run numbers
        run_numbers = []
        for run_path in existing_runs:
            run_name = os.path.basename(run_path)
            match = re.match(r"run(\d{4})", run_name)
            if match:
                run_numbers.append(int(match.group(1)))
        
        # Determine next run number
        if run_numbers:
            next_run_number = max(run_numbers) + 1
        else:
            next_run_number = 1
        
        # Create run directory
        self.run_number = next_run_number
        run_dir = os.path.join(base_dir, f"run{next_run_number:04d}")
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir
    
    def save_training_statistics(self, filepath: str = None) -> str:
        """Save comprehensive training statistics to file"""
        if filepath is None:
            filepath = os.path.join(self.run_dir, f"training_stats_episode_{len(self.episode_rewards['total_reward'])}.json")
        
        stats = self.get_training_statistics(window=self.moving_avg_window)
        
        # Add raw data for plotting
        stats['raw_data'] = {
            'episode_rewards': dict(self.episode_rewards),
            'losses': dict(self.losses),
            'component_weights_history': getattr(self, 'component_weights_history', [])
        }
        
        # Add configuration
        stats['config'] = {
            'moving_avg_window': self.moving_avg_window,
            'log_every': self.log_every,
            'entropy_bonus_coeff': self.entropy_bonus_coeff,
            'weight_momentum': self.weight_update_momentum,
            'normalize_weights_every': self.normalize_weights_every
        }
        
        import json
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_stats = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    serializable_stats[key] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in value.items()}
                elif hasattr(value, 'tolist'):
                    serializable_stats[key] = value.tolist()
                else:
                    serializable_stats[key] = value
            
            json.dump(serializable_stats, f, indent=2)
        
        return filepath
    
    def get_checkpoint_info(self) -> Dict[str, any]:
        """Get information about checkpoint configuration and saved models"""
        import glob
        
        # Find saved models in run directory
        best_models = glob.glob(os.path.join(self.run_dir, "best_model_*.pt"))
        checkpoints = glob.glob(os.path.join(self.run_dir, "checkpoint_*.pt"))
        final_model = os.path.join(self.run_dir, "final_model.pt")
        
        return {
            'checkpoint_frequency': self.checkpoint_every if self.checkpoint_every is not None else 'disabled',
            'run_number': self.run_number,
            'run_directory': self.run_dir,
            'best_models_count': len(best_models),
            'periodic_checkpoints_count': len(checkpoints),
            'final_model_exists': os.path.exists(final_model),
            'total_saved_models': len(best_models) + len(checkpoints) + (1 if os.path.exists(final_model) else 0),
            'best_reward_so_far': self.best_total_reward,
            'save_directory': self.save_dir
        }
    
    def _extract_detailed_node_data(self, episode_num: int, step_num: int, executed_actions: Dict = None) -> Dict:
        """
        Extract comprehensive node data for current step
        
        Returns detailed information about each node including:
        - Node positions (x, y)
        - Spawn parameters (gamma, alpha, noise, theta) if available
        - Persistent node IDs
        - Graph connectivity information
        - Substrate information at node positions
        """
        if not self.enable_detailed_logging:
            return {}
        
        detailed_data = {
            'episode': episode_num,
            'step': step_num,
            'timestamp': time.time(),
            'nodes': [],
            'graph_info': {},
            'substrate_info': {},
            'action_info': {
                'spawn_actions': 0,
                'delete_actions': 0,
                'total_actions': 0
            }
        }
        
        # Count actions if provided
        if executed_actions:
            spawn_count = sum(1 for action in executed_actions.values() if action == 'spawn')
            delete_count = sum(1 for action in executed_actions.values() if action == 'delete')
            detailed_data['action_info'] = {
                'spawn_actions': spawn_count,
                'delete_actions': delete_count,
                'total_actions': spawn_count + delete_count
            }
        
        try:
            # Get current topology and state
            topology = self.env.topology if hasattr(self.env, 'topology') else None
            if not topology or not hasattr(topology, 'graph'):
                return detailed_data
            
            graph = topology.graph
            num_nodes = graph.num_nodes()
            
            # Extract graph-level information
            detailed_data['graph_info'] = {
                'num_nodes': int(num_nodes),
                'num_edges': int(graph.num_edges()),
                'node_count_history': list(self.current_episode_node_counts) if hasattr(self, 'current_episode_node_counts') else []
            }
            
            # Extract substrate information
            if hasattr(self.env, 'substrate') and hasattr(self.env.substrate, 'substrate_kind'):
                substrate_params = getattr(self, 'current_substrate_params', {})
                detailed_data['substrate_info'] = {
                    'type': substrate_params.get('kind', 'unknown'),
                    'parameters': {
                        'm': substrate_params.get('m', 0.0),
                        'b': substrate_params.get('b', 0.0)
                    }
                }
            
            # Extract detailed node information
            if num_nodes > 0:
                # Get node data from the graph
                node_data = graph.ndata
                
                # Initialize centroid calculation variables
                centroid_x_sum = 0.0
                centroid_y_sum = 0.0
                valid_position_count = 0
                
                for node_id in range(num_nodes):
                    node_info = {
                        'node_id': int(node_id),
                    }
                    
                    # Add persistent ID if enabled
                    if self.log_persistent_ids:
                        node_info['persistent_id'] = int(node_id)  # Default to node_id, will try to get actual persistent ID
                        # Try to get persistent ID if available
                        if 'id' in node_data:
                            node_info['persistent_id'] = int(node_data['id'][node_id])
                        elif 'node_id' in node_data:
                            node_info['persistent_id'] = int(node_data['node_id'][node_id])
                        elif hasattr(topology, 'node_ids') and node_id < len(topology.node_ids):
                            node_info['persistent_id'] = int(topology.node_ids[node_id])
                    
                    # Add position information if enabled
                    if self.log_node_positions:
                        node_info['position'] = {'x': 0.0, 'y': 0.0}  # Initialize default position
                        if 'pos' in node_data:
                            pos = node_data['pos'][node_id]
                            if len(pos) >= 2:
                                node_info['position']['x'] = float(pos[0])
                                node_info['position']['y'] = float(pos[1])
                                # Accumulate for centroid calculation
                                centroid_x_sum += node_info['position']['x']
                                centroid_y_sum += node_info['position']['y']
                                valid_position_count += 1
                        elif 'x' in node_data and 'y' in node_data:
                            node_info['position']['x'] = float(node_data['x'][node_id])
                            node_info['position']['y'] = float(node_data['y'][node_id])
                            # Accumulate for centroid calculation
                            centroid_x_sum += node_info['position']['x']
                            centroid_y_sum += node_info['position']['y']
                            valid_position_count += 1
                    
                    # Add spawn parameters if enabled
                    if self.log_spawn_parameters:
                        node_info['spawn_parameters'] = {
                            'gamma': None,
                            'alpha': None, 
                            'noise': None,
                            'theta': None
                        }
                        spawn_params = ['gamma', 'alpha', 'noise', 'theta']
                        for param in spawn_params:
                            if param in node_data:
                                value = float(node_data[param][node_id])
                                # Convert NaN to None for JSON serialization
                                node_info['spawn_parameters'][param] = None if np.isnan(value) else value
                    
                    # Add connectivity information if enabled
                    if self.log_connectivity:
                        node_info['connectivity'] = {
                            'degree': 0,
                            'neighbors': []
                        }
                        if 'degree' in node_data:
                            node_info['connectivity']['degree'] = int(node_data['degree'][node_id])
                        
                        # Get neighbors (edges from this node)
                        try:
                            # Get edges and find neighbors
                            edges = graph.edges()
                            if len(edges) >= 2:
                                src_nodes, dst_nodes = edges
                                # Find edges where this node is the source
                                neighbors = []
                                for i in range(len(src_nodes)):
                                    if int(src_nodes[i]) == node_id:
                                        neighbors.append(int(dst_nodes[i]))
                                    elif int(dst_nodes[i]) == node_id:  # Also check reverse direction for undirected graphs
                                        neighbors.append(int(src_nodes[i]))
                                node_info['connectivity']['neighbors'] = list(set(neighbors))  # Remove duplicates
                        except Exception:
                            # Failed to extract neighbor information - leave empty
                            node_info['connectivity']['neighbors'] = []
                    
                    # Add substrate value if enabled
                    if self.log_substrate_values:
                        node_info['substrate_value'] = 0.0
                        try:
                            if hasattr(self.env, 'substrate') and self.log_node_positions and 'position' in node_info:
                                x_pos = node_info['position']['x']
                                y_pos = node_info['position']['y']
                                substrate_val = self.env.substrate.get_value(x_pos, y_pos)
                                node_info['substrate_value'] = float(substrate_val)
                        except Exception:
                            node_info['substrate_value'] = 0.0
                    
                    detailed_data['nodes'].append(node_info)
                
                # Calculate and add graph centroid
                if valid_position_count > 0:
                    centroid_x = centroid_x_sum / valid_position_count
                    centroid_y = centroid_y_sum / valid_position_count
                    detailed_data['graph_info']['centroid'] = {
                        'x': float(centroid_x),
                        'y': float(centroid_y),
                        'node_count': valid_position_count
                    }
                else:
                    detailed_data['graph_info']['centroid'] = {
                        'x': 0.0,
                        'y': 0.0,
                        'node_count': 0
                    }
            else:
                # No nodes - set centroid to origin
                detailed_data['graph_info']['centroid'] = {
                    'x': 0.0,
                    'y': 0.0,
                    'node_count': 0
                }
        
        except Exception as e:
            # Log error but don't fail training
            print(f"⚠️  Warning: Failed to extract detailed node data at episode {episode_num}, step {step_num}: {e}")
            
        return detailed_data
    
    def save_detailed_node_logs(self, episode_num: int) -> str:
        """
        Save detailed node logs for the current episode
        
        Supports two modes:
        1. Single file mode: Append episode data to one comprehensive JSON file
        2. Separate file mode: Create individual JSON file per episode (legacy)
        
        Returns the filepath of the saved JSON file
        """
        if not self.save_detailed_logs_to_json or not self.detailed_node_logs:
            return ""
        
        try:
            # Prepare episode data
            episode_data = {
                'episode': episode_num,
                'total_steps': len(self.detailed_node_logs),
                'timestamp': time.time(),
                'configuration': {
                    'log_node_positions': self.log_node_positions,
                    'log_spawn_parameters': self.log_spawn_parameters,
                    'log_persistent_ids': self.log_persistent_ids,
                    'log_connectivity': self.log_connectivity,
                    'log_substrate_values': self.log_substrate_values,
                    'compressed': self.compress_logs
                },
                'summary_statistics': {
                    'total_nodes_tracked': sum(len(step_data.get('nodes', [])) for step_data in self.detailed_node_logs),
                    'max_nodes_per_step': max((len(step_data.get('nodes', [])) for step_data in self.detailed_node_logs), default=0),
                    'min_nodes_per_step': min((len(step_data.get('nodes', [])) for step_data in self.detailed_node_logs), default=0),
                    'avg_nodes_per_step': sum(len(step_data.get('nodes', [])) for step_data in self.detailed_node_logs) / len(self.detailed_node_logs) if self.detailed_node_logs else 0
                },
                'steps': list(self.detailed_node_logs)  # Copy the step data
            }
            
            if self.single_file_mode:
                return self._save_to_single_file(episode_data, episode_num)
            else:
                return self._save_to_separate_file(episode_data, episode_num)
                
        except Exception as e:
            print(f"❌ Failed to save detailed node logs for episode {episode_num}: {e}")
            return ""
    
    def _save_to_single_file(self, episode_data: Dict, episode_num: int) -> str:
        """Save episode data to a single comprehensive JSON file"""
        try:
            # Initialize single file path if not set
            if self.detailed_logs_file_path is None:
                filename = "detailed_nodes_all_episodes.json"
                if self.compress_logs:
                    filename += ".gz"
                self.detailed_logs_file_path = os.path.join(self.run_dir, filename)
                
                # Initialize the file with metadata if it doesn't exist
                self.all_episodes_data['training_metadata']['configuration'] = episode_data['configuration']
                self.all_episodes_data['training_metadata']['run_directory'] = self.run_dir
                self.all_episodes_data['training_metadata']['single_file_mode'] = True
            
            # Add episode data to the collection
            self.all_episodes_data['episodes'].append(episode_data)
            self.all_episodes_data['training_metadata']['total_episodes_logged'] += 1
            self.all_episodes_data['training_metadata']['last_updated'] = time.time()
            
            # Save the entire data structure
            if self.compress_logs:
                import gzip
                with gzip.open(self.detailed_logs_file_path, 'wt') as f:
                    json.dump(self.all_episodes_data, f, indent=2, default=str)
            else:
                with open(self.detailed_logs_file_path, 'w') as f:
                    json.dump(self.all_episodes_data, f, indent=2, default=str)
            
            filename = os.path.basename(self.detailed_logs_file_path)
            total_episodes = self.all_episodes_data['training_metadata']['total_episodes_logged']
            
            print(f"📄 Appended episode {episode_num} to {filename} ({len(self.detailed_node_logs)} steps)")
            print(f"   └─ Total episodes in file: {total_episodes}")
            
            # Print summary statistics for this episode
            if episode_data['summary_statistics']['total_nodes_tracked'] > 0:
                print(f"   └─ Episode summary: {episode_data['summary_statistics']['total_nodes_tracked']} total nodes tracked, "
                      f"avg {episode_data['summary_statistics']['avg_nodes_per_step']:.1f} nodes/step, "
                      f"range {episode_data['summary_statistics']['min_nodes_per_step']}-{episode_data['summary_statistics']['max_nodes_per_step']} nodes")
            
            return self.detailed_logs_file_path
            
        except Exception as e:
            print(f"❌ Failed to save to single file: {e}")
            return ""
    
    def _save_to_separate_file(self, episode_data: Dict, episode_num: int) -> str:
        """Save episode data to a separate JSON file (legacy mode)"""
        try:
            # Create filename with episode number
            filename = f"detailed_nodes_episode_{episode_num:04d}.json"
            if self.compress_logs:
                filename += ".gz"
            filepath = os.path.join(self.run_dir, filename)
            
            # Save to JSON file (with optional compression)
            if self.compress_logs:
                import gzip
                with gzip.open(filepath, 'wt') as f:
                    json.dump(episode_data, f, indent=2, default=str)
            else:
                with open(filepath, 'w') as f:
                    json.dump(episode_data, f, indent=2, default=str)
            
            print(f"📄 Saved detailed node logs: {filename} ({len(self.detailed_node_logs)} steps)")
            
            # Print summary statistics
            if episode_data['summary_statistics']['total_nodes_tracked'] > 0:
                print(f"   └─ Summary: {episode_data['summary_statistics']['total_nodes_tracked']} total nodes tracked, "
                      f"avg {episode_data['summary_statistics']['avg_nodes_per_step']:.1f} nodes/step, "
                      f"range {episode_data['summary_statistics']['min_nodes_per_step']}-{episode_data['summary_statistics']['max_nodes_per_step']} nodes")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to save to separate file: {e}")
            return ""
    
    def finalize_detailed_logging(self) -> str:
        """
        Finalize detailed logging by adding training completion metadata
        Only applies to single file mode
        """
        if not self.single_file_mode or not self.save_detailed_logs_to_json:
            return ""
        
        if self.detailed_logs_file_path and hasattr(self, 'all_episodes_data'):
            try:
                # Add completion metadata
                self.all_episodes_data['training_metadata']['training_completed'] = True
                self.all_episodes_data['training_metadata']['completion_timestamp'] = time.time()
                self.all_episodes_data['training_metadata']['final_episode_count'] = self.all_episodes_data['training_metadata']['total_episodes_logged']
                
                # Calculate overall statistics
                total_steps_all_episodes = sum(ep['total_steps'] for ep in self.all_episodes_data['episodes'])
                total_nodes_all_episodes = sum(ep['summary_statistics']['total_nodes_tracked'] for ep in self.all_episodes_data['episodes'])
                
                self.all_episodes_data['training_metadata']['overall_statistics'] = {
                    'total_steps_all_episodes': total_steps_all_episodes,
                    'total_nodes_all_episodes': total_nodes_all_episodes,
                    'avg_steps_per_episode': total_steps_all_episodes / len(self.all_episodes_data['episodes']) if self.all_episodes_data['episodes'] else 0,
                    'avg_nodes_per_episode': total_nodes_all_episodes / len(self.all_episodes_data['episodes']) if self.all_episodes_data['episodes'] else 0
                }
                
                # Save final version
                if self.compress_logs:
                    import gzip
                    with gzip.open(self.detailed_logs_file_path, 'wt') as f:
                        json.dump(self.all_episodes_data, f, indent=2, default=str)
                else:
                    with open(self.detailed_logs_file_path, 'w') as f:
                        json.dump(self.all_episodes_data, f, indent=2, default=str)
                
                print(f"🏁 Finalized detailed logging: {os.path.basename(self.detailed_logs_file_path)}")
                print(f"   └─ Total episodes: {self.all_episodes_data['training_metadata']['final_episode_count']}")
                print(f"   └─ Total steps: {total_steps_all_episodes}")
                print(f"   └─ Total nodes tracked: {total_nodes_all_episodes}")
                
                return self.detailed_logs_file_path
                
            except Exception as e:
                print(f"❌ Failed to finalize detailed logging: {e}")
                return ""
        
        return ""
    
    def get_detailed_logging_summary(self) -> Dict:
        """
        Get a summary of the detailed logging configuration and current state
        """
        return {
            'enabled': self.enable_detailed_logging,
            'save_to_json': self.save_detailed_logs_to_json,
            'single_file_mode': self.single_file_mode,
            'configuration': {
                'log_node_positions': self.log_node_positions,
                'log_spawn_parameters': self.log_spawn_parameters,
                'log_persistent_ids': self.log_persistent_ids,
                'log_connectivity': self.log_connectivity,
                'log_substrate_values': self.log_substrate_values,
                'compress_logs': self.compress_logs,
                'max_log_file_size_mb': self.max_log_file_size_mb
            },
            'current_episode_steps_logged': len(self.detailed_node_logs),
            'run_directory': getattr(self, 'run_dir', None),
            'file_info': {
                'single_file_path': getattr(self, 'detailed_logs_file_path', None),
                'total_episodes_logged': self.all_episodes_data['training_metadata'].get('total_episodes_logged', 0) if hasattr(self, 'all_episodes_data') else 0,
                'filename_format': 'detailed_nodes_all_episodes.json' if self.single_file_mode else 'detailed_nodes_episode_XXXX.json'
            }
        }
    
    def _get_curriculum_config(self, episode_num: int) -> Dict[str, any]:
        """Get curriculum-adjusted configuration based on training progress."""
        if not self.enable_curriculum:
            return {}

        # If we built scaled curriculum stages, use them to select the config
        if self.curriculum_stages:
            for stage in self.curriculum_stages:
                if episode_num >= stage['start'] and episode_num <= stage['end']:
                    # Return a shallow copy of the stage config to avoid accidental mutation
                    out = stage.get('config', {}).copy() if isinstance(stage.get('config', {}), dict) else {}
                    # include stage metadata
                    out['_stage_name'] = stage.get('name')
                    out['_stage_start'] = stage.get('start')
                    out['_stage_end'] = stage.get('end')
                    out['_stage_focus'] = stage.get('focus')
                    return out

        # Fallback: legacy phase_1/2/3 durations
        curriculum_config = {}
        if episode_num < self.phase_1_episodes:
            curriculum_config.update(self.phase_1_config)
        elif episode_num < self.phase_2_episodes:
            curriculum_config.update(self.phase_2_config)
        else:
            curriculum_config.update(self.phase_3_config)

        return curriculum_config

    def _build_scaled_curriculum(self, curriculum_config: Dict[str, any]) -> None:
        """Build curriculum stages scaled to the trainer's total_episodes.

        The method looks for explicit stage definitions under the provided
        curriculum_config (keys like 'stage_1_navigation', 'stage_2_management', ...)
        or a 'stages' mapping. If explicit episode ranges are present, their
        relative sizes are used to compute proportions and scale them to
        self.total_episodes. The resulting stages are stored in
        self.curriculum_stages as a list of dicts.
        """
        total_eps = max(1, int(self.total_episodes))

        # Prefer an explicit 'stages' list if present
        stages = []
        if 'stages' in curriculum_config and isinstance(curriculum_config['stages'], list):
            stages = curriculum_config['stages']
        else:
            # Otherwise, gather keys that look like stage definitions
            for key, value in curriculum_config.items():
                if isinstance(value, dict) and key.lower().startswith('stage'):
                    # Keep the stage name and the dict
                    stages.append({'name': key, 'config': value})

        if not stages:
            # Nothing to build
            self.curriculum_stages = []
            return

        # Extract absolute lengths if episode_start/episode_end provided; otherwise use equal weights
        abs_lengths = []
        stage_meta = []
        for entry in stages:
            name = entry.get('name') if isinstance(entry, dict) and 'name' in entry else entry
            cfg = entry.get('config') if isinstance(entry, dict) and 'config' in entry else entry

            start = cfg.get('episode_start') if isinstance(cfg, dict) else None
            end = cfg.get('episode_end') if isinstance(cfg, dict) else None
            focus = cfg.get('focus') if isinstance(cfg, dict) else None

            if start is not None and end is not None:
                length = max(0, int(end) - int(start) + 1)
            else:
                length = None

            abs_lengths.append(length)
            stage_meta.append({'name': name, 'config': cfg if isinstance(cfg, dict) else {}, 'focus': focus, 'start': start, 'end': end})

        # If any absolute lengths are present, use relative proportions of those lengths
        if any(l is not None for l in abs_lengths):
            # Replace None with average of present lengths
            present = [l for l in abs_lengths if l is not None]
            avg = int(sum(present) / len(present)) if present else 1
            rel_lengths = [l if l is not None else avg for l in abs_lengths]
        else:
            # Equal weights
            rel_lengths = [1] * len(stage_meta)

        total_rel = sum(rel_lengths)
        # Compute scaled lengths (floor), then distribute remainder
        scaled = [int((rl * total_eps) / total_rel) for rl in rel_lengths]
        remainder = total_eps - sum(scaled)
        # Distribute remainder to first stages
        for i in range(remainder):
            scaled[i % len(scaled)] += 1

        # Build start/end indices
        stages_out = []
        cur = 0
        for meta, length in zip(stage_meta, scaled):
            start = cur
            end = max(cur, cur + length - 1)
            stages_out.append({
                'name': meta['name'],
                'start': start,
                'end': end,
                'config': meta.get('config', {}),
                'focus': meta.get('focus')
            })
            cur = end + 1

        # Store built curriculum
        # Apply overlap: prefer curriculum_overlap_pct (fraction), otherwise use stage_overlap (absolute)
        overlap_pct = curriculum_config.get('curriculum_overlap_pct', None)
        stage_overlap_abs = curriculum_config.get('stage_overlap', None)

        if overlap_pct is not None:
            # Interpret as fraction of each stage length to overlap with next stage
            adjusted = []
            for s in stages_out:
                length = s['end'] - s['start'] + 1
                ov = int(round(length * float(overlap_pct)))
                adjusted.append((s, ov))

            # Apply overlaps by shrinking next stage start accordingly
            final = []
            for i, (s, ov) in enumerate(adjusted):
                start = s['start']
                end = s['end']
                # shrink end by ov for current stage, but ensure non-negative length
                new_end = max(start, end - ov)
                final.append({**s, 'start': start, 'end': new_end})

            # Ensure contiguous coverage (next stage start is previous end+1)
            for i in range(1, len(final)):
                final[i]['start'] = final[i-1]['end'] + 1
                if final[i]['start'] > final[i]['end']:
                    final[i]['end'] = final[i]['start']

            self.curriculum_stages = final
        elif stage_overlap_abs is not None:
            # Absolute overlap in episodes; shrink current stage end by overlap, leave next stage start
            adjusted = []
            for s in stages_out:
                start = s['start']
                end = s['end']
                new_end = max(start, end - int(stage_overlap_abs))
                adjusted.append({**s, 'start': start, 'end': new_end})

            # Contiguous coverage
            for i in range(1, len(adjusted)):
                adjusted[i]['start'] = adjusted[i-1]['end'] + 1
                if adjusted[i]['start'] > adjusted[i]['end']:
                    adjusted[i]['end'] = adjusted[i]['start']

            self.curriculum_stages = adjusted
        else:
            # No overlap adjustment
            self.curriculum_stages = stages_out
    
    def _apply_curriculum_to_env(self, curriculum_config: Dict[str, any]):
        """Apply curriculum configuration to environment."""
        if not curriculum_config:
            return
            
        # Apply curriculum settings to environment
        self.env.apply_curriculum_config(curriculum_config)
        
        # Also store survival config for reward calculation
        survival_config = curriculum_config.get('survival_rewards', {})
        self.env._survival_config = survival_config
            
        # Update critical node thresholds
        if 'max_critical_nodes' in curriculum_config:
            self.env.max_critical_nodes = curriculum_config['max_critical_nodes']
            
        # Update initial node count
        if 'init_num_nodes' in curriculum_config:
            self.init_num_nodes = curriculum_config['init_num_nodes']
    
    def _evaluate_episode_success(self, episode_rewards: List[Dict], episode_length: int, 
                                 final_state: Dict) -> Tuple[bool, Dict[str, bool]]:
        """Evaluate if an episode was successful using multiple criteria."""
        if not self.enable_multiple_criteria:
            # Use traditional success evaluation (if any)
            return False, {}
            
        total_reward = sum(r.get('total_reward', 0) for r in episode_rewards)
        num_nodes = final_state.get('num_nodes', 0)
        
        # Multiple success criteria (easier to achieve)
        success_criteria = {
            'survival_success': episode_length >= self.survival_success_steps and num_nodes > 0,
            'reward_success': total_reward > self.reward_success_threshold,
            'growth_success': episode_length >= 5 and num_nodes >= self.growth_success_nodes,
            'exploration_success': episode_length >= self.exploration_success_steps,
        }
        
        # Episode is successful if it meets any criterion
        is_successful = any(success_criteria.values())
        
        return is_successful, success_criteria
    
    def collect_episode(self, episode_num: int = 0) -> Tuple[List[Dict], List[torch.Tensor], List[Dict], List[Dict], bool, bool]:
        """Collect one episode of experience with curriculum learning support"""
        # Apply curriculum learning configuration
        curriculum_config = self._get_curriculum_config(episode_num)
        self._apply_curriculum_to_env(curriculum_config)
        
        # Reset tracking for this episode
        self.current_episode_spawn_params = {
            'gamma': [],
            'alpha': [],
            'noise': [],
            'theta': []
        }
        
        self.current_episode_rewards = {
            'graph_reward': [],
            'spawn_reward': [],
            'delete_reward': [],
            'edge_reward': [],
            'total_node_reward': [],
            'total_reward': []
        }
        
        # Clear detailed node logs for new episode
        self.detailed_node_logs = []

        # Update substrate if using random type
        if self.substrate_type == 'random':
            self._update_random_substrate()

        # Reset per-step node counts for this episode
        self.current_episode_node_counts = []
        
        states = []
        actions_taken = []
        rewards = []
        values = []
        log_probs_list = []
        
        obs, info = self.env.reset()
        # Update state extractor with current topology
        self.state_extractor.set_topology(self.env.topology)
        done = False
        episode_length = 0
        terminated = False
        truncated = False
        
        # Track previous step's executed actions for detailed logging
        previous_executed_actions = None
        
        while not done and episode_length < self.max_steps:
            # Extract detailed node data at the beginning of each step (with previous step's actions)
            if self.enable_detailed_logging:
                step_node_data = self._extract_detailed_node_data(episode_num, episode_length, previous_executed_actions)
                self.detailed_node_logs.append(step_node_data)
            # Get current state
            state_dict = self.state_extractor.get_state_features(include_substrate=True)

            # Move tensors in state_dict to the trainer device. Handle tuple edge_index specially.
            moved_state = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    moved_state[k] = v.to(self.device)
                elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, torch.Tensor) for x in v):
                    # edge_index as tuple (src, dst)
                    src, dst = v
                    moved_state[k] = (src.to(self.device), dst.to(self.device))
                else:
                    moved_state[k] = v

            state_dict = moved_state

            # Track node count for this step
            try:
                self.current_episode_node_counts.append(int(state_dict.get('num_nodes', 0)))
            except Exception:
                # Ignore if node count is not available
                pass
            
            if state_dict['num_nodes'] == 0:
                if self.enable_graceful_recovery:
                    # Graceful empty graph handling: add minimal dummy node to preserve training continuity
                    if self.log_recoveries:
                        print(f"⚠️  Empty graph detected at episode step {episode_length}, adding {self.recovery_num_nodes} node(s) to continue training...")
                    self.env.topology.reset(init_num_nodes=self.recovery_num_nodes)
                    self.empty_graph_recovery_count += 1
                    # Get updated state after reset
                    state_dict = self.state_extractor.get_state_features(include_substrate=True)
                    state_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in state_dict.items()}
                    # Update state extractor with reset topology
                    self.state_extractor.set_topology(self.env.topology)
                else:
                    # Fallback to original behavior: break episode
                    if self.log_recoveries:
                        print(f"⚠️  Empty graph detected at episode step {episode_length}, ending episode...")
                    break
                
            # Create action mask
            action_mask = self.create_action_mask(state_dict)
            
            # Forward pass through network
            with torch.no_grad():
                output = self.network(state_dict, deterministic=False, action_mask=action_mask)
            
            # Store state and predictions
            states.append(state_dict)
            values.append(output['value_predictions'])
            
            # Extract actions and log probs
            if len(output.get('discrete_actions', [])) > 0:
                discrete_actions = output['discrete_actions']
                continuous_actions = output['continuous_actions']
                log_probs = output['total_log_probs']
                
                actions_taken.append({
                    'discrete': discrete_actions,
                    'continuous': continuous_actions,
                    'mask': action_mask
                })
                
                # Store separate log probs for hybrid policy loss
                log_probs_list.append({
                    'discrete': output['discrete_log_probs'],
                    'continuous': output['continuous_log_probs'],
                    'total': log_probs  # Keep for backward compatibility
                })
                
                # Execute actions in environment
                topology_actions = self.network.get_topology_actions(output)
                
                for node_id, action_type in topology_actions.items():
                    try:
                        if action_type == 'spawn':
                            params = self.network.get_spawn_parameters(output, node_id)
                            # Track spawn parameters for statistics
                            self.current_episode_spawn_params['gamma'].append(params[0])
                            self.current_episode_spawn_params['alpha'].append(params[1])
                            self.current_episode_spawn_params['noise'].append(params[2])
                            self.current_episode_spawn_params['theta'].append(params[3])
                            
                            self.env.topology.spawn(node_id, gamma=params[0], alpha=params[1], 
                                                  noise=params[2], theta=params[3])
                        elif action_type == 'delete':
                            self.env.topology.delete(node_id)
                    except Exception as e:
                        # Action failed - this is learning signal
                        continue
                
                # Store executed actions for next step's detailed logging
                previous_executed_actions = topology_actions
            else:
                # No actions taken this step
                previous_executed_actions = {}
            
            # Environment step
            next_obs, reward_components, terminated, truncated, info = self.env.step(0)
            done = terminated or truncated

            # Track environment-side empty graph recoveries for logging consistency
            if info.get('empty_graph_recovered'):
                self.empty_graph_recovery_count += 1

            # Enhanced milestone-based reward shaping
            if 'milestone_bonus' not in reward_components:
                reward_components['milestone_bonus'] = 0.0
            
            # Progressive milestone rewards based on episode length and achievements
            episode_progress = episode_length / self.max_steps
            num_nodes = state_dict.get('num_nodes', 0)
            
            # Stage 1: Survival and basic navigation rewards
            if episode_length >= 10 and episode_progress < 0.3:
                reward_components['milestone_bonus'] += 2.0  # Early survival bonus
            
            # Stage 2: Spatial exploration rewards
            substrate_x = state_dict.get('substrate_x', 0.0)
            if substrate_x > 0.7 and num_nodes >= 2:
                reward_components['milestone_bonus'] += 5.0  # Right-side exploration
            elif substrate_x > 0.9 and num_nodes >= 3:
                reward_components['milestone_bonus'] += 10.0  # Deep exploration with nodes
            
            # Stage 3: Node management skill rewards
            spawn_count = len(self.current_episode_rewards['spawn_reward'])
            delete_count = len(self.current_episode_rewards['delete_reward'])
            
            if spawn_count >= 1 and delete_count >= 1:
                reward_components['milestone_bonus'] += 3.0  # Basic node management
            if spawn_count >= 3 and delete_count >= 2:
                reward_components['milestone_bonus'] += 7.0  # Advanced node management
            
            # Stage 4: Efficiency and optimization rewards
            if episode_length >= 30 and num_nodes <= 10:  # Efficient long episodes
                reward_components['milestone_bonus'] += 4.0
            
            # Stage 5: Consistency rewards (avoid large negative spikes)
            recent_rewards = self.current_episode_rewards['total_reward'][-5:] if len(self.current_episode_rewards['total_reward']) >= 5 else []
            if recent_rewards and all(r >= -5.0 for r in recent_rewards):
                reward_components['milestone_bonus'] += 1.0  # Consistency bonus
            
            # Add milestone bonus to total_reward
            if reward_components['milestone_bonus'] > 0:
                reward_components['total_reward'] += reward_components['milestone_bonus']

            # Track reward components for statistics
            for component, value in reward_components.items():
                if component in self.current_episode_rewards:
                    self.current_episode_rewards[component].append(value)

            # Store reward components
            rewards.append(reward_components)
            episode_length += 1
        
        # Get final state value for bootstrapping (if episode was truncated, not terminated)
        final_values = None
        if truncated and not terminated and states:
            # Episode was truncated (timeout) - bootstrap from final state value
            final_state_dict = self.state_extractor.get_state_features(include_substrate=True)
            final_state_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in final_state_dict.items()}
            
            if final_state_dict['num_nodes'] > 0:
                with torch.no_grad():
                    final_output = self.network(final_state_dict, deterministic=True)
                    final_values = final_output['value_predictions']
        
        success = not done or episode_length >= 150  # Consider long episodes successful
        
        return states, actions_taken, rewards, values, log_probs_list, final_values, terminated, success
    
    def compute_returns_and_advantages(self, rewards: List[Dict], values: List[Dict], 
                                     final_values: Optional[Dict] = None, terminated: bool = False,
                                     gamma: float = 0.99, gae_lambda: float = 0.95) -> Dict[str, torch.Tensor]:
        """Compute returns and advantages for each reward component with improved normalization"""
        if not rewards or not values:
            return {}, {}
        
        # Tier 1: Per-episode reward normalization (prevents component domination within episode)
        normalized_rewards = self.normalize_episode_rewards(rewards)
        
        # Tier 2: Get adaptive scaling factors (for cross-episode balance)  
        scaling_factors = self.get_component_scaling_factors()
        
        returns = {}
        advantages = {}
        
        for component in self.component_names:
            # Extract normalized component rewards
            component_rewards = torch.tensor(normalized_rewards[component], 
                                           dtype=torch.float32, device=self.device)
            
            # Apply adaptive scaling for cross-episode consistency
            scaling_factor = scaling_factors.get(component, 1.0)
            component_rewards = component_rewards * scaling_factor
            
            component_values = torch.stack([v[component] for v in values])
            
            # Compute returns using GAE with proper bootstrapping
            T = len(component_rewards)
            component_returns = torch.zeros_like(component_rewards)
            component_advantages = torch.zeros_like(component_rewards)
            
            # Determine next value for bootstrapping
            if terminated:
                # Episode terminated naturally - next value is 0
                next_value = 0.0
            elif final_values is not None and component in final_values:
                # Episode was truncated - bootstrap from critic's final prediction
                next_value = final_values[component].item()
            else:
                # Fallback: assume terminal (conservative approach)
                next_value = 0.0
            
            # GAE computation (backward pass)
            gae = 0.0
            
            for t in reversed(range(T)):
                if t == T - 1:
                    # Last step: use determined next_value (0.0 for terminal, critic prediction for truncated)
                    next_value_t = next_value
                else:
                    # Intermediate step: next state's value
                    next_value_t = component_values[t + 1].item()
                
                # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                delta = component_rewards[t] + gamma * next_value_t - component_values[t]
                
                # GAE: Â_t = δ_t + (γλ) * Â_{t+1}
                gae = delta + gamma * gae_lambda * gae
                component_advantages[t] = gae
                
                # Return: R_t = Â_t + V(s_t)
                component_returns[t] = gae + component_values[t]
            
            returns[component] = component_returns
            advantages[component] = component_advantages
        
        return returns, advantages
    
    def compute_hybrid_policy_loss(self, old_log_probs_dict: Dict[str, torch.Tensor], 
                                  eval_output: Dict[str, torch.Tensor], 
                                  advantage: float, episode: int = 0) -> Dict[str, torch.Tensor]:
        """Compute balanced policy loss for hybrid discrete+continuous actions with adaptive gradient scaling"""
        policy_losses = {}
        
        # Extract base weights and clipping
        clip_eps = self.policy_loss_weights['clip_epsilon']
        
        # === DISCRETE POLICY LOSS ===
        discrete_loss_raw = torch.tensor(0.0, device=self.device)
        if ('discrete' in old_log_probs_dict and 
            'discrete_log_probs' in eval_output and 
            len(old_log_probs_dict['discrete']) > 0 and 
            len(eval_output['discrete_log_probs']) > 0):
            
            # Compute ratio for discrete actions
            old_discrete_log_prob = old_log_probs_dict['discrete'].sum()
            new_discrete_log_prob = eval_output['discrete_log_probs'].sum()
            
            ratio_discrete = torch.exp(new_discrete_log_prob - old_discrete_log_prob)
            clipped_ratio_discrete = torch.clamp(ratio_discrete, 1 - clip_eps, 1 + clip_eps)
            
            # PPO discrete policy loss (before scaling)
            discrete_loss_raw = -torch.min(
                ratio_discrete * advantage,
                clipped_ratio_discrete * advantage
            )
        
        # === CONTINUOUS POLICY LOSS ===
        continuous_loss_raw = torch.tensor(0.0, device=self.device)
        if ('continuous' in old_log_probs_dict and 
            'continuous_log_probs' in eval_output and 
            len(old_log_probs_dict['continuous']) > 0 and 
            len(eval_output['continuous_log_probs']) > 0):
            
            # Compute ratio for continuous actions
            old_continuous_log_prob = old_log_probs_dict['continuous'].sum()
            new_continuous_log_prob = eval_output['continuous_log_probs'].sum()
            
            ratio_continuous = torch.exp(new_continuous_log_prob - old_continuous_log_prob)
            clipped_ratio_continuous = torch.clamp(ratio_continuous, 1 - clip_eps, 1 + clip_eps)
            
            # PPO continuous policy loss (before scaling)
            continuous_loss_raw = -torch.min(
                ratio_continuous * advantage,
                clipped_ratio_continuous * advantage
            )
        
        # === ADAPTIVE GRADIENT SCALING ===
        # Compute adaptive weights based on gradient magnitudes
        adaptive_discrete_weight, adaptive_continuous_weight = self.compute_adaptive_gradient_scaling(
            discrete_loss_raw, continuous_loss_raw
        )
        
        # Apply adaptive weights
        policy_losses['discrete'] = adaptive_discrete_weight * discrete_loss_raw
        policy_losses['continuous'] = adaptive_continuous_weight * continuous_loss_raw
        
        # === ENHANCED ENTROPY REGULARIZATION ===
        # Use enhanced entropy system instead of basic entropy handling
        entropy_losses = self.compute_enhanced_entropy_loss(eval_output, episode)
        
        # Total policy loss
        total_policy_loss = sum(policy_losses.values())
        
        # Ensure entropy loss is always a tensor
        if entropy_losses:
            total_entropy_loss = sum(entropy_losses.values())
        else:
            total_entropy_loss = torch.tensor(0.0, device=self.device)
            
        # Ensure it's a tensor, not a scalar
        if not isinstance(total_entropy_loss, torch.Tensor):
            total_entropy_loss = torch.tensor(total_entropy_loss, device=self.device)
        
        return {
            'policy_loss_discrete': policy_losses['discrete'],
            'policy_loss_continuous': policy_losses['continuous'],
            'total_policy_loss': total_policy_loss,
            'entropy_loss': total_entropy_loss,
            'discrete_weight_used': adaptive_discrete_weight,
            'continuous_weight_used': adaptive_continuous_weight,
            'discrete_grad_norm': getattr(self, 'discrete_grad_norm_ema', 0.0),
            'continuous_grad_norm': getattr(self, 'continuous_grad_norm_ema', 0.0),
            'gradient_scaling_active': self.enable_gradient_scaling and self.gradient_step_count >= self.gradient_warmup_steps
        }
    
    def update_policy(self, states: List[Dict], actions: List[Dict], 
                     returns: Dict[str, torch.Tensor], advantages: Dict[str, torch.Tensor],
                     old_log_probs: List[Dict], episode: int = 0) -> Dict[str, float]:
        """Update policy using PPO with efficient batched re-evaluation and enhanced entropy"""
        if not states or not actions:
            return {}
        
        losses = {}
        
        # === ENHANCED LEARNABLE ADVANTAGE WEIGHTING ===
        # Use enhanced learnable weighting system instead of traditional approach
        total_advantages = self.compute_enhanced_advantage_weights(advantages)
        
        # === EFFICIENT BATCHED RE-EVALUATION ===
        
        # Step 1: Collate all states into a single batch
        batched_states = self.collate_graph_batch(states)
        
        # Step 2: Collate action masks
        action_masks = []
        for i, action_dict in enumerate(actions):
            if 'mask' in action_dict and action_dict['mask'] is not None:
                action_masks.append(action_dict['mask'])
            elif states[i].get('num_nodes', 0) > 0:
                # Create default mask if not provided
                num_nodes = states[i]['num_nodes']
                default_mask = torch.ones(num_nodes, 2, dtype=torch.bool, device=self.device)
                action_masks.append(default_mask)
            else:
                action_masks.append(None)
        
        # Batch action masks
        if any(mask is not None for mask in action_masks):
            valid_masks = [mask for mask in action_masks if mask is not None]
            if valid_masks:
                batched_action_mask = torch.cat(valid_masks, dim=0)
            else:
                batched_action_mask = None
        else:
            batched_action_mask = None
        
        # Step 3: Collate discrete and continuous actions
        all_discrete_actions = []
        all_continuous_actions = []
        
        for action_dict in actions:
            if 'discrete' in action_dict and action_dict['discrete'].shape[0] > 0:
                all_discrete_actions.append(action_dict['discrete'])
                all_continuous_actions.append(action_dict['continuous'])
        
        if all_discrete_actions:
            batched_discrete_actions = torch.cat(all_discrete_actions, dim=0)
            batched_continuous_actions = torch.cat(all_continuous_actions, dim=0)
        else:
            # Handle empty actions case
            batched_discrete_actions = torch.empty(0, dtype=torch.long, device=self.device)
            batched_continuous_actions = torch.empty(0, 4, device=self.device)
        
        # Step 4: Single batched re-evaluation (🚀 MUCH FASTER!)
        if batched_states['num_nodes'] > 0:
            batched_eval_output = self.network.evaluate_actions(
                batched_states,
                batched_discrete_actions,
                batched_continuous_actions,
                action_mask=batched_action_mask
            )
        else:
            # Handle empty batch
            batched_eval_output = {
                'discrete_log_probs': torch.empty(0, device=self.device),
                'continuous_log_probs': torch.empty(0, device=self.device),
                'total_log_probs': torch.empty(0, device=self.device),
                'value_predictions': {k: torch.empty(0, device=self.device) for k in self.component_names},
                'entropy': torch.tensor(0.0, device=self.device)
            }
        
        # Step 5: Unbatch the evaluation results
        node_counts = batched_states['node_counts']
        individual_eval_outputs = self.unbatch_network_output(batched_eval_output, node_counts)
        
        # === COMPUTE LOSSES USING BATCHED RESULTS ===
        
        hybrid_policy_losses = []
        value_losses = {component: [] for component in self.component_names}
        
        for i, (eval_output, old_log_probs_dict) in enumerate(zip(individual_eval_outputs, old_log_probs)):
            if i >= len(actions) or 'discrete' not in actions[i]:
                continue
            
            # === HYBRID POLICY LOSS ===
            advantage = total_advantages[i]
            
            # Compute hybrid policy loss with separate discrete/continuous handling
            hybrid_loss_dict = self.compute_hybrid_policy_loss(
                old_log_probs_dict, eval_output, advantage, episode
            )
            
            hybrid_policy_losses.append(hybrid_loss_dict)
            
            # === VALUE LOSSES (Component-specific) ===
            for component in self.component_names:
                if component in eval_output['value_predictions'] and component in returns:
                    predicted_value = eval_output['value_predictions'][component]
                    target_return = returns[component][i]
                    value_loss = F.mse_loss(predicted_value, target_return)
                    value_losses[component].append(value_loss)
        
        # === AGGREGATE LOSSES ===
        
        # Policy losses
        if hybrid_policy_losses:
            # Average across batch
            avg_discrete_loss = torch.stack([h['policy_loss_discrete'] for h in hybrid_policy_losses]).mean()
            avg_continuous_loss = torch.stack([h['policy_loss_continuous'] for h in hybrid_policy_losses]).mean()
            avg_total_policy_loss = torch.stack([h['total_policy_loss'] for h in hybrid_policy_losses]).mean()
            # Handle entropy loss - ensure all are tensors
            entropy_losses_list = []
            for h in hybrid_policy_losses:
                entropy_loss = h['entropy_loss']
                if not isinstance(entropy_loss, torch.Tensor):
                    entropy_loss = torch.tensor(entropy_loss, device=self.device)
                entropy_losses_list.append(entropy_loss)
            avg_entropy_loss = torch.stack(entropy_losses_list).mean()
            
            losses['policy_loss_discrete'] = avg_discrete_loss.item()
            losses['policy_loss_continuous'] = avg_continuous_loss.item()
            losses['total_policy_loss'] = avg_total_policy_loss.item()
            losses['entropy_loss'] = avg_entropy_loss.item()
            
            # Show weight usage
            losses['discrete_weight'] = hybrid_policy_losses[0]['discrete_weight_used']
            losses['continuous_weight'] = hybrid_policy_losses[0]['continuous_weight_used']
        else:
            avg_total_policy_loss = torch.tensor(0.0, device=self.device)
            avg_entropy_loss = torch.tensor(0.0, device=self.device)
        
        # Component-weighted value loss
        total_value_loss = torch.tensor(0.0, device=self.device)
        for component, component_losses in value_losses.items():
            if component_losses:
                component_loss = torch.stack(component_losses).mean()
                weight = self.component_weights.get(component, 1.0)
                total_value_loss += weight * component_loss
                losses[f'value_loss_{component}'] = component_loss.item()
        
        losses['total_value_loss'] = total_value_loss.item()
        
        # === ENTROPY BONUS FOR EXPLORATION ===
        # Encourage exploration to prevent premature convergence
        entropy_bonus = torch.tensor(0.0, device=self.device)
        if 'entropy' in batched_eval_output:
            entropy_bonus = -self.entropy_bonus_coeff * batched_eval_output['entropy']
            losses['entropy_bonus'] = entropy_bonus.item()
        else:
            losses['entropy_bonus'] = 0.0
        
        # === TOTAL LOSS AND OPTIMIZATION ===
        total_loss = avg_total_policy_loss + 0.5 * total_value_loss + avg_entropy_loss + entropy_bonus
        losses['total_loss'] = total_loss.item()
        
        # Update learnable component weights based on policy performance
        self.update_learnable_weights(advantages, avg_total_policy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return losses

    def update_policy_minibatch(self, states: List[Dict], actions: List[Dict], 
                               returns: List[torch.Tensor], advantages: List[torch.Tensor],
                               old_log_probs: List[Dict], old_values: List[Dict], episode: int = 0) -> Dict[str, float]:
        """Update policy using a minibatch of data from the trajectory buffer"""
        if not states or not actions:
            return {}
        
        # Convert list data back to the format expected by original update_policy
        # Group returns and advantages by component
        returns_dict = {component: [] for component in self.component_names}
        advantages_dict = {component: [] for component in self.component_names}
        old_values_dict = {component: [] for component in self.component_names}
        
        # For minibatch training, we need to reconstruct the component-wise format
        # Since returns and advantages are already computed as scalars in the buffer,
        # we'll use them directly as total values
        for i, (ret_val, adv_val, old_val) in enumerate(zip(returns, advantages, old_values)):
            # Use the total values for all components (they're already weighted)
            for component in self.component_names:
                returns_dict[component].append(ret_val)
                advantages_dict[component].append(adv_val)
                # Handle old values - if it's a dict, extract component, otherwise use as total
                if isinstance(old_val, dict):
                    old_values_dict[component].append(old_val.get(component, old_val.get('total_reward', ret_val)))
                else:
                    old_values_dict[component].append(old_val)
        
        # Convert to tensors
        for component in self.component_names:
            returns_dict[component] = torch.stack(returns_dict[component])
            advantages_dict[component] = torch.stack(advantages_dict[component])
            old_values_dict[component] = torch.stack(old_values_dict[component])
        
        # Call the original update_policy method with old values
        return self.update_policy_with_value_clipping(states, actions, returns_dict, advantages_dict, old_log_probs, old_values_dict, episode)

    def update_policy_with_value_clipping(self, states: List[Dict], actions: List[Dict], 
                                        returns: Dict[str, torch.Tensor], advantages: Dict[str, torch.Tensor],
                                        old_log_probs: List[Dict], old_values: Dict[str, torch.Tensor], episode: int = 0) -> Dict[str, float]:
        """Update policy using PPO with value clipping for stable critic updates"""
        if not states or not actions:
            return {}
        
        losses = {}
        
        # === ENHANCED LEARNABLE ADVANTAGE WEIGHTING ===
        # Use enhanced learnable weighting system instead of traditional approach
        total_advantages = self.compute_enhanced_advantage_weights(advantages)
        
        # === EFFICIENT BATCHED RE-EVALUATION ===
        
        # Step 1: Collate all states into a single batch
        batched_states = self.collate_graph_batch(states)
        
        # Step 2: Collate action masks
        action_masks = []
        for i, action_dict in enumerate(actions):
            if 'mask' in action_dict and action_dict['mask'] is not None:
                action_masks.append(action_dict['mask'])
            elif states[i].get('num_nodes', 0) > 0:
                # Create default mask if not provided
                num_nodes = states[i]['num_nodes']
                default_mask = torch.ones(num_nodes, 2, dtype=torch.bool, device=self.device)
                action_masks.append(default_mask)
            else:
                action_masks.append(None)
        
        # Batch action masks
        if any(mask is not None for mask in action_masks):
            valid_masks = [mask for mask in action_masks if mask is not None]
            if valid_masks:
                batched_action_mask = torch.cat(valid_masks, dim=0)
            else:
                batched_action_mask = None
        else:
            batched_action_mask = None
        
        # Step 3: Collate discrete and continuous actions
        all_discrete_actions = []
        all_continuous_actions = []
        
        for action_dict in actions:
            if 'discrete' in action_dict and action_dict['discrete'].shape[0] > 0:
                all_discrete_actions.append(action_dict['discrete'])
                all_continuous_actions.append(action_dict['continuous'])
        
        if all_discrete_actions:
            batched_discrete_actions = torch.cat(all_discrete_actions, dim=0)
            batched_continuous_actions = torch.cat(all_continuous_actions, dim=0)
        else:
            # Handle empty actions case
            batched_discrete_actions = torch.empty(0, dtype=torch.long, device=self.device)
            batched_continuous_actions = torch.empty(0, 4, device=self.device)
        
        # Step 4: Single batched re-evaluation (🚀 MUCH FASTER!)
        if batched_states['num_nodes'] > 0:
            batched_eval_output = self.network.evaluate_actions(
                batched_states,
                batched_discrete_actions,
                batched_continuous_actions,
                action_mask=batched_action_mask
            )
        else:
            # Handle empty batch
            batched_eval_output = {
                'discrete_log_probs': torch.empty(0, device=self.device),
                'continuous_log_probs': torch.empty(0, device=self.device),
                'total_log_probs': torch.empty(0, device=self.device),
                'value_predictions': {k: torch.empty(0, device=self.device) for k in self.component_names},
                'entropy': torch.tensor(0.0, device=self.device)
            }
        
        # Step 5: Unbatch the evaluation results
        node_counts = batched_states['node_counts']
        individual_eval_outputs = self.unbatch_network_output(batched_eval_output, node_counts)
        
        # === COMPUTE LOSSES USING BATCHED RESULTS ===
        
        hybrid_policy_losses = []
        value_losses = {component: [] for component in self.component_names}
        
        for i, (eval_output, old_log_probs_dict) in enumerate(zip(individual_eval_outputs, old_log_probs)):
            if i >= len(actions) or 'discrete' not in actions[i]:
                continue
            
            # === HYBRID POLICY LOSS ===
            advantage = total_advantages[i]
            
            # Compute hybrid policy loss with separate discrete/continuous handling
            hybrid_loss_dict = self.compute_hybrid_policy_loss(
                old_log_probs_dict, eval_output, advantage, episode
            )
            
            hybrid_policy_losses.append(hybrid_loss_dict)
            
            # === PPO VALUE LOSSES WITH CLIPPING (Component-specific) ===
            for component in self.component_names:
                if component in eval_output['value_predictions'] and component in returns:
                    predicted_value = eval_output['value_predictions'][component]
                    target_return = returns[component][i]
                    old_value = old_values[component][i]
                    
                    if self.enable_value_clipping:
                        # PPO-style value clipping to prevent large critic updates
                        value_pred_clipped = old_value + torch.clamp(
                            predicted_value - old_value, 
                            -self.value_clip_epsilon, 
                            self.value_clip_epsilon
                        )
                        
                        # Compute both clipped and unclipped value losses
                        v_loss1 = (predicted_value - target_return) ** 2
                        v_loss2 = (value_pred_clipped - target_return) ** 2
                        
                        # Take the maximum to ensure we don't make updates that are too large
                        value_loss = torch.max(v_loss1, v_loss2)
                    else:
                        # Standard MSE loss (fallback)
                        value_loss = F.mse_loss(predicted_value, target_return)
                    
                    value_losses[component].append(value_loss)
        
        # === AGGREGATE LOSSES ===
        
        # Policy losses
        if hybrid_policy_losses:
            # Average across batch
            avg_discrete_loss = torch.stack([h['policy_loss_discrete'] for h in hybrid_policy_losses]).mean()
            avg_continuous_loss = torch.stack([h['policy_loss_continuous'] for h in hybrid_policy_losses]).mean()
            avg_total_policy_loss = torch.stack([h['total_policy_loss'] for h in hybrid_policy_losses]).mean()
            # Handle entropy loss - ensure all are tensors
            entropy_losses_list = []
            for h in hybrid_policy_losses:
                entropy_loss = h['entropy_loss']
                if not isinstance(entropy_loss, torch.Tensor):
                    entropy_loss = torch.tensor(entropy_loss, device=self.device)
                entropy_losses_list.append(entropy_loss)
            avg_entropy_loss = torch.stack(entropy_losses_list).mean()
            
            losses['policy_loss_discrete'] = avg_discrete_loss.item()
            losses['policy_loss_continuous'] = avg_continuous_loss.item()
            losses['total_policy_loss'] = avg_total_policy_loss.item()
            losses['entropy_loss'] = avg_entropy_loss.item()
            
            # Show weight usage
            losses['discrete_weight'] = hybrid_policy_losses[0]['discrete_weight_used']
            losses['continuous_weight'] = hybrid_policy_losses[0]['continuous_weight_used']
        else:
            avg_total_policy_loss = torch.tensor(0.0, device=self.device)
            avg_entropy_loss = torch.tensor(0.0, device=self.device)
        
        # Component-weighted value loss with clipping information
        total_value_loss = torch.tensor(0.0, device=self.device)
        for component, component_losses in value_losses.items():
            if component_losses:
                component_loss = torch.stack(component_losses).mean()
                weight = self.component_weights.get(component, 1.0)
                total_value_loss += weight * component_loss
                losses[f'value_loss_{component}'] = component_loss.item()
        
        losses['total_value_loss'] = total_value_loss.item()
        losses['value_clipping_enabled'] = self.enable_value_clipping
        losses['value_clip_epsilon'] = self.value_clip_epsilon
        
        # === ENTROPY BONUS FOR EXPLORATION ===
        # Encourage exploration to prevent premature convergence
        entropy_bonus = torch.tensor(0.0, device=self.device)
        if 'entropy' in batched_eval_output:
            entropy_bonus = -self.entropy_bonus_coeff * batched_eval_output['entropy']
            losses['entropy_bonus'] = entropy_bonus.item()
        else:
            losses['entropy_bonus'] = 0.0
        
        # === TOTAL LOSS AND OPTIMIZATION ===
        total_loss = avg_total_policy_loss + 0.5 * total_value_loss + avg_entropy_loss + entropy_bonus
        losses['total_loss'] = total_loss.item()
        
        # Update learnable component weights based on policy performance
        self.update_learnable_weights(advantages, avg_total_policy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return losses
    
    def train(self):
        """Main training loop with batch updates"""
        # Run directory already created in __init__, just confirm it exists
        print(f"🏋️ Starting training for {self.total_episodes} episodes (Run #{self.run_number:04d})")
        print(f"📁 Saving to: {self.run_dir}")
        print(f"📊 Batch Training: {self.rollout_batch_size} episodes per batch, {self.update_epochs} update epochs, {self.minibatch_size} minibatch size")
        print(f"📊 Progress format: Batch | R: Current (MA: MovingAvg, Best: Best) | Loss | Entropy | LR: Learning Rate | Episodes | Success Rate | Focus: Component(Weight)")
        
        episode_count = 0
        batch_count = 0
        
        while episode_count < self.total_episodes:
            # ==========================================
            # PHASE 1: COLLECT ROLLOUT BATCH
            # ==========================================
            self.trajectory_buffer.clear()
            batch_episode_rewards = []
            batch_successes = []
            
            # Collect rollout_batch_size episodes
            for batch_episode in range(self.rollout_batch_size):
                if episode_count >= self.total_episodes:
                    break
                    
                # Start new episode in buffer
                self.trajectory_buffer.start_episode()
                
                # Collect episode using existing method
                states, actions, rewards, values, log_probs, final_values, terminated, success = self.collect_episode(episode_count)
                
                if not rewards:
                    continue
                
                # Add episode data to buffer
                for i in range(len(states)):
                    self.trajectory_buffer.add_step(
                        states[i], actions[i], rewards[i], values[i], log_probs[i], values[i]  # old_values = values for first collection
                    )
                
                # Finish episode in buffer
                self.trajectory_buffer.finish_episode(final_values, terminated, success)
                
                # Update component statistics for adaptive scaling
                self.update_component_stats(rewards)
                
                # Track episode rewards and success
                episode_total_reward = sum(r.get('total_reward', 0.0) for r in rewards)
                batch_episode_rewards.append(episode_total_reward)
                batch_successes.append(success)
                
                # Update episode tracking
                self.episode_rewards['total_reward'].append(episode_total_reward)
                for component in self.component_names:
                    component_reward = sum(r.get(component, 0.0) for r in rewards)
                    self.episode_rewards[component].append(component_reward)
                
                # Enhanced reward smoothing with multiple methods
                window = self.moving_avg_window if hasattr(self, 'moving_avg_window') else 20
                simple_ma = moving_average(self.episode_rewards['total_reward'], window)
                robust_ma = robust_moving_average(self.episode_rewards['total_reward'], window)
                ema = exponential_moving_average(self.episode_rewards['total_reward'])
                
                # Use robust moving average as primary smoothed metric
                self.smoothed_rewards.append(robust_ma)
                
                # Track smoothed rewards for milestone_bonus component too
                if 'milestone_bonus' in self.episode_rewards:
                    milestone_smoothed = robust_moving_average(self.episode_rewards['milestone_bonus'], window//2)
                else:
                    milestone_rewards = [sum(r.get('milestone_bonus', 0.0) for r in rewards)]
                    self.episode_rewards['milestone_bonus'] = self.episode_rewards.get('milestone_bonus', []) + milestone_rewards
                    milestone_smoothed = milestone_rewards[0]
                
                # Update learning rate schedule
                self.scheduler.step()
                
                # Compute and save spawn parameter statistics
                self.save_spawn_statistics(episode_count)
                
                # Compute and save reward component statistics
                self.save_reward_statistics(episode_count)
                
                # Save detailed node logs for this episode
                if self.enable_detailed_logging and self.detailed_node_logs:
                    self.save_detailed_node_logs(episode_count)
                
                # Enhanced episode progress print with smoothed metrics
                latest_loss = self.losses['total_loss'][-1] if self.losses['total_loss'] else 0.0
                smoothed_reward = self.smoothed_rewards[-1] if self.smoothed_rewards else episode_total_reward
                milestone_bonus = sum(r.get('milestone_bonus', 0.0) for r in rewards)
                print(f"Episode {episode_count:4d}: R={episode_total_reward:7.3f} (Smooth={smoothed_reward:6.2f}) | MB={milestone_bonus:4.1f} | Steps={len(states):3d} | Success={success} | Loss={latest_loss:7.4f}")
                
                episode_count += 1
            
            # ==========================================
            # PHASE 2: COMPUTE RETURNS AND ADVANTAGES
            # ==========================================
            if len(self.trajectory_buffer) > 0:
                # Compute returns and advantages for all episodes in buffer
                algorithm_config = self.config_loader.get_algorithm_config()
                gamma = algorithm_config.get('gamma', 0.99)
                gae_lambda = algorithm_config.get('gae_lambda', 0.95)
                
                self.trajectory_buffer.compute_returns_and_advantages_for_all_episodes(gamma, gae_lambda)
                
                # ==========================================
                # PHASE 3: BATCH POLICY UPDATES
                # ==========================================
                
                # Perform multiple update epochs on the collected batch
                total_losses = {}
                
                for epoch in range(self.update_epochs):
                    # Create random minibatches from buffer
                    minibatches = self.trajectory_buffer.create_minibatches(self.minibatch_size)
                    
                    epoch_losses = {}
                    for minibatch in minibatches:
                        # Update policy on this minibatch
                        losses = self.update_policy_minibatch(
                            minibatch['states'], 
                            minibatch['actions'],
                            minibatch['returns'], 
                            minibatch['advantages'], 
                            minibatch['log_probs'],
                            minibatch['old_values'], 
                            episode_count
                        )
                        
                        # Accumulate losses
                        for loss_name, loss_value in losses.items():
                            if loss_name not in epoch_losses:
                                epoch_losses[loss_name] = []
                            epoch_losses[loss_name].append(loss_value)
                    
                    # Average losses across minibatches for this epoch
                    for loss_name, loss_values in epoch_losses.items():
                        if loss_name not in total_losses:
                            total_losses[loss_name] = []
                        total_losses[loss_name].append(np.mean(loss_values))
                
                # Average losses across all epochs
                final_losses = {}
                for loss_name, loss_values in total_losses.items():
                    final_losses[loss_name] = np.mean(loss_values)
                
                # Track losses
                for loss_name, loss_value in final_losses.items():
                    self.losses[loss_name].append(loss_value)
                # Update smoothed losses
                window = self.moving_avg_window if hasattr(self, 'moving_avg_window') else 20
                if 'total_loss' in final_losses:
                    smoothed_loss = moving_average(self.losses['total_loss'], window)
                    self.smoothed_losses.append(smoothed_loss)

                # ---- Update per-episode JSON entries with computed episode loss ----
                self.save_loss_statistics(episode_count)
                try:
                    episode_loss_value = float(final_losses.get('total_policy_loss', None)) if final_losses else None
                    num_episodes_in_batch = len(batch_episode_rewards)

                    if episode_loss_value is not None and num_episodes_in_batch > 0:
                        # Helper to update the last N entries in a JSON history file
                        def update_last_n_entries(filepath, key_name='episode_loss'):
                            if not os.path.exists(filepath):
                                return
                            try:
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                            except Exception:
                                return

                            if isinstance(data, dict):
                                # Old format: single dict, just set the key
                                data[key_name] = episode_loss_value
                                new_data = data
                            elif isinstance(data, list):
                                # Update the last num_episodes_in_batch entries
                                for i in range(1, num_episodes_in_batch + 1):
                                    if len(data) - i >= 0:
                                        if isinstance(data[-i], dict):
                                            data[-i][key_name] = episode_loss_value
                                new_data = data
                            else:
                                return

                            # Write back
                            try:
                                with open(filepath, 'w') as f:
                                    json.dump(new_data, f, indent=2)
                            except Exception:
                                pass

                        # Update both spawn & reward stat files with episode_loss and recent_total_loss
                        update_last_n_entries(os.path.join(self.run_dir, 'spawn_parameters_stats.json'))
                        update_last_n_entries(os.path.join(self.run_dir, 'reward_components_stats.json'))
                        
                        # Also update recent_total_loss field with the batch's total_loss
                        if 'total_loss' in final_losses:
                            total_loss_value = float(final_losses['total_loss'])
                            
                            # Define a version that updates with total_loss_value instead of episode_loss_value
                            def update_recent_total_loss(filepath):
                                if not os.path.exists(filepath):
                                    return
                                try:
                                    with open(filepath, 'r') as f:
                                        data = json.load(f)
                                except Exception:
                                    return

                                if isinstance(data, dict):
                                    # Old format: single dict, just set the key
                                    data['recent_total_loss'] = total_loss_value
                                    new_data = data
                                elif isinstance(data, list):
                                    # Update the last num_episodes_in_batch entries
                                    for i in range(1, num_episodes_in_batch + 1):
                                        if len(data) - i >= 0:
                                            if isinstance(data[-i], dict):
                                                data[-i]['recent_total_loss'] = total_loss_value
                                    new_data = data
                                else:
                                    return

                                # Write back
                                try:
                                    with open(filepath, 'w') as f:
                                        json.dump(new_data, f, indent=2)
                                except Exception:
                                    pass
                            
                            update_recent_total_loss(os.path.join(self.run_dir, 'spawn_parameters_stats.json'))
                            update_recent_total_loss(os.path.join(self.run_dir, 'reward_components_stats.json'))
                            def update_recent_total_loss(filepath):
                                if not os.path.exists(filepath):
                                    return
                                try:
                                    with open(filepath, 'r') as f:
                                        data = json.load(f)
                                except Exception:
                                    return

                                if isinstance(data, list):
                                    # Update the last num_episodes_in_batch entries with total_loss_value
                                    for i in range(1, num_episodes_in_batch + 1):
                                        if len(data) - i >= 0:
                                            if isinstance(data[-i], dict):
                                                data[-i]['recent_total_loss'] = total_loss_value

                                # Write back
                                try:
                                    with open(filepath, 'w') as f:
                                        json.dump(data, f, indent=2)
                                except Exception:
                                    pass
                            
                            update_recent_total_loss(os.path.join(self.run_dir, 'spawn_parameters_stats.json'))
                            update_recent_total_loss(os.path.join(self.run_dir, 'reward_components_stats.json'))
                except Exception:
                    # Non-critical: do not fail training if we can't update files
                    pass
                
                # ==========================================
                # PHASE 4: BATCH PROGRESS REPORTING
                # ==========================================
                
                # Get batch statistics
                batch_stats = self.trajectory_buffer.get_episode_stats()
                
                # Progress reporting (every few batches or early batches)
                if batch_count % max(1, self.progress_print_every // self.rollout_batch_size) == 0 or batch_count < 3:
                    recent_reward = moving_average(self.episode_rewards['total_reward'], min(20, len(self.episode_rewards['total_reward'])))
                    best_so_far = max(self.episode_rewards['total_reward']) if self.episode_rewards['total_reward'] else 0.0
                    total_loss = final_losses.get('total_loss', 0.0)
                    entropy_bonus = final_losses.get('entropy_bonus', 0.0)
                    dominant_comp = max(self.component_weights.items(), key=lambda x: x[1])[0]
                    dominant_weight = self.component_weights[dominant_comp]
                    
                    # Build substrate info
                    substrate_info = ""
                    if self.substrate_type == 'random':
                        params = self.current_substrate_params
                        substrate_info = f" | Sub: {params['kind'][:3]}(m:{params['m']:.3f} b:{params['b']:.2f})"
                    elif self.substrate_type in ['linear', 'exponential']:
                        substrate_info = f" | Sub: {self.substrate_type[:3]}"
                    
                    # Build empty graph recovery info
                    recovery_info = ""
                    if self.empty_graph_recovery_count > 0:
                        recovery_info = f" | Recoveries: {self.empty_graph_recovery_count}"
                    
                    print(f"Batch {batch_count:3d} | R: {batch_stats['avg_reward']:6.3f} (MA: {recent_reward:6.3f}, Best: {best_so_far:6.3f}) | "
                          f"Loss: {total_loss:7.4f} | Entropy: {entropy_bonus:7.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e} | Episodes: {batch_stats['num_episodes']:2d} | "
                          f"Success: {batch_stats['success_rate']:.2f} | Focus: {dominant_comp[:8]}({dominant_weight:.3f}){substrate_info}{recovery_info}")
                
                # Save best model (check against best episode in this batch)
                best_batch_reward = max(batch_episode_rewards) if batch_episode_rewards else 0.0
                if best_batch_reward > self.best_total_reward:
                    self.best_total_reward = best_batch_reward
                    self.save_model(f"best_model_batch{batch_count}.pt")
                
                # Periodic saves (only if checkpoint_every is configured)
                if self.checkpoint_every is not None and batch_count % (self.checkpoint_every // self.rollout_batch_size) == 0 and batch_count > 0:
                    self.save_model(f"checkpoint_batch{batch_count}.pt")
                    print(f"💾 Checkpoint saved at batch {batch_count} (episode {episode_count})")
                
                batch_count += 1
        
        print(f"🎉 Training completed!")
        print(f"   Episodes trained: {episode_count}")
        print(f"   Batches completed: {batch_count}")
        print(f"   Best total reward: {self.best_total_reward:.3f}")
        
        # Finalize detailed logging if enabled
        if self.enable_detailed_logging:
            self.finalize_detailed_logging()
        
        self.save_model("final_model.pt")
        self.save_metrics()
        
        # Generate training visualization plots
        self.generate_training_plots()
    
    def create_run_directory(self, base_dir: str) -> str:
        """Create a new run directory with automatic numbering (run0001, run0002, etc.)"""
        import glob
        import re
        
        # Ensure base directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Find existing run directories
        existing_runs = glob.glob(os.path.join(base_dir, "run[0-9][0-9][0-9][0-9]"))
        
        # Extract run numbers
        run_numbers = []
        for run_path in existing_runs:
            run_name = os.path.basename(run_path)
            match = re.match(r"run(\d{4})", run_name)
            if match:
                run_numbers.append(int(match.group(1)))
        
        # Determine next run number
        if run_numbers:
            next_run_number = max(run_numbers) + 1
        else:
            next_run_number = 1
        
        # Create run directory
        self.run_number = next_run_number
        run_dir = os.path.join(base_dir, f"run{next_run_number:04d}")
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir
    
    def compute_spawn_statistics(self, episode: int) -> dict:
        """Compute statistics for spawn parameters from current episode (compact format)"""
        import statistics
        
        stats = {
            'episode': episode,
            'timestamp': time.time(),
            'parameters': {}
        }
        
        for param_name, values in self.current_episode_spawn_params.items():
            if values:  # Only compute stats if we have values
                try:
                    # Compact format - only essential statistics
                    param_stats = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'sum': sum(values)
                    }
                    
                    stats['parameters'][param_name] = param_stats
                    
                except statistics.StatisticsError:
                    # Handle edge cases
                    stats['parameters'][param_name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'sum': sum(values)
                    }
            else:
                # No spawn actions in this episode
                stats['parameters'][param_name] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'sum': 0.0
                }
                stats['parameters'][param_name] = {
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'median': None,
                    'range': None,
                    'variance': None
                }
        
        return stats
    
    def save_spawn_statistics(self, episode: int) -> None:
        """Compute and save spawn parameter statistics to JSON file (appends episode history)"""
        stats = self.compute_spawn_statistics(episode)
        
        # Save to run directory
        stats_filepath = os.path.join(self.run_dir, 'spawn_parameters_stats.json')
        
        # Enrich stats with additional episode diagnostics
        # Loss summary - use most recent available loss from completed batches
        try:
            # Get most recent loss from completed batches (self.losses is populated after batch completion)
            if self.losses.get('total_loss'):
                recent_total_loss = self.losses['total_loss'][-1]
            # Fallback to using smoothed losses if available
            elif hasattr(self, 'smoothed_losses') and self.smoothed_losses:
                recent_total_loss = self.smoothed_losses[-1]
            else:
                # For episodes before first batch completion, recent_total_loss will be null
                # This will be updated via the post-processing mechanism in save_loss_statistics
                recent_total_loss = None
        except Exception:
            recent_total_loss = None

        # Node statistics for this episode (we track current_spawn_summary['count'] and can sample node counts if available)
        node_stats = {
            'current_nodes': getattr(self.env.topology.graph, 'num_nodes', lambda: None)() if hasattr(self.env, 'topology') else None,
            'init_num_nodes': getattr(self, 'init_num_nodes', None)
        }

        # If we tracked per-step node counts for this episode, compute summary stats
        try:
            counts = list(self.current_episode_node_counts) if hasattr(self, 'current_episode_node_counts') else []
            if counts:
                node_stats['min'] = int(min(counts))
                node_stats['max'] = int(max(counts))
                node_stats['mean'] = float(np.mean(counts))
                node_stats['std'] = float(np.std(counts))
                node_stats['samples'] = len(counts)
            else:
                node_stats['min'] = None
                node_stats['max'] = None
                node_stats['mean'] = None
                node_stats['std'] = None
                node_stats['samples'] = 0
        except Exception:
            # Non-critical: leave node_stats as minimal
            pass

        # Entropy & LR & component weights snapshot
        # Short-term moving averages for monitoring convergence
        try:
            ma_window = int(getattr(self, 'moving_avg_window', 20))
        except Exception:
            ma_window = 20

        def moving_avg(seq, window):
            if not seq:
                return None
            seq = list(seq)
            return float(np.mean(seq[-window:])) if len(seq) >= 1 else None

        extra_metadata = {
            'recent_total_loss': recent_total_loss,
            'ma_total_loss': moving_avg(self.losses.get('total_loss', []), ma_window),
            'ma_total_reward': moving_avg(self.episode_rewards.get('total_reward', []), ma_window),
            'node_stats': node_stats,
            'entropy_coeff': getattr(self, 'current_entropy_coeff', None),
            'learning_rate': float(self.optimizer.param_groups[0]['lr']) if hasattr(self, 'optimizer') else None,
            'component_weights_snapshot': dict(self.component_weights) if hasattr(self, 'component_weights') else None,
            'spawn_summary': dict(self.current_spawn_summary) if hasattr(self, 'current_spawn_summary') else None,
            'success': None  # will be set by training loop when known
        }

        # Load existing data if file exists, otherwise start with empty list
        if os.path.exists(stats_filepath):
            try:
                with open(stats_filepath, 'r') as f:
                    existing_data = json.load(f)
                # Handle both old format (dict) and new format (list)
                if isinstance(existing_data, dict):
                    episode_history = [existing_data]  # Convert old format
                else:
                    episode_history = existing_data
            except (json.JSONDecodeError, FileNotFoundError):
                episode_history = []
        else:
            episode_history = []
        
        # Merge extra metadata and append current episode data
        if isinstance(stats, dict):
            stats.update(extra_metadata)
        episode_history.append(stats)
        
        # Save updated history
        with open(stats_filepath, 'w') as f:
            json.dump(episode_history, f, indent=2)
        
        # Optional: Print summary if there were spawn actions (silent for one-line logging)
        total_spawns = sum(stats['parameters'][param]['count'] for param in stats['parameters'] 
                          if stats['parameters'][param]['count'] > 0)
        
        # Store spawn summary for consolidated logging
        if total_spawns > 0:
            self.current_spawn_summary = {
                'count': total_spawns,
                'gamma': f"{stats['parameters']['gamma']['mean']:.2f}±{stats['parameters']['gamma']['std']:.2f}",
                'alpha': f"{stats['parameters']['alpha']['mean']:.2f}±{stats['parameters']['alpha']['std']:.2f}",
                'noise': f"{stats['parameters']['noise']['mean']:.2f}±{stats['parameters']['noise']['std']:.2f}",
                'theta': f"{stats['parameters']['theta']['mean']:.2f}±{stats['parameters']['theta']['std']:.2f}"
            }
        else:
            self.current_spawn_summary = {'count': 0}
    
    def compute_reward_statistics(self, episode: int) -> dict:
        """Compute statistics for reward components from current episode (compact format)"""
        import statistics
        
        stats = {
            'episode': episode,
            'timestamp': time.time(),
            'reward_components': {}
        }
        
        for component_name, values in self.current_episode_rewards.items():
            if values:  # Only compute stats if we have values
                try:
                    # Compact format - only essential statistics
                    component_stats = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'sum': sum(values)  # Total reward for this component
                    }
                        
                    stats['reward_components'][component_name] = component_stats
                    
                except statistics.StatisticsError:
                    # Handle edge cases
                    stats['reward_components'][component_name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'sum': sum(values)
                    }
            else:
                # No rewards recorded for this component
                stats['reward_components'][component_name] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'sum': 0.0
                }
        
        return stats
    
    def save_loss_statistics(self, episode: int) -> None:
        """Save per-episode and smoothed losses to JSON file for analysis"""
        metrics_path = os.path.join(self.run_dir, 'loss_metrics.json')
        metrics = {
            'episode': episode,
            'loss': self.losses['total_loss'][-1] if self.losses['total_loss'] else None,
            'smoothed_loss': self.smoothed_losses[-1] if self.smoothed_losses else None
        }
        # Append to file
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = []
        else:
            data = []
        data.append(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_reward_statistics(self, episode: int) -> None:
        """Compute and save reward component statistics to JSON file (appends episode history)"""
        stats = self.compute_reward_statistics(episode)
        
        # Save to run directory
        stats_filepath = os.path.join(self.run_dir, 'reward_components_stats.json')
        
        # Enrich stats with additional episode diagnostics
        try:
            # Get most recent loss from completed batches (self.losses is populated after batch completion)
            if self.losses.get('total_loss'):
                recent_total_loss = self.losses['total_loss'][-1]
            # Fallback to using smoothed losses if available
            elif hasattr(self, 'smoothed_losses') and self.smoothed_losses:
                recent_total_loss = self.smoothed_losses[-1]
            else:
                # For episodes before first batch completion, recent_total_loss will be null
                # This will be updated via the post-processing mechanism in save_loss_statistics  
                recent_total_loss = None
        except Exception:
            recent_total_loss = None

        node_stats = {
            'current_nodes': getattr(self.env.topology.graph, 'num_nodes', lambda: None)() if hasattr(self, 'env') and hasattr(self.env, 'topology') else None,
            'init_num_nodes': getattr(self, 'init_num_nodes', None)
        }

        extra_metadata = {
            'recent_total_loss': recent_total_loss,
            'node_stats': node_stats,
            'entropy_coeff': getattr(self, 'current_entropy_coeff', None),
            'learning_rate': float(self.optimizer.param_groups[0]['lr']) if hasattr(self, 'optimizer') else None,
            'component_weights_snapshot': dict(self.component_weights) if hasattr(self, 'component_weights') else None,
            'spawn_summary': dict(self.current_spawn_summary) if hasattr(self, 'current_spawn_summary') else None,
            'success': None
        }

        # Load existing data if file exists, otherwise start with empty list
        if os.path.exists(stats_filepath):
            try:
                with open(stats_filepath, 'r') as f:
                    existing_data = json.load(f)
                # Handle both old format (dict) and new format (list)
                if isinstance(existing_data, dict):
                    episode_history = [existing_data]  # Convert old format
                else:
                    episode_history = existing_data
            except (json.JSONDecodeError, FileNotFoundError):
                episode_history = []
        else:
            episode_history = []
        
        # Merge extra metadata and append current episode data
        if isinstance(stats, dict):
            stats.update(extra_metadata)
        episode_history.append(stats)
        
        # Save updated history
        with open(stats_filepath, 'w') as f:
            json.dump(episode_history, f, indent=2)
        
        # Store reward summary for consolidated logging (silent for one-line logging)
        key_components = ['total_reward', 'graph_reward', 'spawn_reward', 'edge_reward']
        active_components = []
        
        for comp in key_components:
            if comp in stats['reward_components'] and stats['reward_components'][comp]['count'] > 0:
                comp_stats = stats['reward_components'][comp]
                active_components.append({
                    'name': comp,
                    'sum': comp_stats['sum'],
                    'mean': comp_stats['mean'],
                    'std': comp_stats['std']
                })
        
        self.current_reward_summary = active_components
    
    def save_model(self, filename: str):
        """Save model checkpoint to run directory"""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode_rewards': dict(self.episode_rewards),
            'losses': dict(self.losses),
            'best_reward': self.best_total_reward,
            'component_weights': self.component_weights,
            'run_number': self.run_number
        }
        
        filepath = os.path.join(self.run_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Saved: {filename} (Run #{self.run_number})")
    
    def load_model(self, filename: str):
        """Load model checkpoint from file"""
        filepath = os.path.join(self.run_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ Checkpoint file not found: {filename}")
            return False
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network and optimizer states
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available (for backward compatibility)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("⚠️ No scheduler state found in checkpoint (older checkpoint format)")
        
        # Load training progress
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = defaultdict(list, checkpoint['episode_rewards'])
        if 'losses' in checkpoint:
            self.losses = defaultdict(list, checkpoint['losses'])
        if 'best_reward' in checkpoint:
            self.best_total_reward = checkpoint['best_reward']
        if 'component_weights' in checkpoint:
            self.component_weights = checkpoint['component_weights']
            
        print(f"📂 Loaded: {filename} (Run #{checkpoint.get('run_number', 'unknown')})")
        return True
    
    def save_metrics(self):
        """Save training metrics"""
        import json
        
        # Prepare scaling statistics for saving
        scaling_stats = {}
        for component, stats in self.component_running_stats.items():
            scaling_stats[component] = {
                'final_mean': float(stats['mean']),
                'final_std': float(stats.get('std', 1.0)),
                'sample_count': int(stats['count']),
                'recent_rewards': [float(x) for x in list(stats['raw_rewards'])]
            }
        
        metrics = {
            'episode_rewards': {k: [float(x) for x in v] for k, v in self.episode_rewards.items()},
            'losses': {k: [float(x) for x in v] for k, v in self.losses.items()},
            'component_weights': self.component_weights,
            'policy_loss_weights': self.policy_loss_weights,  # Save hybrid policy config
            'best_reward': float(self.best_total_reward),
            'adaptive_scaling_enabled': self.enable_adaptive_scaling,
            'scaling_warmup_episodes': self.scaling_warmup_episodes,
            'component_scaling_stats': scaling_stats,
            'final_scaling_factors': self.get_component_scaling_factors()
        }
        
        with open(os.path.join(self.run_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Saved training metrics to run{self.run_number:04d}")
    
    def generate_training_plots(self):
        """Generate plots for spawn parameters, reward components, and loss evolution after training completion"""
        try:
            print(f"📈 Generating training visualization plots...")
            
            # Call plotter.py with spawn parameters, reward components, and loss evolution
            cmd = [
                sys.executable, 'plotter.py',
                '--input', self.run_dir,
                '--combined', '--rewards', '--loss'
            ]
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                print(f"✅ Training plots generated successfully!")
                print(f"   📁 Plots saved to: {self.run_dir}/")
                print(f"   📊 Generated files:")
                print(f"      • spawn_parameters_evolution_run{self.run_number:04d}.png")
                print(f"      • spawn_parameters_combined_run{self.run_number:04d}.png") 
                print(f"      • reward_components_run{self.run_number:04d}.png")
                print(f"      • loss_evolution_run{self.run_number:04d}.png")
            else:
                print(f"⚠️  Warning: Plot generation failed")
                print(f"   Error: {result.stderr}")
                print(f"   Stdout: {result.stdout}")
                print(f"   You can manually generate plots with:")
                print(f"   python plotter.py --input {self.run_dir} --combined --rewards --loss")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots automatically: {e}")
            print(f"   You can manually generate plots with:")
            print(f"   python plotter.py --input {self.run_dir} --combined --rewards --loss")
    
    def _initialize_substrate(self):
        """Initialize substrate based on substrate_type"""
        # Validate substrate type
        if self.substrate_type not in ['linear', 'exponential', 'random']:
            raise ValueError(f"Invalid substrate_type: {self.substrate_type}. "
                           f"Supported types: ['linear', 'exponential', 'random']")
        
        if self.substrate_type == 'random':
            # For random, we'll update the substrate each episode
            # Start with a default linear substrate
            self.env.substrate.create('linear', m=0.05, b=1.0)
            print(f"   Substrate initialized as random (starting with linear)")
        else:
            # Fixed substrate type
            self.env.substrate.create(self.substrate_type, m=0.05, b=1.0)
            print(f"   Substrate initialized as fixed {self.substrate_type}")
        
        # Initialize tracking for random substrate parameters
        self.current_substrate_params = {
            'kind': self.substrate_type,
            'm': 0.05,
            'b': 1.0
        }
    
    def _update_random_substrate(self):
        """Update substrate with random parameters (only for substrate_type='random')"""
        import random
        
        # Random substrate type selection
        substrate_types = ['linear', 'exponential']
        chosen_type = random.choice(substrate_types)
        
        # Random parameter ranges based on substrate type (now configurable)
        if chosen_type == 'linear':
            # Linear substrate: y = mx + b
            m = random.uniform(*self.linear_m_range)   # Configurable slope range
            b = random.uniform(*self.linear_b_range)   # Configurable intercept range
        else:  # exponential
            # Exponential substrate: y = b * exp(mx)
            m = random.uniform(*self.exponential_m_range)  # Configurable growth rate range
            b = random.uniform(*self.exponential_b_range)  # Configurable base value range
        
        # Update the substrate
        self.env.substrate.create(chosen_type, m=m, b=b)
        
        # Store current parameters for logging
        self.current_substrate_params = {
            'kind': chosen_type,
            'm': m,
            'b': b
        }


def main():
    """Run the training"""
    print("🎯 Multi-Component Durotaxis Training")
    print("=" * 50)
    
    trainer = DurotaxisTrainer(
        config_path="config.yaml",
        # Example of parameter overrides (optional)
        # total_episodes=500,
        # learning_rate=1e-4,
        # substrate_type='linear'
    )
    
    trainer.train()


if __name__ == "__main__":
    main()