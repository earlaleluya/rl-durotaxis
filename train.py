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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


from durotaxis_env import DurotaxisEnv
from state import TopologyState  
from encoder import GraphInputEncoder
from actor_critic import HybridActorCritic


def moving_average(data, window=20):
    """Calculate moving average for smoother trend tracking"""
    if len(data) < window:
        return np.mean(data) if data else 0.0
    return np.mean(data[-window:])


class DurotaxisTrainer:
    """Streamlined trainer for hybrid actor-critic with component rewards"""
    
    def __init__(self, 
                 total_episodes: int = 1000,
                 learning_rate: float = 3e-4,
                 hidden_dim: int = 128,
                 save_dir: str = "./training_results",
                 entropy_bonus_coeff: float = 0.01,
                 weight_momentum: float = 0.9,
                 normalize_weights_every: int = 10,
                 moving_avg_window: int = 20,
                 log_every: int = 50,
                 progress_print_every: int = 5,
                 checkpoint_every: Optional[int] = None,
                 substrate_type: str = 'random',
                 max_episode_length: int = 200,
                 # Environment setup parameters (HIGH PRIORITY)
                 substrate_size: tuple = (200, 200),
                 init_num_nodes: int = 1,
                 max_critical_nodes: int = 50,
                 # Random substrate parameter ranges (HIGH PRIORITY)
                 linear_m_range: tuple = (0.01, 0.1),
                 linear_b_range: tuple = (0.5, 2.0),
                 exponential_m_range: tuple = (0.01, 0.05),
                 exponential_b_range: tuple = (0.5, 1.5)):
        
        self.total_episodes = total_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.moving_avg_window = moving_avg_window  # Window size for moving averages
        self.log_every = log_every                  # Logging frequency
        self.progress_print_every = progress_print_every  # One-line progress frequency
        self.checkpoint_every = checkpoint_every    # Model checkpoint frequency
        self.substrate_type = substrate_type        # Substrate type for training
        self.max_episode_length = max_episode_length  # Maximum episode length
        
        # Environment setup parameters
        self.substrate_size = substrate_size
        self.init_num_nodes = init_num_nodes
        self.max_critical_nodes = max_critical_nodes
        
        # Random substrate parameter ranges
        self.linear_m_range = linear_m_range
        self.linear_b_range = linear_b_range
        self.exponential_m_range = exponential_m_range
        self.exponential_b_range = exponential_b_range
        
        # Create run directory with automatic numbering
        self.run_dir = self.create_run_directory(save_dir)
        print(f"📁 Created run directory: {self.run_dir} (Run #{self.run_number})")
        
        # Environment setup
        self.env = DurotaxisEnv(
            substrate_size=self.substrate_size,
            init_num_nodes=self.init_num_nodes,
            max_critical_nodes=self.max_critical_nodes,
            max_steps=self.max_episode_length,
            # Balanced reward settings for learning
            graph_rewards={
                'connectivity_penalty': 5.0,
                'growth_penalty': 2.0,
                'survival_reward': 0.1,
                'action_reward': 0.02
            },
            spawn_rewards={
                'spawn_success_reward': 1.0,
                'spawn_failure_penalty': 0.5
            }
        )
        
        # Initialize substrate based on type
        self._initialize_substrate()
        
        # Initialize the state extractor with the environment's topology
        self.state_extractor = TopologyState()
        
        # Test environment setup
        test_obs, test_info = self.env.reset()
        self.state_extractor.set_topology(self.env.topology)
        print(f"   Environment initialized: {self.env.topology.graph.num_nodes()} nodes")
        
        # Component configuration - match your environment's components
        self.component_names = [
            'total_reward',
            'graph_reward', 
            'spawn_reward',
            'delete_reward',
            'edge_reward',
            'total_node_reward'
        ]
        
        # Component weights for balanced learning (adaptive)
        self.component_weights = {
            'total_reward': 1.0,      # Main optimization target
            'graph_reward': 0.4,      # Topology structure
            'spawn_reward': 0.3,      # Spawning decisions
            'delete_reward': 0.2,     # Deletion strategy
            'edge_reward': 0.2,       # Movement direction
            'total_node_reward': 0.3  # Node behaviors
        }
        
        # Adaptive weighting parameters
        self.weight_update_momentum = weight_momentum     # Momentum for weight updates
        self.weight_updates_count = 0                     # Track number of updates for normalization
        self.normalize_weights_every = normalize_weights_every  # Normalize weights every N updates
        
        # Hybrid policy loss weights - balance discrete vs continuous actions
        self.policy_loss_weights = {
            'discrete_weight': 0.7,    # Spawn/delete decisions (typically more important)
            'continuous_weight': 0.3,  # Continuous parameters (fine-tuning)
            'entropy_weight': 0.01,    # Exploration bonus
            'clip_epsilon': 0.2        # PPO clipping parameter
        }
        
        # Create network
        self.encoder = GraphInputEncoder(
            hidden_dim=64,
            out_dim=64, 
            num_layers=2
        ).to(self.device)
        
        self.network = HybridActorCritic(
            encoder=self.encoder,
            hidden_dim=hidden_dim,
            value_components=self.component_names,
            dropout_rate=0.1
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training tracking
        self.episode_rewards = defaultdict(list)
        self.losses = defaultdict(list)
        self.best_total_reward = float('-inf')
        
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
    
    def create_action_mask(self, state_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Create action mask to prevent invalid topology operations"""
        num_nodes = state_dict.get('num_nodes', 0)
        if num_nodes == 0:
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
        """Get adaptive scaling factors for reward components"""
        scaling_factors = {}
        
        if not self.enable_adaptive_scaling:
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
                        edge_index = torch.stack([src, dst], dim=0)
                    
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
    
    def collect_episode(self) -> Tuple[List[Dict], List[torch.Tensor], List[Dict], List[Dict], bool, bool]:
        """Collect one episode of experience"""
        # Reset spawn parameter tracking for this episode
        self.current_episode_spawn_params = {
            'gamma': [],
            'alpha': [],
            'noise': [],
            'theta': []
        }
        
        # Reset reward component tracking for this episode
        self.current_episode_rewards = {
            'graph_reward': [],
            'spawn_reward': [],
            'delete_reward': [],
            'edge_reward': [],
            'total_node_reward': [],
            'total_reward': []
        }
        
        # Update substrate if using random type
        if self.substrate_type == 'random':
            self._update_random_substrate()
        
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
        
        while not done and episode_length < self.max_episode_length:
            # Get current state
            state_dict = self.state_extractor.get_state_features(include_substrate=True)
            state_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in state_dict.items()}
            
            if state_dict['num_nodes'] == 0:
                # Handle empty graph case
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
            
            # Environment step
            next_obs, reward_components, terminated, truncated, info = self.env.step(0)
            done = terminated or truncated
            
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
        """Compute returns and advantages for each reward component with proper GAE bootstrapping"""
        if not rewards or not values:
            return {}, {}
        
        # Get adaptive scaling factors
        scaling_factors = self.get_component_scaling_factors()
        
        returns = {}
        advantages = {}
        
        for component in self.component_names:
            # Extract component rewards and values
            raw_component_rewards = torch.tensor([r.get(component, 0.0) for r in rewards], 
                                               dtype=torch.float32, device=self.device)
            
            # Apply adaptive scaling
            scaling_factor = scaling_factors.get(component, 1.0)
            component_rewards = raw_component_rewards * scaling_factor
            
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
                                  advantage: float) -> Dict[str, torch.Tensor]:
        """Compute balanced policy loss for hybrid discrete+continuous actions"""
        policy_losses = {}
        entropy_losses = {}
        
        # Extract weights
        discrete_weight = self.policy_loss_weights['discrete_weight']
        continuous_weight = self.policy_loss_weights['continuous_weight']
        clip_eps = self.policy_loss_weights['clip_epsilon']
        
        # === DISCRETE POLICY LOSS ===
        if ('discrete' in old_log_probs_dict and 
            'discrete_log_probs' in eval_output and 
            len(old_log_probs_dict['discrete']) > 0 and 
            len(eval_output['discrete_log_probs']) > 0):
            
            # Compute ratio for discrete actions
            old_discrete_log_prob = old_log_probs_dict['discrete'].sum()
            new_discrete_log_prob = eval_output['discrete_log_probs'].sum()
            
            ratio_discrete = torch.exp(new_discrete_log_prob - old_discrete_log_prob)
            clipped_ratio_discrete = torch.clamp(ratio_discrete, 1 - clip_eps, 1 + clip_eps)
            
            # PPO discrete policy loss
            discrete_loss = -torch.min(
                ratio_discrete * advantage,
                clipped_ratio_discrete * advantage
            )
            
            policy_losses['discrete'] = discrete_weight * discrete_loss
        else:
            policy_losses['discrete'] = torch.tensor(0.0, device=self.device)
        
        # === CONTINUOUS POLICY LOSS ===
        if ('continuous' in old_log_probs_dict and 
            'continuous_log_probs' in eval_output and 
            len(old_log_probs_dict['continuous']) > 0 and 
            len(eval_output['continuous_log_probs']) > 0):
            
            # Compute ratio for continuous actions
            old_continuous_log_prob = old_log_probs_dict['continuous'].sum()
            new_continuous_log_prob = eval_output['continuous_log_probs'].sum()
            
            ratio_continuous = torch.exp(new_continuous_log_prob - old_continuous_log_prob)
            clipped_ratio_continuous = torch.clamp(ratio_continuous, 1 - clip_eps, 1 + clip_eps)
            
            # PPO continuous policy loss
            continuous_loss = -torch.min(
                ratio_continuous * advantage,
                clipped_ratio_continuous * advantage
            )
            
            policy_losses['continuous'] = continuous_weight * continuous_loss
        else:
            policy_losses['continuous'] = torch.tensor(0.0, device=self.device)
        
        # === ENTROPY LOSSES (Separate for discrete and continuous) ===
        entropy_weight = self.policy_loss_weights['entropy_weight']
        
        if 'entropy' in eval_output:
            # Use provided entropy (combined)
            entropy_losses['total'] = -entropy_weight * eval_output['entropy']
        else:
            # Compute separate entropies if available
            if 'discrete_entropy' in eval_output:
                entropy_losses['discrete'] = -entropy_weight * eval_output['discrete_entropy']
            if 'continuous_entropy' in eval_output:
                entropy_losses['continuous'] = -entropy_weight * eval_output['continuous_entropy']
        
        # Total policy loss
        total_policy_loss = sum(policy_losses.values())
        total_entropy_loss = sum(entropy_losses.values())
        
        return {
            'policy_loss_discrete': policy_losses['discrete'],
            'policy_loss_continuous': policy_losses['continuous'],
            'total_policy_loss': total_policy_loss,
            'entropy_loss': total_entropy_loss,
            'discrete_weight_used': discrete_weight,
            'continuous_weight_used': continuous_weight
        }
    
    def update_policy(self, states: List[Dict], actions: List[Dict], 
                     returns: Dict[str, torch.Tensor], advantages: Dict[str, torch.Tensor],
                     old_log_probs: List[Dict]) -> Dict[str, float]:
        """Update policy using PPO with efficient batched re-evaluation"""
        if not states or not actions:
            return {}
        
        losses = {}
        
        # === ADAPTIVE COMPONENT WEIGHTING ===
        # Update component weights based on advantage magnitude
        self.update_adaptive_component_weights(advantages)
        
        # Combine advantages with adaptive component weighting
        total_advantages = torch.zeros(len(states), device=self.device)
        for component, adv in advantages.items():
            weight = self.component_weights.get(component, 1.0)
            total_advantages += weight * adv
        
        # Normalize advantages
        if len(total_advantages) > 1:
            total_advantages = (total_advantages - total_advantages.mean()) / (total_advantages.std() + 1e-8)
        
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
                old_log_probs_dict, eval_output, advantage
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
            avg_entropy_loss = torch.stack([h['entropy_loss'] for h in hybrid_policy_losses]).mean()
            
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
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return losses
    
    def train(self):
        """Main training loop"""
        # Create run directory for this training session
        self.run_dir = self.create_run_directory(self.save_dir)
        print(f"🏋️ Starting training for {self.total_episodes} episodes (Run #{self.run_number:04d})")
        print(f"📁 Saving to: {self.run_dir}")
        print(f"📊 Progress format: Ep | R: Current (MA: MovingAvg, Best: Best) Trend | Loss | Entropy | Steps | Success | Focus: Component(Weight)")
        
        for episode in range(self.total_episodes):
            # Collect episode
            states, actions, rewards, values, log_probs, final_values, terminated, success = self.collect_episode()
            
            if not rewards:
                continue
            
            # Update component statistics for adaptive scaling
            self.update_component_stats(rewards)
                
            # Compute returns and advantages (with proper GAE bootstrapping)
            returns, advantages = self.compute_returns_and_advantages(
                rewards, values, final_values, terminated
            )
            
            # Update policy
            losses = self.update_policy(states, actions, returns, advantages, log_probs)
            
            # Track episode rewards
            episode_total_reward = sum(r.get('total_reward', 0.0) for r in rewards)
            self.episode_rewards['total_reward'].append(episode_total_reward)
            
            for component in self.component_names:
                component_reward = sum(r.get(component, 0.0) for r in rewards)
                self.episode_rewards[component].append(component_reward)
            
            # Track losses
            for loss_name, loss_value in losses.items():
                self.losses[loss_name].append(loss_value)
            
            # Compute and save spawn parameter statistics
            self.save_spawn_statistics(episode)
            
            # Compute and save reward component statistics
            self.save_reward_statistics(episode)
            
            # One-line comprehensive progress print (includes spawn & reward stats)
            if episode % self.progress_print_every == 0 or episode < 10:
                recent_reward = moving_average(self.episode_rewards['total_reward'], min(10, len(self.episode_rewards['total_reward'])))
                best_so_far = max(self.episode_rewards['total_reward']) if self.episode_rewards['total_reward'] else 0.0
                total_loss = losses.get('total_loss', 0.0)
                entropy_bonus = losses.get('entropy_bonus', 0.0)
                dominant_comp = max(self.component_weights.items(), key=lambda x: x[1])[0]
                dominant_weight = self.component_weights[dominant_comp]
                
                # Show progress with trend indicator
                trend = "↗" if len(self.episode_rewards['total_reward']) > 1 and recent_reward > self.episode_rewards['total_reward'][-2] else "→"
                
                # Build spawn info
                spawn_info = ""
                if hasattr(self, 'current_spawn_summary') and self.current_spawn_summary['count'] > 0:
                    spawn_info = f" | Spawns: {self.current_spawn_summary['count']}(γ:{self.current_spawn_summary['gamma']} α:{self.current_spawn_summary['alpha']})"
                
                # Build reward breakdown info  
                reward_info = ""
                if hasattr(self, 'current_reward_summary') and self.current_reward_summary:
                    total_r = next((r for r in self.current_reward_summary if r['name'] == 'total_reward'), None)
                    graph_r = next((r for r in self.current_reward_summary if r['name'] == 'graph_reward'), None)
                    if total_r and graph_r:
                        reward_info = f" | Rewards: T:{total_r['sum']:.1f} G:{graph_r['sum']:.1f}"
                
                # Build substrate info
                substrate_info = ""
                if self.substrate_type == 'random':
                    params = self.current_substrate_params
                    substrate_info = f" | Sub: {params['kind'][:3]}(m:{params['m']:.3f} b:{params['b']:.2f})"
                elif self.substrate_type in ['linear', 'exponential']:
                    substrate_info = f" | Sub: {self.substrate_type[:3]}"
                
                print(f"Ep {episode:4d} | R: {episode_total_reward:6.3f} (MA: {recent_reward:6.3f}, Best: {best_so_far:6.3f}) {trend} | "
                      f"Loss: {total_loss:7.4f} | Entropy: {entropy_bonus:7.4f} | Steps: {len(rewards):3d} | "
                      f"Success: {'✓' if success else '✗'} | Focus: {dominant_comp[:8]}({dominant_weight:.3f}){spawn_info}{reward_info}{substrate_info}")
            
            # Save best model
            if episode_total_reward > self.best_total_reward:
                self.best_total_reward = episode_total_reward
                self.save_model(f"best_model_ep{episode}.pt")
            
            # Periodic saves (only if checkpoint_every is configured)
            if self.checkpoint_every is not None and episode % self.checkpoint_every == 0 and episode > 0:
                self.save_model(f"checkpoint_ep{episode}.pt")
                print(f"💾 Checkpoint saved at episode {episode}")
        
        print(f"🎉 Training completed!")
        print(f"   Best total reward: {self.best_total_reward:.3f}")
        self.save_model("final_model.pt")
        self.save_metrics()
    
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
        """Compute statistics for spawn parameters from current episode"""
        import statistics
        
        stats = {
            'episode': episode,
            'timestamp': time.time(),
            'parameters': {}
        }
        
        for param_name, values in self.current_episode_spawn_params.items():
            if values:  # Only compute stats if we have values
                try:
                    param_stats = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'median': statistics.median(values),
                        'mode': statistics.mode(values) if len(values) > 0 else values[0],
                        'range': max(values) - min(values),
                        'variance': statistics.variance(values) if len(values) > 1 else 0.0
                    }
                    
                    # Add quartiles if we have enough data
                    if len(values) >= 4:
                        sorted_vals = sorted(values)
                        n = len(sorted_vals)
                        param_stats['q25'] = sorted_vals[n//4]
                        param_stats['q75'] = sorted_vals[3*n//4]
                        param_stats['iqr'] = param_stats['q75'] - param_stats['q25']
                    
                    stats['parameters'][param_name] = param_stats
                    
                except statistics.StatisticsError:
                    # Handle edge cases (e.g., mode with no clear mode)
                    stats['parameters'][param_name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'median': statistics.median(values),
                        'range': max(values) - min(values),
                        'variance': statistics.variance(values) if len(values) > 1 else 0.0
                    }
            else:
                # No spawn actions in this episode
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
        """Compute and save spawn parameter statistics to JSON file (overwrites each episode)"""
        stats = self.compute_spawn_statistics(episode)
        
        # Save to run directory
        stats_filepath = os.path.join(self.run_dir, 'spawn_parameters_stats.json')
        
        with open(stats_filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
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
        """Compute statistics for reward components from current episode"""
        import statistics
        
        stats = {
            'episode': episode,
            'timestamp': time.time(),
            'substrate_params': self.current_substrate_params,
            'reward_components': {}
        }
        
        for component_name, values in self.current_episode_rewards.items():
            if values:  # Only compute stats if we have values
                try:
                    component_stats = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'median': statistics.median(values),
                        'mode': statistics.mode(values) if len(values) > 0 else values[0],
                        'range': max(values) - min(values),
                        'variance': statistics.variance(values) if len(values) > 1 else 0.0,
                        'sum': sum(values)  # Total reward for this component
                    }
                    
                    # Add quartiles if we have enough data
                    if len(values) >= 4:
                        sorted_vals = sorted(values)
                        n = len(sorted_vals)
                        component_stats['q25'] = sorted_vals[n//4]
                        component_stats['q75'] = sorted_vals[3*n//4]
                        component_stats['iqr'] = component_stats['q75'] - component_stats['q25']
                    
                    # Add percentiles for detailed analysis
                    if len(values) >= 10:
                        sorted_vals = sorted(values)
                        n = len(sorted_vals)
                        component_stats['p10'] = sorted_vals[n//10]
                        component_stats['p90'] = sorted_vals[9*n//10]
                        
                    stats['reward_components'][component_name] = component_stats
                    
                except statistics.StatisticsError:
                    # Handle edge cases (e.g., mode with no clear mode)
                    stats['reward_components'][component_name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'median': statistics.median(values),
                        'range': max(values) - min(values),
                        'variance': statistics.variance(values) if len(values) > 1 else 0.0,
                        'sum': sum(values)
                    }
            else:
                # No rewards recorded for this component
                stats['reward_components'][component_name] = {
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'median': None,
                    'range': None,
                    'variance': None,
                    'sum': 0.0
                }
        
        return stats
    
    def save_reward_statistics(self, episode: int) -> None:
        """Compute and save reward component statistics to JSON file (overwrites each episode)"""
        stats = self.compute_reward_statistics(episode)
        
        # Save to run directory
        stats_filepath = os.path.join(self.run_dir, 'reward_components_stats.json')
        
        with open(stats_filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
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
            'episode_rewards': dict(self.episode_rewards),
            'losses': dict(self.losses),
            'best_reward': self.best_total_reward,
            'component_weights': self.component_weights,
            'run_number': self.run_number
        }
        
        filepath = os.path.join(self.run_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Saved: {filename} (Run #{self.run_number})")
    
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
        total_episodes=1000,
        learning_rate=3e-4,
        hidden_dim=128,
        max_episode_length=200,  # Now configurable!
        substrate_size=(400, 300),  # Now configurable!
        init_num_nodes=2,  # Now configurable!
        max_critical_nodes=15,  # Now configurable!
        substrate_type='random',  # Try 'linear', 'exponential', or 'random'
        # Custom random substrate ranges (optional)
        linear_m_range=(0.01, 0.1),
        linear_b_range=(0.5, 2.0),
        exponential_m_range=(0.01, 0.05),
        exponential_b_range=(0.5, 1.5)
    )
    
    trainer.train()


if __name__ == "__main__":
    main()