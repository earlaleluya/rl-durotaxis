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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Add project to path
sys.path.append('/home/arl_eifer/github/rl-durotaxis')

from durotaxis_sim import Durotaxis
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
                 log_every: int = 50):
        
        self.total_episodes = total_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.moving_avg_window = moving_avg_window  # Window size for moving averages
        self.log_every = log_every                  # Logging frequency
        os.makedirs(save_dir, exist_ok=True)
        
        # Environment setup
        self.env = Durotaxis(
            substrate_size=(400, 300),
            init_num_nodes=2,
            max_critical_nodes=15,
            max_steps=200,
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
    
    def save_training_statistics(self, filepath: str = None) -> str:
        """Save comprehensive training statistics to file"""
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"training_stats_episode_{len(self.episode_rewards['total_reward'])}.json")
        
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
    
    def collect_episode(self) -> Tuple[List[Dict], List[torch.Tensor], List[Dict], List[Dict], bool, bool]:
        """Collect one episode of experience"""
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
        
        while not done and episode_length < 200:
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
        print(f"🏋️ Starting training for {self.total_episodes} episodes")
        
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
            
            # Logging
            if episode % self.log_every == 0:
                # Use moving averages for smoother trend tracking
                recent_reward = moving_average(self.episode_rewards['total_reward'], self.moving_avg_window)
                
                print(f"\n📊 Episode {episode}")
                print(f"   Total reward (moving avg): {recent_reward:.3f}")
                print(f"   Episode length: {len(rewards)}")
                print(f"   Success: {success}")
                
                # Show component reward trends with moving averages
                print(f"   📈 Component Rewards (moving avg, window={self.moving_avg_window}):")
                for comp in self.component_names:
                    comp_avg = moving_average(self.episode_rewards[comp], self.moving_avg_window)
                    print(f"      {comp}: {comp_avg:.3f}")
                
                # Show adaptive component weights
                print(f"   🎯 Adaptive Component Weights:")
                for comp, weight in self.component_weights.items():
                    print(f"      {comp}: {weight:.4f}")
                
                # Show loss trends with moving averages
                print(f"   📉 Loss Trends (moving avg, window={self.moving_avg_window}):")
                for loss_name in ['total_loss', 'policy_loss_discrete', 'policy_loss_continuous', 'entropy_bonus']:
                    if loss_name in self.losses and self.losses[loss_name]:
                        loss_avg = moving_average(self.losses[loss_name], self.moving_avg_window)
                        print(f"      {loss_name}: {loss_avg:.6f}")
                
                # Show recent weight normalization if it happened
                if self.weight_updates_count % self.normalize_weights_every == 0:
                    print(f"   🔄 Weights normalized (update #{self.weight_updates_count})")
                
                # Component breakdown
                print("   Component rewards (recent avg):")
                for component in ['graph_reward', 'spawn_reward', 'edge_reward', 'total_node_reward']:
                    if self.episode_rewards[component]:
                        avg_reward = np.mean(self.episode_rewards[component][-10:])
                        print(f"     {component}: {avg_reward:.3f}")
                
                # Show adaptive scaling information
                if self.enable_adaptive_scaling and episode >= self.scaling_warmup_episodes:
                    scaling_factors = self.get_component_scaling_factors()
                    print("   🎯 Adaptive scaling factors:")
                    for component in ['graph_reward', 'spawn_reward', 'edge_reward', 'total_node_reward']:
                        if component in scaling_factors:
                            factor = scaling_factors[component]
                            stats = self.component_running_stats[component]
                            recent_std = np.std(list(stats['raw_rewards'])) if len(stats['raw_rewards']) >= 10 else 0
                            print(f"     {component}: {factor:.2f}x (std: {recent_std:.3f})")
                
                if losses:
                    print(f"   📉 Losses:")
                    if 'policy_loss_discrete' in losses:
                        print(f"     Policy (discrete): {losses['policy_loss_discrete']:.4f} (weight: {losses.get('discrete_weight', 0.7):.1f})")
                    if 'policy_loss_continuous' in losses:
                        print(f"     Policy (continuous): {losses['policy_loss_continuous']:.4f} (weight: {losses.get('continuous_weight', 0.3):.1f})")
                    if 'total_policy_loss' in losses:
                        print(f"     Policy (total): {losses['total_policy_loss']:.4f}")
                    if 'total_value_loss' in losses:
                        print(f"     Value (total): {losses['total_value_loss']:.4f}")
                    if 'entropy_loss' in losses:
                        print(f"     Entropy: {losses['entropy_loss']:.4f}")
            
            # Save best model
            if episode_total_reward > self.best_total_reward:
                self.best_total_reward = episode_total_reward
                self.save_model(f"best_model_ep{episode}.pt")
            
            # Periodic saves
            if episode % 200 == 0 and episode > 0:
                self.save_model(f"checkpoint_ep{episode}.pt")
        
        print(f"🎉 Training completed!")
        print(f"   Best total reward: {self.best_total_reward:.3f}")
        self.save_model("final_model.pt")
        self.save_metrics()
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': dict(self.episode_rewards),
            'losses': dict(self.losses),
            'best_reward': self.best_total_reward,
            'component_weights': self.component_weights
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Saved: {filepath}")
    
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
        
        with open(os.path.join(self.save_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Saved training metrics")


def main():
    """Run the training"""
    print("🎯 Multi-Component Durotaxis Training")
    print("=" * 50)
    
    trainer = DurotaxisTrainer(
        total_episodes=1000,
        learning_rate=3e-4,
        hidden_dim=128
    )
    
    trainer.train()


if __name__ == "__main__":
    main()