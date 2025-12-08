#!/usr/bin/env python3
"""
Deployment script for trained Durotaxis RL agent (Delete Ratio Architecture)

This script allows you to load a trained delete ratio model and run it on custom 
substrates with configurable parameters for evaluation and demonstration purposes.

Delete Ratio Architecture:
- Single global continuous action: [delete_ratio] (1D action space)
- Spawn parameters (gamma, alpha, noise) are fixed from config.yaml
- Sorts nodes by x-position, deletes leftmost fraction
- Applies global spawn parameters to remaining nodes with directional Hill function

Usage:
    python deploy.py --substrate_type linear --m 0.05 --b 1.0 \
                     --substrate_width 600 --substrate_height 600 \
                     --deterministic --max_episodes 3 --max_steps 1000 \
                     --max_critical_nodes 40 --threshold_critical_nodes 400 \
                     --model_path ./training_results/run0047/succ_model_batch19.pt \
                     --init-nodes 10 
    
    # Without visualization, custom substrate size
    python deploy.py --model_path ./training_results/run0018/best_model_batch2.pt \
                     --substrate_type linear --m 0.05 --b 1.0 \
                     --substrate_width 400 --substrate_height 300 \
                     --deterministic --max_episodes 10 --max_steps 1000 --no_viz \
                     --max_critical_nodes 100 --threshold_critical_nodes 600
"""

import argparse
import os
import sys
import torch
from device import cpu_numpy
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time

# Import project modules
from durotaxis_env import DurotaxisEnv
from state import TopologyState
from encoder import GraphInputEncoder
from actor_critic import HybridActorCritic
from config_loader import ConfigLoader


class DurotaxisDeployment:
    """
    Deployment class for running trained Delete Ratio Durotaxis agents.
    
    Loads a trained model and executes the delete ratio strategy:
    - Sorts nodes by x-position
    - Deletes leftmost fraction based on delete_ratio
    - Spawns from remaining nodes with global parameters
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config.yaml",
                 device: Optional[str] = None):
        """
        Initialize deployment with trained model.
        
        Args:
            model_path: Path to saved model (.pt file)
            config_path: Path to configuration file
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        from device import get_device
        
        self.model_path = model_path
        self.config_path = config_path
        
        # Set device using centralized logic
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing Durotaxis Deployment")
        print(f"   Model: {model_path}")
        print(f"   Device: {self.device}")
        
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        # Component names for value prediction
        self.component_names = [
            'total_reward',
            'graph_reward', 
            'spawn_reward',
            'delete_reward',
            'edge_reward',
            'total_node_reward'
        ]
        
        # Initialize network architecture
        self._initialize_network()
        
        # Load trained model
        self._load_model()
        
        # Initialize state extractor
        self.state_extractor = TopologyState()
        
        print(f"✅ Deployment ready!")
    
    def _initialize_network(self):
        """Initialize network architecture matching training configuration"""
        # Get configuration
        encoder_config = self.config_loader.get_encoder_config()
        actor_critic_config = self.config_loader.get_actor_critic_config()
        
        # Create encoder
        self.encoder = GraphInputEncoder(
            hidden_dim=encoder_config.get('hidden_dim', 128),
            out_dim=encoder_config.get('out_dim', 64),
            num_layers=encoder_config.get('num_layers', 4)
        ).to(self.device)
        
        # Create actor-critic network with proper constructor signature
        self.network = HybridActorCritic(
            encoder=self.encoder,
            config_path=self.config_path,
            hidden_dim=actor_critic_config.get('hidden_dim', 128),
            value_components=actor_critic_config.get('value_components', self.component_names),
            dropout_rate=actor_critic_config.get('dropout_rate', 0.1)
        ).to(self.device)
        
        print(f"   Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
    
    def _load_model(self):
        """Load trained model weights"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model state dict
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Format: {'model_state_dict': ..., 'episode': ..., 'best_reward': ...}
            self.network.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded model from epoch/episode: {checkpoint.get('episode', 'unknown')}")
            if 'best_reward' in checkpoint:
                print(f"   Best reward achieved: {checkpoint['best_reward']:.3f}")
        elif 'network_state_dict' in checkpoint:
            # Format: {'network_state_dict': ..., 'optimizer_state_dict': ..., ...}
            self.network.load_state_dict(checkpoint['network_state_dict'])
            print(f"   Loaded network from checkpoint")
            if 'best_reward' in checkpoint:
                print(f"   Best reward achieved: {checkpoint['best_reward']:.3f}")
        else:
            # Direct state dict format
            self.network.load_state_dict(checkpoint)
            print(f"   Loaded model state dict")
        
        # Set to evaluation mode
        self.network.eval()
    
    def create_action_mask(self, state_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Create action mask (unused in delete ratio architecture, kept for API compatibility).
        
        Note: Delete ratio architecture uses global actions, not per-node masks.
        """
        # Action masking not used in delete ratio architecture
        return None
    
    def run_episode(self, 
                   env: DurotaxisEnv,
                   max_steps: int = 100,
                   deterministic: bool = True,
                   verbose: bool = True,
                   enable_visualization: bool = True,
                   episode_num: int = None) -> Dict:
        """
        Run a single episode with the trained agent
        
        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic policy
            verbose: Whether to print step-by-step information
            enable_visualization: Whether to show topology visualization
            episode_num: Episode number for visualization labeling
            
        Returns:
            Dictionary with episode statistics
        """
        # Reset environment
        obs, info = env.reset()
        self.state_extractor.set_topology(env.topology)
        
        # Episode tracking
        episode_rewards = []
        episode_actions = []
        episode_values = []
        centroid_data = []  # Track centroid positions and substrate intensity
        step_count = 0
        done = False
        
        if verbose:
            print(f"\n🎯 Starting episode (max_steps={max_steps})")
            print(f"   Initial nodes (step 0): {env.topology.graph.num_nodes()}")
        
        initial_node_count = env.topology.graph.num_nodes()  # Track initial count
        
        # Log initial state at step 0 (before any actions)
        if initial_node_count > 0:
            state_features = self.state_extractor.get_state_features(
                include_substrate=True,
                node_age=env._node_age,
                node_stagnation=env._node_stagnation
            )
            graph_features = state_features.get('graph_features', None)
            
            if graph_features is not None and len(graph_features) >= 11:
                num_nodes_feature = int(graph_features[0])
                cx = float(graph_features[3])
                cy = float(graph_features[4])
                bbox_width = float(graph_features[9])
                bbox_height = float(graph_features[10])
                intensity = env.substrate.get_intensity((cx, cy))
                
                # Record each node at initial state
                node_positions = env.topology.graph.ndata['pos'].cpu().numpy()
                persistent_ids = env.topology.graph.ndata['persistent_id'].cpu().numpy() if 'persistent_id' in env.topology.graph.ndata else list(range(len(node_positions)))
                
                for i in range(len(node_positions)):
                    node_x, node_y = float(node_positions[i][0]), float(node_positions[i][1])
                    centroid_data.append({
                        'episode': episode_num if episode_num is not None else 0,
                        'step': 0,
                        'num_nodes': num_nodes_feature,
                        'cx': cx,
                        'cy': cy,
                        'intensity': float(intensity),
                        'bbox_height': bbox_height,
                        'bbox_width': bbox_width,
                        'delete_ratio': 0.0,
                        'node_id': int(persistent_ids[i]),
                        'node_x': node_x,
                        'node_y': node_y,
                        'action': '',  # No action at initial state
                        'spawn_node_id': '',
                        'spawn_node_x': '',
                        'spawn_node_y': ''
                    })
        
        while not done and step_count < max_steps:
            # Get current state with age/stagnation tracking from environment
            state_dict = self.state_extractor.get_state_features(
                include_substrate=True,
                node_age=env._node_age,
                node_stagnation=env._node_stagnation
            )
            
            # Move tensors to device (handle edge_index tuple specially)
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.to(self.device)
                elif k == 'edge_index' and isinstance(v, tuple):
                    # edge_index is a tuple of (src, dst) tensors
                    state_dict[k] = tuple(t.to(self.device) for t in v)
            
            # Check for empty graph
            if state_dict['num_nodes'] == 0:
                if verbose:
                    print(f"⚠️  Empty graph at step {step_count}, ending episode")
                break
            
            # Create action mask
            action_mask = self.create_action_mask(state_dict)
            
            # Forward pass through network
            with torch.no_grad():
                output = self.network(state_dict, deterministic=deterministic, action_mask=action_mask)
            
            # Store predictions
            episode_values.append(output['value_predictions'])
            
            # Initialize action parameters with default values (will be overwritten if actions exist)
            delete_ratio = 0.0
            gamma = 0.0
            alpha = 0.0
            noise_param = 0.0
            topology_actions = {}      # Maps node_id -> action_type ('spawn' or 'delete')
            node_pid_to_action = {}    # Maps persistent_id -> action_type (for logging after deletions)
            nodes_before_action = {}   # Maps persistent_id -> (x, y) before actions (for deleted nodes)
            spawn_tracking = {}        # Maps parent_persistent_id -> (spawned_node_id, spawn_x, spawn_y)
            
            # Extract and execute delete ratio actions
            if 'continuous_actions' in output:
                continuous_actions = output['continuous_actions']
                
                episode_actions.append({
                    'continuous': cpu_numpy(continuous_actions),
                    'num_nodes': state_dict['num_nodes']
                })
                
                # Get node positions for sorting (delete ratio strategy)
                node_features = state_dict['node_features']
                node_positions = [(i, node_features[i][0].item()) for i in range(state_dict['num_nodes'])]
                node_positions.sort(key=lambda x: x[1])  # Sort by x-position (ascending)
                
                # Execute topology actions using delete ratio strategy
                topology_actions = self.network.get_topology_actions(output, node_positions)
                
                # Get single global spawn parameters
                spawn_params = self.network.get_spawn_parameters(output)
                
                # Extract action parameters for logging
                delete_ratio = float(continuous_actions[0].item()) if len(continuous_actions) > 0 else 0.0
                gamma = float(spawn_params[0])
                alpha = float(spawn_params[1])
                noise_param = float(spawn_params[2])
                
                # Capture ALL node states BEFORE any modifications (for logging deleted nodes)
                # Build mapping: persistent_id -> (action_type, node_x, node_y)
                node_pid_to_action = {}
                nodes_before_action = {}  # Maps persistent_id -> (x, y) for nodes before actions
                
                for node_id, action_type in topology_actions.items():
                    if node_id < env.topology.graph.num_nodes():
                        node_pid = int(env.topology.graph.ndata['persistent_id'][node_id].item()) if 'persistent_id' in env.topology.graph.ndata else node_id
                        node_pos = env.topology.graph.ndata['pos'][node_id].cpu().numpy()
                        node_pid_to_action[node_pid] = action_type
                        nodes_before_action[node_pid] = (float(node_pos[0]), float(node_pos[1]))
                
                # Track spawn actions for logging: maps parent_persistent_id -> (spawned_node_id, spawn_x, spawn_y)
                spawn_tracking = {}
                
                for node_id, action_type in topology_actions.items():
                    try:
                        # Get persistent_id before any modifications
                        parent_pid = int(env.topology.graph.ndata['persistent_id'][node_id].item()) if 'persistent_id' in env.topology.graph.ndata else node_id
                        
                        if action_type == 'spawn':
                            spawned_node_id = env.topology.spawn(node_id, gamma=spawn_params[0], alpha=spawn_params[1], 
                                             noise=spawn_params[2])
                            if spawned_node_id is not None:
                                # Get the spawned node's position and persistent_id
                                spawn_pos = env.topology.graph.ndata['pos'][spawned_node_id].cpu().numpy()
                                spawn_pid = int(env.topology.graph.ndata['persistent_id'][spawned_node_id].item()) if 'persistent_id' in env.topology.graph.ndata else spawned_node_id
                                spawn_tracking[parent_pid] = (spawn_pid, float(spawn_pos[0]), float(spawn_pos[1]))
                            if verbose:
                                print(f"   Step {step_count+1}: Spawned from node {node_id} (γ={spawn_params[0]:.3f}, α={spawn_params[1]:.3f})")
                        elif action_type == 'delete':
                            env.topology.delete(node_id)
                            if verbose:
                                print(f"   Step {step_count+1}: Deleted node {node_id}")
                    except Exception as e:
                        if verbose:
                            print(f"   Step {step_count+1}: Action failed: {e}")
                        continue
            
            # Environment step
            next_obs, reward_components, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            
            # Add node count to reward components for tracking
            num_nodes = env.topology.graph.num_nodes()
            num_edges = env.topology.graph.num_edges()
            reward_components['num_nodes'] = num_nodes
            reward_components['num_edges'] = num_edges
            
            # Store rewards
            episode_rewards.append(reward_components)
            
            # Track centroid position and substrate intensity (after step execution)
            if num_nodes > 0:
                # Get graph features which contain centroid coordinates
                state_features = self.state_extractor.get_state_features(
                    include_substrate=True,
                    node_age=env._node_age,
                    node_stagnation=env._node_stagnation
                )
                graph_features = state_features.get('graph_features', None)
                
                if graph_features is not None and len(graph_features) >= 11:
                    # graph_features structure: [num_nodes, num_edges, avg_degree, 
                    #                            centroid_x, centroid_y, bbox_min_x, bbox_min_y,
                    #                            bbox_max_x, bbox_max_y, bbox_size_x, bbox_size_y, ...]
                    num_nodes_feature = int(graph_features[0])  # num_nodes from graph features
                    cx = float(graph_features[3])
                    cy = float(graph_features[4])
                    
                    # Extract bounding box size (width and height)
                    bbox_width = float(graph_features[9])   # bbox_size_x
                    bbox_height = float(graph_features[10])  # bbox_size_y
                    
                    # Get substrate intensity at centroid position
                    intensity = env.substrate.get_intensity((cx, cy))
                    
                    # Record each node with its action
                    node_positions = env.topology.graph.ndata['pos'].cpu().numpy()
                    persistent_ids = env.topology.graph.ndata['persistent_id'].cpu().numpy() if 'persistent_id' in env.topology.graph.ndata else list(range(len(node_positions)))
                    
                    # First, log all nodes that still exist (survived or were spawned)
                    for i in range(len(node_positions)):
                        node_x, node_y = float(node_positions[i][0]), float(node_positions[i][1])
                        node_pid = int(persistent_ids[i])
                        
                        # Determine action for this node using persistent_id
                        action_str = ''
                        spawn_node_id_val = ''
                        spawn_node_x_val = ''
                        spawn_node_y_val = ''
                        
                        if node_pid in node_pid_to_action:
                            action_type = node_pid_to_action[node_pid]
                            if action_type == 'spawn':
                                action_str = 'spawn'
                                # Check if spawn was tracked
                                if node_pid in spawn_tracking:
                                    spawn_pid, spawn_x, spawn_y = spawn_tracking[node_pid]
                                    spawn_node_id_val = int(spawn_pid)
                                    spawn_node_x_val = spawn_x
                                    spawn_node_y_val = spawn_y
                            # Note: 'delete' won't appear here since deleted nodes are gone
                        
                        centroid_data.append({
                            'episode': episode_num if episode_num is not None else 0,
                            'step': step_count + 1,
                            'num_nodes': num_nodes_feature,
                            'cx': cx,
                            'cy': cy,
                            'intensity': float(intensity),
                            'bbox_height': bbox_height,
                            'bbox_width': bbox_width,
                            'delete_ratio': delete_ratio,
                            'node_id': node_pid,
                            'node_x': node_x,
                            'node_y': node_y,
                            'action': action_str,
                            'spawn_node_id': spawn_node_id_val,
                            'spawn_node_x': spawn_node_x_val,
                            'spawn_node_y': spawn_node_y_val
                        })
                    
                    # Second, log deleted nodes using the saved positions from before deletion
                    for node_pid, action_type in node_pid_to_action.items():
                        if action_type == 'delete':
                            # Get the position from before deletion
                            if node_pid in nodes_before_action:
                                node_x, node_y = nodes_before_action[node_pid]
                                
                                centroid_data.append({
                                    'episode': episode_num if episode_num is not None else 0,
                                    'step': step_count + 1,
                                    'num_nodes': num_nodes_feature,
                                    'cx': cx,
                                    'cy': cy,
                                    'intensity': float(intensity),
                                    'bbox_height': bbox_height,
                                    'bbox_width': bbox_width,
                                    'delete_ratio': delete_ratio,
                                    'node_id': node_pid,
                                    'node_x': node_x,
                                    'node_y': node_y,
                                    'action': 'delete',
                                    'spawn_node_id': '',
                                    'spawn_node_x': '',
                                    'spawn_node_y': ''
                                })
            
            if verbose:
                total_reward = reward_components.get('total_reward', 0.0)
                nodes_delta = num_nodes - (initial_node_count if step_count == 0 else episode_rewards[-1]['num_nodes'])
                print(f"   Step {step_count+1}: N={num_nodes:3d} ({nodes_delta:+d}) E={num_edges:3d} | R={total_reward:+7.3f} | Done={done}")
            
            # Show topology visualization if enabled (independent of verbose)
            if enable_visualization:
                try:
                    if verbose:
                        print(f"   📊 Topology visualization:")
                    # Set step counter on topology for proper visualization labeling
                    env.topology._step_counter = step_count + 1
                    env.topology.show(episode_num=episode_num, highlight_outmost=True)
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Visualization failed: {e}")
            
            step_count += 1
        
        # Compute episode statistics
        total_reward = sum(r.get('total_reward', 0.0) for r in episode_rewards)
        component_rewards = {}
        for component in self.component_names:
            component_rewards[component] = sum(r.get(component, 0.0) for r in episode_rewards)
        
        stats = {
            'total_reward': total_reward,
            'component_rewards': component_rewards,
            'episode_length': step_count,
            'final_nodes': env.topology.graph.num_nodes(),
            'final_edges': env.topology.graph.num_edges(),
            'terminated': done,
            'rewards_per_step': episode_rewards,
            'actions_per_step': episode_actions,
            'values_per_step': episode_values,
            'centroid_data': centroid_data
        }
        
        if verbose:
            print(f"📊 Episode completed:")
            print(f"   Total reward: {total_reward:.3f}")
            print(f"   Episode length: {step_count} steps")
            print(f"   Final topology: {stats['final_nodes']} nodes, {stats['final_edges']} edges")
        
        return stats
    
    def run_evaluation(self,
                      substrate_type: str = "linear",
                      substrate_size: Tuple[int, int] = (200, 200),
                      m: float = 0.05,
                      b: float = 1.0,
                      max_episodes: int = 10,
                      max_steps: int = 100,
                      deterministic: bool = True,
                      save_results: bool = True,
                      enable_visualization: bool = True,
                      max_critical_nodes: int = 75,
                      threshold_critical_nodes: int = 500,
                      init_num_nodes: int = None) -> Dict:
        """
        Run evaluation over multiple episodes
        
        Args:
            substrate_type: Type of substrate ('linear', 'exponential', 'random')
            substrate_size: Tuple of (width, height) for substrate dimensions
            m: Substrate parameter m
            b: Substrate parameter b
            max_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic policy
            save_results: Whether to save results to file
            enable_visualization: Whether to enable visualization during episodes
            max_critical_nodes: Maximum allowed nodes before growth penalties
            threshold_critical_nodes: Critical threshold for episode termination
            init_num_nodes: Initial number of nodes (default from config.yaml)
            
        Returns:
            Dictionary with evaluation statistics
        """
        verbose_eval = not enable_visualization  # Verbose evaluation output when visualization is disabled
        
        if verbose_eval:
            print(f"\n🧪 Running evaluation:")
            print(f"   Substrate: {substrate_type} (size={substrate_size}, m={m}, b={b})")
            print(f"   Episodes: {max_episodes}, Max steps: {max_steps}")
            print(f"   Policy: {'Deterministic' if deterministic else 'Stochastic'}")
            print(f"   Node limits: max_critical={max_critical_nodes}, threshold={threshold_critical_nodes}")
            if init_num_nodes is not None:
                print(f"   Initial nodes: {init_num_nodes}")
            print(f"   Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
        
        # Create environment with specified substrate
        env_kwargs = {
            'config_path': self.config_path,
            'substrate_type': substrate_type,
            'substrate_size': substrate_size,
            'substrate_params': {'m': m, 'b': b},
            'max_critical_nodes': max_critical_nodes,
            'threshold_critical_nodes': threshold_critical_nodes
        }
        if init_num_nodes is not None:
            env_kwargs['init_num_nodes'] = init_num_nodes
        
        env = DurotaxisEnv(**env_kwargs)
        
        # Run episodes
        episode_stats = []
        total_rewards = []
        episode_lengths = []
        
        for episode in range(max_episodes):
            if verbose_eval:
                print(f"\n--- Episode {episode + 1}/{max_episodes} ---")
            
            stats = self.run_episode(
                env=env,
                max_steps=max_steps,
                deterministic=deterministic,
                verbose=not enable_visualization,  # Verbose when visualization is disabled
                enable_visualization=enable_visualization,
                episode_num=episode + 1  # Pass episode number for visualization
            )
            
            episode_stats.append(stats)
            total_rewards.append(stats['total_reward'])
            episode_lengths.append(stats['episode_length'])
            
            # Brief summary for each episode
            if verbose_eval:
                print(f"Episode {episode + 1}: R={stats['total_reward']:7.3f} | Steps={stats['episode_length']:3d} | Nodes={stats['final_nodes']:3d}")
        
        # Compute overall statistics
        evaluation_results = {
            'substrate_config': {
                'type': substrate_type,
                'size': substrate_size,
                'm': m,
                'b': b
            },
            'evaluation_config': {
                'max_episodes': max_episodes,
                'max_steps': max_steps,
                'deterministic': deterministic,
                'enable_visualization': enable_visualization
            },
            'overall_stats': {
                'mean_reward': np.mean(total_rewards),
                'std_reward': np.std(total_rewards),
                'min_reward': np.min(total_rewards),
                'max_reward': np.max(total_rewards),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths),
                'success_rate': np.mean([s['episode_length'] >= 10 for s in episode_stats])  # Episodes with >=10 steps
            },
            'episode_details': episode_stats
        }
        
        # Print summary
        if verbose_eval:
            print(f"\n📊 Evaluation Summary:")
            print(f"   Mean reward: {evaluation_results['overall_stats']['mean_reward']:.3f} ± {evaluation_results['overall_stats']['std_reward']:.3f}")
            print(f"   Reward range: [{evaluation_results['overall_stats']['min_reward']:.3f}, {evaluation_results['overall_stats']['max_reward']:.3f}]")
            print(f"   Mean episode length: {evaluation_results['overall_stats']['mean_length']:.1f} ± {evaluation_results['overall_stats']['std_length']:.1f}")
            print(f"   Success rate (≥10 steps): {evaluation_results['overall_stats']['success_rate']:.1%}")
        
        # Save results
        if save_results:
            self._save_comprehensive_results(
                evaluation_results, 
                episode_stats, 
                substrate_type, 
                m, 
                b, 
                verbose_eval
            )
            
            # Save centroid trajectory CSV
            self._save_centroid_csv(episode_stats, substrate_type, m, b, verbose_eval)
        
        return evaluation_results
    
    def _save_comprehensive_results(self, 
                                    evaluation_results: Dict,
                                    episode_stats: List[Dict],
                                    substrate_type: str,
                                    m: float,
                                    b: float,
                                    verbose: bool = True):
        """
        Save comprehensive evaluation results in multiple JSON files for analysis.
        
        Creates the following files:
        - evaluation_[config]_[timestamp].json: Overall evaluation summary
        - detailed_nodes_all_episodes.json: Node count evolution per episode
        - spawn_parameters_stats.json: Spawn parameter statistics per episode
        - reward_components_stats.json: Reward component statistics per episode
        - training_metrics.json: Episode-level metrics (rewards, lengths)
        - loss_metrics.json: Placeholder for consistency with training format
        
        Args:
            evaluation_results: Overall evaluation results dictionary
            episode_stats: List of per-episode statistics
            substrate_type: Type of substrate used
            m: Substrate parameter m
            b: Substrate parameter b
            verbose: Whether to print save confirmations
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(self.model_path)
        
        # Helper function to convert numpy/torch types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return cpu_numpy(obj).tolist()
            return obj
        
        # 1. Main evaluation results file
        results_filename = f"evaluation_{substrate_type}_m{m}_b{b}_{timestamp}.json"
        results_path = os.path.join(output_dir, results_filename)
        serializable_results = json.loads(json.dumps(evaluation_results, default=convert_numpy))
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if verbose:
            print(f"\n💾 Saving comprehensive evaluation results:")
            print(f"   Main results: {results_filename}")
        
        # 2. Detailed nodes evolution (all episodes)
        detailed_nodes = []
        for ep_idx, ep_stats in enumerate(episode_stats):
            episode_entry = {
                'episode': ep_idx,
                'nodes_per_step': []
            }
            
            # Extract node counts from rewards_per_step if available
            for step_idx, reward_dict in enumerate(ep_stats.get('rewards_per_step', [])):
                # Node count should be tracked; if not available, use final count
                episode_entry['nodes_per_step'].append({
                    'step': step_idx,
                    'nodes': reward_dict.get('num_nodes', ep_stats.get('final_nodes', 0))
                })
            
            detailed_nodes.append(episode_entry)
        
        detailed_nodes_path = os.path.join(output_dir, f"detailed_nodes_all_episodes_{timestamp}.json")
        with open(detailed_nodes_path, 'w') as f:
            json.dump(detailed_nodes, f, indent=2)
        
        if verbose:
            print(f"   Node evolution: detailed_nodes_all_episodes_{timestamp}.json")
        
        # 3. Spawn parameters statistics per episode
        spawn_params_stats = []
        
        # Get fixed spawn parameters from config (same for all episodes)
        env_config = self.config_loader.get_environment_config()
        spawn_params_config = env_config.get('spawn_parameters', {})
        fixed_gamma = float(spawn_params_config.get('gamma', 5.0))
        fixed_alpha = float(spawn_params_config.get('alpha', 2.0))
        fixed_noise = float(spawn_params_config.get('noise', 0.5))
        
        for ep_idx, ep_stats in enumerate(episode_stats):
            actions_per_step = ep_stats.get('actions_per_step', [])
            
            if actions_per_step:
                # Spawn parameters are fixed from config, not varying per episode
                # Record the fixed values used throughout the episode
                episode_entry = {
                    'episode': ep_idx,
                    'parameters': {
                        'gamma': {
                            'value': fixed_gamma,
                            'fixed': True,
                            'note': 'Fixed from config.yaml'
                        },
                        'alpha': {
                            'value': fixed_alpha,
                            'fixed': True,
                            'note': 'Fixed from config.yaml'
                        },
                        'noise': {
                            'value': fixed_noise,
                            'fixed': True,
                            'note': 'Fixed from config.yaml'
                        }
                    }
                }
                spawn_params_stats.append(episode_entry)
        
        spawn_params_path = os.path.join(output_dir, f"spawn_parameters_stats_{timestamp}.json")
        with open(spawn_params_path, 'w') as f:
            json.dump(spawn_params_stats, f, indent=2)
        
        if verbose:
            print(f"   Spawn parameters: spawn_parameters_stats_{timestamp}.json")
        
        # 4. Reward components statistics per episode
        reward_components_stats = []
        for ep_idx, ep_stats in enumerate(episode_stats):
            rewards_per_step = ep_stats.get('rewards_per_step', [])
            
            # Aggregate rewards across steps
            component_aggregates = {}
            for component in self.component_names:
                values = [r.get(component, 0.0) for r in rewards_per_step]
                component_aggregates[component] = {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'min': float(np.min(values)) if values else 0.0,
                    'max': float(np.max(values)) if values else 0.0,
                    'sum': float(np.sum(values)) if values else 0.0
                }
            
            episode_entry = {
                'episode': ep_idx,
                'reward_components': component_aggregates
            }
            reward_components_stats.append(episode_entry)
        
        reward_components_path = os.path.join(output_dir, f"reward_components_stats_{timestamp}.json")
        with open(reward_components_path, 'w') as f:
            json.dump(reward_components_stats, f, indent=2)
        
        if verbose:
            print(f"   Reward components: reward_components_stats_{timestamp}.json")
        
        # 5. Training metrics (episode-level summary)
        training_metrics = []
        for ep_idx, ep_stats in enumerate(episode_stats):
            episode_entry = {
                'episode': ep_idx,
                'total_reward': float(ep_stats['total_reward']),
                'episode_length': int(ep_stats['episode_length']),
                'final_nodes': int(ep_stats['final_nodes']),
                'final_edges': int(ep_stats['final_edges']),
                'terminated': bool(ep_stats['terminated']),
                'component_rewards': {k: float(v) for k, v in ep_stats['component_rewards'].items()}
            }
            training_metrics.append(episode_entry)
        
        training_metrics_path = os.path.join(output_dir, f"training_metrics_{timestamp}.json")
        with open(training_metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        if verbose:
            print(f"   Training metrics: training_metrics_{timestamp}.json")
        
        # 6. Loss metrics (placeholder for evaluation - no actual loss in deployment)
        loss_metrics = []
        for ep_idx in range(len(episode_stats)):
            # In evaluation mode, we don't have loss, but create placeholder for compatibility
            episode_entry = {
                'episode': ep_idx,
                'loss': 0.0,  # No loss in evaluation
                'smoothed_loss': 0.0,
                'note': 'Evaluation mode - no loss computed'
            }
            loss_metrics.append(episode_entry)
        
        loss_metrics_path = os.path.join(output_dir, f"loss_metrics_{timestamp}.json")
        with open(loss_metrics_path, 'w') as f:
            json.dump(loss_metrics, f, indent=2)
        
        if verbose:
            print(f"   Loss metrics: loss_metrics_{timestamp}.json (placeholder)")
            print(f"   ✅ All evaluation files saved successfully!")
    
    def _save_centroid_csv(self,
                          episode_stats: List[Dict],
                          substrate_type: str,
                          m: float,
                          b: float,
                          verbose: bool = True):
        """
        Save centroid trajectory data to CSV file.
        
        Creates a CSV file with per-node records including:
        - episode, step, num_nodes, cx, cy, intensity, bbox_height, bbox_width
        - delete_ratio: action parameter
        - node_id, node_x, node_y: current node's persistent_id and position
        - action: 'spawn', 'delete', or empty string
        - spawn_node_id, spawn_node_x, spawn_node_y: spawned node details (if action='spawn')
        
        Args:
            episode_stats: List of per-episode statistics
            substrate_type: Type of substrate used
            m: Substrate parameter m
            b: Substrate parameter b
            verbose: Whether to print save confirmation
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(self.model_path)
        
        # Collect all centroid data from all episodes
        all_centroid_data = []
        for ep_stats in episode_stats:
            centroid_data = ep_stats.get('centroid_data', [])
            all_centroid_data.extend(centroid_data)
        
        if not all_centroid_data:
            if verbose:
                print(f"   ⚠️  No centroid data to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_centroid_data)
        
        # Save to CSV
        csv_filename = f"centroid_trajectory_{substrate_type}_m{m}_b{b}_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        if verbose:
            print(f"   📊 Centroid trajectory: {csv_filename} ({len(df)} data points)")


def main():
    """Main deployment function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Deploy trained Durotaxis RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and configuration
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.pt)')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    # Substrate parameters
    parser.add_argument('--substrate_type', type=str, default='linear',
                       choices=['linear', 'exponential', 'random'],
                       help='Type of substrate')
    parser.add_argument('--substrate_width', type=int, default=600,
                       help='Substrate width')
    parser.add_argument('--substrate_height', type=int, default=400,
                       help='Substrate height')
    parser.add_argument('--m', type=float, default=0.05,
                       help='Substrate parameter m')
    parser.add_argument('--b', type=float, default=1.0,
                       help='Substrate parameter b')
    
    # Evaluation parameters
    parser.add_argument('--max_episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic policy (default: stochastic)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to run on (auto-detect if not specified)')
    
    # Environment node limits
    parser.add_argument('--init-nodes', type=int, default=None,
                       help='Initial number of nodes (default from config.yaml)')
    parser.add_argument('--max_critical_nodes', type=int, default=75,
                       help='Maximum allowed nodes before growth penalties (default: 75)')
    parser.add_argument('--threshold_critical_nodes', type=int, default=500,
                       help='Critical threshold - episode terminates if exceeded (default: 500)')
    
    # Output options
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save evaluation results to JSON file')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results (overrides --save_results)')
    parser.add_argument('--enable_visualization', action='store_true', default=True,
                       help='Enable visualization during evaluation (default: True)')
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable visualization (overrides --enable_visualization)')
    
    args = parser.parse_args()
    
    # Handle save_results flag
    if args.no_save:
        args.save_results = False
    
    # Handle visualization flag
    if args.no_viz:
        args.enable_visualization = False
    
    try:
        # Initialize deployment
        deployment = DurotaxisDeployment(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device
        )
        
        # Run evaluation
        results = deployment.run_evaluation(
            substrate_type=args.substrate_type,
            substrate_size=(args.substrate_width, args.substrate_height),
            m=args.m,
            b=args.b,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            save_results=args.save_results,
            enable_visualization=args.enable_visualization,
            max_critical_nodes=args.max_critical_nodes,
            threshold_critical_nodes=args.threshold_critical_nodes,
            init_num_nodes=args.init_nodes
        )
        
        # Only print completion message when not using visualization
        if not args.enable_visualization:
            print(f"\n✅ Deployment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
