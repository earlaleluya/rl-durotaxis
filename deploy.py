#!/usr/bin/env python3
"""
Deployment script for trained Durotaxis RL agent

This script allows you to load a trained model and run it on custom substrates
with configurable parameters for evaluation and demonstration purposes.

Usage:
    python deploy.py --model_path ./training_results/run0004/best_model_batch4.pt \
                     --substrate_type linear --m 0.05 --b 1.0 \
                     --deterministic --max_episodes 10 --max_steps 1000
    
    # Without visualization
    python deploy.py --model_path ./training_results/run0004/best_model_batch4.pt \
                     --substrate_type linear --m 0.05 --b 1.0 \
                     --deterministic --max_episodes 10 --max_steps 1000 --no_viz
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
import time

# Import project modules
from durotaxis_env import DurotaxisEnv
from state import TopologyState
from encoder import GraphInputEncoder
from actor_critic import HybridActorCritic
from config_loader import ConfigLoader


class DurotaxisDeployment:
    """Deployment class for running trained Durotaxis agents"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config.yaml",
                 device: Optional[str] = None):
        """
        Initialize deployment with trained model
        
        Args:
            model_path: Path to saved model (.pt file)
            config_path: Path to configuration file
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        # Create actor-critic network
        self.network = HybridActorCritic(
            encoder=self.encoder,
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
        """Create action mask to prevent invalid topology operations"""
        num_nodes = state_dict.get('num_nodes', 0)
        if num_nodes == 0:
            return None
        
        # Basic action masking rules
        mask = torch.ones(num_nodes, 2, dtype=torch.bool, device=self.device)  # [spawn, delete]
        
        # Don't delete if too few nodes (prevent disconnection)
        if num_nodes <= 2:
            mask[:, 1] = False  # No deletion allowed
        
        return mask
    
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
        step_count = 0
        done = False
        
        if verbose:
            print(f"\n🎯 Starting episode (max_steps={max_steps})")
            print(f"   Initial nodes: {env.topology.graph.num_nodes()}")
        
        while not done and step_count < max_steps:
            # Get current state
            state_dict = self.state_extractor.get_state_features(include_substrate=True)
            state_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in state_dict.items()}
            
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
            
            # Extract and execute actions
            if len(output.get('discrete_actions', [])) > 0:
                discrete_actions = output['discrete_actions']
                continuous_actions = output['continuous_actions']
                
                episode_actions.append({
                    'discrete': discrete_actions.cpu().numpy(),
                    'continuous': continuous_actions.cpu().numpy(),
                    'num_nodes': state_dict['num_nodes']
                })
                
                # Execute topology actions
                topology_actions = self.network.get_topology_actions(output)
                
                for node_id, action_type in topology_actions.items():
                    try:
                        if action_type == 'spawn':
                            params = self.network.get_spawn_parameters(output, node_id)
                            env.topology.spawn(node_id, gamma=params[0], alpha=params[1], 
                                             noise=params[2], theta=params[3])
                            if verbose:
                                print(f"   Step {step_count+1}: Spawned from node {node_id} (γ={params[0]:.3f}, α={params[1]:.3f})")
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
            
            # Store rewards
            episode_rewards.append(reward_components)
            
            if verbose:
                total_reward = reward_components.get('total_reward', 0.0)
                num_nodes = env.topology.graph.num_nodes()
                num_edges = env.topology.graph.num_edges()
                print(f"   Step {step_count+1}: N={num_nodes:3d} E={num_edges:3d} | R={total_reward:+7.3f} | Done={done}")
            
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
            'values_per_step': episode_values
        }
        
        if verbose:
            print(f"📊 Episode completed:")
            print(f"   Total reward: {total_reward:.3f}")
            print(f"   Episode length: {step_count} steps")
            print(f"   Final topology: {stats['final_nodes']} nodes, {stats['final_edges']} edges")
        
        return stats
    
    def run_evaluation(self,
                      substrate_type: str = "linear",
                      m: float = 0.05,
                      b: float = 1.0,
                      max_episodes: int = 10,
                      max_steps: int = 100,
                      deterministic: bool = True,
                      save_results: bool = True,
                      enable_visualization: bool = True) -> Dict:
        """
        Run evaluation over multiple episodes
        
        Args:
            substrate_type: Type of substrate ('linear', 'exponential', 'random')
            m: Substrate parameter m
            b: Substrate parameter b
            max_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic policy
            save_results: Whether to save results to file
            enable_visualization: Whether to enable visualization during episodes
            
        Returns:
            Dictionary with evaluation statistics
        """
        verbose_eval = not enable_visualization  # Verbose evaluation output when visualization is disabled
        
        if verbose_eval:
            print(f"\n🧪 Running evaluation:")
            print(f"   Substrate: {substrate_type} (m={m}, b={b})")
            print(f"   Episodes: {max_episodes}, Max steps: {max_steps}")
            print(f"   Policy: {'Deterministic' if deterministic else 'Stochastic'}")
            print(f"   Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
        
        # Create environment with specified substrate
        env = DurotaxisEnv(
            config_path=self.config_path,
            substrate_type=substrate_type,
            substrate_m=m,
            substrate_b=b,
            threshold_critical_nodes=500  # Increased from 200 to allow longer episodes
        )
        
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
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_filename = f"evaluation_{substrate_type}_m{m}_b{b}_{timestamp}.json"
            results_path = os.path.join(os.path.dirname(self.model_path), results_filename)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.cpu().numpy().tolist()
                return obj
            
            # Deep convert the results
            import json
            serializable_results = json.loads(json.dumps(evaluation_results, default=convert_numpy))
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            if verbose_eval:
                print(f"💾 Results saved to: {results_path}")
        
        return evaluation_results


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
            m=args.m,
            b=args.b,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            save_results=args.save_results,
            enable_visualization=args.enable_visualization
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
