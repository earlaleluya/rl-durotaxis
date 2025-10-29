#!/usr/bin/env python3
"""
Simple Visualization Test for Delete Ratio Rightward Movement

This script creates a simple test environment and visualizes how the agent
moves rightward using the delete ratio parameter architecture.

ARCHITECTURE:
- Action Space: [delete_ratio, gamma, alpha, noise, theta] (5D continuous)
- Delete Strategy: Sort nodes by x-position, delete leftmost fraction
- Spawn Strategy: Apply global parameters to remaining nodes

USAGE:
    # Random agent (no trained model)
    python tools/test_visualize_rightward.py
    
    # With trained model
    python tools/test_visualize_rightward.py --model_path training_results/run0007/best_model.pt
    
    # Longer episode
    python tools/test_visualize_rightward.py --max_steps 500
    
    # Different substrate
    python tools/test_visualize_rightward.py --substrate_type sigmoid --m 0.1
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

from durotaxis_env import DurotaxisEnv
from state import TopologyState
from encoder import GraphInputEncoder
from actor_critic import HybridActorCritic
from config_loader import ConfigLoader


class SimpleVisualizer:
    """Simple visualizer for rightward movement with delete ratio"""
    
    def __init__(self, env, model=None, device='cpu'):
        """
        Initialize visualizer
        
        Args:
            env: DurotaxisEnv instance
            model: Trained model (optional, uses random if None)
            device: torch device
        """
        self.env = env
        self.model = model
        self.device = device
        self.state_extractor = TopologyState()
        
        # Setup figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Delete Ratio Rightward Movement Visualization', fontsize=16)
        
        # Flatten axes for easier access
        self.ax_topology = self.axes[0, 0]  # Top-left: topology
        self.ax_substrate = self.axes[0, 1]  # Top-right: substrate
        self.ax_centroid = self.axes[1, 0]   # Bottom-left: centroid trajectory
        self.ax_actions = self.axes[1, 1]    # Bottom-right: action values
        
        # History tracking
        self.centroid_history = []
        self.action_history = {
            'delete_ratio': [],
            'gamma': [],
            'alpha': [],
            'noise': [],
            'theta': []
        }
        self.step_count = 0
        
    def setup_plots(self):
        """Setup initial plot configurations"""
        # Topology plot
        self.ax_topology.set_title('Graph Topology')
        self.ax_topology.set_xlabel('X Position')
        self.ax_topology.set_ylabel('Y Position')
        self.ax_topology.set_xlim(0, self.env.topology.substrate.width)
        self.ax_topology.set_ylim(0, self.env.topology.substrate.height)
        self.ax_topology.grid(True, alpha=0.3)
        
        # Add goal region
        goal_x = self.env.goal_x
        goal_y = self.env.topology.substrate.height / 2
        goal_circle = plt.Circle((goal_x, goal_y), 20, color='green', alpha=0.2, label='Goal')
        self.ax_topology.add_patch(goal_circle)
        self.ax_topology.axvline(x=goal_x, color='green', linestyle='--', alpha=0.5, label='Goal X')
        
        # Substrate plot
        self.ax_substrate.set_title('Stiffness Substrate')
        self.ax_substrate.set_xlabel('X Position')
        self.ax_substrate.set_ylabel('Stiffness (Pa)')
        
        # Centroid trajectory plot
        self.ax_centroid.set_title('Centroid Trajectory')
        self.ax_centroid.set_xlabel('Step')
        self.ax_centroid.set_ylabel('Centroid X Position')
        self.ax_centroid.grid(True, alpha=0.3)
        self.ax_centroid.axhline(y=goal_x, color='green', linestyle='--', alpha=0.5, label='Goal')
        
        # Actions plot
        self.ax_actions.set_title('Action Values Over Time')
        self.ax_actions.set_xlabel('Step')
        self.ax_actions.set_ylabel('Action Value')
        self.ax_actions.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def get_action(self, obs):
        """Get action from model or random"""
        if self.model is not None:
            # Use trained model
            with torch.no_grad():
                self.model.eval()
                
                # Extract state features
                self.state_extractor.set_topology(self.env.topology)
                state_dict = self.state_extractor.get_state_features()
                
                # Prepare batch
                graph_batch = state_dict['graph'].to(self.device)
                substrate_tensor = torch.FloatTensor(state_dict['substrate']).unsqueeze(0).to(self.device)
                global_features = torch.FloatTensor(state_dict['global_features']).unsqueeze(0).to(self.device)
                
                # Get action from model
                action, _, _, _ = self.model(graph_batch, substrate_tensor, global_features)
                action = action.cpu().numpy()[0]
        else:
            # Random action for visualization
            # Use reasonable defaults biased toward rightward movement
            action = np.array([
                np.random.uniform(0.1, 0.3),  # delete_ratio: delete 10-30%
                np.random.uniform(0.8, 1.0),  # gamma: high stiffness preference
                np.random.uniform(0.7, 0.9),  # alpha: strong stiffness influence
                np.random.uniform(0.05, 0.15), # noise: low to moderate
                np.random.uniform(0.0, np.pi/4)  # theta: rightward bias
            ])
        
        return action
    
    def update_plots(self):
        """Update all plots with current state"""
        # Clear plots
        self.ax_topology.clear()
        self.ax_substrate.clear()
        self.ax_centroid.clear()
        self.ax_actions.clear()
        
        # Re-setup static elements
        goal_x = self.env.goal_x
        
        # ===== TOPOLOGY PLOT =====
        self.ax_topology.set_title(f'Graph Topology (Step {self.step_count})')
        self.ax_topology.set_xlabel('X Position')
        self.ax_topology.set_ylabel('Y Position')
        self.ax_topology.set_xlim(0, self.env.topology.substrate.width)
        self.ax_topology.set_ylim(0, self.env.topology.substrate.height)
        self.ax_topology.grid(True, alpha=0.3)
        
        # Plot goal region
        goal_y = self.env.topology.substrate.height / 2
        goal_circle = plt.Circle((goal_x, goal_y), 20, color='green', alpha=0.2)
        self.ax_topology.add_patch(goal_circle)
        self.ax_topology.axvline(x=goal_x, color='green', linestyle='--', alpha=0.5)
        
        # Plot nodes
        if self.env.topology.graph.number_of_nodes() > 0:
            positions = self.env.topology.graph.ndata['pos'].cpu().numpy()
            self.ax_topology.scatter(positions[:, 0], positions[:, 1], 
                                   c='blue', s=50, alpha=0.6, label='Nodes')
            
            # Plot edges
            if self.env.topology.graph.number_of_edges() > 0:
                src, dst = self.env.topology.graph.edges()
                src_np = src.cpu().numpy()
                dst_np = dst.cpu().numpy()
                
                for s, d in zip(src_np, dst_np):
                    x = [positions[s, 0], positions[d, 0]]
                    y = [positions[s, 1], positions[d, 1]]
                    self.ax_topology.plot(x, y, 'gray', alpha=0.3, linewidth=0.5)
            
            # Plot centroid
            centroid_x = positions[:, 0].mean()
            centroid_y = positions[:, 1].mean()
            self.ax_topology.scatter([centroid_x], [centroid_y], 
                                   c='red', s=200, marker='*', 
                                   edgecolors='black', linewidths=2,
                                   label='Centroid', zorder=5)
            
            # Add info text
            num_nodes = self.env.topology.graph.number_of_nodes()
            info_text = f'Nodes: {num_nodes}\nCentroid X: {centroid_x:.1f}'
            self.ax_topology.text(0.02, 0.98, info_text,
                                transform=self.ax_topology.transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax_topology.legend(loc='upper right')
        
        # ===== SUBSTRATE PLOT =====
        self.ax_substrate.set_title('Stiffness Substrate')
        self.ax_substrate.set_xlabel('X Position')
        self.ax_substrate.set_ylabel('Stiffness (Pa)')
        
        # Plot substrate profile
        x_positions = np.linspace(0, self.env.topology.substrate.width, 200)
        y_center = self.env.topology.substrate.height / 2
        stiffness_values = [self.env.topology.substrate.get_stiffness(x, y_center) 
                          for x in x_positions]
        
        self.ax_substrate.plot(x_positions, stiffness_values, 'b-', linewidth=2)
        self.ax_substrate.axvline(x=goal_x, color='green', linestyle='--', alpha=0.5)
        self.ax_substrate.fill_between(x_positions, stiffness_values, alpha=0.3)
        
        # ===== CENTROID TRAJECTORY =====
        self.ax_centroid.set_title('Centroid Trajectory')
        self.ax_centroid.set_xlabel('Step')
        self.ax_centroid.set_ylabel('Centroid X Position')
        self.ax_centroid.grid(True, alpha=0.3)
        self.ax_centroid.axhline(y=goal_x, color='green', linestyle='--', alpha=0.5)
        
        if len(self.centroid_history) > 0:
            steps = range(len(self.centroid_history))
            self.ax_centroid.plot(steps, self.centroid_history, 'b-', linewidth=2, marker='o')
            
            # Show progress
            if self.centroid_history:
                progress = (self.centroid_history[-1] / goal_x) * 100
                self.ax_centroid.text(0.02, 0.98, f'Progress: {progress:.1f}%',
                                    transform=self.ax_centroid.transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ===== ACTIONS PLOT =====
        self.ax_actions.set_title('Action Values Over Time')
        self.ax_actions.set_xlabel('Step')
        self.ax_actions.set_ylabel('Action Value')
        self.ax_actions.grid(True, alpha=0.3)
        
        if len(self.action_history['delete_ratio']) > 0:
            steps = range(len(self.action_history['delete_ratio']))
            
            self.ax_actions.plot(steps, self.action_history['delete_ratio'], 
                               label='Delete Ratio', linewidth=2)
            self.ax_actions.plot(steps, self.action_history['gamma'], 
                               label='Gamma', linewidth=2)
            self.ax_actions.plot(steps, self.action_history['alpha'], 
                               label='Alpha', linewidth=2)
            self.ax_actions.plot(steps, self.action_history['noise'], 
                               label='Noise', linewidth=2)
            
            # Theta normalized to [0, 1] for visibility
            theta_normalized = np.array(self.action_history['theta']) / np.pi
            self.ax_actions.plot(steps, theta_normalized, 
                               label='Theta (normalized)', linewidth=2)
            
            self.ax_actions.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
    def run_episode(self, max_steps=200):
        """Run one episode with visualization"""
        print("\n" + "="*60)
        print("🎮 Starting Visualization Episode")
        print("="*60)
        
        obs = self.env.reset()
        self.setup_plots()
        
        done = False
        total_reward = 0.0
        self.step_count = 0
        
        # Initial centroid
        if self.env.topology.graph.number_of_nodes() > 0:
            positions = self.env.topology.graph.ndata['pos'].cpu().numpy()
            centroid_x = positions[:, 0].mean()
            self.centroid_history.append(centroid_x)
        
        while not done and self.step_count < max_steps:
            # Get action
            action = self.get_action(obs)
            
            # Store action
            self.action_history['delete_ratio'].append(action[0])
            self.action_history['gamma'].append(action[1])
            self.action_history['alpha'].append(action[2])
            self.action_history['noise'].append(action[3])
            self.action_history['theta'].append(action[4])
            
            # Take step (Gymnasium returns 5 values)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Extract total reward (reward can be dict or scalar)
            if isinstance(reward, dict):
                reward_scalar = reward.get('total_reward', 0.0)
            else:
                reward_scalar = reward
            total_reward += reward_scalar
            self.step_count += 1
            
            # Store centroid position
            if self.env.topology.graph.number_of_nodes() > 0:
                positions = self.env.topology.graph.ndata['pos'].cpu().numpy()
                centroid_x = positions[:, 0].mean()
                self.centroid_history.append(centroid_x)
            
            # Update plots
            self.update_plots()
            plt.pause(0.1)  # Small pause for animation
            
            # Print step info
            if self.step_count % 10 == 0:
                print(f"Step {self.step_count:3d} | "
                      f"Reward: {reward_scalar:7.2f} | "
                      f"Delete Ratio: {action[0]:.3f} | "
                      f"Nodes: {self.env.topology.graph.number_of_nodes()}")
        
        print("\n" + "="*60)
        print(f"✅ Episode Complete!")
        print(f"   Total Steps: {self.step_count}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Final Nodes: {self.env.topology.graph.number_of_nodes()}")
        
        if self.centroid_history:
            final_centroid = self.centroid_history[-1]
            goal_x = self.env.goal_x
            progress = (final_centroid / goal_x) * 100
            print(f"   Final Centroid X: {final_centroid:.1f}")
            print(f"   Goal X: {goal_x:.1f}")
            print(f"   Progress: {progress:.1f}%")
            
            if info.get('success', False):
                print(f"   🎉 SUCCESS! Reached goal!")
        
        print("="*60)
        
        plt.show()


def load_model(model_path, config_path='config.yaml', device='cpu'):
    """Load trained model"""
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        return None
    
    print(f"📦 Loading model from: {model_path}")
    
    # Load config
    config_loader = ConfigLoader(config_path)
    encoder_config = config_loader.get_encoder_config()
    actor_critic_config = config_loader.get_actor_critic_config()
    
    # Component names
    component_names = [
        'total_reward',
        'graph_reward', 
        'spawn_reward',
        'delete_reward',
        'edge_reward',
        'total_node_reward'
    ]
    
    # Create encoder
    encoder = GraphInputEncoder(
        hidden_dim=encoder_config.get('hidden_dim', 128),
        out_dim=encoder_config.get('out_dim', 64),
        num_layers=encoder_config.get('num_layers', 4)
    ).to(device)
    
    # Create network
    network = HybridActorCritic(
        encoder=encoder,
        hidden_dim=actor_critic_config.get('hidden_dim', 128),
        value_components=actor_critic_config.get('value_components', component_names),
        dropout_rate=actor_critic_config.get('dropout_rate', 0.1)
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        network.load_state_dict(checkpoint['model_state_dict'])
    elif 'network_state_dict' in checkpoint:
        network.load_state_dict(checkpoint['network_state_dict'])
    else:
        network.load_state_dict(checkpoint)
    
    network.eval()
    print(f"✅ Model loaded successfully!")
    
    return network


def main():
    parser = argparse.ArgumentParser(description='Visualize Delete Ratio Rightward Movement')
    
    # Model options
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (optional, uses random if not provided)')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                       help='Path to config file')
    
    # Environment options
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximum steps per episode')
    parser.add_argument('--substrate_type', type=str, default='linear',
                       choices=['linear', 'sigmoid', 'exponential', 'step'],
                       help='Substrate stiffness profile')
    parser.add_argument('--m', type=float, default=0.05,
                       help='Substrate gradient parameter')
    parser.add_argument('--b', type=float, default=1.0,
                       help='Substrate offset parameter')
    parser.add_argument('--initial_nodes', type=int, default=20,
                       help='Initial number of nodes')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load model if provided
    model = None
    if args.model_path:
        model = load_model(args.model_path, args.config_path, device)
    else:
        print("⚠️  No model provided, using random actions")
    
    # Create environment
    print(f"\n🌍 Creating environment...")
    print(f"   Substrate: {args.substrate_type} (m={args.m}, b={args.b})")
    print(f"   Initial nodes: {args.initial_nodes}")
    
    env = DurotaxisEnv(
        config_path=args.config_path,
        substrate_type=args.substrate_type,
        init_num_nodes=args.initial_nodes,
        max_critical_nodes=100,
        max_steps=args.max_steps,
        enable_visualization=False  # We handle visualization in this script
    )
    
    # Create visualizer
    visualizer = SimpleVisualizer(env, model, device)
    
    # Run episode
    visualizer.run_episode(max_steps=args.max_steps)


if __name__ == '__main__':
    main()
