#!/usr/bin/env python3
"""
Test script for centroid-to-goal distance-only mode.
Verifies that ONLY the distance penalty is computed when the mode is enabled.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from durotaxis_env import DurotaxisEnv

def test_centroid_distance_mode():
    """Test that centroid distance mode computes only distance penalty."""
    print("=" * 80)
    print("TEST: Centroid-to-Goal Distance-Only Mode")
    print("=" * 80)
    
    # Load config
    config_path = project_root / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable centroid distance mode
    env_config = config['environment']
    env_config['centroid_distance_only_mode'] = True
    env_config['simple_delete_only_mode'] = False
    env_config['enable_visualization'] = False
    
    # Save modified config temporarily
    temp_config_path = project_root / 'config_test_centroid.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Create environment with temp config
        env = DurotaxisEnv(str(temp_config_path))
        print(f"✓ Environment created with goal_x = {env.goal_x}")
        print(f"✓ centroid_distance_only_mode = {env.centroid_distance_only_mode}")
        print(f"✓ simple_delete_only_mode = {env.simple_delete_only_mode}")
        print()
        
        # Test Case 1: Initial state
        print("-" * 80)
        print("TEST CASE 1: Initial State Reward")
        print("-" * 80)
        obs, info = env.reset()
        
        # Get initial state to extract centroid
        state = env.state_extractor.get_state_features(
            include_substrate=True,
            node_age=env._node_age,
            node_stagnation=env._node_stagnation
        )
        
        # Get initial centroid position
        centroid_x = state['graph_features'][3].item() if state['num_nodes'] > 0 else 0.0
        print(f"Initial centroid_x: {centroid_x:.2f}")
        print(f"Goal x: {env.goal_x}")
        
        # Take a step (spawn action to test reward calculation)
        action = {
            'discrete': torch.tensor([1, 0]),  # Spawn
            'continuous': torch.tensor([5.0, 1.0, 0.1, 0.0])  # gamma, alpha, noise, theta
        }
        
        obs, reward_components, terminated, truncated, info = env.step(action)
        
        # Get new state to extract centroid
        new_state = env.state_extractor.get_state_features(
            include_substrate=True,
            node_age=env._node_age,
            node_stagnation=env._node_stagnation
        )
        
        # Extract reward components
        total_reward = reward_components['total_reward']
        graph_reward = reward_components.get('graph_reward', 0.0)
        spawn_reward = reward_components.get('spawn_reward', 0.0)
        delete_reward = reward_components.get('delete_reward', 0.0)
        edge_reward = reward_components.get('edge_reward', 0.0)
        milestone_reward = reward_components.get('milestone_reward', 0.0)
        node_reward = reward_components.get('total_node_reward', 0.0)
        survival_reward = reward_components.get('survival_reward', 0.0)
        
        # Get new centroid position
        new_centroid_x = new_state['graph_features'][3].item() if new_state['num_nodes'] > 0 else 0.0
        
        # Calculate expected distance penalty
        expected_penalty = -(env.goal_x - new_centroid_x) / env.goal_x
        
        print(f"\nNew centroid_x: {new_centroid_x:.2f}")
        print(f"Expected distance penalty: {expected_penalty:.6f}")
        print(f"Formula: -({env.goal_x} - {new_centroid_x:.2f}) / {env.goal_x}")
        print()
        print("Reward Components:")
        print(f"  total_reward:    {total_reward:.6f}")
        print(f"  graph_reward:    {graph_reward:.6f}")
        print(f"  spawn_reward:    {spawn_reward:.6f}")
        print(f"  delete_reward:   {delete_reward:.6f}")
        print(f"  edge_reward:     {edge_reward:.6f}")
        print(f"  milestone_reward: {milestone_reward:.6f}")
        print(f"  node_reward:     {node_reward:.6f}")
        print(f"  survival_reward: {survival_reward:.6f}")
        
        # Verify only distance penalty is non-zero
        assert abs(total_reward - expected_penalty) < 1e-5, \
            f"Total reward should equal distance penalty! Got {total_reward}, expected {expected_penalty}"
        assert spawn_reward == 0.0, f"Spawn reward should be 0! Got {spawn_reward}"
        assert delete_reward == 0.0, f"Delete reward should be 0! Got {delete_reward}"
        assert edge_reward == 0.0, f"Edge reward should be 0! Got {edge_reward}"
        assert milestone_reward == 0.0, f"Milestone reward should be 0! Got {milestone_reward}"
        assert node_reward == 0.0, f"Node reward should be 0! Got {node_reward}"
        assert survival_reward == 0.0, f"Survival reward should be 0! Got {survival_reward}"
        
        print(f"\n✅ PASSED: Only distance penalty is active ({total_reward:.6f})")
        print()
        
        # Test Case 2: Progress toward goal
        print("-" * 80)
        print("TEST CASE 2: Progress Toward Goal")
        print("-" * 80)
        
        penalties = []
        centroids = []
        
        for i in range(5):
            # Take spawn action to encourage rightward movement
            action = {
                'discrete': torch.tensor([1, 0]),
                'continuous': torch.tensor([5.0, 1.0, 0.1, 0.0])  # theta=0 for rightward
            }
            
            obs, reward_components, terminated, truncated, info = env.step(action)
            
            # Get state to extract centroid
            new_state = env.state_extractor.get_state_features(
                include_substrate=True,
                node_age=env._node_age,
                node_stagnation=env._node_stagnation
            )
            
            centroid_x = new_state['graph_features'][3].item() if new_state['num_nodes'] > 0 else 0.0
            total_reward = reward_components['total_reward']
            
            centroids.append(centroid_x)
            penalties.append(total_reward)
            
            if terminated or truncated:
                print(f"Step {i+1}: Episode terminated/truncated")
                break
        
        print(f"\nCentroid progression: {[f'{c:.2f}' for c in centroids]}")
        print(f"Penalty progression:  {[f'{p:.4f}' for p in penalties]}")
        print()
        
        # Verify penalties improve as centroid moves right
        if len(penalties) > 1:
            penalty_improved = penalties[-1] > penalties[0]
            print(f"✓ Penalty change: {penalties[0]:.4f} → {penalties[-1]:.4f}")
            
            if penalty_improved:
                print("✅ PASSED: Penalty improves as centroid moves right")
            else:
                print("⚠️  WARNING: Penalty did not improve")
        print()
        
        print("=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nSummary:")
        print("✅ Distance penalty computed correctly")
        print("✅ All other reward components zeroed out")
        print("\n🎉 Centroid-to-goal distance-only mode is working correctly!")
        
    finally:
        # Clean up temp config file
        if temp_config_path.exists():
            temp_config_path.unlink()
            print(f"\n✓ Cleaned up temporary config file")

if __name__ == "__main__":
    test_centroid_distance_mode()
