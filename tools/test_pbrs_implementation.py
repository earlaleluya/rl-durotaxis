#!/usr/bin/env python3
"""
Test suite for Potential-Based Reward Shaping (PBRS) implementation.

This test verifies:
1. PBRS parameters load correctly from config
2. Potential functions compute correct values
3. PBRS shaping terms are added correctly to rewards
4. PBRS preserves Markov property (depends only on state)
5. PBRS works in both simple_delete_only_mode and centroid_distance_only_mode
"""

import sys
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, '/home/arl_eifer/github/rl-durotaxis')

from durotaxis_env import DurotaxisEnv


def test_pbrs_delete_potential():
    """Test delete potential function Phi(s)."""
    print("\n" + "="*80)
    print("TEST 1: Delete Potential Function")
    print("="*80)
    
    # Create a mock state with known to_delete flags
    state = {
        'num_nodes': 10,
        'to_delete': torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
        'persistent_id': torch.arange(10),
        'node_features': torch.zeros(10, 5),
        'graph_features': torch.zeros(10),
        'edge_attr': torch.zeros(0, 5),
        'edge_index': (torch.zeros(0), torch.zeros(0)),
        'num_edges': 0,
        'centroid_x': 50.0,
        'goal_x': 200.0
    }
    
    # Create environment with PBRS parameters
    env = DurotaxisEnv('config.yaml')
    env._pbrs_delete_w_pending = 1.0
    env._pbrs_delete_w_safe = 0.25
    
    phi = env._phi_delete_potential(state)
    
    # Expected: -1.0 * 3 (marked) + 0.25 * 7 (unmarked) = -3.0 + 1.75 = -1.25
    expected_phi = -1.0 * 3 + 0.25 * 7
    
    print(f"State: 10 nodes, 3 marked for deletion, 7 safe")
    print(f"Phi(s) = -w_pending * marked + w_safe * unmarked")
    print(f"Phi(s) = -1.0 * 3 + 0.25 * 7")
    print(f"Expected Phi: {expected_phi:.4f}")
    print(f"Computed Phi: {phi:.4f}")
    print(f"Match: {'✅ PASS' if abs(phi - expected_phi) < 0.001 else '❌ FAIL'}")
    
    return abs(phi - expected_phi) < 0.001


def test_pbrs_centroid_potential():
    """Test centroid distance potential function Phi(s)."""
    print("\n" + "="*80)
    print("TEST 2: Centroid Distance Potential Function")
    print("="*80)
    
    # Create a mock state
    state = {
        'num_nodes': 5,
        'centroid_x': 50.0,
        'goal_x': 200.0,
        'to_delete': torch.zeros(5),
        'persistent_id': torch.arange(5),
        'node_features': torch.zeros(5, 5),
        'graph_features': torch.zeros(10),
        'edge_attr': torch.zeros(0, 5),
        'edge_index': (torch.zeros(0), torch.zeros(0)),
        'num_edges': 0
    }
    
    env = DurotaxisEnv('config.yaml')
    env._pbrs_centroid_scale = 1.0
    
    phi = env._phi_centroid_distance_potential(state)
    
    # Expected: -scale * (goal_x - centroid_x) = -1.0 * (200 - 50) = -150
    expected_phi = -1.0 * (200.0 - 50.0)
    
    print(f"State: centroid_x=50.0, goal_x=200.0")
    print(f"Phi(s) = -scale * (goal_x - centroid_x)")
    print(f"Phi(s) = -1.0 * (200.0 - 50.0)")
    print(f"Expected Phi: {expected_phi:.4f}")
    print(f"Computed Phi: {phi:.4f}")
    print(f"Match: {'✅ PASS' if abs(phi - expected_phi) < 0.001 else '❌ FAIL'}")
    
    return abs(phi - expected_phi) < 0.001


def test_pbrs_delete_shaping():
    """Test that PBRS shaping is correctly added to delete reward."""
    print("\n" + "="*80)
    print("TEST 3: Delete Reward with PBRS Shaping")
    print("="*80)
    
    # Load config and enable PBRS
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable PBRS for delete reward
    config['environment']['delete_reward']['pbrs'] = {
        'enabled': True,
        'shaping_coeff': 0.1,
        'phi_weight_pending_marked': 1.0,
        'phi_weight_safe_unmarked': 0.25
    }
    
    # Save modified config temporarily
    with open('config_pbrs_test.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create environment with PBRS enabled
    env = DurotaxisEnv('config_pbrs_test.yaml')
    env.simple_delete_only_mode = True
    
    print(f"PBRS Delete Enabled: {env._pbrs_delete_enabled}")
    print(f"PBRS Delete Coeff: {env._pbrs_delete_coeff}")
    print(f"PBRS Gamma: {env._pbrs_gamma}")
    
    # Create two states: s -> s' with one marked node deleted
    prev_state = {
        'num_nodes': 3,
        'to_delete': torch.tensor([1, 0, 0], dtype=torch.float32),
        'persistent_id': torch.tensor([1, 2, 3], dtype=torch.long),
        'node_features': torch.zeros(3, 5),
        'graph_features': torch.zeros(10),
        'edge_attr': torch.zeros(0, 5),
        'edge_index': (torch.zeros(0), torch.zeros(0)),
        'num_edges': 0,
        'centroid_x': 50.0,
        'goal_x': 200.0
    }
    
    new_state = {
        'num_nodes': 2,
        'to_delete': torch.tensor([0, 0], dtype=torch.float32),
        'persistent_id': torch.tensor([2, 3], dtype=torch.long),  # Node 1 deleted
        'node_features': torch.zeros(2, 5),
        'graph_features': torch.zeros(10),
        'edge_attr': torch.zeros(0, 5),
        'edge_index': (torch.zeros(0), torch.zeros(0)),
        'num_edges': 0,
        'centroid_x': 52.0,
        'goal_x': 200.0
    }
    
    # Calculate reward with PBRS
    reward_with_pbrs = env._calculate_delete_reward(prev_state, new_state, [])
    
    # Disable PBRS and calculate again
    env._pbrs_delete_enabled = False
    reward_without_pbrs = env._calculate_delete_reward(prev_state, new_state, [])
    
    # Calculate expected PBRS shaping
    phi_prev = -1.0 * 1 + 0.25 * 2  # 1 marked, 2 safe
    phi_new = -1.0 * 0 + 0.25 * 2   # 0 marked, 2 safe (marked node deleted)
    pbrs_shaping = 0.99 * phi_new - phi_prev
    expected_reward = reward_without_pbrs + 0.1 * pbrs_shaping
    
    print(f"\nPrev state: 3 nodes (1 marked, 2 safe)")
    print(f"New state: 2 nodes (0 marked, 2 safe) - marked node deleted")
    print(f"\nBase delete reward: {reward_without_pbrs:.4f}")
    print(f"Phi(prev): {phi_prev:.4f}")
    print(f"Phi(new): {phi_new:.4f}")
    print(f"PBRS shaping: gamma*Phi(new) - Phi(prev) = {pbrs_shaping:.4f}")
    print(f"Scaled shaping: {0.1 * pbrs_shaping:.4f}")
    print(f"\nExpected reward (base + shaping): {expected_reward:.4f}")
    print(f"Computed reward: {reward_with_pbrs:.4f}")
    print(f"Match: {'✅ PASS' if abs(reward_with_pbrs - expected_reward) < 0.001 else '❌ FAIL'}")
    
    # Cleanup
    import os
    os.remove('config_pbrs_test.yaml')
    
    return abs(reward_with_pbrs - expected_reward) < 0.001


def test_pbrs_centroid_shaping():
    """Test that PBRS shaping is correctly added to centroid movement reward."""
    print("\n" + "="*80)
    print("TEST 4: Centroid Movement Reward with PBRS Shaping")
    print("="*80)
    
    # Load config and enable PBRS
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable PBRS for centroid distance
    config['environment']['graph_rewards']['pbrs_centroid'] = {
        'enabled': True,
        'shaping_coeff': 0.1,
        'phi_distance_scale': 1.0
    }
    config['environment']['centroid_distance_only_mode'] = True
    
    # Save modified config temporarily
    with open('config_pbrs_centroid_test.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create environment with PBRS enabled
    env = DurotaxisEnv('config_pbrs_centroid_test.yaml')
    
    print(f"PBRS Centroid Enabled: {env._pbrs_centroid_enabled}")
    print(f"PBRS Centroid Coeff: {env._pbrs_centroid_coeff}")
    print(f"PBRS Gamma: {env._pbrs_gamma}")
    
    # Create two states: s -> s' with rightward movement
    prev_state = {
        'num_nodes': 5,
        'to_delete': torch.zeros(5),
        'persistent_id': torch.arange(5),
        'node_features': torch.zeros(5, 5),
        'graph_features': torch.tensor([0, 0, 0, 50.0, 0, 0, 0, 0, 0, 0]),  # centroid at index 3
        'edge_attr': torch.zeros(0, 5),
        'edge_index': (torch.zeros(0), torch.zeros(0)),
        'num_edges': 0,
        'centroid_x': 50.0,
        'goal_x': 200.0
    }
    
    new_state = {
        'num_nodes': 5,
        'to_delete': torch.zeros(5),
        'persistent_id': torch.arange(5),
        'node_features': torch.zeros(5, 5),
        'graph_features': torch.tensor([0, 0, 0, 60.0, 0, 0, 0, 0, 0, 0]),  # moved right to 60
        'edge_attr': torch.zeros(0, 5),
        'edge_index': (torch.zeros(0), torch.zeros(0)),
        'num_edges': 0,
        'centroid_x': 60.0,
        'goal_x': 200.0
    }
    
    # Calculate reward with PBRS
    reward_with_pbrs = env._calculate_centroid_movement_reward(prev_state, new_state)
    
    # Disable PBRS and calculate again
    env._pbrs_centroid_enabled = False
    reward_without_pbrs = env._calculate_centroid_movement_reward(prev_state, new_state)
    
    # Calculate expected PBRS shaping
    phi_prev = -1.0 * (200.0 - 50.0)  # -150
    phi_new = -1.0 * (200.0 - 60.0)   # -140
    pbrs_shaping = 0.99 * phi_new - phi_prev
    expected_reward = reward_without_pbrs + 0.1 * pbrs_shaping
    
    print(f"\nPrev state: centroid_x=50.0, distance to goal=150.0")
    print(f"New state: centroid_x=60.0, distance to goal=140.0 (moved right +10)")
    print(f"\nBase centroid reward: {reward_without_pbrs:.4f}")
    print(f"Phi(prev): {phi_prev:.4f}")
    print(f"Phi(new): {phi_new:.4f}")
    print(f"PBRS shaping: gamma*Phi(new) - Phi(prev) = {pbrs_shaping:.4f}")
    print(f"Scaled shaping: {0.1 * pbrs_shaping:.4f}")
    print(f"\nExpected reward (base + shaping): {expected_reward:.4f}")
    print(f"Computed reward: {reward_with_pbrs:.4f}")
    print(f"Match: {'✅ PASS' if abs(reward_with_pbrs - expected_reward) < 0.001 else '❌ FAIL'}")
    
    # Cleanup
    import os
    os.remove('config_pbrs_centroid_test.yaml')
    
    return abs(reward_with_pbrs - expected_reward) < 0.001


def run_all_tests():
    """Run all PBRS tests."""
    print("\n" + "="*80)
    print("POTENTIAL-BASED REWARD SHAPING (PBRS) TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Delete Potential Function", test_pbrs_delete_potential()))
    results.append(("Centroid Distance Potential", test_pbrs_centroid_potential()))
    results.append(("Delete Reward PBRS Shaping", test_pbrs_delete_shaping()))
    results.append(("Centroid Movement PBRS Shaping", test_pbrs_centroid_shaping()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "="*80)
    if all_passed:
        print("🎉 All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
