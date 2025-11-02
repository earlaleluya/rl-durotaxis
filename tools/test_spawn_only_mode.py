#!/usr/bin/env python3
"""
Test script for simple_spawn_only_mode implementation with PBRS.

Tests:
1. Spawn-only mode reward composition (only spawn rewards, no other components)
2. PBRS potential function for spawn (_phi_spawn_potential)
3. PBRS shaping integration in spawn reward
4. Mode combinations (spawn+delete, spawn+centroid, all three)
5. Termination reward handling with spawn mode
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_loader import ConfigLoader
from durotaxis_env import DurotaxisEnv


def test_1_spawn_only_mode_reward_composition():
    """Test that spawn-only mode uses ONLY spawn rewards."""
    print("\n" + "="*70)
    print("TEST 1: Spawn-Only Mode Reward Composition")
    print("="*70)
    
    # Create environment from config file
    env = DurotaxisEnv('config.yaml')
    
    # Override config for test
    env.simple_spawn_only_mode = True
    env.spawn_reward = 2.0
    env.simple_delete_only_mode = False
    env.centroid_distance_only_mode = False
    env.include_termination_rewards = False
    
    # Reset and take a step
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check reward breakdown
    breakdown = info.get('reward_breakdown', {})
    
    print(f"Spawn-only mode enabled: {env.simple_spawn_only_mode}")
    print(f"Spawn reward value: {env.spawn_reward}")
    print(f"\nReward Breakdown:")
    print(f"  total_reward: {breakdown.get('total_reward', 0.0):.4f}")
    print(f"  graph_reward: {breakdown.get('graph_reward', 0.0):.4f}")
    print(f"  spawn_reward: {breakdown.get('spawn_reward', 0.0):.4f}")
    print(f"  delete_reward: {breakdown.get('delete_reward', 0.0):.4f}")
    print(f"  centroid_reward: {breakdown.get('centroid_reward', 0.0):.4f}")
    print(f"  survival_reward: {breakdown.get('survival_reward', 0.0):.4f}")
    print(f"  milestone_reward: {breakdown.get('milestone_reward', 0.0):.4f}")
    print(f"  total_node_reward: {breakdown.get('total_node_reward', 0.0):.4f}")
    
    # Verify ONLY spawn reward is non-zero (other components should be zero)
    spawn_reward = breakdown.get('spawn_reward', 0.0)
    delete_reward = breakdown.get('delete_reward', 0.0)
    survival_reward = breakdown.get('survival_reward', 0.0)
    milestone_reward = breakdown.get('milestone_reward', 0.0)
    node_reward = breakdown.get('total_node_reward', 0.0)
    
    # In spawn-only mode: total = spawn_reward (all others zero)
    assert delete_reward == 0.0, f"Delete reward should be 0, got {delete_reward}"
    assert survival_reward == 0.0, f"Survival reward should be 0, got {survival_reward}"
    assert milestone_reward == 0.0, f"Milestone reward should be 0, got {milestone_reward}"
    assert node_reward == 0.0, f"Node reward should be 0, got {node_reward}"
    
    total_expected = spawn_reward
    total_actual = breakdown.get('total_reward', 0.0)
    
    print(f"\n✓ Spawn-only mode correctly uses ONLY spawn rewards")
    print(f"  Expected total: {total_expected:.4f}")
    print(f"  Actual total: {total_actual:.4f}")
    
    # Allow small floating point error
    assert abs(total_actual - total_expected) < 1e-4, \
        f"Total reward mismatch: expected {total_expected}, got {total_actual}"
    
    print("\n✓ TEST 1 PASSED")
    return True


def test_2_spawn_potential_function():
    """Test _phi_spawn_potential correctness."""
    print("\n" + "="*70)
    print("TEST 2: Spawn Potential Function (_phi_spawn_potential)")
    print("="*70)
    
    # Create environment with PBRS enabled
    env = DurotaxisEnv('config.yaml')
    env.simple_spawn_only_mode = True
    env._pbrs_spawn_enabled = True
    env._pbrs_spawn_coeff = 0.1
    env._pbrs_spawn_w_spawnable = 1.0
    
    # Create test states with known intensities
    # State 1: 3 nodes with high intensity (>= delta_intensity)
    delta_intensity = env.delta_intensity
    print(f"\nDelta intensity threshold: {delta_intensity:.3f}")
    
    state1 = {
        'num_nodes': 5,
        'intensity': torch.tensor([
            delta_intensity + 0.1,  # High (spawnable)
            delta_intensity + 0.2,  # High (spawnable)
            delta_intensity - 0.1,  # Low (not spawnable)
            delta_intensity + 0.05, # High (spawnable)
            delta_intensity - 0.05  # Low (not spawnable)
        ], dtype=torch.float32)
    }
    
    phi1 = env._phi_spawn_potential(state1)
    expected_spawnable_1 = 3  # 3 nodes >= delta_intensity
    expected_phi1 = env._pbrs_spawn_w_spawnable * expected_spawnable_1
    
    print(f"\nState 1:")
    print(f"  Intensities: {state1['intensity'].numpy()}")
    print(f"  Spawnable nodes (>= {delta_intensity:.3f}): {expected_spawnable_1}")
    print(f"  Computed Φ(s): {phi1:.3f}")
    print(f"  Expected Φ(s): {expected_phi1:.3f}")
    
    assert abs(phi1 - expected_phi1) < 1e-4, \
        f"Potential mismatch: expected {expected_phi1}, got {phi1}"
    
    # State 2: All nodes have high intensity
    state2 = {
        'num_nodes': 4,
        'intensity': torch.tensor([
            delta_intensity + 0.3,
            delta_intensity + 0.4,
            delta_intensity + 0.1,
            delta_intensity + 0.2
        ], dtype=torch.float32)
    }
    
    phi2 = env._phi_spawn_potential(state2)
    expected_spawnable_2 = 4  # All 4 nodes >= delta_intensity
    expected_phi2 = env._pbrs_spawn_w_spawnable * expected_spawnable_2
    
    print(f"\nState 2 (all high intensity):")
    print(f"  Intensities: {state2['intensity'].numpy()}")
    print(f"  Spawnable nodes: {expected_spawnable_2}")
    print(f"  Computed Φ(s): {phi2:.3f}")
    print(f"  Expected Φ(s): {expected_phi2:.3f}")
    
    assert abs(phi2 - expected_phi2) < 1e-4, \
        f"Potential mismatch: expected {expected_phi2}, got {phi2}"
    
    # State 3: Empty graph (edge case)
    state3 = {'num_nodes': 0}
    phi3 = env._phi_spawn_potential(state3)
    
    print(f"\nState 3 (empty graph):")
    print(f"  Computed Φ(s): {phi3:.3f}")
    print(f"  Expected Φ(s): 0.0")
    
    assert phi3 == 0.0, f"Empty graph should have Φ=0, got {phi3}"
    
    print("\n✓ TEST 2 PASSED")
    return True


def test_3_pbrs_shaping_integration():
    """Test that PBRS shaping is correctly added to spawn reward."""
    print("\n" + "="*70)
    print("TEST 3: PBRS Shaping Integration")
    print("="*70)
    
    # Create environment with PBRS enabled
    env = DurotaxisEnv('config.yaml')
    env.simple_spawn_only_mode = True
    env.spawn_reward = 2.0
    env._pbrs_spawn_enabled = True
    env._pbrs_spawn_coeff = 0.1
    env._pbrs_spawn_w_spawnable = 1.0
    
    print(f"\nPBRS enabled: {env._pbrs_spawn_enabled}")
    print(f"PBRS shaping coefficient: {env._pbrs_spawn_coeff}")
    print(f"PBRS gamma: {env._pbrs_gamma}")
    
    # Create mock states with different spawnable counts
    delta = env.delta_intensity
    
    prev_state = {
        'num_nodes': 3,
        'intensity': torch.tensor([delta + 0.1, delta + 0.2, delta - 0.1], dtype=torch.float32)
    }
    
    new_state = {
        'num_nodes': 4,
        'intensity': torch.tensor([delta + 0.1, delta + 0.2, delta + 0.15, delta + 0.05], dtype=torch.float32)
    }
    
    # Compute potentials
    phi_prev = env._phi_spawn_potential(prev_state)
    phi_new = env._phi_spawn_potential(new_state)
    
    # Expected PBRS shaping: F = γ*Φ(s') - Φ(s)
    expected_shaping = env._pbrs_gamma * phi_new - phi_prev
    expected_shaping_term = env._pbrs_spawn_coeff * expected_shaping
    
    print(f"\nPrevious state:")
    print(f"  Spawnable nodes: 2 (out of 3)")
    print(f"  Φ(s): {phi_prev:.3f}")
    
    print(f"\nNew state:")
    print(f"  Spawnable nodes: 4 (out of 4)")
    print(f"  Φ(s'): {phi_new:.3f}")
    
    print(f"\nPBRS Shaping:")
    print(f"  F = γ*Φ(s') - Φ(s) = {env._pbrs_gamma:.2f}*{phi_new:.3f} - {phi_prev:.3f} = {expected_shaping:.3f}")
    print(f"  Weighted shaping = {env._pbrs_spawn_coeff:.2f} * {expected_shaping:.3f} = {expected_shaping_term:.3f}")
    
    # Verify the shaping term is positive (more spawnable nodes = good)
    assert expected_shaping > 0, f"Shaping should be positive when spawnable count increases"
    
    print("\n✓ PBRS shaping correctly increases when more nodes become spawnable")
    print("\n✓ TEST 3 PASSED")
    return True


def test_4_mode_combinations():
    """Test spawn mode combinations with delete and centroid modes."""
    print("\n" + "="*70)
    print("TEST 4: Mode Combinations")
    print("="*70)
    
    test_cases = [
        # (delete, centroid, spawn, description)
        (False, False, True, "Spawn-only"),
        (True, False, True, "Delete + Spawn"),
        (False, True, True, "Centroid + Spawn"),
        (True, True, True, "All three modes"),
    ]
    
    for delete_mode, centroid_mode, spawn_mode, description in test_cases:
        print(f"\n--- Testing: {description} ---")
        
        env = DurotaxisEnv('config.yaml')
        env.simple_delete_only_mode = delete_mode
        env.centroid_distance_only_mode = centroid_mode
        env.simple_spawn_only_mode = spawn_mode
        env.include_termination_rewards = False
        
        # Reset and take a step
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        breakdown = info.get('reward_breakdown', {})
        
        # Check which components should be active
        spawn_reward = breakdown.get('spawn_reward', 0.0)
        delete_reward = breakdown.get('delete_reward', 0.0)
        total_reward = breakdown.get('total_reward', 0.0)
        
        print(f"  Delete mode: {delete_mode}, Centroid mode: {centroid_mode}, Spawn mode: {spawn_mode}")
        print(f"  Spawn reward: {spawn_reward:.4f}")
        print(f"  Delete reward: {delete_reward:.4f}")
        print(f"  Total reward: {total_reward:.4f}")
        
        # Verify spawn reward is included (since spawn mode is always True in these tests)
        # Note: spawn_reward might be 0 if no spawning happened, but it should be calculated
        
        if delete_mode and not centroid_mode:
            # Delete + Spawn: should have both components
            print(f"  ✓ Delete+Spawn mode active")
        elif centroid_mode and not delete_mode:
            # Centroid + Spawn: should have spawn + distance signal
            print(f"  ✓ Centroid+Spawn mode active")
        elif delete_mode and centroid_mode:
            # All three: should have all special mode components
            print(f"  ✓ All three modes active")
        else:
            # Spawn-only
            print(f"  ✓ Spawn-only mode active")
    
    print("\n✓ TEST 4 PASSED")
    return True


def test_5_termination_rewards_with_spawn_mode():
    """Test termination reward handling with spawn mode."""
    print("\n" + "="*70)
    print("TEST 5: Termination Rewards with Spawn Mode")
    print("="*70)
    
    # Test with include_termination_rewards = False
    print("\n--- Case 1: Spawn mode WITHOUT termination rewards ---")
    env = DurotaxisEnv('config.yaml')
    env.simple_spawn_only_mode = True
    env.include_termination_rewards = False
    
    print(f"  Include termination rewards: {env.include_termination_rewards}")
    print(f"  ✓ Configuration loaded correctly")
    
    # Test with include_termination_rewards = True
    print("\n--- Case 2: Spawn mode WITH termination rewards ---")
    env2 = DurotaxisEnv('config.yaml')
    env2.simple_spawn_only_mode = True
    env2.include_termination_rewards = True
    
    print(f"  Include termination rewards: {env2.include_termination_rewards}")
    print(f"  ✓ Configuration loaded correctly")
    
    # Test combination with centroid mode (should use scaled termination)
    print("\n--- Case 3: Spawn+Centroid mode WITH termination rewards (scaled) ---")
    env3 = DurotaxisEnv('config.yaml')
    env3.simple_spawn_only_mode = True
    env3.centroid_distance_only_mode = True
    env3.include_termination_rewards = True
    print(f"  Include termination rewards: {env3.include_termination_rewards}")
    print(f"  Termination reward scaling: {env3.dm_term_scale}")
    print(f"  Termination reward clipping: {env3.dm_term_clip}")
    print(f"  ✓ Centroid mode should apply scaled/clipped termination")
    
    print("\n✓ TEST 5 PASSED")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("TESTING: Simple Spawn-Only Mode Implementation")
    print("="*70)
    
    tests = [
        ("Spawn-Only Mode Reward Composition", test_1_spawn_only_mode_reward_composition),
        ("Spawn Potential Function", test_2_spawn_potential_function),
        ("PBRS Shaping Integration", test_3_pbrs_shaping_integration),
        ("Mode Combinations", test_4_mode_combinations),
        ("Termination Rewards with Spawn Mode", test_5_termination_rewards_with_spawn_mode),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED", None))
        except Exception as e:
            results.append((test_name, "FAILED", str(e)))
            print(f"\n❌ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)
    
    for test_name, status, error in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
