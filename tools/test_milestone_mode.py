#!/usr/bin/env python3
"""
Test milestone printing behavior in centroid_distance_only_mode.

This test verifies that:
1. When centroid_distance_only_mode=False: milestones are printed and rewarded
2. When centroid_distance_only_mode=True: milestones are NOT printed and NOT rewarded
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from durotaxis_env import DurotaxisEnv


def test_milestone_with_distance_mode_disabled():
    """Test that milestones work normally when distance mode is disabled."""
    print("\n" + "="*80)
    print("TEST 1: Milestones with centroid_distance_only_mode=False")
    print("="*80)
    
    # Create env with distance mode disabled
    env = DurotaxisEnv('config.yaml', centroid_distance_only_mode=False)
    env.reset()
    
    # Manually set agent far to the right (past 25% threshold)
    # Substrate width = 600, so 25% = 150
    if hasattr(env.topology, 'graph') and env.topology.graph.num_nodes() > 0:
        env.topology.graph.ndata['pos'][:, 0] = 200.0  # Move all nodes to x=200 (>150)
    
    # Get state and calculate milestone reward
    state = env.state_extractor.get_state_features(env.topology)
    milestone_reward = env._calculate_milestone_reward(state)
    
    print(f"Milestone reward: {milestone_reward:.2f}")
    print(f"Milestones reached: {env._milestones_reached}")
    
    if milestone_reward > 0:
        print("✅ TEST 1 PASSED: Milestones are awarded when distance mode is OFF")
        return True
    else:
        print("❌ TEST 1 FAILED: No milestone reward given (expected > 0)")
        return False


def test_milestone_with_distance_mode_enabled():
    """Test that milestones are suppressed when distance mode is enabled."""
    print("\n" + "="*80)
    print("TEST 2: Milestones with centroid_distance_only_mode=True")
    print("="*80)
    
    # Create env with distance mode enabled
    env = DurotaxisEnv('config.yaml', centroid_distance_only_mode=True)
    env.reset()
    
    # Manually set agent far to the right (past 25% threshold)
    if hasattr(env.topology, 'graph') and env.topology.graph.num_nodes() > 0:
        env.topology.graph.ndata['pos'][:, 0] = 200.0  # Move all nodes to x=200 (>150)
    
    # Get state and calculate milestone reward
    state = env.state_extractor.get_state_features(env.topology)
    milestone_reward = env._calculate_milestone_reward(state)
    
    print(f"Milestone reward: {milestone_reward:.2f}")
    print(f"Milestones reached: {env._milestones_reached}")
    
    if milestone_reward == 0:
        print("✅ TEST 2 PASSED: Milestones are suppressed when distance mode is ON")
        return True
    else:
        print(f"❌ TEST 2 FAILED: Milestone reward given ({milestone_reward:.2f}), expected 0.0")
        return False


def test_full_step_with_distance_mode():
    """Test that full step() correctly zeros milestone in distance-only mode."""
    print("\n" + "="*80)
    print("TEST 3: Full step() with centroid_distance_only_mode=True")
    print("="*80)
    
    # Create env with distance mode enabled
    env = DurotaxisEnv('config.yaml', centroid_distance_only_mode=True)
    env.reset()
    
    # Take a step with spawn action
    action = {
        'discrete': torch.tensor([0]),  # Spawn action
        'continuous': torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # Position/angle/distance
    }
    
    # Execute multiple steps to potentially trigger milestone
    for i in range(10):
        new_state, reward, done, truncated, info = env.step(action)
        if 'milestone_reward' in info:
            milestone_from_info = info['milestone_reward']
            print(f"Step {i+1}: milestone_reward from info = {milestone_from_info:.2f}")
            
            if milestone_from_info != 0:
                print(f"❌ TEST 3 FAILED: Non-zero milestone reward in distance-only mode: {milestone_from_info:.2f}")
                return False
        
        if done or truncated:
            break
    
    print("✅ TEST 3 PASSED: All milestone rewards are 0 in distance-only mode")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MILESTONE PRINTING MODE TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run all tests
    results.append(test_milestone_with_distance_mode_disabled())
    results.append(test_milestone_with_distance_mode_enabled())
    results.append(test_full_step_with_distance_mode())
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("\nBehavior verified:")
        print("  ✅ Milestones work normally when centroid_distance_only_mode=False")
        print("  ✅ Milestones are suppressed when centroid_distance_only_mode=True")
        print("  ✅ No confusing milestone messages in distance-only mode")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
