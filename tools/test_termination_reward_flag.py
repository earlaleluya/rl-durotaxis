#!/usr/bin/env python3
"""
Test termination reward flag behavior.

This test verifies that:
1. Normal mode (both special modes off): Always includes termination rewards
2. Special mode + include_termination_rewards=False: Excludes termination rewards
3. Special mode + include_termination_rewards=True: Includes termination rewards
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from durotaxis_env import DurotaxisEnv


def test_normal_mode_always_includes_termination():
    """Test that normal mode always includes termination rewards regardless of flag."""
    print("\n" + "="*80)
    print("TEST 1: Normal mode (both special modes OFF) - Always includes termination")
    print("="*80)
    
    # Test with flag=False (should still include termination in normal mode)
    env = DurotaxisEnv('config.yaml', 
                       simple_delete_only_mode=False,
                       centroid_distance_only_mode=False,
                       include_termination_rewards=False)  # Flag ignored in normal mode
    env.reset()
    
    # Force success by moving centroid to goal
    if hasattr(env.topology, 'graph') and env.topology.graph.num_nodes() > 0:
        env.topology.graph.ndata['pos'][:, 0] = env.goal_x  # Move to goal
    
    # Take a step to trigger termination check
    action = {
        'discrete': torch.tensor([0]),  # Spawn action
        'continuous': torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    }
    
    # Take steps until success
    for _ in range(10):
        new_state, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    termination_reward = info['reward_breakdown'].get('termination_reward', 0.0)
    total_reward = info['reward_breakdown']['total_reward']
    
    print(f"Done: {done}")
    print(f"Termination reward: {termination_reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    
    if done and termination_reward != 0:
        print("✅ TEST 1 PASSED: Normal mode includes termination rewards (flag ignored)")
        return True
    else:
        print("❌ TEST 1 FAILED: Normal mode should include termination rewards")
        return False


def test_simple_delete_mode_excludes_by_default():
    """Test that simple_delete_only_mode excludes termination rewards by default."""
    print("\n" + "="*80)
    print("TEST 2: Simple delete mode + include_termination_rewards=False")
    print("="*80)
    
    env = DurotaxisEnv('config.yaml',
                       simple_delete_only_mode=True,
                       centroid_distance_only_mode=False,
                       include_termination_rewards=False)  # Exclude termination
    env.reset()
    
    # Force success by moving centroid to goal
    if hasattr(env.topology, 'graph') and env.topology.graph.num_nodes() > 0:
        env.topology.graph.ndata['pos'][:, 0] = env.goal_x  # Move to goal
    
    # Take steps until success termination
    action = {
        'discrete': torch.tensor([0]),  # Spawn action
        'continuous': torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    }
    
    for _ in range(5):
        new_state, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    termination_reward = info['reward_breakdown'].get('termination_reward', 0.0)
    total_reward = info['reward_breakdown']['total_reward']
    graph_reward = info['reward_breakdown'].get('graph_reward', 0.0)
    
    print(f"Done: {done}")
    print(f"Termination reward: {termination_reward:.2f}")
    print(f"Graph reward: {graph_reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    
    # In simple delete mode with flag=False, total should equal graph_reward only
    if abs(total_reward - graph_reward) < 0.01:  # No termination added
        print("✅ TEST 2 PASSED: Simple delete mode excludes termination by default")
        return True
    else:
        print(f"❌ TEST 2 FAILED: Expected total={graph_reward:.2f}, got {total_reward:.2f}")
        return False


def test_simple_delete_mode_includes_with_flag():
    """Test that simple_delete_only_mode includes termination when flag=True."""
    print("\n" + "="*80)
    print("TEST 3: Simple delete mode + include_termination_rewards=True")
    print("="*80)
    
    env = DurotaxisEnv('config.yaml',
                       simple_delete_only_mode=True,
                       centroid_distance_only_mode=False,
                       include_termination_rewards=True)  # Include termination
    env.reset()
    
    # Force success
    if hasattr(env.topology, 'graph') and env.topology.graph.num_nodes() > 0:
        env.topology.graph.ndata['pos'][:, 0] = env.goal_x
    
    action = {
        'discrete': torch.tensor([0]),
        'continuous': torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    }
    
    for _ in range(5):
        new_state, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    termination_reward = info['reward_breakdown'].get('termination_reward', 0.0)
    total_reward = info['reward_breakdown']['total_reward']
    graph_reward = info['reward_breakdown'].get('graph_reward', 0.0)
    
    print(f"Done: {done}")
    print(f"Termination reward: {termination_reward:.2f}")
    print(f"Graph reward: {graph_reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Expected: graph + termination = {graph_reward + termination_reward:.2f}")
    
    # With flag=True, total should equal graph_reward + termination_reward
    expected_total = graph_reward + termination_reward
    if abs(total_reward - expected_total) < 0.01:
        print("✅ TEST 3 PASSED: Simple delete mode includes termination with flag=True")
        return True
    else:
        print(f"❌ TEST 3 FAILED: Expected {expected_total:.2f}, got {total_reward:.2f}")
        return False


def test_centroid_mode_excludes_by_default():
    """Test that centroid_distance_only_mode excludes termination by default."""
    print("\n" + "="*80)
    print("TEST 4: Centroid distance mode + include_termination_rewards=False")
    print("="*80)
    
    env = DurotaxisEnv('config.yaml',
                       simple_delete_only_mode=False,
                       centroid_distance_only_mode=True,
                       include_termination_rewards=False)  # Exclude termination
    env.reset()
    
    # Get initial distance penalty
    action = {
        'discrete': torch.tensor([0]),
        'continuous': torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    }
    
    new_state, reward, done, truncated, info = env.step(action)
    
    # Take another step to ensure we have distance penalty
    if not done:
        new_state, reward, done, truncated, info = env.step(action)
    
    termination_reward = info['reward_breakdown'].get('termination_reward', 0.0)
    total_reward = info['reward_breakdown']['total_reward']
    graph_reward = info['reward_breakdown'].get('graph_reward', 0.0)
    
    print(f"Done: {done}")
    print(f"Termination reward: {termination_reward:.2f}")
    print(f"Graph reward (distance penalty): {graph_reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    
    # Total should equal graph_reward (distance penalty only)
    if abs(total_reward - graph_reward) < 0.01:
        print("✅ TEST 4 PASSED: Centroid mode excludes termination by default")
        return True
    else:
        print(f"❌ TEST 4 FAILED: Expected total={graph_reward:.2f}, got {total_reward:.2f}")
        return False


def test_centroid_mode_includes_with_flag():
    """Test that centroid_distance_only_mode includes termination when flag=True."""
    print("\n" + "="*80)
    print("TEST 5: Centroid distance mode + include_termination_rewards=True")
    print("="*80)
    
    env = DurotaxisEnv('config.yaml',
                       simple_delete_only_mode=False,
                       centroid_distance_only_mode=True,
                       include_termination_rewards=True)  # Include termination
    env.reset()
    
    # Force success by moving to goal
    if hasattr(env.topology, 'graph') and env.topology.graph.num_nodes() > 0:
        env.topology.graph.ndata['pos'][:, 0] = env.goal_x
    
    action = {
        'discrete': torch.tensor([0]),
        'continuous': torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    }
    
    for _ in range(5):
        new_state, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    termination_reward = info['reward_breakdown'].get('termination_reward', 0.0)
    total_reward = info['reward_breakdown']['total_reward']
    graph_reward = info['reward_breakdown'].get('graph_reward', 0.0)
    
    print(f"Done: {done}")
    print(f"Termination reward: {termination_reward:.2f}")
    print(f"Graph reward (distance penalty): {graph_reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Expected: distance + termination = {graph_reward + termination_reward:.2f}")
    
    # With flag=True, total should equal distance_penalty + termination_reward
    expected_total = graph_reward + termination_reward
    if abs(total_reward - expected_total) < 0.01:
        print("✅ TEST 5 PASSED: Centroid mode includes termination with flag=True")
        return True
    else:
        print(f"❌ TEST 5 FAILED: Expected {expected_total:.2f}, got {total_reward:.2f}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TERMINATION REWARD FLAG TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run all tests
    results.append(test_normal_mode_always_includes_termination())
    results.append(test_simple_delete_mode_excludes_by_default())
    results.append(test_simple_delete_mode_includes_with_flag())
    results.append(test_centroid_mode_excludes_by_default())
    results.append(test_centroid_mode_includes_with_flag())
    
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
        print("  ✅ Normal mode: Always includes termination (flag ignored)")
        print("  ✅ Special modes + flag=False: Excludes termination rewards")
        print("  ✅ Special modes + flag=True: Includes termination rewards")
        print("\nYou now have full control over termination reward inclusion!")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
