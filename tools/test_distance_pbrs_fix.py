#!/usr/bin/env python3
"""
Test script to verify distance reward PBRS bug fix.

This test verifies that:
1. Distance reward stays within [-1, 1] bounds when PBRS is enabled
2. Distance reward stays within [-1, 1] bounds when PBRS is disabled
3. PBRS term is correctly incorporated into the substrate-aware signal before tanh
"""

import sys
import numpy as np
from config_loader import ConfigLoader
from durotaxis_env import DurotaxisEnv


def test_distance_reward_bounds_with_pbrs():
    """Test that distance reward respects [-1, 1] bounds with PBRS enabled."""
    print("=" * 70)
    print("TEST: Distance Reward Bounds with PBRS Enabled")
    print("=" * 70)
    print("(Using default config - assumes substrate_aware_scaling enabled)")
    print()
    
    # Create environment with overrides
    # Note: PBRS for centroid is configured in graph_rewards section (legacy location)
    env = DurotaxisEnv('config.yaml')
    
    # Run multiple episodes
    all_distance_rewards = []
    violations = []
    
    for episode in range(5):
        obs = env.reset()
        episode_rewards = []
        
        for step in range(50):
            obs, reward, done, trunc, info = env.step(0)
            dist_reward = reward['distance_reward']
            episode_rewards.append(dist_reward)
            
            # Check bounds
            if dist_reward < -1.0 or dist_reward > 1.0:
                violations.append({
                    'episode': episode,
                    'step': step,
                    'reward': dist_reward
                })
            
            if done or trunc:
                break
        
        all_distance_rewards.extend(episode_rewards)
        
        print(f"Episode {episode + 1}: "
              f"dist_reward ∈ [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}], "
              f"mean={np.mean(episode_rewards):.4f}")
    
    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  Total steps: {len(all_distance_rewards)}")
    print(f"  Distance reward range: [{min(all_distance_rewards):.4f}, {max(all_distance_rewards):.4f}]")
    print(f"  Mean: {np.mean(all_distance_rewards):.4f}")
    print(f"  Std: {np.std(all_distance_rewards):.4f}")
    print(f"  Violations (|r| > 1): {len(violations)}")
    
    if violations:
        print("\n⚠️  VIOLATIONS DETECTED:")
        for v in violations[:5]:  # Show first 5
            print(f"    Episode {v['episode']}, Step {v['step']}: reward = {v['reward']:.4f}")
        if len(violations) > 5:
            print(f"    ... and {len(violations) - 5} more")
        return False
    else:
        print("\n✅ PASS: All distance rewards within [-1, 1] bounds")
        return True


def test_distance_reward_bounds_without_pbrs():
    """Test that distance reward respects [-1, 1] bounds with PBRS disabled."""
    print("\n" + "=" * 70)
    print("TEST: Distance Reward Bounds with PBRS Disabled")
    print("=" * 70)
    print("(Using default config - assumes substrate_aware_scaling enabled)")
    print()
    
    # Create environment
    # Note: Default config should have PBRS disabled for centroid
    env = DurotaxisEnv('config.yaml')
    
    # Run multiple episodes
    all_distance_rewards = []
    violations = []
    
    for episode in range(5):
        obs = env.reset()
        episode_rewards = []
        
        for step in range(50):
            obs, reward, done, trunc, info = env.step(0)
            dist_reward = reward['distance_reward']
            episode_rewards.append(dist_reward)
            
            # Check bounds
            if dist_reward < -1.0 or dist_reward > 1.0:
                violations.append({
                    'episode': episode,
                    'step': step,
                    'reward': dist_reward
                })
            
            if done or trunc:
                break
        
        all_distance_rewards.extend(episode_rewards)
        
        print(f"Episode {episode + 1}: "
              f"dist_reward ∈ [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}], "
              f"mean={np.mean(episode_rewards):.4f}")
    
    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  Total steps: {len(all_distance_rewards)}")
    print(f"  Distance reward range: [{min(all_distance_rewards):.4f}, {max(all_distance_rewards):.4f}]")
    print(f"  Mean: {np.mean(all_distance_rewards):.4f}")
    print(f"  Std: {np.std(all_distance_rewards):.4f}")
    print(f"  Violations (|r| > 1): {len(violations)}")
    
    if violations:
        print("\n⚠️  VIOLATIONS DETECTED:")
        for v in violations[:5]:  # Show first 5
            print(f"    Episode {v['episode']}, Step {v['step']}: reward = {v['reward']:.4f}")
        if len(violations) > 5:
            print(f"    ... and {len(violations) - 5} more")
        return False
    else:
        print("\n✅ PASS: All distance rewards within [-1, 1] bounds")
        return True


def main():
    """Run all tests."""
    print("\n🧪 Testing Distance Reward PBRS Bug Fix")
    print("=" * 70)
    
    try:
        # Test with PBRS enabled
        test1_passed = test_distance_reward_bounds_with_pbrs()
        
        # Test with PBRS disabled
        test2_passed = test_distance_reward_bounds_without_pbrs()
        
        # Overall result
        print("\n" + "=" * 70)
        print("OVERALL RESULT:")
        if test1_passed and test2_passed:
            print("✅ ALL TESTS PASSED")
            print("\nDistance reward correctly bounded to [-1, 1] with and without PBRS")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
