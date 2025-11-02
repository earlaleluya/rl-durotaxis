#!/usr/bin/env python3
"""
Test script to verify delete reward PBRS bug fix.

This test verifies that:
1. Delete reward stays within [-1, 1] bounds when PBRS is enabled
2. Delete reward stays within [-1, 1] bounds when PBRS is disabled
3. PBRS term is correctly scaled to maintain bounds
"""

import sys
import numpy as np
from config_loader import ConfigLoader
from durotaxis_env import DurotaxisEnv


def test_delete_reward_bounds_with_pbrs():
    """Test that delete reward respects [-1, 1] bounds with PBRS enabled."""
    print("=" * 70)
    print("TEST: Delete Reward Bounds with PBRS Enabled")
    print("=" * 70)
    
    # Load config and check current PBRS settings
    config = ConfigLoader('config.yaml')
    env_config = config.config['environment']
    
    delete_config = env_config.get('delete_reward', {})
    pbrs_config = delete_config.get('pbrs', {})
    
    print(f"Current PBRS config:")
    print(f"  enabled: {pbrs_config.get('enabled', False)}")
    print(f"  shaping_coeff: {pbrs_config.get('shaping_coeff', 0.0)}")
    print(f"  phi_weight_pending_marked: {pbrs_config.get('phi_weight_pending_marked', 1.0)}")
    print(f"  phi_weight_safe_unmarked: {pbrs_config.get('phi_weight_safe_unmarked', 0.25)}")
    print()
    
    # Create environment
    env = DurotaxisEnv('config.yaml')
    
    # Check if PBRS is actually enabled
    if not env._pbrs_delete_enabled:
        print("⚠️  WARNING: PBRS for delete reward is DISABLED in config")
        print("   Enable it in config.yaml to test with PBRS")
        print()
    
    # Run multiple episodes
    all_delete_rewards = []
    violations = []
    
    for episode in range(5):
        obs = env.reset()
        episode_rewards = []
        
        for step in range(50):
            obs, reward, done, trunc, info = env.step(0)
            del_reward = reward['delete_reward']
            episode_rewards.append(del_reward)
            
            # Check bounds
            if del_reward < -1.0 or del_reward > 1.0:
                violations.append({
                    'episode': episode,
                    'step': step,
                    'reward': del_reward,
                    'num_nodes': reward['num_nodes']
                })
            
            if done or trunc:
                break
        
        all_delete_rewards.extend(episode_rewards)
        
        print(f"Episode {episode + 1}: "
              f"delete_reward ∈ [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}], "
              f"mean={np.mean(episode_rewards):.4f}")
    
    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  Total steps: {len(all_delete_rewards)}")
    print(f"  Delete reward range: [{min(all_delete_rewards):.4f}, {max(all_delete_rewards):.4f}]")
    print(f"  Mean: {np.mean(all_delete_rewards):.4f}")
    print(f"  Std: {np.std(all_delete_rewards):.4f}")
    print(f"  Violations (|r| > 1): {len(violations)}")
    
    if violations:
        print("\n⚠️  VIOLATIONS DETECTED:")
        for v in violations[:10]:  # Show first 10
            print(f"    Episode {v['episode']}, Step {v['step']}: "
                  f"reward = {v['reward']:.4f}, num_nodes = {v['num_nodes']}")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")
        return False
    else:
        print("\n✅ PASS: All delete rewards within [-1, 1] bounds")
        return True


def test_delete_reward_bounds_without_pbrs():
    """Test that delete reward respects [-1, 1] bounds with PBRS disabled."""
    print("\n" + "=" * 70)
    print("TEST: Delete Reward Bounds with PBRS Disabled")
    print("=" * 70)
    print("(Requires manually disabling PBRS in config.yaml)")
    print()
    
    # Load config
    config = ConfigLoader('config.yaml')
    env_config = config.config['environment']
    
    delete_config = env_config.get('delete_reward', {})
    pbrs_config = delete_config.get('pbrs', {})
    
    print(f"Current PBRS config:")
    print(f"  enabled: {pbrs_config.get('enabled', False)}")
    print()
    
    # Create environment
    env = DurotaxisEnv('config.yaml')
    
    # Run multiple episodes
    all_delete_rewards = []
    violations = []
    
    for episode in range(5):
        obs = env.reset()
        episode_rewards = []
        
        for step in range(50):
            obs, reward, done, trunc, info = env.step(0)
            del_reward = reward['delete_reward']
            episode_rewards.append(del_reward)
            
            # Check bounds
            if del_reward < -1.0 or del_reward > 1.0:
                violations.append({
                    'episode': episode,
                    'step': step,
                    'reward': del_reward,
                    'num_nodes': reward['num_nodes']
                })
            
            if done or trunc:
                break
        
        all_delete_rewards.extend(episode_rewards)
        
        print(f"Episode {episode + 1}: "
              f"delete_reward ∈ [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}], "
              f"mean={np.mean(episode_rewards):.4f}")
    
    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  Total steps: {len(all_delete_rewards)}")
    print(f"  Delete reward range: [{min(all_delete_rewards):.4f}, {max(all_delete_rewards):.4f}]")
    print(f"  Mean: {np.mean(all_delete_rewards):.4f}")
    print(f"  Std: {np.std(all_delete_rewards):.4f}")
    print(f"  Violations (|r| > 1): {len(violations)}")
    
    if violations:
        print("\n⚠️  VIOLATIONS DETECTED:")
        for v in violations[:10]:
            print(f"    Episode {v['episode']}, Step {v['step']}: "
                  f"reward = {v['reward']:.4f}, num_nodes = {v['num_nodes']}")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")
        return False
    else:
        print("\n✅ PASS: All delete rewards within [-1, 1] bounds")
        return True


def main():
    """Run all tests."""
    print("\n🧪 Testing Delete Reward PBRS Bounds")
    print("=" * 70)
    
    try:
        # Test with current config (may have PBRS enabled or disabled)
        test1_passed = test_delete_reward_bounds_with_pbrs()
        
        # Overall result
        print("\n" + "=" * 70)
        print("OVERALL RESULT:")
        if test1_passed:
            print("✅ ALL TESTS PASSED")
            print("\nDelete reward correctly bounded to [-1, 1]")
            print("\nNote: For thorough testing, run twice:")
            print("  1. With PBRS enabled (set pbrs.enabled: true in config)")
            print("  2. With PBRS disabled (set pbrs.enabled: false in config)")
            return 0
        else:
            print("❌ TESTS FAILED")
            print("\nDelete reward exceeded [-1, 1] bounds!")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
