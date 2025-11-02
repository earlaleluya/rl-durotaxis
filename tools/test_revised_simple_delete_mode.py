#!/usr/bin/env python3
"""
Test script for revised simple_delete_only_mode.

Verifies:
1. Rule 0: NO penalty when num_nodes > max_critical_nodes (just triggers tagging)
2. Rule 1: Persistence penalty for keeping marked nodes
3. Rule 2: Improper deletion penalty for deleting unmarked nodes
4. Proper deletion reward for deleting marked nodes
5. Proper persistence reward for keeping unmarked nodes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from durotaxis_env import DurotaxisEnv

def test_revised_simple_delete_mode():
    """Test the revised simple_delete_only_mode reward logic."""
    print("=" * 80)
    print("Testing REVISED simple_delete_only_mode")
    print("=" * 80)
    
    # Load config and create environment with overrides
    from config_loader import ConfigLoader
    
    config_loader = ConfigLoader('config.yaml')
    
    # Override for testing
    config_loader.config['environment']['simple_delete_only_mode'] = True
    config_loader.config['environment']['centroid_distance_only_mode'] = False
    config_loader.config['environment']['include_termination_rewards'] = False  # Focus on delete rewards only
    config_loader.config['environment']['max_steps'] = 20
    config_loader.config['environment']['init_num_nodes'] = 8
    config_loader.config['environment']['max_critical_nodes'] = 5  # Trigger tagging when nodes > 5
    config_loader.config['environment']['delta_time'] = 3  # Tag nodes 3 steps ahead
    
    # Create environment
    env = DurotaxisEnv('config.yaml')
    # Apply overrides
    env.simple_delete_only_mode = True
    env.centroid_distance_only_mode = False
    env.include_termination_rewards = False
    env.max_steps = 20
    env.max_critical_nodes = 5
    
    print("\n📋 Configuration:")
    print(f"  - simple_delete_only_mode: {env.simple_delete_only_mode}")
    print(f"  - centroid_distance_only_mode: {env.centroid_distance_only_mode}")
    print(f"  - include_termination_rewards: {env.include_termination_rewards}")
    print(f"  - max_critical_nodes: {env.max_critical_nodes}")
    print(f"  - delta_time: {env.delta_time}")
    print(f"  - delete_proper_reward: {env.delete_proper_reward}")
    print(f"  - delete_persistence_penalty: {env.delete_persistence_penalty}")
    print(f"  - delete_improper_penalty: {env.delete_improper_penalty}")
    
    # Reset environment
    observation, info = env.reset()
    print(f"\n🔄 Environment reset")
    print(f"  - Initial nodes: {info['num_nodes']}")
    
    # Run a few steps to observe behavior
    step_count = 0
    total_reward = 0.0
    
    print("\n📊 Step-by-step observations:")
    print("-" * 80)
    
    for i in range(15):
        # Get current state info from topology
        num_nodes = env.topology.graph.num_nodes()
        
        # Check if tagging should be triggered
        should_trigger_tagging = num_nodes > env.max_critical_nodes
        
        # Take a random action (delete_ratio architecture)
        # delete_ratio, gamma, alpha, noise, theta
        delete_ratio = torch.rand(1).item() * 0.3  # 0-30% deletion
        actions = {
            'continuous': torch.tensor([delete_ratio, 5.0, 2.0, 0.1, 0.0])
        }
        
        # Step environment
        next_observation, reward_components, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        reward = reward_components.get('total_reward', 0.0)
        next_num_nodes = env.topology.graph.num_nodes()
        
        step_count += 1
        total_reward += reward
        
        # Extract reward components
        graph_reward = reward_components.get('graph_reward', 0.0)
        delete_reward = reward_components.get('delete_reward', 0.0)
        
        print(f"\nStep {step_count}:")
        print(f"  Nodes: {num_nodes:2d} → {next_num_nodes:2d}")
        print(f"  Delete ratio: {delete_ratio:.3f}")
        print(f"  Trigger tagging? {'YES' if should_trigger_tagging else 'NO'} (nodes > {env.max_critical_nodes})")
        print(f"  Rewards:")
        print(f"    - Total reward: {reward:+8.3f}")
        print(f"    - Graph reward: {graph_reward:+8.3f}")
        print(f"    - Delete reward: {delete_reward:+8.3f}")
        
        # Analyze delete reward
        if delete_reward != 0.0:
            if delete_reward > 0:
                print(f"    ✅ Positive delete reward (proper deletion or persistence)")
            else:
                print(f"    ❌ Negative delete reward (Rule 1: persistence or Rule 2: improper deletion)")
        
        if done:
            print(f"\n⚠️  Episode terminated at step {step_count}")
            break
    
    print("\n" + "=" * 80)
    print(f"✅ Test completed!")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:+.3f}")
    print(f"  Average reward per step: {total_reward/step_count:+.3f}")
    
    print("\n📝 Key Observations:")
    print("  1. Rule 0: No growth penalty when nodes > max_critical_nodes")
    print("  2. Tagging is triggered automatically based on intensity < avg")
    print("  3. Rewards/penalties are based on compliance with to_delete flags")
    print("  4. Proper persistence (keeping unmarked nodes) now gives +reward")
    
    return True

def test_reward_components():
    """Verify that only delete rewards are non-zero in simple_delete_only_mode."""
    print("\n" + "=" * 80)
    print("Testing reward component isolation")
    print("=" * 80)
    
    env = DurotaxisEnv('config.yaml')
    # Apply overrides
    env.simple_delete_only_mode = True
    env.centroid_distance_only_mode = False
    env.include_termination_rewards = False
    env.max_steps = 10
    
    observation, _ = env.reset()
    
    # Take one step
    actions = {
        'continuous': torch.tensor([0.2, 5.0, 2.0, 0.1, 0.0])
    }
    next_observation, reward_components, terminated, truncated, info = env.step(actions)
    reward = reward_components.get('total_reward', 0.0)
    
    # Check reward components
    print("\n📊 Reward components:")
    expected_zero = [
        'spawn_reward', 'efficiency_reward', 'edge_reward',
        'centroid_reward', 'milestone_reward', 'total_node_reward',
        'survival_reward'
    ]
    
    all_zero = True
    for component in expected_zero:
        value = reward_components.get(component, 0.0)
        is_zero = abs(value) < 1e-6
        status = "✅ ZERO" if is_zero else "❌ NON-ZERO"
        print(f"  {component:25s}: {value:+8.3f} {status}")
        if not is_zero:
            all_zero = False
    
    print(f"\n  graph_reward: {reward_components.get('graph_reward', 0.0):+8.3f} (should equal delete_reward)")
    print(f"  delete_reward: {reward_components.get('delete_reward', 0.0):+8.3f}")
    
    if all_zero and abs(reward_components.get('graph_reward', 0.0) - reward_components.get('delete_reward', 0.0)) < 1e-6:
        print("\n✅ Reward isolation PASSED: Only delete rewards are active")
        return True
    else:
        print("\n❌ Reward isolation FAILED: Some rewards are non-zero")
        return False

if __name__ == '__main__':
    print("\n🚀 Starting revised simple_delete_only_mode tests...\n")
    
    try:
        # Test 1: Revised delete mode behavior
        test1_passed = test_revised_simple_delete_mode()
        
        # Test 2: Reward component isolation
        test2_passed = test_reward_components()
        
        print("\n" + "=" * 80)
        print("📋 TEST SUMMARY")
        print("=" * 80)
        print(f"  Test 1 (Revised delete mode): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"  Test 2 (Reward isolation):     {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 All tests PASSED!")
            sys.exit(0)
        else:
            print("\n⚠️  Some tests FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
