#!/usr/bin/env python3
"""
Test the simple_delete_only_mode flag to ensure:
0. Growth penalty is applied when num_nodes > max_critical_nodes (Rule 0)
1. Only delete penalties are computed (Rule 1 & 2)
2. All other rewards are zeroed out
3. No positive rewards are given (proper deletions give 0)
"""
import os
import sys
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from durotaxis_env import DurotaxisEnv
from state import TopologyState


def test_simple_delete_mode():
    print("=" * 80)
    print("SIMPLE DELETE-ONLY MODE TEST")
    print("=" * 80)
    
    # Test 1: Normal mode (flag disabled)
    print("\n" + "-" * 80)
    print("TEST 1: Normal Mode (simple_delete_only_mode=False)")
    print("-" * 80)
    
    env = DurotaxisEnv(config_path="config.yaml", substrate_type='linear', simple_delete_only_mode=False)
    env.reset()
    
    state_extractor = TopologyState(env.topology)
    
    # Ensure we have at least 3 nodes
    while env.topology.graph.num_nodes() < 3:
        env.topology.spawn(0, gamma=5.0, alpha=0.0, noise=0.0, theta=0.0)
    
    print(f"Initial setup: {env.topology.graph.num_nodes()} nodes")
    print(f"Simple delete-only mode: {env.simple_delete_only_mode}")
    
    # Mark node 0 for deletion but don't delete (persistence penalty - Rule 1)
    env.topology.graph.ndata['to_delete'][0] = 1.0
    
    prev_state = state_extractor.get_state_features(
        include_substrate=True, node_age=env._node_age, node_stagnation=env._node_stagnation
    )
    
    # Don't delete, just take a step
    new_state = state_extractor.get_state_features(
        include_substrate=True, node_age=env._node_age, node_stagnation=env._node_stagnation
    )
    
    # Calculate rewards
    reward_breakdown = env._calculate_reward(prev_state, new_state, [])
    
    print(f"\nReward Breakdown (Normal Mode):")
    print(f"  Total Reward: {reward_breakdown['total_reward']:.4f}")
    print(f"  Graph Reward: {reward_breakdown['graph_reward']:.4f}")
    print(f"  Delete Reward: {reward_breakdown['delete_reward']:.4f}")
    print(f"  Spawn Reward: {reward_breakdown['spawn_reward']:.4f}")
    print(f"  Edge Reward: {reward_breakdown['edge_reward']:.4f}")
    print(f"  Milestone Reward: {reward_breakdown['milestone_reward']:.4f}")
    print(f"  Survival Reward: {reward_breakdown['survival_reward']:.4f}")
    print(f"  Total Node Reward: {reward_breakdown['total_node_reward']:.4f}")
    
    expected_delete_penalty = -env.delete_persistence_penalty
    print(f"\nExpected delete penalty: {expected_delete_penalty:.4f}")
    print(f"Actual delete reward: {reward_breakdown['delete_reward']:.4f}")
    
    if abs(reward_breakdown['delete_reward'] - expected_delete_penalty) < 0.01:
        print("✓ Delete penalty matches expected value")
    else:
        print("✗ Delete penalty does NOT match expected value")
    
    # In normal mode, other rewards should NOT be zero
    if reward_breakdown['survival_reward'] != 0.0:
        print("✓ Survival reward is non-zero (normal mode)")
    else:
        print("✗ Survival reward is zero (unexpected in normal mode)")
    
    # Test 2: Simple delete-only mode (flag enabled)
    print("\n" + "-" * 80)
    print("TEST 2: Simple Delete-Only Mode (simple_delete_only_mode=True)")
    print("-" * 80)
    
    env2 = DurotaxisEnv(config_path="config.yaml", substrate_type='linear', simple_delete_only_mode=True)
    env2.reset()
    
    state_extractor2 = TopologyState(env2.topology)
    
    # Ensure we have at least 3 nodes
    while env2.topology.graph.num_nodes() < 3:
        env2.topology.spawn(0, gamma=5.0, alpha=0.0, noise=0.0, theta=0.0)
    
    print(f"Initial setup: {env2.topology.graph.num_nodes()} nodes")
    print(f"Simple delete-only mode: {env2.simple_delete_only_mode}")
    
    # Mark node 0 for deletion but don't delete (persistence penalty - Rule 1)
    env2.topology.graph.ndata['to_delete'][0] = 1.0
    
    prev_state2 = state_extractor2.get_state_features(
        include_substrate=True, node_age=env2._node_age, node_stagnation=env2._node_stagnation
    )
    
    # Don't delete, just take a step
    new_state2 = state_extractor2.get_state_features(
        include_substrate=True, node_age=env2._node_age, node_stagnation=env2._node_stagnation
    )
    
    # Calculate rewards
    reward_breakdown2 = env2._calculate_reward(prev_state2, new_state2, [])
    
    print(f"\nReward Breakdown (Simple Delete-Only Mode):")
    print(f"  Total Reward: {reward_breakdown2['total_reward']:.4f}")
    print(f"  Graph Reward: {reward_breakdown2['graph_reward']:.4f}")
    print(f"  Delete Reward: {reward_breakdown2['delete_reward']:.4f}")
    print(f"  Spawn Reward: {reward_breakdown2['spawn_reward']:.4f}")
    print(f"  Edge Reward: {reward_breakdown2['edge_reward']:.4f}")
    print(f"  Milestone Reward: {reward_breakdown2['milestone_reward']:.4f}")
    print(f"  Survival Reward: {reward_breakdown2['survival_reward']:.4f}")
    print(f"  Total Node Reward: {reward_breakdown2['total_node_reward']:.4f}")
    
    expected_delete_penalty2 = -env2.delete_persistence_penalty
    print(f"\nExpected delete penalty: {expected_delete_penalty2:.4f}")
    print(f"Actual total reward: {reward_breakdown2['total_reward']:.4f}")
    
    # Verify all rewards except delete are zero
    print("\nVerifying Simple Delete-Only Mode:")
    
    if abs(reward_breakdown2['spawn_reward']) < 0.01:
        print("✓ Spawn reward is zero")
    else:
        print(f"✗ Spawn reward is NOT zero: {reward_breakdown2['spawn_reward']:.4f}")
    
    if abs(reward_breakdown2['edge_reward']) < 0.01:
        print("✓ Edge reward is zero")
    else:
        print(f"✗ Edge reward is NOT zero: {reward_breakdown2['edge_reward']:.4f}")
    
    if abs(reward_breakdown2['milestone_reward']) < 0.01:
        print("✓ Milestone reward is zero")
    else:
        print(f"✗ Milestone reward is NOT zero: {reward_breakdown2['milestone_reward']:.4f}")
    
    if abs(reward_breakdown2['survival_reward']) < 0.01:
        print("✓ Survival reward is zero")
    else:
        print(f"✗ Survival reward is NOT zero: {reward_breakdown2['survival_reward']:.4f}")
    
    if abs(reward_breakdown2['total_node_reward']) < 0.01:
        print("✓ Total node reward is zero")
    else:
        print(f"✗ Total node reward is NOT zero: {reward_breakdown2['total_node_reward']:.4f}")
    
    # Total reward should equal delete penalty
    if abs(reward_breakdown2['total_reward'] - expected_delete_penalty2) < 0.01:
        print(f"✓ Total reward equals delete penalty: {reward_breakdown2['total_reward']:.4f}")
    else:
        print(f"✗ Total reward does NOT equal delete penalty")
        print(f"  Expected: {expected_delete_penalty2:.4f}")
        print(f"  Actual: {reward_breakdown2['total_reward']:.4f}")
    
    # Test 3: No positive rewards in simple mode
    print("\n" + "-" * 80)
    print("TEST 3: No Positive Rewards for Proper Deletion (simple mode)")
    print("-" * 80)
    
    env3 = DurotaxisEnv(config_path="config.yaml", substrate_type='linear', simple_delete_only_mode=True)
    env3.reset()
    
    state_extractor3 = TopologyState(env3.topology)
    
    # Ensure we have at least 3 nodes
    while env3.topology.graph.num_nodes() < 3:
        env3.topology.spawn(0, gamma=5.0, alpha=0.0, noise=0.0, theta=0.0)
    
    # Mark node 0 for deletion AND delete it (proper deletion)
    env3.topology.graph.ndata['to_delete'][0] = 1.0
    
    prev_state3 = state_extractor3.get_state_features(
        include_substrate=True, node_age=env3._node_age, node_stagnation=env3._node_stagnation
    )
    
    # Delete node 0
    env3.topology.delete(0)
    
    new_state3 = state_extractor3.get_state_features(
        include_substrate=True, node_age=env3._node_age, node_stagnation=env3._node_stagnation
    )
    
    # Calculate rewards
    reward_breakdown3 = env3._calculate_reward(prev_state3, new_state3, [])
    
    print(f"Proper deletion in simple mode:")
    print(f"  Delete Reward: {reward_breakdown3['delete_reward']:.4f}")
    print(f"  Total Reward: {reward_breakdown3['total_reward']:.4f}")
    
    # In simple mode, proper deletion should give 0 (not positive reward)
    if abs(reward_breakdown3['delete_reward']) < 0.01:
        print("✓ Proper deletion gives 0 reward (no positive reward)")
    else:
        print(f"✗ Proper deletion gives non-zero: {reward_breakdown3['delete_reward']:.4f}")
    
    if abs(reward_breakdown3['total_reward']) < 0.01:
        print("✓ Total reward is 0 for proper deletion")
    else:
        print(f"✗ Total reward is NOT 0: {reward_breakdown3['total_reward']:.4f}")
    
    # Test 4: Improper deletion penalty (Rule 2)
    print("\n" + "-" * 80)
    print("TEST 4: Improper Deletion Penalty (Rule 2 - simple mode)")
    print("-" * 80)
    
    env4 = DurotaxisEnv(config_path="config.yaml", substrate_type='linear', simple_delete_only_mode=True)
    env4.reset()
    
    state_extractor4 = TopologyState(env4.topology)
    
    # Ensure we have at least 3 nodes
    while env4.topology.graph.num_nodes() < 3:
        env4.topology.spawn(0, gamma=5.0, alpha=0.0, noise=0.0, theta=0.0)
    
    # Do NOT mark node 0 for deletion
    env4.topology.graph.ndata['to_delete'][0] = 0.0
    
    prev_state4 = state_extractor4.get_state_features(
        include_substrate=True, node_age=env4._node_age, node_stagnation=env4._node_stagnation
    )
    
    # Delete node 0 anyway (improper deletion)
    env4.topology.delete(0)
    
    new_state4 = state_extractor4.get_state_features(
        include_substrate=True, node_age=env4._node_age, node_stagnation=env4._node_stagnation
    )
    
    # Calculate rewards
    reward_breakdown4 = env4._calculate_reward(prev_state4, new_state4, [])
    
    expected_improper_penalty = -env4.delete_improper_penalty
    print(f"Improper deletion (NOT marked but deleted):")
    print(f"  Expected penalty: {expected_improper_penalty:.4f}")
    print(f"  Delete Reward: {reward_breakdown4['delete_reward']:.4f}")
    print(f"  Total Reward: {reward_breakdown4['total_reward']:.4f}")
    
    if abs(reward_breakdown4['delete_reward'] - expected_improper_penalty) < 0.01:
        print("✓ Improper deletion penalty matches expected (Rule 2)")
    else:
        print(f"✗ Improper deletion penalty does NOT match")
    
    if abs(reward_breakdown4['total_reward'] - expected_improper_penalty) < 0.01:
        print("✓ Total reward equals improper deletion penalty")
    else:
        print(f"✗ Total reward does NOT equal improper deletion penalty")
    
    # Test 5: Growth penalty (Rule 0)
    print("\n" + "-" * 80)
    print("TEST 5: Growth Penalty (Rule 0 - simple mode)")
    print("-" * 80)
    
    env5 = DurotaxisEnv(config_path="config.yaml", substrate_type='linear', simple_delete_only_mode=True)
    env5.reset()
    
    state_extractor5 = TopologyState(env5.topology)
    
    # Spawn many nodes to exceed max_critical_nodes
    max_critical = env5.max_critical_nodes
    print(f"Max critical nodes: {max_critical}")
    
    # Spawn nodes until we exceed the limit
    while env5.topology.graph.num_nodes() <= max_critical:
        env5.topology.spawn(0, gamma=5.0, alpha=0.0, noise=0.0, theta=0.0)
    
    print(f"Current nodes: {env5.topology.graph.num_nodes()}")
    
    prev_state5 = state_extractor5.get_state_features(
        include_substrate=True, node_age=env5._node_age, node_stagnation=env5._node_stagnation
    )
    
    # Don't delete anything, just check growth penalty
    new_state5 = state_extractor5.get_state_features(
        include_substrate=True, node_age=env5._node_age, node_stagnation=env5._node_stagnation
    )
    
    # Calculate rewards
    reward_breakdown5 = env5._calculate_reward(prev_state5, new_state5, [])
    
    # Calculate expected growth penalty
    num_nodes = env5.topology.graph.num_nodes()
    excess_nodes = num_nodes - max_critical
    expected_growth_penalty = -env5.growth_penalty * (1 + excess_nodes / max_critical)
    
    print(f"Growth penalty (Rule 0):")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Max critical: {max_critical}")
    print(f"  Excess nodes: {excess_nodes}")
    print(f"  Expected growth penalty: {expected_growth_penalty:.4f}")
    print(f"  Total Reward: {reward_breakdown5['total_reward']:.4f}")
    print(f"  Graph Reward: {reward_breakdown5['graph_reward']:.4f}")
    
    if abs(reward_breakdown5['total_reward'] - expected_growth_penalty) < 0.01:
        print("✓ Growth penalty matches expected (Rule 0)")
    else:
        print(f"✗ Growth penalty does NOT match")
        print(f"  Expected: {expected_growth_penalty:.4f}")
        print(f"  Actual: {reward_breakdown5['total_reward']:.4f}")
    
    # Test 6: Combined penalties (Rule 0 + Rule 1 + Rule 2)
    print("\n" + "-" * 80)
    print("TEST 6: Combined Penalties (Rules 0 + 1 + 2)")
    print("-" * 80)
    
    env6 = DurotaxisEnv(config_path="config.yaml", substrate_type='linear', simple_delete_only_mode=True)
    env6.reset()
    
    state_extractor6 = TopologyState(env6.topology)
    
    # Spawn enough nodes to exceed max_critical_nodes
    max_critical6 = env6.max_critical_nodes
    while env6.topology.graph.num_nodes() <= max_critical6 + 2:
        env6.topology.spawn(0, gamma=5.0, alpha=0.0, noise=0.0, theta=0.0)
    
    # Mark node 0 for deletion but don't delete (Rule 1)
    env6.topology.graph.ndata['to_delete'][0] = 1.0
    # Don't mark node 1 but we'll delete it (Rule 2)
    env6.topology.graph.ndata['to_delete'][1] = 0.0
    
    prev_state6 = state_extractor6.get_state_features(
        include_substrate=True, node_age=env6._node_age, node_stagnation=env6._node_stagnation
    )
    
    # Delete node 1 (improper deletion - Rule 2)
    env6.topology.delete(1)
    
    new_state6 = state_extractor6.get_state_features(
        include_substrate=True, node_age=env6._node_age, node_stagnation=env6._node_stagnation
    )
    
    # Calculate rewards
    reward_breakdown6 = env6._calculate_reward(prev_state6, new_state6, [])
    
    # Calculate expected penalties
    num_nodes6 = env6.topology.graph.num_nodes()
    excess_nodes6 = num_nodes6 - max_critical6
    expected_growth6 = -env6.growth_penalty * (1 + excess_nodes6 / max_critical6)
    expected_persistence6 = -env6.delete_persistence_penalty  # Rule 1
    expected_improper6 = -env6.delete_improper_penalty  # Rule 2
    expected_total6 = expected_growth6 + expected_persistence6 + expected_improper6
    
    print(f"Combined penalties:")
    print(f"  Rule 0 (growth): {expected_growth6:.4f}")
    print(f"  Rule 1 (persistence): {expected_persistence6:.4f}")
    print(f"  Rule 2 (improper): {expected_improper6:.4f}")
    print(f"  Expected total: {expected_total6:.4f}")
    print(f"  Actual total: {reward_breakdown6['total_reward']:.4f}")
    
    if abs(reward_breakdown6['total_reward'] - expected_total6) < 0.01:
        print("✓ Combined penalties match expected")
    else:
        print(f"✗ Combined penalties do NOT match")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_simple_delete_mode()
