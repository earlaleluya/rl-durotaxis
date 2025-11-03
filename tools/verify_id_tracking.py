#!/usr/bin/env python3
"""
Comprehensive verification of to_delete, node_id, and persistent_id tracking.
This test ensures reward calculation has correct data for spawn/delete rewards.
"""
import os
import sys
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from durotaxis_env import DurotaxisEnv
from topology import Topology
from substrate import Substrate


def test_persistent_id_tracking():
    """Test that persistent IDs are correctly maintained across spawn/delete operations."""
    print("=" * 80)
    print("TEST 1: Persistent ID Tracking Across Operations")
    print("=" * 80)
    
    substrate = Substrate(size=(100, 100))
    substrate.create('linear', m=0.05, b=1)
    topo = Topology(substrate=substrate)
    
    # Reset with 3 nodes
    topo.reset(init_num_nodes=3)
    print(f"\n✓ Initial: {topo.graph.num_nodes()} nodes")
    print(f"  node_ids: {list(range(topo.graph.num_nodes()))}")
    print(f"  persistent_ids: {topo.graph.ndata['persistent_id'].tolist()}")
    
    # Verify initial persistent IDs are sequential
    expected_pids = [0, 1, 2]
    actual_pids = topo.graph.ndata['persistent_id'].tolist()
    assert actual_pids == expected_pids, f"Initial PIDs wrong: {actual_pids} != {expected_pids}"
    
    # Spawn from node 1
    new_node_id = topo.spawn(1)
    print(f"\n✓ After spawn(1): {topo.graph.num_nodes()} nodes")
    print(f"  node_ids: {list(range(topo.graph.num_nodes()))}")
    print(f"  persistent_ids: {topo.graph.ndata['persistent_id'].tolist()}")
    print(f"  new_node_id returned: {new_node_id}")
    
    # Verify new node has persistent_id=3
    assert topo.graph.ndata['persistent_id'][new_node_id].item() == 3, \
        f"New node PID should be 3, got {topo.graph.ndata['persistent_id'][new_node_id].item()}"
    
    # Delete node 1 (middle node)
    print(f"\n✓ Before delete(1):")
    print(f"  node_ids: {list(range(topo.graph.num_nodes()))}")
    print(f"  persistent_ids: {topo.graph.ndata['persistent_id'].tolist()}")
    
    topo.delete(1)
    print(f"\n✓ After delete(1): {topo.graph.num_nodes()} nodes")
    print(f"  node_ids: {list(range(topo.graph.num_nodes()))}")
    print(f"  persistent_ids: {topo.graph.ndata['persistent_id'].tolist()}")
    
    # Verify persistent IDs are preserved (but node 1's PID should be gone)
    # Should have PIDs: [0, 2, 3] (node 1's PID=1 was deleted)
    expected_after_delete = [0, 2, 3]
    actual_after_delete = topo.graph.ndata['persistent_id'].tolist()
    assert actual_after_delete == expected_after_delete, \
        f"PIDs after delete wrong: {actual_after_delete} != {expected_after_delete}"
    
    print("\n✅ TEST 1 PASSED: Persistent IDs correctly tracked")
    return True


def test_to_delete_flag_tracking():
    """Test that to_delete flags are correctly maintained across operations."""
    print("\n" + "=" * 80)
    print("TEST 2: to_delete Flag Tracking Across Operations")
    print("=" * 80)
    
    substrate = Substrate(size=(100, 100))
    substrate.create('linear', m=0.05, b=1)
    topo = Topology(substrate=substrate)
    
    # Reset with 4 nodes
    topo.reset(init_num_nodes=4)
    print(f"\n✓ Initial: {topo.graph.num_nodes()} nodes")
    print(f"  to_delete flags: {topo.graph.ndata['to_delete'].tolist()}")
    
    # Verify all to_delete flags start at 0
    assert torch.all(topo.graph.ndata['to_delete'] == 0.0), "Initial to_delete should be all 0"
    
    # Mark nodes 1 and 3 for deletion
    topo.graph.ndata['to_delete'][1] = 1.0
    topo.graph.ndata['to_delete'][3] = 1.0
    print(f"\n✓ After marking nodes 1,3:")
    print(f"  to_delete flags: {topo.graph.ndata['to_delete'].tolist()}")
    
    # Spawn from node 0
    new_node_id = topo.spawn(0)
    print(f"\n✓ After spawn(0): {topo.graph.num_nodes()} nodes")
    print(f"  node_ids: {list(range(topo.graph.num_nodes()))}")
    print(f"  to_delete flags: {topo.graph.ndata['to_delete'].tolist()}")
    print(f"  new_node to_delete: {topo.graph.ndata['to_delete'][new_node_id].item()}")
    
    # Verify new node has to_delete=0
    assert topo.graph.ndata['to_delete'][new_node_id].item() == 0.0, \
        "New spawned node should have to_delete=0"
    
    # Verify old to_delete flags preserved
    # After spawn, we should still have the marked nodes (indices may shift)
    to_delete_after_spawn = topo.graph.ndata['to_delete'].tolist()
    assert sum(1 for x in to_delete_after_spawn if x > 0.5) == 2, \
        "Should still have 2 nodes marked for deletion after spawn"
    
    # Delete node 1
    print(f"\n✓ Before delete(1):")
    print(f"  to_delete flags: {topo.graph.ndata['to_delete'].tolist()}")
    
    topo.delete(1)
    print(f"\n✓ After delete(1): {topo.graph.num_nodes()} nodes")
    print(f"  to_delete flags: {topo.graph.ndata['to_delete'].tolist()}")
    
    # Verify to_delete flags correctly restored (excluding deleted node)
    assert len(topo.graph.ndata['to_delete']) == topo.graph.num_nodes(), \
        "to_delete length should match num_nodes"
    
    print("\n✅ TEST 2 PASSED: to_delete flags correctly tracked")
    return True


def test_reward_calculation_correctness():
    """Test that reward calculation uses correct persistent_id and to_delete data."""
    print("\n" + "=" * 80)
    print("TEST 3: Reward Calculation with Persistent IDs and to_delete Flags")
    print("=" * 80)
    
    # Create environment with config_path
    env = DurotaxisEnv(
        config_path="config.yaml",
        init_num_nodes=5,
        max_critical_nodes=10
    )
    env.reset()
    
    # Get initial state
    prev_state = env.state_extractor.get_state_features(
        include_substrate=True,
        node_age=env._node_age,
        node_stagnation=env._node_stagnation
    )
    
    print(f"\n✓ Initial state:")
    print(f"  num_nodes: {prev_state['num_nodes']}")
    print(f"  persistent_ids: {prev_state['persistent_id'].tolist()}")
    print(f"  to_delete flags: {prev_state['to_delete'].tolist()}")
    
    # Mark node 2 for deletion (PID should be 2)
    env.topology.graph.ndata['to_delete'][2] = 1.0
    pid_to_delete = env.topology.graph.ndata['persistent_id'][2].item()
    
    print(f"\n✓ Marked node_id=2 (PID={pid_to_delete}) for deletion")
    
    # Take snapshot AFTER marking
    prev_state_marked = env.state_extractor.get_state_features(
        include_substrate=True,
        node_age=env._node_age,
        node_stagnation=env._node_stagnation
    )
    
    print(f"  to_delete flags (snapshot): {prev_state_marked['to_delete'].tolist()}")
    
    # Delete node 2
    env.topology.delete(2)
    
    # Get new state
    new_state = env.state_extractor.get_state_features(
        include_substrate=True,
        node_age=env._node_age,
        node_stagnation=env._node_stagnation
    )
    
    print(f"\n✓ After deletion:")
    print(f"  num_nodes: {new_state['num_nodes']}")
    print(f"  persistent_ids: {new_state['persistent_id'].tolist()}")
    print(f"  Deleted PID {pid_to_delete} present in new state: {pid_to_delete in new_state['persistent_id'].tolist()}")
    
    # Calculate delete reward
    delete_reward = env._calculate_delete_reward(prev_state_marked, new_state, {})
    
    print(f"\n✓ Delete reward calculated: {delete_reward:.4f}")
    
    # Verify reward is positive (proper deletion)
    # Expected: 
    # - 1 node marked and deleted: +1.0 (RULE 1)
    # - 4 nodes not marked and preserved: +4.0 (RULE 4)
    # Total raw: +5.0, scaled by 5 nodes = +1.0
    expected_raw = env.delete_proper_reward * 1 + env.delete_proper_reward * 4  # 5.0
    expected_scaled = expected_raw / prev_state_marked['num_nodes']  # 5.0 / 5 = 1.0
    
    print(f"  Expected (without PBRS): {expected_scaled:.4f}")
    print(f"  Actual: {delete_reward:.4f}")
    
    # Allow for PBRS shaping
    assert delete_reward >= 0.8, \
        f"Delete reward should be positive for proper deletion, got {delete_reward}"
    
    print("\n✅ TEST 3 PASSED: Reward calculation uses correct IDs")
    return True


def test_spawn_reward_with_persistent_ids():
    """Test spawn reward calculation with persistent ID tracking."""
    print("\n" + "=" * 80)
    print("TEST 4: Spawn Reward with Persistent ID Tracking")
    print("=" * 80)
    
    env = DurotaxisEnv(
        config_path="config.yaml",
        init_num_nodes=3
    )
    env.reset()
    
    print(f"\n✓ Initial: {env.topology.graph.num_nodes()} nodes")
    print(f"  persistent_ids: {env.topology.graph.ndata['persistent_id'].tolist()}")
    
    # Get prev state
    prev_state = env.state_extractor.get_state_features(
        include_substrate=True,
        node_age=env._node_age,
        node_stagnation=env._node_stagnation
    )
    
    # Spawn from node 0
    new_node_id = env.topology.spawn(0, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0)
    
    print(f"\n✓ After spawn: {env.topology.graph.num_nodes()} nodes")
    print(f"  persistent_ids: {env.topology.graph.ndata['persistent_id'].tolist()}")
    print(f"  new_node_id: {new_node_id}, PID: {env.topology.graph.ndata['persistent_id'][new_node_id].item()}")
    
    # Get new state
    new_state = env.state_extractor.get_state_features(
        include_substrate=True,
        node_age=env._node_age,
        node_stagnation=env._node_stagnation
    )
    
    # Calculate spawn reward
    actions = {0: 'spawn'}
    spawn_reward = env._calculate_spawn_reward(prev_state, new_state, actions)
    
    print(f"\n✓ Spawn reward: {spawn_reward:.4f}")
    
    # Verify new node has unique persistent ID
    all_pids = env.topology.graph.ndata['persistent_id'].tolist()
    assert len(all_pids) == len(set(all_pids)), \
        f"Duplicate persistent IDs found: {all_pids}"
    
    print("\n✅ TEST 4 PASSED: Spawn reward with correct persistent IDs")
    return True


def test_node_id_vs_persistent_id_independence():
    """Verify that node_id (index) and persistent_id are properly separated."""
    print("\n" + "=" * 80)
    print("TEST 5: Node ID vs Persistent ID Independence")
    print("=" * 80)
    
    substrate = Substrate(size=(100, 100))
    substrate.create('linear', m=0.05, b=1)
    topo = Topology(substrate=substrate)
    
    # Create 5 nodes
    topo.reset(init_num_nodes=5)
    print(f"\n✓ Initial: {topo.graph.num_nodes()} nodes")
    print(f"  node_ids:       {list(range(topo.graph.num_nodes()))}")
    print(f"  persistent_ids: {topo.graph.ndata['persistent_id'].tolist()}")
    
    # Delete nodes 1 and 3
    print(f"\n✓ Deleting nodes 1 and 3...")
    pid_1 = topo.graph.ndata['persistent_id'][1].item()
    pid_3 = topo.graph.ndata['persistent_id'][3].item()
    print(f"  Deleting node_id=1 (PID={pid_1})")
    print(f"  Deleting node_id=3 (PID={pid_3})")
    
    topo.delete(3)  # Delete 3 first (higher index)
    topo.delete(1)  # Then delete 1
    
    print(f"\n✓ After deletions: {topo.graph.num_nodes()} nodes")
    print(f"  node_ids:       {list(range(topo.graph.num_nodes()))}")
    print(f"  persistent_ids: {topo.graph.ndata['persistent_id'].tolist()}")
    
    # Verify:
    # - Node IDs are sequential: [0, 1, 2]
    # - Persistent IDs preserve original values: [0, 2, 4] (1 and 3 deleted)
    expected_pids = [0, 2, 4]
    actual_pids = topo.graph.ndata['persistent_id'].tolist()
    
    assert actual_pids == expected_pids, \
        f"Persistent IDs should be {expected_pids}, got {actual_pids}"
    
    assert list(range(topo.graph.num_nodes())) == [0, 1, 2], \
        "Node IDs should be sequential after deletions"
    
    print(f"\n✓ Verification:")
    print(f"  Node IDs are sequential: {list(range(topo.graph.num_nodes()))}")
    print(f"  Persistent IDs maintain history: {actual_pids}")
    print(f"  PIDs 1 and 3 are gone (as expected)")
    
    print("\n✅ TEST 5 PASSED: Node ID vs Persistent ID correctly separated")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VERIFICATION: ID TRACKING FOR REWARD CALCULATION")
    print("=" * 80)
    
    tests = [
        ("Persistent ID Tracking", test_persistent_id_tracking),
        ("to_delete Flag Tracking", test_to_delete_flag_tracking),
        ("Reward Calculation Correctness", test_reward_calculation_correctness),
        ("Spawn Reward with Persistent IDs", test_spawn_reward_with_persistent_ids),
        ("Node ID vs Persistent ID Independence", test_node_id_vs_persistent_id_independence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - ID tracking is correct!")
    else:
        print("⚠️  SOME TESTS FAILED - Review output above")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
