#!/usr/bin/env python3
"""
Test script to verify state immutability - ensures snapshots don't change after graph mutations.
This prevents aliasing bugs where prev_state and new_state share references.
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

def test_state_immutability():
    """Test that state snapshots remain unchanged after graph mutations."""
    print("=" * 80)
    print("TEST: State Immutability (Aliasing Prevention)")
    print("=" * 80)
    
    # Load config
    config_path = project_root / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Disable special modes and visualization
    env_config = config['environment']
    env_config['centroid_distance_only_mode'] = False
    env_config['simple_delete_only_mode'] = False
    env_config['enable_visualization'] = False
    
    # Save modified config temporarily
    temp_config_path = project_root / 'config_test_immutability.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Create environment
        env = DurotaxisEnv(str(temp_config_path))
        print(f"✓ Environment created")
        print()
        
        # Test 1: State snapshot immutability after node spawn
        print("-" * 80)
        print("TEST 1: State Snapshot Immutability After Spawn")
        print("-" * 80)
        
        obs, info = env.reset()
        
        # Get initial state
        prev_state = env.state_extractor.get_state_features(
            include_substrate=True,
            node_age=env._node_age,
            node_stagnation=env._node_stagnation
        )
        
        # Extract initial values
        prev_num_nodes = prev_state['num_nodes']
        prev_num_edges = prev_state['num_edges']
        prev_centroid_x = prev_state['centroid_x']
        prev_persistent_ids = set(prev_state['persistent_id'].cpu().tolist())
        prev_graph_features = prev_state['graph_features'].clone()
        prev_node_features = prev_state['node_features'].clone()
        
        print(f"Initial state:")
        print(f"  num_nodes: {prev_num_nodes}")
        print(f"  num_edges: {prev_num_edges}")
        print(f"  centroid_x: {prev_centroid_x:.2f}")
        print(f"  persistent_ids: {prev_persistent_ids}")
        print(f"  graph_features device: {prev_graph_features.device}")
        print(f"  node_features device: {prev_node_features.device}")
        
        # Mutate graph by spawning nodes
        action = {
            'discrete': torch.tensor([1, 0]),  # Spawn
            'continuous': torch.tensor([5.0, 1.0, 0.1, 0.0])
        }
        
        obs, reward_components, terminated, truncated, info = env.step(action)
        
        # Get new state after mutation
        new_state = env.state_extractor.get_state_features(
            include_substrate=True,
            node_age=env._node_age,
            node_stagnation=env._node_stagnation
        )
        
        print(f"\nAfter spawn action:")
        print(f"  num_nodes: {new_state['num_nodes']}")
        print(f"  num_edges: {new_state['num_edges']}")
        print(f"  centroid_x: {new_state['centroid_x']:.2f}")
        print(f"  persistent_ids: {set(new_state['persistent_id'].cpu().tolist())}")
        
        # Verify prev_state tensors remain unchanged
        print(f"\nVerifying prev_state immutability:")
        
        # Check num_nodes didn't change in prev_state
        assert prev_state['num_nodes'] == prev_num_nodes, \
            f"prev_state['num_nodes'] changed! Was {prev_num_nodes}, now {prev_state['num_nodes']}"
        print(f"  ✓ num_nodes unchanged: {prev_state['num_nodes']}")
        
        # Check persistent_ids didn't change
        current_prev_pids = set(prev_state['persistent_id'].cpu().tolist())
        assert current_prev_pids == prev_persistent_ids, \
            f"prev_state['persistent_id'] changed! Was {prev_persistent_ids}, now {current_prev_pids}"
        print(f"  ✓ persistent_ids unchanged: {len(current_prev_pids)} nodes")
        
        # Check graph_features tensor didn't change
        assert torch.allclose(prev_state['graph_features'], prev_graph_features, atol=1e-6), \
            "prev_state['graph_features'] tensor changed!"
        print(f"  ✓ graph_features tensor unchanged")
        
        # Check node_features tensor didn't change
        assert torch.allclose(prev_state['node_features'], prev_node_features, atol=1e-6), \
            "prev_state['node_features'] tensor changed!"
        print(f"  ✓ node_features tensor unchanged")
        
        # Check device consistency
        assert prev_state['graph_features'].device == prev_state['node_features'].device, \
            "Device mismatch between graph and node features!"
        assert prev_state['persistent_id'].device == prev_state['node_features'].device, \
            "Device mismatch for persistent_id!"
        print(f"  ✓ All tensors on consistent device: {prev_state['graph_features'].device}")
        
        print(f"\n✅ TEST 1 PASSED: State snapshots are immutable")
        print()
        
        # Test 2: State snapshot immutability after node deletion
        print("-" * 80)
        print("TEST 2: State Snapshot Immutability After Deletion")
        print("-" * 80)
        
        # Get state before deletion
        prev_state2 = env.state_extractor.get_state_features(
            include_substrate=True,
            node_age=env._node_age,
            node_stagnation=env._node_stagnation
        )
        
        prev_num_nodes2 = prev_state2['num_nodes']
        prev_persistent_ids2 = set(prev_state2['persistent_id'].cpu().tolist())
        prev_to_delete2 = prev_state2['to_delete'].clone()
        
        print(f"Before deletion:")
        print(f"  num_nodes: {prev_num_nodes2}")
        print(f"  persistent_ids: {prev_persistent_ids2}")
        
        # Try to delete a node
        if env.topology.graph.num_nodes() > 0:
            env.topology.delete(0)  # Delete first node
        
        # Get new state after deletion
        new_state2 = env.state_extractor.get_state_features(
            include_substrate=True,
            node_age=env._node_age,
            node_stagnation=env._node_stagnation
        )
        
        print(f"\nAfter deletion:")
        print(f"  num_nodes: {new_state2['num_nodes']}")
        print(f"  persistent_ids: {set(new_state2['persistent_id'].cpu().tolist())}")
        
        # Verify prev_state2 didn't change
        print(f"\nVerifying prev_state2 immutability:")
        
        assert prev_state2['num_nodes'] == prev_num_nodes2, \
            f"prev_state2['num_nodes'] changed after deletion!"
        print(f"  ✓ num_nodes unchanged: {prev_state2['num_nodes']}")
        
        current_prev_pids2 = set(prev_state2['persistent_id'].cpu().tolist())
        assert current_prev_pids2 == prev_persistent_ids2, \
            f"prev_state2['persistent_id'] changed after deletion!"
        print(f"  ✓ persistent_ids unchanged: {len(current_prev_pids2)} nodes")
        
        assert torch.allclose(prev_state2['to_delete'], prev_to_delete2, atol=1e-6), \
            "prev_state2['to_delete'] tensor changed after deletion!"
        print(f"  ✓ to_delete flags unchanged")
        
        print(f"\n✅ TEST 2 PASSED: State snapshots remain immutable after deletion")
        print()
        
        # Test 3: Verify no 'topology' reference in state
        print("-" * 80)
        print("TEST 3: No Topology Reference in State")
        print("-" * 80)
        
        state = env.state_extractor.get_state_features(
            include_substrate=True,
            node_age=env._node_age,
            node_stagnation=env._node_stagnation
        )
        
        assert 'topology' not in state, \
            "State dict contains 'topology' reference - this causes aliasing!"
        print(f"  ✓ 'topology' key not in state dict")
        
        # Verify all expected keys are present
        required_keys = [
            'graph_features', 'node_features', 'edge_attr', 'edge_index',
            'persistent_id', 'to_delete', 'num_nodes', 'num_edges',
            'centroid_x', 'goal_x', 'delta_centroid_x', 'delta_num_nodes', 'delta_avg_intensity'
        ]
        
        for key in required_keys:
            assert key in state, f"Missing required key: {key}"
        print(f"  ✓ All {len(required_keys)} required keys present")
        print(f"  ✓ Keys: {list(state.keys())}")
        
        print(f"\n✅ TEST 3 PASSED: No aliasing references in state")
        print()
        
        print("=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nSummary:")
        print("✅ State snapshots are immutable (clone/detach works)")
        print("✅ Tensors remain on consistent device")
        print("✅ No topology references (aliasing prevented)")
        print("✅ Delta features included for temporal context")
        print("\n🎉 State immutability verified - no aliasing bugs!")
        
    finally:
        # Clean up temp config file
        if temp_config_path.exists():
            temp_config_path.unlink()
            print(f"\n✓ Cleaned up temporary config file")

if __name__ == "__main__":
    test_state_immutability()
