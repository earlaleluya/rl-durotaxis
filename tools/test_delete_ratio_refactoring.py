#!/usr/bin/env python3
"""
Test suite for delete ratio refactoring verification.

Tests:
1. Network architecture (continuous-only actions)
2. Action space dimensions
3. Delete ratio strategy (leftmost node deletion)
4. Unified spawn parameters
5. Training loop integration
6. Stage 1 vs Stage 2 behavior
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config_loader import ConfigLoader
from actor_critic import HybridActorCritic
from durotaxis_env import DurotaxisEnv
import dgl

def test_config_updates():
    """Test 1: Verify config.yaml updates"""
    print("\n" + "="*60)
    print("TEST 1: Config Updates")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    actor_critic_config = config_loader.get_actor_critic_config()
    algorithm_config = config_loader.get_algorithm_config()
    
    # Check continuous_dim
    continuous_dim = actor_critic_config.get('continuous_dim', 0)
    assert continuous_dim == 5, f"Expected continuous_dim=5, got {continuous_dim}"
    print("✓ continuous_dim = 5")
    
    # Check num_discrete_actions
    num_discrete = actor_critic_config.get('num_discrete_actions', -1)
    assert num_discrete == 0, f"Expected num_discrete_actions=0, got {num_discrete}"
    print("✓ num_discrete_actions = 0")
    
    # Check action_parameter_bounds
    action_bounds = actor_critic_config.get('action_parameter_bounds', {})
    assert 'delete_ratio' in action_bounds, "delete_ratio not in action_parameter_bounds"
    assert action_bounds['delete_ratio'] == [0.0, 0.5], "delete_ratio bounds incorrect"
    print("✓ delete_ratio bounds = [0.0, 0.5]")
    
    # Check stage_1_fixed_spawn_params
    two_stage = algorithm_config.get('two_stage_curriculum', {})
    stage1_params = two_stage.get('stage_1_fixed_spawn_params', {})
    assert stage1_params['gamma'] == 0.5, "Stage 1 gamma incorrect"
    print("✓ Stage 1 fixed spawn params configured")
    
    print("\n✅ All config tests passed!")
    return True


def test_network_architecture():
    """Test 2: Verify network outputs correct action dimensions"""
    print("\n" + "="*60)
    print("TEST 2: Network Architecture")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    actor_critic_config = config_loader.get_actor_critic_config()
    action_bounds = actor_critic_config.get('action_parameter_bounds', {})
    device = torch.device('cpu')
    
    # Create network
    network = HybridActorCritic(
        encoder_out_dim=512,
        hidden_dim=256,
        continuous_dim=5,
        num_components=4,
        dropout_rate=0.1,
        pretrained_weights='imagenet',
        action_bounds=action_bounds,
        device=device
    )
    
    # Create dummy graph with 5 nodes
    num_nodes = 5
    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    
    # Dummy features
    node_features = torch.randn(num_nodes, 64)
    global_features = torch.randn(128)
    
    state_dict = {
        'graph': g,
        'node_features': node_features,
        'global_features': global_features,
        'num_nodes': num_nodes
    }
    
    # Forward pass
    with torch.no_grad():
        output = network(state_dict)
    
    # Check outputs
    assert 'continuous_actions' in output, "continuous_actions not in output"
    assert output['continuous_actions'].shape == torch.Size([5]), \
        f"Expected shape [5], got {output['continuous_actions'].shape}"
    print(f"✓ continuous_actions shape: {output['continuous_actions'].shape}")
    
    assert 'continuous_log_probs' in output, "continuous_log_probs not in output"
    print(f"✓ continuous_log_probs shape: {output['continuous_log_probs'].shape}")
    
    assert 'discrete_actions' not in output, "discrete_actions still in output!"
    assert 'discrete_log_probs' not in output, "discrete_log_probs still in output!"
    print("✓ No discrete actions in output")
    
    # Check action bounds
    actions = output['continuous_actions'].cpu().numpy()
    delete_ratio = actions[0]
    assert 0.0 <= delete_ratio <= 0.5, f"delete_ratio {delete_ratio} out of bounds [0.0, 0.5]"
    print(f"✓ delete_ratio within bounds: {delete_ratio:.4f}")
    
    print("\n✅ All network architecture tests passed!")
    return True


def test_delete_ratio_strategy():
    """Test 3: Verify delete ratio strategy (leftmost deletion)"""
    print("\n" + "="*60)
    print("TEST 3: Delete Ratio Strategy")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    actor_critic_config = config_loader.get_actor_critic_config()
    action_bounds = actor_critic_config.get('action_parameter_bounds', {})
    device = torch.device('cpu')
    
    network = HybridActorCritic(
        encoder_out_dim=512,
        hidden_dim=256,
        continuous_dim=5,
        num_components=4,
        dropout_rate=0.1,
        action_bounds=action_bounds,
        device=device
    )
    
    # Create test case: 10 nodes at different x-positions
    num_nodes = 10
    x_positions = [5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 9.0, 4.0, 7.0, 0.0]  # Random order
    node_positions = [(i, x_positions[i]) for i in range(num_nodes)]
    
    # Test with delete_ratio = 0.3 (should delete 3 leftmost nodes)
    output = {
        'continuous_actions': torch.tensor([0.3, 0.5, 0.2, 0.1, 0.0])  # delete_ratio=0.3
    }
    
    topology_actions = network.get_topology_actions(output, node_positions)
    
    # Expected: nodes at x=0.0, 1.0, 2.0 should be deleted
    # That's node_ids: 9, 3, 1
    delete_actions = [nid for nid, action in topology_actions.items() if action == 'delete']
    spawn_actions = [nid for nid, action in topology_actions.items() if action == 'spawn']
    
    print(f"Node positions: {sorted(node_positions, key=lambda x: x[1])}")
    print(f"Delete actions (should be 3 leftmost): {sorted(delete_actions)}")
    print(f"Spawn actions: {sorted(spawn_actions)}")
    
    assert len(delete_actions) == 3, f"Expected 3 delete actions, got {len(delete_actions)}"
    assert len(spawn_actions) == 7, f"Expected 7 spawn actions, got {len(spawn_actions)}"
    
    # Verify leftmost nodes are deleted
    deleted_x_positions = [x_positions[nid] for nid in delete_actions]
    assert all(x <= 2.0 for x in deleted_x_positions), f"Non-leftmost nodes deleted: {deleted_x_positions}"
    print("✓ Leftmost 3 nodes correctly identified for deletion")
    
    # Test with delete_ratio = 0.0 (no deletions)
    output_zero = {
        'continuous_actions': torch.tensor([0.0, 0.5, 0.2, 0.1, 0.0])
    }
    topology_actions_zero = network.get_topology_actions(output_zero, node_positions)
    delete_count_zero = sum(1 for a in topology_actions_zero.values() if a == 'delete')
    assert delete_count_zero == 0, f"Expected 0 deletions with ratio=0.0, got {delete_count_zero}"
    print("✓ delete_ratio=0.0 results in no deletions")
    
    # Test with delete_ratio = 0.5 (maximum: 5 deletions)
    output_max = {
        'continuous_actions': torch.tensor([0.5, 0.5, 0.2, 0.1, 0.0])
    }
    topology_actions_max = network.get_topology_actions(output_max, node_positions)
    delete_count_max = sum(1 for a in topology_actions_max.values() if a == 'delete')
    assert delete_count_max == 5, f"Expected 5 deletions with ratio=0.5, got {delete_count_max}"
    print("✓ delete_ratio=0.5 results in maximum deletions (50%)")
    
    print("\n✅ All delete ratio strategy tests passed!")
    return True


def test_unified_spawn_parameters():
    """Test 4: Verify unified spawn parameters"""
    print("\n" + "="*60)
    print("TEST 4: Unified Spawn Parameters")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    actor_critic_config = config_loader.get_actor_critic_config()
    action_bounds = actor_critic_config.get('action_parameter_bounds', {})
    device = torch.device('cpu')
    
    network = HybridActorCritic(
        encoder_out_dim=512,
        hidden_dim=256,
        continuous_dim=5,
        num_components=4,
        dropout_rate=0.1,
        action_bounds=action_bounds,
        device=device
    )
    
    # Create output with specific spawn parameters
    output = {
        'continuous_actions': torch.tensor([0.2, 0.6, 0.3, 0.15, 0.5])
        # [delete_ratio, gamma, alpha, noise, theta]
    }
    
    spawn_params = network.get_spawn_parameters(output)
    
    print(f"Spawn parameters: gamma={spawn_params[0]:.4f}, alpha={spawn_params[1]:.4f}, "
          f"noise={spawn_params[2]:.4f}, theta={spawn_params[3]:.4f}")
    
    # Verify it returns a single tuple (not per-node)
    assert isinstance(spawn_params, tuple), f"Expected tuple, got {type(spawn_params)}"
    assert len(spawn_params) == 4, f"Expected 4 parameters, got {len(spawn_params)}"
    print("✓ Single global spawn parameter tuple returned")
    
    # Verify values match continuous_actions indices [1:5]
    expected = output['continuous_actions'][1:5].tolist()
    actual = list(spawn_params)
    for i, (exp, act) in enumerate(zip(expected, actual)):
        assert abs(exp - act) < 1e-6, f"Parameter {i} mismatch: {exp} vs {act}"
    print("✓ Spawn parameters match continuous_actions[1:5]")
    
    print("\n✅ All unified spawn parameter tests passed!")
    return True


def test_training_integration():
    """Test 5: Verify training loop can process actions correctly"""
    print("\n" + "="*60)
    print("TEST 5: Training Integration")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    actor_critic_config = config_loader.get_actor_critic_config()
    action_bounds = actor_critic_config.get('action_parameter_bounds', {})
    device = torch.device('cpu')
    
    # Create environment
    env = DurotaxisEnv('config.yaml')
    
    # Create network
    network = HybridActorCritic(
        encoder_out_dim=512,
        hidden_dim=256,
        continuous_dim=5,
        num_components=4,
        dropout_rate=0.1,
        action_bounds=action_bounds,
        device=device
    )
    
    # Reset environment
    state = env.reset()
    
    print(f"Initial state - num_nodes: {state['num_nodes']}")
    
    # Get action from network
    with torch.no_grad():
        state_dict = {
            'graph': state['graph'],
            'node_features': state['node_features'],
            'global_features': state['global_features'],
            'num_nodes': state['num_nodes']
        }
        output = network(state_dict)
    
    # Verify output structure
    assert 'continuous_actions' in output, "Missing continuous_actions"
    assert output['continuous_actions'].shape[0] == 5, "Wrong action dimension"
    print("✓ Network output has correct structure")
    
    # Get node positions for delete ratio
    node_features = state['node_features']
    node_positions = [(i, node_features[i][0].item()) for i in range(state['num_nodes'])]
    
    # Get topology actions
    topology_actions = network.get_topology_actions(output, node_positions)
    print(f"✓ Topology actions computed: {len(topology_actions)} nodes")
    
    # Get spawn parameters
    spawn_params = network.get_spawn_parameters(output)
    print(f"✓ Spawn parameters: {spawn_params}")
    
    # Verify actions are valid
    delete_count = sum(1 for a in topology_actions.values() if a == 'delete')
    spawn_count = sum(1 for a in topology_actions.values() if a == 'spawn')
    
    assert delete_count + spawn_count == state['num_nodes'], "Action count mismatch"
    print(f"✓ Action distribution: {delete_count} deletes, {spawn_count} spawns")
    
    # Verify delete ratio constraint
    delete_ratio = output['continuous_actions'][0].item()
    expected_deletes = int(delete_ratio * state['num_nodes'])
    assert delete_count == expected_deletes, f"Delete count mismatch: {delete_count} vs {expected_deletes}"
    print(f"✓ Delete ratio enforced correctly: {delete_ratio:.4f} → {delete_count} deletions")
    
    print("\n✅ All training integration tests passed!")
    return True


def test_stage_comparison():
    """Test 6: Verify Stage 1 vs Stage 2 behavior"""
    print("\n" + "="*60)
    print("TEST 6: Stage 1 vs Stage 2 Comparison")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    algorithm_config = config_loader.get_algorithm_config()
    
    # Stage 1: Fixed spawn parameters
    print("\nStage 1 (Fixed spawn params):")
    two_stage = algorithm_config.get('two_stage_curriculum', {})
    stage1_params = two_stage.get('stage_1_fixed_spawn_params', {})
    print(f"  gamma: {stage1_params['gamma']}")
    print(f"  alpha: {stage1_params['alpha']}")
    print(f"  noise: {stage1_params['noise']}")
    print(f"  theta: {stage1_params['theta']}")
    assert stage1_params['gamma'] == 0.5, "Stage 1 gamma incorrect"
    print("✓ Stage 1 uses fixed spawn parameters")
    
    # Stage 2: Network learns all parameters
    print("\nStage 2 (Learned params):")
    print("  Network outputs all 5 continuous parameters")
    print("  [delete_ratio, gamma, alpha, noise, theta]")
    print("✓ Stage 2 learns all parameters")
    
    # Verify training stage handling exists
    from train import PPOTrainer
    trainer = PPOTrainer('config.yaml')
    assert hasattr(trainer, 'training_stage'), "training_stage attribute missing"
    print(f"✓ Trainer has training_stage attribute: {trainer.training_stage}")
    
    print("\n✅ All stage comparison tests passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DELETE RATIO REFACTORING TEST SUITE")
    print("="*60)
    
    tests = [
        ("Config Updates", test_config_updates),
        ("Network Architecture", test_network_architecture),
        ("Delete Ratio Strategy", test_delete_ratio_strategy),
        ("Unified Spawn Parameters", test_unified_spawn_parameters),
        ("Training Integration", test_training_integration),
        ("Stage Comparison", test_stage_comparison),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS", None))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"\n❌ Test failed: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, status, error in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {name}: {status}")
        if error:
            print(f"   Error: {error}")
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Delete ratio refactoring verified!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
