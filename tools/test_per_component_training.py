#!/usr/bin/env python3
"""
Test per-component training to verify critic learns component-specific values.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from train import TrajectoryBuffer

def test_per_component_gae():
    """Test that GAE computes separate returns for each component"""
    print("Testing per-component GAE computation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    buffer = TrajectoryBuffer(device=device)
    
    # Define component names
    component_names = ['total_reward', 'graph_reward', 'spawn_reward', 'delete_reward']
    
    # Create mock episode with different component rewards
    buffer.start_episode()
    
    # Step 1: High delete, low spawn
    state1 = {'num_nodes': 5}
    action1 = {'continuous': torch.tensor([0.2, 1.0, 2.0, 0.1, 0.0])}
    reward1 = {
        'total_reward': 10.0,
        'graph_reward': 10.0,
        'spawn_reward': -5.0,  # Negative spawn
        'delete_reward': 15.0,  # Positive delete
    }
    value1 = {
        'total_reward': torch.tensor(8.0),
        'graph_reward': torch.tensor(8.0),
        'spawn_reward': torch.tensor(-3.0),
        'delete_reward': torch.tensor(11.0),
    }
    log_prob1 = {'continuous': torch.tensor(-1.0), 'total': torch.tensor(-1.0)}
    buffer.add_step(state1, action1, reward1, value1, log_prob1, value1)
    
    # Step 2: High spawn, low delete
    state2 = {'num_nodes': 8}
    action2 = {'continuous': torch.tensor([0.1, 2.0, 3.0, 0.2, 0.5])}
    reward2 = {
        'total_reward': 12.0,
        'graph_reward': 12.0,
        'spawn_reward': 18.0,  # Positive spawn
        'delete_reward': -6.0,  # Negative delete
    }
    value2 = {
        'total_reward': torch.tensor(10.0),
        'graph_reward': torch.tensor(10.0),
        'spawn_reward': torch.tensor(14.0),
        'delete_reward': torch.tensor(-4.0),
    }
    log_prob2 = {'continuous': torch.tensor(-1.2), 'total': torch.tensor(-1.2)}
    buffer.add_step(state2, action2, reward2, value2, log_prob2, value2)
    
    # Finish episode
    final_values = {
        'total_reward': torch.tensor(5.0),
        'graph_reward': torch.tensor(5.0),
        'spawn_reward': torch.tensor(3.0),
        'delete_reward': torch.tensor(2.0),
    }
    buffer.finish_episode(final_values, terminated=False, success=False)
    
    # Compute returns and advantages
    gamma = 0.99
    gae_lambda = 0.95
    buffer.compute_returns_and_advantages_for_all_episodes(gamma, gae_lambda, component_names)
    
    # Verify structure
    episode = buffer.episodes[0]
    assert isinstance(episode['returns'], dict), "Returns should be a dict"
    assert isinstance(episode['advantages'], dict), "Advantages should be a dict"
    
    print("✅ Returns and advantages are dicts")
    
    # Check each component exists
    for component in component_names:
        assert component in episode['returns'], f"Missing {component} in returns"
        assert component in episode['advantages'], f"Missing {component} in advantages"
        assert len(episode['returns'][component]) == 2, f"{component} returns should have 2 steps"
        assert len(episode['advantages'][component]) == 2, f"{component} advantages should have 2 steps"
    
    print("✅ All components have returns and advantages")
    
    # Verify component-specific values are different
    spawn_return_0 = episode['returns']['spawn_reward'][0].item()
    delete_return_0 = episode['returns']['delete_reward'][0].item()
    
    print(f"\nComponent-specific returns at step 0:")
    print(f"  Spawn return:  {spawn_return_0:.3f}")
    print(f"  Delete return: {delete_return_0:.3f}")
    
    # They should be different since spawn reward was -5 and delete was +15
    assert abs(spawn_return_0 - delete_return_0) > 1.0, "Component returns should be significantly different"
    
    print("✅ Component-specific returns are different (as expected)")
    
    # Test batch data structure
    batch_data = buffer.get_batch_data()
    assert isinstance(batch_data['returns'], dict), "Batch returns should be a dict"
    assert isinstance(batch_data['advantages'], dict), "Batch advantages should be a dict"
    
    for component in component_names:
        assert component in batch_data['returns'], f"Missing {component} in batch returns"
        assert len(batch_data['returns'][component]) == 2, f"{component} batch should have 2 steps"
    
    print("✅ Batch data structure is correct")
    
    # Test minibatch creation
    minibatches = buffer.create_minibatches(minibatch_size=1)
    assert len(minibatches) == 2, "Should create 2 minibatches"
    
    for mb in minibatches:
        assert isinstance(mb['returns'], dict), "Minibatch returns should be dict"
        assert isinstance(mb['advantages'], dict), "Minibatch advantages should be dict"
        for component in component_names:
            assert component in mb['returns'], f"Missing {component} in minibatch"
            assert len(mb['returns'][component]) == 1, f"Minibatch {component} should have 1 step"
    
    print("✅ Minibatch creation preserves per-component structure")
    
    print("\n🎉 All per-component training tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_per_component_gae()
        print("\n✅ Per-component training is working correctly!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
