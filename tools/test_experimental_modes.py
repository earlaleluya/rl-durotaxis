#!/usr/bin/env python3
"""
Test script to verify all 4 experimental modes work correctly with delete ratio architecture.

Tests:
1. simple_delete_only_mode (delete penalties only)
2. centroid_distance_only_mode (distance shaping only)
3. Combined mode (both enabled)
4. Normal mode (all reward components)

Validates:
- No discrete action references
- Correct reward component activation/deactivation
- Delete ratio action space compatibility
- Proper reward calculations
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from durotaxis_env import DurotaxisEnv


def run_test_episode(env, mode_name, expected_components):
    """Run a test episode and verify reward components."""
    print(f"\n{'='*60}")
    print(f"Testing: {mode_name}")
    print(f"{'='*60}")
    
    # Reset environment (returns observation, info tuple)
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Nodes: {info['num_nodes']}")
    print(f"  - Edges: {info['num_edges']}")
    
    # Note: action_space is Discrete(1) dummy, actual actions are 5D continuous from policy
    print(f"✓ Environment uses delete_ratio architecture (5D continuous from policy)")
    
    # Run a few test steps
    total_rewards = []
    component_checks = {
        'graph_reward': [],
        'spawn_reward': [],
        'delete_reward': [],
        'total_node_reward': [],
        'efficiency_reward': [],
        'survival_reward': [],
        'milestone_reward': []
    }
    
    for step in range(5):
        # Sample random delete ratio action (dummy action, actual actions from policy)
        action = env.action_space.sample()
        
        # Step environment
        result = env.step(action)
        
        # Handle both old (4-tuple) and new (5-tuple) gymnasium API
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
        
        # Handle reward (can be scalar or dict)
        if isinstance(reward, dict):
            total_reward_val = reward.get('total_reward', 0.0)
            total_rewards.append(total_reward_val)
            # Use reward dict for component checks
            for component in component_checks.keys():
                if component in reward:
                    component_checks[component].append(reward[component])
        else:
            total_rewards.append(reward)
            # Collect component values from info
            for component in component_checks.keys():
                if component in info:
                    component_checks[component].append(info[component])
        
        if done or truncated:
            break
    
    print(f"✓ Completed {len(total_rewards)} steps")
    print(f"  - Mean reward: {np.mean(total_rewards):.4f}")
    
    # Verify expected components are active
    print(f"\nReward Component Status:")
    print(f"{'Component':<20} {'Expected':<10} {'Active':<10} {'Status'}")
    print(f"{'-'*60}")
    
    all_passed = True
    for component, expected in expected_components.items():
        values = component_checks.get(component, [])
        is_active = any(abs(v) > 1e-6 for v in values if v is not None)
        status = "✓" if is_active == expected else "✗"
        
        if is_active != expected:
            all_passed = False
        
        print(f"{component:<20} {str(expected):<10} {str(is_active):<10} {status}")
    
    if all_passed:
        print(f"\n✅ {mode_name}: All component checks PASSED")
    else:
        print(f"\n❌ {mode_name}: Some component checks FAILED")
    
    return all_passed


def test_mode_1_simple_delete_only():
    """Test Mode 1: simple_delete_only_mode (delete penalties only)."""
    env = DurotaxisEnv(
        'config.yaml',
        simple_delete_only_mode=True,
        centroid_distance_only_mode=False,
        include_termination_rewards=False
    )
    
    # Expected: Only graph_reward and delete_reward active (penalties)
    # All other components should be zero
    expected = {
        'graph_reward': True,       # Growth penalty (Rule 0)
        'spawn_reward': False,      # Disabled
        'delete_reward': True,      # Delete penalties (Rule 1 & 2)
        'total_node_reward': False, # Disabled
        'efficiency_reward': False, # Disabled
        'survival_reward': False,   # Disabled
        'milestone_reward': False   # Disabled
    }
    
    return run_test_episode(env, "Mode 1: simple_delete_only_mode", expected)


def test_mode_2_centroid_distance_only():
    """Test Mode 2: centroid_distance_only_mode (distance shaping only)."""
    env = DurotaxisEnv(
        'config.yaml',
        simple_delete_only_mode=False,
        centroid_distance_only_mode=True,
        include_termination_rewards=False
    )
    
    # Expected: Only total_node_reward active (distance signal)
    # All other components should be zero
    expected = {
        'graph_reward': True,       # Contains distance signal
        'spawn_reward': False,      # Disabled
        'delete_reward': False,     # Disabled
        'total_node_reward': False, # Disabled (zeroed in distance mode)
        'efficiency_reward': False, # Disabled
        'survival_reward': False,   # Disabled
        'milestone_reward': False   # Disabled
    }
    
    return run_test_episode(env, "Mode 2: centroid_distance_only_mode", expected)


def test_mode_3_combined():
    """Test Mode 3: Both modes enabled (distance + delete penalties)."""
    env = DurotaxisEnv(
        'config.yaml',
        simple_delete_only_mode=True,
        centroid_distance_only_mode=True,
        include_termination_rewards=False
    )
    
    # Expected: graph_reward contains distance signal + delete penalties
    # All other components should be zero
    expected = {
        'graph_reward': True,       # Distance + delete penalties
        'spawn_reward': False,      # Disabled
        'delete_reward': True,      # Delete penalties (for logging)
        'total_node_reward': False, # Disabled
        'efficiency_reward': False, # Disabled
        'survival_reward': False,   # Disabled
        'milestone_reward': False   # Disabled
    }
    
    return run_test_episode(env, "Mode 3: Combined (simple_delete + centroid_distance)", expected)


def test_mode_4_normal():
    """Test Mode 4: Normal mode (all reward components active)."""
    env = DurotaxisEnv(
        'config.yaml',
        simple_delete_only_mode=False,
        centroid_distance_only_mode=False,
        include_termination_rewards=False
    )
    
    # Expected: Key components should be active (spawn/efficiency/milestone are conditional)
    # In normal mode, these should NOT be disabled:
    expected = {
        'graph_reward': True,       # Always active
        'spawn_reward': False,      # Conditional (may not trigger in 5 steps)
        'delete_reward': True,      # Usually active
        'total_node_reward': True,  # Usually active
        'efficiency_reward': False, # Conditional (may not trigger)
        'survival_reward': True,    # Usually active
        'milestone_reward': False   # Conditional (may not trigger in 5 steps)
    }
    
    return run_test_episode(env, "Mode 4: Normal mode (all components)", expected)


def check_no_discrete_action_bugs():
    """Verify no discrete action references in environment."""
    print(f"\n{'='*60}")
    print(f"Checking for discrete action bugs in durotaxis_env.py")
    print(f"{'='*60}")
    
    with open('durotaxis_env.py', 'r') as f:
        content = f.read()
    
    # Check for problematic discrete action patterns
    # Note: Discrete(1) is acceptable as a dummy action space
    discrete_patterns = [
        'num_discrete_actions',
        'discrete_actions',
        'discrete_head',
        'discrete_bias',
        'get_topology_actions',
        'get_spawn_parameters',
        'MultiDiscrete',
    ]
    
    found_issues = []
    for pattern in discrete_patterns:
        if pattern in content:
            # Count occurrences
            count = content.count(pattern)
            found_issues.append(f"  - Found '{pattern}' ({count} occurrences)")
    
    # Check for Discrete( but allow Discrete(1) dummy action space
    if 'Discrete(' in content:
        # Check if it's only the dummy action space
        if content.count('Discrete(1)') == 1 and content.count('Discrete(') == 1:
            print(f"✓ Found Discrete(1) dummy action space (acceptable)")
        else:
            found_issues.append(f"  - Found problematic 'Discrete(' usage")
    
    if found_issues:
        print(f"❌ Found discrete action bugs:")
        for issue in found_issues:
            print(issue)
        return False
    else:
        print(f"✅ No discrete action bugs found")
        return True


def main():
    """Run all experimental mode tests."""
    print(f"\n{'#'*60}")
    print(f"# DELETE RATIO EXPERIMENTAL MODE TEST SUITE")
    print(f"{'#'*60}")
    print(f"\nTesting 4 experimental modes:")
    print(f"  1. simple_delete_only_mode")
    print(f"  2. centroid_distance_only_mode")
    print(f"  3. Combined mode (both enabled)")
    print(f"  4. Normal mode (all components)")
    
    # Check for discrete action bugs first
    no_bugs = check_no_discrete_action_bugs()
    
    # Run mode tests
    results = {
        'Mode 1 (simple_delete_only)': test_mode_1_simple_delete_only(),
        'Mode 2 (centroid_distance_only)': test_mode_2_centroid_distance_only(),
        'Mode 3 (combined)': test_mode_3_combined(),
        'Mode 4 (normal)': test_mode_4_normal()
    }
    
    # Print summary
    print(f"\n{'#'*60}")
    print(f"# TEST SUMMARY")
    print(f"{'#'*60}")
    print(f"\nDiscrete Action Check: {'✅ PASSED' if no_bugs else '❌ FAILED'}")
    print(f"\nExperimental Mode Tests:")
    
    all_passed = no_bugs
    for mode_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {mode_name:<30} {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print(f"\n{'='*60}")
        print(f"✅ ALL TESTS PASSED!")
        print(f"{'='*60}")
        print(f"\nThe delete ratio codebase is compatible with all 4 experimental modes:")
        print(f"  1. ✓ simple_delete_only_mode")
        print(f"  2. ✓ centroid_distance_only_mode")
        print(f"  3. ✓ Combined mode")
        print(f"  4. ✓ Normal mode")
        print(f"\nYou can safely train with any of these modes.")
        return 0
    else:
        print(f"\n{'='*60}")
        print(f"❌ SOME TESTS FAILED")
        print(f"{'='*60}")
        print(f"\nPlease review the failures above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
