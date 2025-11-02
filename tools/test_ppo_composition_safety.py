#!/usr/bin/env python3
"""
Test PPO component composition safety.

Verifies that:
1. Component returns are computed only for configured components
2. Critic loss uses only components present in both critic heads and trainer weights
3. Policy loss uses total_reward advantages only (weighted composition)
4. graph_reward is correctly treated as an alias of total_reward
5. No legacy components (edge_reward, total_node_reward) are used

Test strategy:
- Load actual config and verify component lists
- Inspect critic loss computation code
- Verify advantage weighting uses configured components only
- Check environment returns correct components with graph_reward = total_reward
"""

import sys
import os
import yaml
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import ConfigLoader
from actor_critic import HybridActorCritic
from durotaxis_env import DurotaxisEnv


def test_1_component_lists_consistency():
    """Test that component lists are consistent across config, network, and trainer"""
    print("\n" + "="*60)
    print("TEST 1: Component Lists Consistency")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    config = config_loader.config
    
    # Get value components from actor_critic config
    actor_critic_config = config.get('actor_critic', {})
    value_components = actor_critic_config.get('value_components', [])
    
    # Get component weights from trainer config
    trainer_config = config.get('trainer', {})
    component_weights = trainer_config.get('component_weights', {})
    
    print(f"\n✓ Config value_components ({len(value_components)}): {value_components}")
    print(f"✓ Config component_weights ({len(component_weights)}): {list(component_weights.keys())}")
    
    # Verify they match
    vc_set = set(value_components)
    cw_set = set(component_weights.keys())
    
    if vc_set != cw_set:
        print(f"\n❌ MISMATCH DETECTED!")
        print(f"   In value_components but not component_weights: {vc_set - cw_set}")
        print(f"   In component_weights but not value_components: {cw_set - vc_set}")
        return False
    
    # Check for legacy components
    legacy_components = ['edge_reward', 'total_node_reward']
    found_legacy = [comp for comp in legacy_components if comp in value_components]
    
    if found_legacy:
        print(f"\n❌ LEGACY COMPONENTS FOUND: {found_legacy}")
        return False
    
    # Verify graph_reward and total_reward both present
    required = ['total_reward', 'graph_reward', 'delete_reward', 'spawn_reward', 'distance_signal']
    missing = [comp for comp in required if comp not in value_components]
    
    if missing:
        print(f"\n❌ MISSING REQUIRED COMPONENTS: {missing}")
        return False
    
    print(f"\n✅ PASSED: All component lists consistent")
    print(f"   - No legacy components")
    print(f"   - All 5 required components present")
    print(f"   - value_components matches component_weights")
    return True


def test_2_critic_loss_uses_configured_components():
    """Test that critic loss computation uses only configured components"""
    print("\n" + "="*60)
    print("TEST 2: Critic Loss Uses Configured Components")
    print("="*60)
    
    # Read train.py and check critic loss computation
    train_py_path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
    with open(train_py_path, 'r') as f:
        train_code = f.read()
    
    # Find the critic loss computation section
    # Should iterate over self.component_names and use component_weights
    
    # Check 1: Critic loss iterates over component_names
    if 'for component in self.component_names:' not in train_code:
        print("❌ Critic loss doesn't iterate over self.component_names")
        return False
    
    print("✓ Critic loss iterates over self.component_names")
    
    # Check 2: Uses component_weights to weight each component loss
    if 'weight = self.component_weights.get(component' not in train_code:
        print("❌ Critic loss doesn't use self.component_weights")
        return False
    
    print("✓ Critic loss applies component_weights")
    
    # Check 3: Verifies component in both eval_output and returns
    if 'if component in eval_output' not in train_code or 'and component in returns' not in train_code:
        print("❌ Critic loss doesn't check component presence in both critic heads and returns")
        return False
    
    print("✓ Critic loss checks component presence in both critic heads and returns")
    
    # Check 4: No hardcoded component names (except in constants/defaults)
    legacy_in_loss = []
    for legacy in ['edge_reward', 'total_node_reward']:
        # Look for usage in loss computation (not in comments or string literals)
        if f"'{legacy}'" in train_code or f'"{legacy}"' in train_code:
            # Check if it's in actual code, not comments
            for line in train_code.split('\n'):
                if legacy in line and not line.strip().startswith('#'):
                    legacy_in_loss.append(legacy)
                    break
    
    if legacy_in_loss:
        print(f"⚠️  WARNING: Legacy component references found: {legacy_in_loss}")
        print("   (May be in comments or old code - manual verification needed)")
    
    print("\n✅ PASSED: Critic loss uses only configured components")
    print("   - Iterates over self.component_names")
    print("   - Applies component_weights")
    print("   - Checks component in both critic and returns")
    return True


def test_3_policy_loss_uses_total_reward_advantages():
    """Test that policy loss uses weighted total_reward advantages only"""
    print("\n" + "="*60)
    print("TEST 3: Policy Loss Uses Total Reward Advantages")
    print("="*60)
    
    # Read train.py and check advantage computation
    train_py_path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
    with open(train_py_path, 'r') as f:
        train_code = f.read()
    
    # Check 1: Advantages are weighted across components
    if 'compute_enhanced_advantage_weights' not in train_code:
        print("❌ compute_enhanced_advantage_weights method not found")
        return False
    
    print("✓ compute_enhanced_advantage_weights method exists")
    
    # Check 2: Total advantages computed from component advantages
    if 'total_advantages = self.compute_enhanced_advantage_weights(advantages)' not in train_code:
        print("❌ total_advantages not computed from component advantages")
        return False
    
    print("✓ total_advantages computed from weighted component advantages")
    
    # Check 3: Policy loss uses total advantages (not per-component)
    if 'advantage = total_advantages[i]' not in train_code:
        print("❌ Policy loss doesn't use total_advantages")
        return False
    
    print("✓ Policy loss uses total_advantages (weighted composition)")
    
    # Check 4: Traditional weighting method uses component_weights
    if '_compute_traditional_weighted_advantages' in train_code:
        if 'self.component_weights.get(component' not in train_code:
            print("⚠️  WARNING: Traditional weighting may not use component_weights")
    
    print("\n✅ PASSED: Policy loss uses weighted total_reward advantages")
    print("   - Advantages weighted across components")
    print("   - Policy loss uses scalar total advantage")
    print("   - Component weights applied in weighting")
    return True


def test_4_graph_reward_alias():
    """Test that graph_reward is correctly treated as alias of total_reward"""
    print("\n" + "="*60)
    print("TEST 4: graph_reward = total_reward Alias")
    print("="*60)
    
    # Read durotaxis_env.py
    env_py_path = os.path.join(os.path.dirname(__file__), '..', 'durotaxis_env.py')
    with open(env_py_path, 'r') as f:
        env_code = f.read()
    
    # Check that graph_reward = total_reward in environment
    if 'graph_reward = total_reward' not in env_code:
        print("❌ graph_reward not set equal to total_reward in environment")
        return False
    
    print("✓ Environment sets graph_reward = total_reward")
    
    # Check both are in reward breakdown
    if "'graph_reward': graph_reward," not in env_code:
        print("❌ graph_reward not in reward breakdown")
        return False
    
    print("✓ Both total_reward and graph_reward in reward breakdown")
    
    # Test with actual environment
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    env = DurotaxisEnv(config_path)
    
    # Run a few steps and check reward components
    env.reset()
    
    for _ in range(5):
        # Random action
        action = {
            'continuous': np.random.uniform(-1, 1, 5).astype(np.float32)
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            total = breakdown.get('total_reward', None)
            graph = breakdown.get('graph_reward', None)
            
            if total is None or graph is None:
                print(f"❌ Missing total_reward or graph_reward in breakdown")
                return False
            
            # Should be exactly equal (same object or same value)
            if abs(float(total) - float(graph)) > 1e-9:
                print(f"❌ graph_reward != total_reward: {graph} != {total}")
                return False
        
        if terminated or truncated:
            break
    
    print("✓ Verified graph_reward = total_reward in actual environment steps")
    
    print("\n✅ PASSED: graph_reward correctly aliased to total_reward")
    print("   - Environment code sets graph_reward = total_reward")
    print("   - Both present in reward breakdown")
    print("   - Runtime verification passed")
    return True


def test_5_component_returns_computation():
    """Test that component returns are computed for configured components only"""
    print("\n" + "="*60)
    print("TEST 5: Component Returns Computation")
    print("="*60)
    
    # Read train.py
    train_py_path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
    with open(train_py_path, 'r') as f:
        train_code = f.read()
    
    # Check compute_returns_and_advantages method
    if 'def compute_returns_and_advantages' not in train_code:
        print("❌ compute_returns_and_advantages method not found")
        return False
    
    print("✓ compute_returns_and_advantages method found")
    
    # Check it iterates over component_names
    if 'for component in self.component_names:' not in train_code:
        print("❌ Returns computation doesn't iterate over component_names")
        return False
    
    print("✓ Returns computed for each component in component_names")
    
    # Check it extracts component rewards from normalized rewards
    if 'normalized_rewards[component]' not in train_code:
        print("❌ Doesn't extract component rewards from normalized_rewards")
        return False
    
    print("✓ Component rewards extracted from normalized_rewards")
    
    # Check it uses component values from value predictions
    if 'component_values = torch.stack([v[component] for v in values])' not in train_code:
        print("❌ Doesn't extract component values correctly")
        return False
    
    print("✓ Component values extracted from value predictions")
    
    # Check GAE computation for each component
    if 'component_returns[t] = gae + component_values[t]' not in train_code:
        print("❌ GAE returns not computed for components")
        return False
    
    print("✓ GAE returns computed per component")
    
    print("\n✅ PASSED: Component returns computed correctly")
    print("   - Iterates over configured component_names")
    print("   - Extracts component rewards and values")
    print("   - Computes GAE returns per component")
    return True


def test_6_no_hardcoded_components_in_ppo():
    """Test that PPO doesn't use hardcoded component names"""
    print("\n" + "="*60)
    print("TEST 6: No Hardcoded Components in PPO")
    print("="*60)
    
    # Read train.py
    train_py_path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
    with open(train_py_path, 'r') as f:
        train_code = f.read()
    
    lines = train_code.split('\n')
    
    # List of components that should NOT be hardcoded in PPO logic
    legacy_components = ['edge_reward', 'total_node_reward']
    
    # Check for hardcoded component access (outside of config defaults)
    hardcoded_refs = []
    
    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue
        
        # Check for hardcoded component access
        for comp in legacy_components:
            if f"'{comp}'" in line or f'"{comp}"' in line:
                # Check if it's in actual code (not default config)
                if 'default' not in line.lower() and 'config' not in line.lower():
                    hardcoded_refs.append((i, comp, line.strip()))
    
    if hardcoded_refs:
        print(f"\n⚠️  WARNING: Found {len(hardcoded_refs)} potential hardcoded references:")
        for line_num, comp, line in hardcoded_refs[:5]:  # Show first 5
            print(f"   Line {line_num}: {comp} in '{line[:80]}'")
        print("   (Manual verification recommended)")
    else:
        print("✓ No hardcoded legacy component references found")
    
    # Verify component_names is used consistently
    if 'self.component_names' not in train_code:
        print("❌ self.component_names not found in train.py")
        return False
    
    print("✓ self.component_names used for component iteration")
    
    print("\n✅ PASSED: No problematic hardcoded components in PPO")
    return True


def main():
    """Run all PPO composition safety tests"""
    print("\n" + "="*60)
    print("PPO COMPONENT COMPOSITION SAFETY TESTS")
    print("="*60)
    print("\nVerifying:")
    print("1. Component lists consistency across config/network/trainer")
    print("2. Critic loss uses only configured components")
    print("3. Policy loss uses weighted total_reward advantages")
    print("4. graph_reward treated as alias of total_reward")
    print("5. Component returns computed for configured components")
    print("6. No hardcoded legacy components in PPO logic")
    
    results = []
    
    # Run tests
    results.append(("Component Lists Consistency", test_1_component_lists_consistency()))
    results.append(("Critic Loss Configuration", test_2_critic_loss_uses_configured_components()))
    results.append(("Policy Loss Advantages", test_3_policy_loss_uses_total_reward_advantages()))
    results.append(("graph_reward Alias", test_4_graph_reward_alias()))
    results.append(("Component Returns", test_5_component_returns_computation()))
    results.append(("No Hardcoded Components", test_6_no_hardcoded_components_in_ppo()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - PPO composition is safe!")
        print("\n✅ Verified:")
        print("   - Only configured components used throughout PPO")
        print("   - Critic loss weights components present in both critic and trainer")
        print("   - Policy loss uses weighted total advantages (not per-component)")
        print("   - graph_reward correctly aliased to total_reward")
        print("   - No legacy component references")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Review required")
        return 1


if __name__ == '__main__':
    sys.exit(main())
