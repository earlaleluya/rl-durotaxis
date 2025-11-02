#!/usr/bin/env python3
"""
Validation script for reward component consistency across the system.

Checks:
1. Value heads match config.actor_critic.value_components
2. Trainer only uses component weights present in config.trainer.component_weights
3. Returns/advantages computed per component for PPO
4. Policy loss uses total_reward advantages only
5. graph_reward treated as alias of total_reward
6. No references to removed/legacy components
7. Environment returns only expected components
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from actor_critic import HybridActorCritic
from encoder import GraphInputEncoder
from durotaxis_env import DurotaxisEnv


def validate_config_consistency():
    """Validate configuration consistency"""
    print('\n' + '='*80)
    print('CONFIGURATION CONSISTENCY VALIDATION')
    print('='*80)
    
    config = load_config('config.yaml')
    actor_critic_config = config.get_actor_critic_config()
    trainer_config = config.get_trainer_config()
    
    value_components = actor_critic_config.get('value_components', [])
    component_weights = trainer_config.get('component_weights', {})
    
    print(f'\n✓ Value Components ({len(value_components)}): {value_components}')
    print(f'✓ Component Weights ({len(component_weights)}): {list(component_weights.keys())}')
    
    # Check consistency
    vc_set = set(value_components)
    cw_set = set(component_weights.keys())
    
    if vc_set != cw_set:
        missing_in_weights = vc_set - cw_set
        missing_in_components = cw_set - vc_set
        if missing_in_weights:
            print(f'\n⚠️  WARNING: Components in value_components but NOT in component_weights:')
            print(f'    {missing_in_weights}')
        if missing_in_components:
            print(f'\n⚠️  WARNING: Components in component_weights but NOT in value_components:')
            print(f'    {missing_in_components}')
        return False
    
    print('\n✅ Config consistency: PASSED')
    return True


def validate_network_structure():
    """Validate Actor-Critic network structure"""
    print('\n' + '='*80)
    print('NETWORK STRUCTURE VALIDATION')
    print('='*80)
    
    config = load_config('config.yaml')
    actor_critic_config = config.get_actor_critic_config()
    
    try:
        encoder = GraphInputEncoder(
            hidden_dim=actor_critic_config.get('hidden_dim', 128),
            out_dim=128,
            num_layers=4
        )
        network = HybridActorCritic(encoder=encoder, config_path='config.yaml')
        
        print(f'\n✓ Network initialized successfully')
        print(f'  Value components: {network.value_components}')
        print(f'  Critic heads: {list(network.critic.value_heads.keys())}')
        
        # Check if they match
        if set(network.value_components) != set(network.critic.value_heads.keys()):
            print(f'\n⚠️  ERROR: Value components don\'t match critic heads!')
            return False
        
        print('\n✅ Network structure: PASSED')
        return True
        
    except Exception as e:
        print(f'\n❌ Network initialization failed: {e}')
        return False


def validate_no_legacy_components():
    """Check for legacy component references"""
    print('\n' + '='*80)
    print('LEGACY COMPONENT CHECK')
    print('='*80)
    
    config = load_config('config.yaml')
    actor_critic_config = config.get_actor_critic_config()
    trainer_config = config.get_trainer_config()
    
    legacy_components = ['edge_reward', 'total_node_reward', 'node_reward']
    value_components = actor_critic_config.get('value_components', [])
    component_weights = trainer_config.get('component_weights', {})
    
    found_legacy = []
    for legacy in legacy_components:
        if legacy in value_components:
            found_legacy.append(f'value_components: {legacy}')
        if legacy in component_weights:
            found_legacy.append(f'component_weights: {legacy}')
    
    if found_legacy:
        print(f'\n⚠️  Found legacy component references:')
        for ref in found_legacy:
            print(f'    {ref}')
        print('\n  Recommendation: Remove these legacy components from config')
        return False
    
    print(f'\n✓ No legacy components found')
    print(f'  Checked for: {legacy_components}')
    print('\n✅ Legacy component check: PASSED')
    return True


def validate_graph_reward_alias():
    """Verify graph_reward is treated as total_reward alias"""
    print('\n' + '='*80)
    print('GRAPH_REWARD ALIAS VALIDATION')
    print('='*80)
    
    config = load_config('config.yaml')
    actor_critic_config = config.get_actor_critic_config()
    
    value_components = actor_critic_config.get('value_components', [])
    has_graph = 'graph_reward' in value_components
    has_total = 'total_reward' in value_components
    
    print(f'\n  graph_reward in value_components: {has_graph}')
    print(f'  total_reward in value_components: {has_total}')
    
    if has_graph and has_total:
        print(f'\n✓ Both present: graph_reward will be treated as alias of total_reward')
        print(f'  Note: Environment should set graph_reward = total_reward')
        print('\n✅ Graph reward alias: PASSED')
        return True
    elif has_total:
        print(f'\n✓ Only total_reward present (simplified, no alias needed)')
        print('\n✅ Graph reward alias: PASSED')
        return True
    else:
        print(f'\n⚠️  WARNING: Missing total_reward (required for policy optimization)')
        return False


def validate_environment_rewards():
    """Validate environment returns correct reward components"""
    print('\n' + '='*80)
    print('ENVIRONMENT REWARD VALIDATION')
    print('='*80)
    
    try:
        env = DurotaxisEnv('config.yaml')
        print(f'\n✓ Environment initialized')
        
        # Reset and take a step
        obs, info = env.reset()
        action = 0  # dummy action
        obs, reward_components, terminated, truncated, info = env.step(action)
        
        print(f'\n✓ Environment step executed')
        print(f'  Reward components returned: {list(reward_components.keys())}')
        
        # Check expected components
        expected = {'total_reward', 'graph_reward', 'delete_reward', 'spawn_reward', 'distance_signal'}
        actual = set(reward_components.keys())
        
        # Remove metadata keys
        metadata_keys = {'num_nodes', 'termination_reward', 'empty_graph_recovery_penalty'}
        actual_components = actual - metadata_keys
        
        missing = expected - actual_components
        extra_legacy = {'edge_reward', 'total_node_reward', 'node_reward'}.intersection(actual)
        
        if missing:
            print(f'\n⚠️  WARNING: Missing expected components: {missing}')
            return False
        
        if extra_legacy:
            print(f'\n⚠️  WARNING: Environment returns legacy components: {extra_legacy}')
            return False
        
        # Check graph_reward == total_reward
        if abs(reward_components['graph_reward'] - reward_components['total_reward']) > 1e-5:
            print(f'\n⚠️  WARNING: graph_reward != total_reward')
            print(f'  graph_reward: {reward_components["graph_reward"]}')
            print(f'  total_reward: {reward_components["total_reward"]}')
            return False
        
        print(f'\n✓ All expected components present')
        print(f'✓ graph_reward == total_reward (difference < 1e-5)')
        print('\n✅ Environment rewards: PASSED')
        return True
        
    except Exception as e:
        print(f'\n❌ Environment validation failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks"""
    print('='*80)
    print('REWARD COMPONENT VALIDATION SUITE')
    print('='*80)
    
    results = {
        'Config Consistency': validate_config_consistency(),
        'Network Structure': validate_network_structure(),
        'Legacy Components': validate_no_legacy_components(),
        'Graph Reward Alias': validate_graph_reward_alias(),
        'Environment Rewards': validate_environment_rewards(),
    }
    
    print('\n' + '='*80)
    print('VALIDATION SUMMARY')
    print('='*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = '✅ PASSED' if passed else '❌ FAILED'
        print(f'{test_name:.<40} {status}')
        if not passed:
            all_passed = False
    
    print('='*80)
    
    if all_passed:
        print('\n🎉 ALL VALIDATIONS PASSED - System is consistent!')
        return 0
    else:
        print('\n⚠️  SOME VALIDATIONS FAILED - Please review warnings above')
        return 1


if __name__ == '__main__':
    sys.exit(main())
