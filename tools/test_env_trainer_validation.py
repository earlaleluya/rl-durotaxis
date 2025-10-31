#!/usr/bin/env python3
"""
Test environment-to-trainer reward component validation.

Verifies that the validation guard correctly:
1. Detects legacy components (edge_reward, total_node_reward)
2. Detects missing expected components
3. Validates graph_reward == total_reward
4. Handles allowed extras (milestone_bonus, termination_reward, etc.)
5. Can be enabled/disabled via config flag
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import ConfigLoader


def test_1_validation_disabled_by_default():
    """Test that validation is disabled by default (no performance impact)"""
    print("\n" + "="*60)
    print("TEST 1: Validation Disabled by Default")
    print("="*60)
    
    config_loader = ConfigLoader('config.yaml')
    config = config_loader.config
    
    # Check config doesn't have validate_env_rewards flag (should be False by default)
    trainer_config = config.get('trainer', {})
    validate_flag = trainer_config.get('validate_env_rewards', False)
    
    if validate_flag:
        print("❌ Validation is enabled by default (should be disabled for production)")
        return False
    
    print("✓ Validation disabled by default (zero performance overhead)")
    print("  To enable validation, set trainer.validate_env_rewards: true in config")
    
    print("\n✅ PASSED: Validation disabled by default")
    return True


def test_2_validation_detects_legacy_components():
    """Test that validation detects legacy components"""
    print("\n" + "="*60)
    print("TEST 2: Detect Legacy Components")
    print("="*60)
    
    from train import DurotaxisTrainer
    
    config_loader = ConfigLoader('config.yaml')
    config_loader.config['trainer']['validate_env_rewards'] = True  # Enable validation
    
    trainer = DurotaxisTrainer(config_loader.config_path)
    
    # Test with legacy component
    bad_reward_components = {
        'total_reward': 1.0,
        'graph_reward': 1.0,
        'delete_reward': 0.5,
        'spawn_reward': 0.3,
        'distance_signal': 0.2,
        'edge_reward': 0.1,  # LEGACY - should trigger error
    }
    
    try:
        trainer.validate_reward_components(bad_reward_components, step_info="Test")
        print("❌ Failed to detect legacy component 'edge_reward'")
        return False
    except ValueError as e:
        if 'Legacy components' in str(e) and 'edge_reward' in str(e):
            print("✓ Correctly detected legacy component 'edge_reward'")
            print(f"  Error message: {str(e)[:100]}...")
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    
    # Test with another legacy component
    bad_reward_components2 = {
        'total_reward': 1.0,
        'graph_reward': 1.0,
        'delete_reward': 0.5,
        'spawn_reward': 0.3,
        'distance_signal': 0.2,
        'total_node_reward': 0.2,  # LEGACY - should trigger error
    }
    
    try:
        trainer.validate_reward_components(bad_reward_components2, step_info="Test")
        print("❌ Failed to detect legacy component 'total_node_reward'")
        return False
    except ValueError as e:
        if 'Legacy components' in str(e) and 'total_node_reward' in str(e):
            print("✓ Correctly detected legacy component 'total_node_reward'")
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    
    print("\n✅ PASSED: Legacy component detection works")
    return True


def test_3_validation_detects_missing_components():
    """Test that validation detects missing expected components"""
    print("\n" + "="*60)
    print("TEST 3: Detect Missing Components")
    print("="*60)
    
    from train import DurotaxisTrainer
    
    config_loader = ConfigLoader('config.yaml')
    config_loader.config['trainer']['validate_env_rewards'] = True
    
    trainer = DurotaxisTrainer(config_loader.config_path)
    
    # Missing delete_reward
    incomplete_rewards = {
        'total_reward': 1.0,
        'graph_reward': 1.0,
        # 'delete_reward': missing!
        'spawn_reward': 0.3,
        'distance_signal': 0.2,
    }
    
    try:
        trainer.validate_reward_components(incomplete_rewards, step_info="Test")
        print("❌ Failed to detect missing component 'delete_reward'")
        return False
    except ValueError as e:
        if 'Missing expected components' in str(e) and 'delete_reward' in str(e):
            print("✓ Correctly detected missing component 'delete_reward'")
            print(f"  Error message: {str(e)[:100]}...")
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    
    print("\n✅ PASSED: Missing component detection works")
    return True


def test_4_validation_checks_graph_reward_consistency():
    """Test that validation checks graph_reward == total_reward"""
    print("\n" + "="*60)
    print("TEST 4: Check graph_reward == total_reward")
    print("="*60)
    
    from train import DurotaxisTrainer
    
    config_loader = ConfigLoader('config.yaml')
    config_loader.config['trainer']['validate_env_rewards'] = True
    
    trainer = DurotaxisTrainer(config_loader.config_path)
    
    # graph_reward != total_reward
    inconsistent_rewards = {
        'total_reward': 1.0,
        'graph_reward': 0.9,  # WRONG - should equal total_reward
        'delete_reward': 0.5,
        'spawn_reward': 0.3,
        'distance_signal': 0.2,
    }
    
    try:
        trainer.validate_reward_components(inconsistent_rewards, step_info="Test")
        print("❌ Failed to detect graph_reward != total_reward")
        return False
    except ValueError as e:
        if 'graph_reward != total_reward' in str(e):
            print("✓ Correctly detected graph_reward != total_reward inconsistency")
            print(f"  Error message: {str(e)[:100]}...")
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    
    print("\n✅ PASSED: graph_reward consistency check works")
    return True


def test_5_validation_allows_extras():
    """Test that validation allows expected extras like milestone_bonus"""
    print("\n" + "="*60)
    print("TEST 5: Allow Expected Extra Components")
    print("="*60)
    
    from train import DurotaxisTrainer
    
    config_loader = ConfigLoader('config.yaml')
    config_loader.config['trainer']['validate_env_rewards'] = True
    
    trainer = DurotaxisTrainer(config_loader.config_path)
    
    # Valid rewards with allowed extras
    rewards_with_extras = {
        'total_reward': 1.0,
        'graph_reward': 1.0,
        'delete_reward': 0.5,
        'spawn_reward': 0.3,
        'distance_signal': 0.2,
        'milestone_bonus': 2.0,  # Allowed extra
        'termination_reward': 500.0,  # Allowed extra
        'num_nodes': 5,  # Allowed extra
    }
    
    try:
        trainer.validate_reward_components(rewards_with_extras, step_info="Test")
        print("✓ Correctly allowed milestone_bonus")
        print("✓ Correctly allowed termination_reward")
        print("✓ Correctly allowed num_nodes")
    except ValueError as e:
        print(f"❌ Incorrectly rejected allowed extras: {e}")
        return False
    
    print("\n✅ PASSED: Allowed extras validation works")
    return True


def test_6_validation_passes_correct_components():
    """Test that validation passes with correct components"""
    print("\n" + "="*60)
    print("TEST 6: Validation Passes Correct Components")
    print("="*60)
    
    from train import DurotaxisTrainer
    
    config_loader = ConfigLoader('config.yaml')
    config_loader.config['trainer']['validate_env_rewards'] = True
    
    trainer = DurotaxisTrainer(config_loader.config_path)
    
    # Correct reward components (what environment should return)
    correct_rewards = {
        'total_reward': 1.0,
        'graph_reward': 1.0,  # Same as total_reward
        'delete_reward': 0.5,
        'spawn_reward': 0.3,
        'distance_signal': 0.2,
    }
    
    try:
        trainer.validate_reward_components(correct_rewards, step_info="Test")
        print("✓ Validation passed with correct components")
    except ValueError as e:
        print(f"❌ Validation failed on correct components: {e}")
        return False
    
    # Test with allowed extras too
    correct_rewards_with_extras = {
        'total_reward': 2.5,
        'graph_reward': 2.5,
        'delete_reward': 1.0,
        'spawn_reward': 0.75,
        'distance_signal': 0.5,
        'milestone_bonus': 0.25,
        'termination_reward': 0.0,
    }
    
    try:
        trainer.validate_reward_components(correct_rewards_with_extras, step_info="Test")
        print("✓ Validation passed with correct components + extras")
    except ValueError as e:
        print(f"❌ Validation failed with extras: {e}")
        return False
    
    print("\n✅ PASSED: Correct components pass validation")
    return True


def main():
    """Run all env-to-trainer validation tests"""
    print("\n" + "="*60)
    print("ENV-TO-TRAINER VALIDATION TESTS")
    print("="*60)
    print("\nVerifying:")
    print("1. Validation disabled by default (no performance impact)")
    print("2. Detects legacy components (edge_reward, total_node_reward)")
    print("3. Detects missing expected components")
    print("4. Validates graph_reward == total_reward")
    print("5. Allows expected extras (milestone_bonus, termination_reward, etc.)")
    print("6. Passes with correct components")
    
    results = []
    
    # Run tests
    results.append(("Disabled by Default", test_1_validation_disabled_by_default()))
    results.append(("Detect Legacy Components", test_2_validation_detects_legacy_components()))
    results.append(("Detect Missing Components", test_3_validation_detects_missing_components()))
    results.append(("Check graph_reward Consistency", test_4_validation_checks_graph_reward_consistency()))
    results.append(("Allow Expected Extras", test_5_validation_allows_extras()))
    results.append(("Pass Correct Components", test_6_validation_passes_correct_components()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Env-trainer validation works correctly!")
        print("\n✅ Validation Features:")
        print("   - Disabled by default (zero overhead in production)")
        print("   - Detects legacy components")
        print("   - Detects missing components")
        print("   - Validates graph_reward == total_reward")
        print("   - Allows expected extras")
        print("   - Can be enabled via config: trainer.validate_env_rewards: true")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Review required")
        return 1


if __name__ == '__main__':
    sys.exit(main())
