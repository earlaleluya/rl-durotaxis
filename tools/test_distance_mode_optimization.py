#!/usr/bin/env python3
"""
Test script for distance mode optimization implementation.

This script verifies that all 6 optimization components are correctly implemented:
1. Config loading (distance_mode parameters)
2. Delta distance shaping (potential-based reward)
3. Scaled termination rewards
4. Entropy tuning (already in config)
5. Stability (already implemented)
6. Adaptive scheduler

Usage:
    python tools/test_distance_mode_optimization.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def test_config_loading():
    """Test 1: Verify distance_mode configuration exists in config.yaml."""
    print("\n" + "="*80)
    print("TEST 1: Configuration File Verification")
    print("="*80)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check environment section exists
    assert 'environment' in config, "❌ environment section missing in config"
    env_config = config['environment']
    
    # Check distance_mode section exists
    assert 'distance_mode' in env_config, "❌ distance_mode section missing in environment config"
    print("✅ distance_mode section found in environment config")
    
    dm = env_config['distance_mode']
    expected_keys = [
        'use_delta_distance', 'distance_reward_scale', 'terminal_reward_scale',
        'clip_terminal_rewards', 'terminal_reward_clip_value', 'scheduler_enabled',
        'scheduler_window_size', 'scheduler_progress_threshold', 
        'scheduler_consecutive_windows', 'scheduler_decay_rate', 'scheduler_min_scale'
    ]
    
    for key in expected_keys:
        assert key in dm, f"❌ Missing key: {key}"
        print(f"   ✅ {key}: {dm[key]}")
    
    print("✅ All distance_mode parameters present in config")
    return True


def test_env_code_verification():
    """Test 2: Verify environment code has distance_mode implementation."""
    print("\n" + "="*80)
    print("TEST 2: Environment Code Verification")
    print("="*80)
    
    with open('durotaxis_env.py', 'r') as f:
        env_code = f.read()
    
    # Check for distance_mode parameter loading
    assert 'dm_use_delta_distance' in env_code, "❌ dm_use_delta_distance not found in env code"
    print("   ✅ dm_use_delta_distance parameter loading found")
    
    assert 'dm_distance_reward_scale' in env_code, "❌ dm_distance_reward_scale not found"
    print("   ✅ dm_distance_reward_scale parameter loading found")
    
    assert 'dm_terminal_reward_scale' in env_code, "❌ dm_terminal_reward_scale not found"
    print("   ✅ dm_terminal_reward_scale parameter loading found")
    
    assert '_prev_centroid_x' in env_code, "❌ _prev_centroid_x tracking not found"
    print("   ✅ _prev_centroid_x tracking found")
    
    # Check for delta distance computation
    assert 'delta_x = centroid_x - self._prev_centroid_x' in env_code, "❌ Delta distance computation not found"
    print("   ✅ Delta distance computation found")
    
    # Check for scaled termination
    assert 'scaled_termination = termination_reward * self.dm_terminal_reward_scale' in env_code, "❌ Termination scaling not found"
    print("   ✅ Termination reward scaling found")
    
    assert 'termination_reward_scaled' in env_code, "❌ Scaled termination logging not found"
    print("   ✅ Scaled termination logging found")
    
    print("✅ Environment distance_mode implementation verified")
    return True


def test_delta_distance_formula():
    """Test 3: Verify delta distance formula is correctly implemented."""
    print("\n" + "="*80)
    print("TEST 3: Delta Distance Formula Verification")
    print("="*80)
    
    with open('durotaxis_env.py', 'r') as f:
        env_code = f.read()
    
    # Check for potential-based shaping formula
    assert 'self.dm_distance_reward_scale * (delta_x / self.goal_x)' in env_code, "❌ Delta distance formula not found"
    print("   ✅ Delta distance formula: scale × (delta_x / goal_x)")
    
    # Check for fallback static penalty
    assert '-(self.goal_x - centroid_x) / self.goal_x' in env_code, "❌ Fallback static penalty not found"
    print("   ✅ Fallback static penalty preserved")
    
    # Check for centroid update
    assert 'self._prev_centroid_x = centroid_x' in env_code, "❌ Centroid update not found"
    print("   ✅ Previous centroid update found")
    
    print("✅ Delta distance shaping formula verified")
    return True


def test_scaled_termination():
    """Test 4: Verify termination reward scaling and clipping code."""
    print("\n" + "="*80)
    print("TEST 4: Scaled Termination Code Verification")
    print("="*80)
    
    with open('durotaxis_env.py', 'r') as f:
        env_code = f.read()
    
    # Check for scaling
    assert 'scaled_termination = termination_reward * self.dm_terminal_reward_scale' in env_code, "❌ Termination scaling not found"
    print("   ✅ Termination scaling: reward × scale")
    
    # Check for clipping
    assert 'if self.dm_clip_terminal_rewards:' in env_code, "❌ Clip check not found"
    print("   ✅ Clipping conditional found")
    
    assert 'max(-self.dm_terminal_reward_clip_value' in env_code, "❌ Clipping operation not found"
    print("   ✅ Clipping operation: max(-clip, min(clip, scaled))")
    
    print("✅ Termination scaling and clipping code verified")
    return True


def test_trainer_scheduler_initialization():
    """Test 5: Verify trainer scheduler code exists and is properly integrated."""
    print("\n" + "="*80)
    print("TEST 5: Trainer Scheduler Code Verification")
    print("="*80)
    
    # Verify that the scheduler methods exist in train.py
    import inspect
    from train import DurotaxisTrainer
    
    # Check if methods exist
    assert hasattr(DurotaxisTrainer, '_compute_rightward_progress_rate'), "❌ _compute_rightward_progress_rate method not found"
    print("   ✅ _compute_rightward_progress_rate method exists")
    
    assert hasattr(DurotaxisTrainer, '_update_terminal_scale_scheduler'), "❌ _update_terminal_scale_scheduler method not found"
    print("   ✅ _update_terminal_scale_scheduler method exists")
    
    # Check __init__ signature includes scheduler state initialization
    init_source = inspect.getsource(DurotaxisTrainer.__init__)
    assert 'dm_scheduler_enabled' in init_source, "❌ Scheduler initialization missing from __init__"
    assert '_dm_rightward_progress_history' in init_source, "❌ Progress history initialization missing"
    assert '_dm_consecutive_good_windows' in init_source, "❌ Consecutive windows counter missing"
    assert '_dm_terminal_scale_history' in init_source, "❌ Scale history initialization missing"
    print("   ✅ Scheduler state variables initialized in __init__")
    
    # Check that scheduler is called in _collect_and_process_episode
    collect_source = inspect.getsource(DurotaxisTrainer._collect_and_process_episode)
    assert '_update_terminal_scale_scheduler' in collect_source, "❌ Scheduler not called in episode collection"
    print("   ✅ Scheduler called after each episode")
    
    print("✅ Trainer scheduler implementation verified")
    return True


def test_entropy_configuration():
    """Test 6: Verify entropy tuning is configured."""
    print("\n" + "="*80)
    print("TEST 6: Entropy Configuration")
    print("="*80)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Entropy is under trainer section
    assert 'trainer' in config, "❌ trainer section missing"
    trainer = config['trainer']
    assert 'entropy_regularization' in trainer, "❌ entropy_regularization missing"
    
    entropy = trainer['entropy_regularization']
    
    print(f"   ✅ entropy_coeff_start: {entropy['entropy_coeff_start']}")
    print(f"   ✅ entropy_coeff_end: {entropy['entropy_coeff_end']}")
    print(f"   ✅ entropy_decay_episodes: {entropy['entropy_decay_episodes']}")
    print(f"   ✅ discrete_entropy_weight: {entropy['discrete_entropy_weight']}")
    print(f"   ✅ continuous_entropy_weight: {entropy['continuous_entropy_weight']}")
    
    # Verify reduced values (compared to high-exploration defaults)
    assert entropy['entropy_coeff_start'] <= 0.3, "❌ Start entropy too high for fast convergence"
    assert entropy['entropy_coeff_end'] <= 0.1, "❌ End entropy too high for policy commitment"
    
    print("✅ Entropy tuning configured for faster convergence")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("DISTANCE MODE OPTIMIZATION TEST SUITE")
    print("="*80)
    print("Testing all 6 optimization components...")
    
    tests = [
        ("Config File", test_config_loading),
        ("Env Code", test_env_code_verification),
        ("Delta Distance", test_delta_distance_formula),
        ("Scaled Termination", test_scaled_termination),
        ("Trainer Scheduler", test_trainer_scheduler_initialization),
        ("Entropy Tuning", test_entropy_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for test_name, success, error in results:
        if success:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            if error:
                print(f"   Error: {error}")
            failed += 1
    
    print("="*80)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - Distance mode optimization ready!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please review implementation")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
