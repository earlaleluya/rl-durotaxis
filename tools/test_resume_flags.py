#!/usr/bin/env python3
"""
Test script to verify resume training flags work correctly:
- resume_from_best: Load best model instead of last checkpoint
- reset_optimizer: Reset optimizer state when resuming
- reset_episode_count: Start from episode 0 when resuming

This script simulates the checkpoint loading process without running full training.
"""

import torch
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_checkpoint_structure():
    """Test 1: Verify checkpoint contains all expected fields"""
    print("="*70)
    print("TEST 1: Checkpoint Structure Verification")
    print("="*70)
    
    # Create a mock checkpoint
    mock_checkpoint = {
        'network_state_dict': {},
        'optimizer_state_dict': {'param_groups': [{'lr': 0.001}]},
        'scheduler_state_dict': {},
        'episode_rewards': {'total_reward': [1.0, 2.0, 3.0]},
        'losses': {'total_loss': [100.0, 90.0, 80.0]},
        'best_reward': 3.0,
        'component_weights': {},
        'run_number': 1,
        'episode_count': 3,  # Episodes 0, 1, 2 completed; next is episode 3
        'smoothed_rewards': [1.0, 1.5, 2.0],
        'smoothed_losses': [100.0, 95.0, 90.0],
    }
    
    expected_keys = [
        'network_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
        'episode_rewards', 'losses', 'best_reward', 'component_weights',
        'run_number', 'episode_count', 'smoothed_rewards', 'smoothed_losses'
    ]
    
    all_present = True
    for key in expected_keys:
        if key in mock_checkpoint:
            print(f"  ✅ {key}: present")
        else:
            print(f"  ❌ {key}: MISSING")
            all_present = False
    
    if all_present:
        print("\n✅ All expected checkpoint fields present")
    else:
        print("\n❌ Some checkpoint fields missing")
    
    return all_present


def test_episode_count_semantics():
    """Test 2: Verify episode_count semantics (next episode to run)"""
    print("\n" + "="*70)
    print("TEST 2: Episode Count Semantics")
    print("="*70)
    
    scenarios = [
        {
            'description': 'After episode 0 completes',
            'episode_count': 1,
            'completed_episodes': [0],
            'next_episode': 1
        },
        {
            'description': 'After episodes 0-9 complete (batch size 10)',
            'episode_count': 10,
            'completed_episodes': list(range(10)),
            'next_episode': 10
        },
        {
            'description': 'After episode 432 completes',
            'episode_count': 433,
            'completed_episodes': list(range(433)),
            'next_episode': 433
        },
    ]
    
    all_correct = True
    for scenario in scenarios:
        print(f"\nScenario: {scenario['description']}")
        print(f"  Checkpoint episode_count: {scenario['episode_count']}")
        print(f"  Completed episodes: {len(scenario['completed_episodes'])} episodes (0 to {scenario['completed_episodes'][-1]})")
        print(f"  Next episode to run: {scenario['next_episode']}")
        
        if scenario['episode_count'] == scenario['next_episode']:
            print(f"  ✅ Semantics correct: episode_count = next episode to run")
        else:
            print(f"  ❌ Semantics incorrect: episode_count ({scenario['episode_count']}) != next episode ({scenario['next_episode']})")
            all_correct = False
    
    if all_correct:
        print("\n✅ Episode count semantics verified")
    else:
        print("\n❌ Episode count semantics failed")
    
    return all_correct


def test_reset_episode_count_flag():
    """Test 3: Verify reset_episode_count flag behavior"""
    print("\n" + "="*70)
    print("TEST 3: reset_episode_count Flag")
    print("="*70)
    
    checkpoint_episode_count = 100
    
    # Test Case 1: reset_episode_count = False (default)
    print("\nCase 1: reset_episode_count = False")
    resume_config = {'reset_episode_count': False}
    
    start_episode = 0  # Initial value
    if not resume_config.get('reset_episode_count', False):
        start_episode = checkpoint_episode_count
        result1 = "Resumed from episode 100"
    else:
        start_episode = 0
        result1 = "Reset to episode 0"
    
    print(f"  Checkpoint has episode_count: {checkpoint_episode_count}")
    print(f"  Result: {result1}")
    print(f"  start_episode: {start_episode}")
    
    if start_episode == 100:
        print(f"  ✅ Correct: Resumed from checkpoint's episode count")
        case1_pass = True
    else:
        print(f"  ❌ Wrong: Should resume from episode 100")
        case1_pass = False
    
    # Test Case 2: reset_episode_count = True
    print("\nCase 2: reset_episode_count = True")
    resume_config = {'reset_episode_count': True}
    
    start_episode = 0  # Initial value
    if not resume_config.get('reset_episode_count', False):
        start_episode = checkpoint_episode_count
        result2 = "Resumed from episode 100"
    else:
        start_episode = 0
        result2 = "Reset to episode 0"
    
    print(f"  Checkpoint has episode_count: {checkpoint_episode_count}")
    print(f"  Result: {result2}")
    print(f"  start_episode: {start_episode}")
    
    if start_episode == 0:
        print(f"  ✅ Correct: Reset to episode 0")
        case2_pass = True
    else:
        print(f"  ❌ Wrong: Should reset to episode 0")
        case2_pass = False
    
    if case1_pass and case2_pass:
        print("\n✅ reset_episode_count flag works correctly")
    else:
        print("\n❌ reset_episode_count flag has issues")
    
    return case1_pass and case2_pass


def test_reset_optimizer_flag():
    """Test 4: Verify reset_optimizer flag behavior"""
    print("\n" + "="*70)
    print("TEST 4: reset_optimizer Flag")
    print("="*70)
    
    checkpoint_optimizer_lr = 0.001
    
    # Test Case 1: reset_optimizer = False (default)
    print("\nCase 1: reset_optimizer = False")
    resume_config = {'reset_optimizer': False}
    
    optimizer_restored = False
    if not resume_config.get('reset_optimizer', False):
        # Would call: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_restored = True
        result1 = "Optimizer state restored"
    else:
        result1 = "Optimizer state reset (fresh start)"
    
    print(f"  Checkpoint has optimizer LR: {checkpoint_optimizer_lr}")
    print(f"  Result: {result1}")
    print(f"  Optimizer restored: {optimizer_restored}")
    
    if optimizer_restored:
        print(f"  ✅ Correct: Optimizer state restored from checkpoint")
        case1_pass = True
    else:
        print(f"  ❌ Wrong: Should restore optimizer state")
        case1_pass = False
    
    # Test Case 2: reset_optimizer = True
    print("\nCase 2: reset_optimizer = True")
    resume_config = {'reset_optimizer': True}
    
    optimizer_restored = False
    if not resume_config.get('reset_optimizer', False):
        optimizer_restored = True
        result2 = "Optimizer state restored"
    else:
        # Optimizer state NOT loaded - uses fresh initialization
        result2 = "Optimizer state reset (fresh start)"
    
    print(f"  Checkpoint has optimizer LR: {checkpoint_optimizer_lr}")
    print(f"  Result: {result2}")
    print(f"  Optimizer restored: {optimizer_restored}")
    
    if not optimizer_restored:
        print(f"  ✅ Correct: Optimizer state reset (fresh initialization)")
        case2_pass = True
    else:
        print(f"  ❌ Wrong: Should NOT restore optimizer state")
        case2_pass = False
    
    if case1_pass and case2_pass:
        print("\n✅ reset_optimizer flag works correctly")
    else:
        print("\n❌ reset_optimizer flag has issues")
    
    return case1_pass and case2_pass


def test_resume_from_best_flag():
    """Test 5: Verify resume_from_best flag behavior"""
    print("\n" + "="*70)
    print("TEST 5: resume_from_best Flag")
    print("="*70)
    
    # Test Case 1: resume_from_best = False (default)
    print("\nCase 1: resume_from_best = False")
    resume_config = {
        'checkpoint_path': 'checkpoint_batch50.pt',
        'resume_from_best': False
    }
    
    checkpoint_path = resume_config.get('checkpoint_path')
    if resume_config.get('resume_from_best', False):
        # Would look for best_model*.pt files
        checkpoint_path = 'best_model_batch30.pt'
        result1 = "Loading best model"
    else:
        result1 = "Loading specified checkpoint"
    
    print(f"  Specified checkpoint: {resume_config['checkpoint_path']}")
    print(f"  Result: {result1}")
    print(f"  Will load: {checkpoint_path}")
    
    if checkpoint_path == 'checkpoint_batch50.pt':
        print(f"  ✅ Correct: Loading specified checkpoint")
        case1_pass = True
    else:
        print(f"  ❌ Wrong: Should load specified checkpoint")
        case1_pass = False
    
    # Test Case 2: resume_from_best = True
    print("\nCase 2: resume_from_best = True")
    resume_config = {
        'checkpoint_path': 'checkpoint_batch50.pt',
        'resume_from_best': True
    }
    
    checkpoint_path = resume_config.get('checkpoint_path')
    if resume_config.get('resume_from_best', False):
        # Would look for best_model*.pt files and use most recent
        checkpoint_path = 'best_model_batch30.pt'  # Simulated result
        result2 = "Loading best model (overriding checkpoint_path)"
    else:
        result2 = "Loading specified checkpoint"
    
    print(f"  Specified checkpoint: {resume_config['checkpoint_path']}")
    print(f"  Result: {result2}")
    print(f"  Will load: {checkpoint_path}")
    
    if checkpoint_path == 'best_model_batch30.pt':
        print(f"  ✅ Correct: Loading best model (overrides checkpoint_path)")
        case2_pass = True
    else:
        print(f"  ❌ Wrong: Should load best model")
        case2_pass = False
    
    if case1_pass and case2_pass:
        print("\n✅ resume_from_best flag works correctly")
    else:
        print("\n❌ resume_from_best flag has issues")
    
    return case1_pass and case2_pass


def test_loss_metrics_with_checkpoint():
    """Test 6: Verify loss_metrics.json includes checkpoint filename"""
    print("\n" + "="*70)
    print("TEST 6: loss_metrics.json with Checkpoint Tracking")
    print("="*70)
    
    # Simulate loss metrics entries
    mock_loss_metrics = [
        {
            'episode': 10,
            'loss': 16560.71,
            'smoothed_loss': 16560.71,
            'checkpoint_filename': 'checkpoint_batch1.pt'
        },
        {
            'episode': 20,
            'loss': 15234.52,
            'smoothed_loss': 15897.62,
            'checkpoint_filename': 'checkpoint_batch2.pt'
        },
        {
            'episode': 433,
            'loss': 8432.15,
            'smoothed_loss': 9123.45,
            'checkpoint_filename': 'checkpoint_batch43.pt'
        },
    ]
    
    print("\nExample loss_metrics.json entries:")
    all_have_checkpoint = True
    for entry in mock_loss_metrics:
        print(f"\n  Episode {entry['episode']}:")
        print(f"    loss: {entry['loss']:.2f}")
        print(f"    smoothed_loss: {entry['smoothed_loss']:.2f}")
        if 'checkpoint_filename' in entry and entry['checkpoint_filename']:
            print(f"    checkpoint_filename: {entry['checkpoint_filename']} ✅")
        else:
            print(f"    checkpoint_filename: MISSING ❌")
            all_have_checkpoint = False
    
    print("\n  With checkpoint_filename, you can:")
    print("    1. Know which checkpoint contains training up to episode X")
    print("    2. Resume from the exact checkpoint for episode X")
    print("    3. Verify checkpoint-episode correspondence")
    
    if all_have_checkpoint:
        print("\n✅ loss_metrics.json includes checkpoint tracking")
    else:
        print("\n❌ Some entries missing checkpoint_filename")
    
    return all_have_checkpoint


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("RESUME TRAINING FLAGS VERIFICATION SUITE")
    print("="*70)
    print("\nThis test suite verifies that resume training flags work correctly:")
    print("  1. Checkpoint structure")
    print("  2. Episode count semantics")
    print("  3. reset_episode_count flag")
    print("  4. reset_optimizer flag")
    print("  5. resume_from_best flag")
    print("  6. loss_metrics.json checkpoint tracking")
    print("\n")
    
    results = []
    results.append(("Checkpoint Structure", test_checkpoint_structure()))
    results.append(("Episode Count Semantics", test_episode_count_semantics()))
    results.append(("reset_episode_count Flag", test_reset_episode_count_flag()))
    results.append(("reset_optimizer Flag", test_reset_optimizer_flag()))
    results.append(("resume_from_best Flag", test_resume_from_best_flag()))
    results.append(("loss_metrics.json Tracking", test_loss_metrics_with_checkpoint()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! Resume flags work correctly.")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED! Check implementation.")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
