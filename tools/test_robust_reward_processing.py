#!/usr/bin/env python3
"""
Test Suite: Robust Reward-to-Loss Processing
Tests the numerical stability improvements in reward processing across all modes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Import safe normalization utilities directly
from train import safe_standardize, safe_zero_center, RunningMeanStd

def test_safe_standardize():
    """Test safe standardization with various edge cases"""
    print("\n" + "="*70)
    print("TEST 1: safe_standardize()")
    print("="*70)
    
    # Test 1: Normal case
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = safe_standardize(x)
    print(f"✓ Normal case: mean={result.mean():.6f}, std={result.std():.6f}")
    assert torch.isfinite(result).all(), "Result contains NaN/Inf"
    
    # Test 2: Zero variance (all same values)
    x_zero_var = torch.tensor([5.0, 5.0, 5.0, 5.0])
    result_zero_var = safe_standardize(x_zero_var)
    print(f"✓ Zero variance: mean={result_zero_var.mean():.6f}, std={result_zero_var.std():.6f}")
    assert torch.isfinite(result_zero_var).all(), "Zero variance case failed"
    assert result_zero_var.std() < 1e-6, "Zero variance should return zero-centered values"
    
    # Test 3: Empty tensor
    x_empty = torch.tensor([])
    result_empty = safe_standardize(x_empty)
    print(f"✓ Empty tensor: shape={result_empty.shape}")
    assert result_empty.numel() == 0, "Empty tensor should remain empty"
    
    # Test 4: Large values (scale mixing test)
    x_large = torch.tensor([-1.0, -0.5, 500.0])
    result_large = safe_standardize(x_large)
    print(f"✓ Large scale mixing: mean={result_large.mean():.6f}, std={result_large.std():.6f}")
    assert torch.isfinite(result_large).all(), "Large value case failed"
    
    # Test 5: Very small variance (near-zero but not exactly zero)
    x_small_var = torch.tensor([1.0, 1.0000001, 1.0000002])
    result_small_var = safe_standardize(x_small_var, eps=1e-8)
    print(f"✓ Small variance: mean={result_small_var.mean():.6f}, std={result_small_var.std():.6f}")
    assert torch.isfinite(result_small_var).all(), "Small variance case failed"
    
    print("\n✅ All safe_standardize tests PASSED")
    return True


def test_safe_zero_center():
    """Test safe zero-centering"""
    print("\n" + "="*70)
    print("TEST 2: safe_zero_center()")
    print("="*70)
    
    # Test 1: Normal case
    x = torch.tensor([10.0, 20.0, 30.0])
    result = safe_zero_center(x)
    print(f"✓ Normal case: mean={result.mean():.6f}")
    assert abs(result.mean()) < 1e-6, "Mean should be near zero"
    
    # Test 2: Empty tensor
    x_empty = torch.tensor([])
    result_empty = safe_zero_center(x_empty)
    print(f"✓ Empty tensor: shape={result_empty.shape}")
    assert result_empty.numel() == 0, "Empty tensor should remain empty"
    
    print("\n✅ All safe_zero_center tests PASSED")
    return True


def test_running_mean_std():
    """Test RunningMeanStd for streaming normalization"""
    print("\n" + "="*70)
    print("TEST 3: RunningMeanStd")
    print("="*70)
    
    rms = RunningMeanStd(shape=())
    
    # Update with batches
    batch1 = np.array([1.0, 2.0, 3.0])
    batch2 = np.array([4.0, 5.0, 6.0])
    
    rms.update(batch1)
    print(f"✓ After batch 1: mean={rms.mean:.4f}, var={rms.var:.4f}")
    
    rms.update(batch2)
    print(f"✓ After batch 2: mean={rms.mean:.4f}, var={rms.var:.4f}")
    
    # Normalize new data
    new_data = np.array([3.5])
    normalized = rms.normalize(new_data)
    print(f"✓ Normalized value: {normalized[0]:.4f}")
    
    assert np.isfinite(rms.mean), "Mean is not finite"
    assert np.isfinite(rms.var), "Variance is not finite"
    assert np.isfinite(normalized).all(), "Normalized values contain NaN/Inf"
    
    print("\n✅ All RunningMeanStd tests PASSED")
    return True


def test_component_masking_simulation():
    """Simulate component masking for special modes"""
    print("\n" + "="*70)
    print("TEST 4: Component Masking (Special Modes)")
    print("="*70)
    
    # Simulate centroid_distance_only_mode: only 1 component active
    print("\nScenario: centroid_distance_only_mode")
    advantages = {
        'total_reward': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        'graph_reward': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        'spawn_reward': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        'delete_reward': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        'edge_reward': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        'total_node_reward': torch.tensor([-0.5, -1.2, -0.8, -1.5])  # Only centroid distance active
    }
    
    # Stack advantages
    advantage_list = [advantages[comp] for comp in advantages.keys()]
    advantage_tensor = torch.stack(advantage_list, dim=1)  # [batch, components]
    
    # Compute variance per component
    component_stds = advantage_tensor.std(dim=0)
    valid_mask = component_stds > 1e-8
    
    print(f"Component STDs: {component_stds}")
    print(f"Valid mask: {valid_mask}")
    print(f"Active components: {valid_mask.sum().item()}/6")
    
    assert valid_mask.sum() == 1, "Only 1 component should be active"
    assert valid_mask[5] == True, "total_node_reward should be active"
    
    # Simulate simple_delete_only_mode: ~3 components active
    print("\nScenario: simple_delete_only_mode")
    advantages_delete = {
        'total_reward': torch.tensor([5.0, 3.0, 4.0, 6.0]),  # Active
        'graph_reward': torch.tensor([2.0, 1.0, 1.5, 2.5]),  # Active
        'spawn_reward': torch.tensor([0.0, 0.0, 0.0, 0.0]),  # Inactive
        'delete_reward': torch.tensor([1.0, 0.5, 0.8, 1.2]),  # Active
        'edge_reward': torch.tensor([0.0, 0.0, 0.0, 0.0]),  # Inactive
        'total_node_reward': torch.tensor([0.0, 0.0, 0.0, 0.0])  # Inactive
    }
    
    advantage_list_delete = [advantages_delete[comp] for comp in advantages_delete.keys()]
    advantage_tensor_delete = torch.stack(advantage_list_delete, dim=1)
    
    component_stds_delete = advantage_tensor_delete.std(dim=0)
    valid_mask_delete = component_stds_delete > 1e-8
    
    print(f"Component STDs: {component_stds_delete}")
    print(f"Valid mask: {valid_mask_delete}")
    print(f"Active components: {valid_mask_delete.sum().item()}/6")
    
    assert valid_mask_delete.sum() == 3, "3 components should be active"
    
    # Simulate normal mode: all components active
    print("\nScenario: normal mode (all components)")
    advantages_normal = {
        'total_reward': torch.tensor([10.0, 8.0, 12.0, 9.0]),
        'graph_reward': torch.tensor([3.0, 2.0, 4.0, 2.5]),
        'spawn_reward': torch.tensor([1.0, 0.5, 1.5, 0.8]),
        'delete_reward': torch.tensor([2.0, 1.0, 2.5, 1.5]),
        'edge_reward': torch.tensor([0.5, 0.3, 0.7, 0.4]),
        'total_node_reward': torch.tensor([-0.5, -0.3, -0.6, -0.4])
    }
    
    advantage_list_normal = [advantages_normal[comp] for comp in advantages_normal.keys()]
    advantage_tensor_normal = torch.stack(advantage_list_normal, dim=1)
    
    component_stds_normal = advantage_tensor_normal.std(dim=0)
    valid_mask_normal = component_stds_normal > 1e-8
    
    print(f"Component STDs: {component_stds_normal}")
    print(f"Valid mask: {valid_mask_normal}")
    print(f"Active components: {valid_mask_normal.sum().item()}/6")
    
    assert valid_mask_normal.sum() == 6, "All 6 components should be active"
    
    print("\n✅ All component masking tests PASSED")
    return True


def test_ratio_guards():
    """Test PPO ratio guards against extreme values"""
    print("\n" + "="*70)
    print("TEST 5: PPO Ratio Guards")
    print("="*70)
    
    # Test 1: Normal ratio
    log_prob_diff_normal = torch.tensor(0.1)
    ratio_normal = torch.exp(log_prob_diff_normal)
    print(f"✓ Normal ratio: log_diff={log_prob_diff_normal:.4f}, ratio={ratio_normal:.4f}")
    assert torch.isfinite(ratio_normal), "Normal ratio should be finite"
    
    # Test 2: Very large positive log diff (would cause exp overflow)
    log_prob_diff_large = torch.tensor(50.0)  # Unclamped would be huge
    log_prob_diff_clamped = torch.clamp(log_prob_diff_large, -20.0, 20.0)
    ratio_clamped = torch.exp(log_prob_diff_clamped)
    ratio_safe = torch.clamp(ratio_clamped, 0.01, 100.0)
    print(f"✓ Large positive: unclamped={log_prob_diff_large:.1f}, clamped={log_prob_diff_clamped:.1f}, ratio={ratio_safe:.4f}")
    assert torch.isfinite(ratio_safe), "Clamped ratio should be finite"
    assert ratio_safe <= 100.0, "Ratio should be clamped to max 100"
    
    # Test 3: Very large negative log diff (would cause exp underflow)
    log_prob_diff_small = torch.tensor(-50.0)
    log_prob_diff_clamped_small = torch.clamp(log_prob_diff_small, -20.0, 20.0)
    ratio_small = torch.exp(log_prob_diff_clamped_small)
    ratio_safe_small = torch.clamp(ratio_small, 0.01, 100.0)
    print(f"✓ Large negative: unclamped={log_prob_diff_small:.1f}, clamped={log_prob_diff_clamped_small:.1f}, ratio={ratio_safe_small:.4f}")
    assert torch.isfinite(ratio_safe_small), "Small ratio should be finite"
    assert ratio_safe_small >= 0.01, "Ratio should be clamped to min 0.01"
    
    # Test 4: NaN handling
    ratio_nan = torch.tensor(float('nan'))
    ratio_fixed = torch.where(torch.isfinite(ratio_nan), ratio_nan, torch.tensor(1.0))
    print(f"✓ NaN handling: original=nan, fixed={ratio_fixed:.4f}")
    assert torch.isfinite(ratio_fixed), "NaN should be replaced with 1.0"
    
    # Test 5: Inf handling
    ratio_inf = torch.tensor(float('inf'))
    ratio_fixed_inf = torch.where(torch.isfinite(ratio_inf), ratio_inf, torch.tensor(1.0))
    print(f"✓ Inf handling: original=inf, fixed={ratio_fixed_inf:.4f}")
    assert torch.isfinite(ratio_fixed_inf), "Inf should be replaced with 1.0"
    
    print("\n✅ All PPO ratio guard tests PASSED")
    return True


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("ROBUST REWARD-TO-LOSS PROCESSING TEST SUITE")
    print("="*70)
    print("Testing numerical stability improvements for all training modes:")
    print("  - normal mode")
    print("  - simple_delete_only_mode")
    print("  - centroid_distance_only_mode")
    print("="*70)
    
    tests = [
        ("Safe Standardize", test_safe_standardize),
        ("Safe Zero Center", test_safe_zero_center),
        ("Running Mean/Std", test_running_mean_std),
        ("Component Masking", test_component_masking_simulation),
        ("PPO Ratio Guards", test_ratio_guards),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with exception: {e}")
            results.append((name, "FAILED"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(status == "PASSED" for _, status in results)
    print("="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Reward-to-loss processing is robust.")
    else:
        print("⚠️  SOME TESTS FAILED! Please review the output above.")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
