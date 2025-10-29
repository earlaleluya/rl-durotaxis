# Experimental Modes Verification

**Date**: 2024-01-XX  
**Status**: ✅ ALL MODES VERIFIED  
**Architecture**: Delete Ratio (5D Continuous Actions)

## Overview

This document verifies that the delete ratio codebase is fully compatible with all 4 experimental reward modes used for research experiments.

## Test Results

### ✅ Discrete Action Check: PASSED
- No discrete action bugs found in `durotaxis_env.py`
- `Discrete(1)` dummy action space is acceptable (actual actions come from policy)
- All references to old discrete action architecture removed

### Experimental Modes Tested

#### 1. ✅ simple_delete_only_mode (PASSED)
**Configuration**:
```yaml
simple_delete_only_mode: true
centroid_distance_only_mode: false
```

**Active Components**:
- ✓ `graph_reward`: Growth penalty (Rule 0)
- ✓ `delete_reward`: Delete penalties (Rules 1 & 2)
- ✗ All other components disabled

**Purpose**: Learn efficient node deletion policies without other reward signals  
**Use Case**: Ablation studies focusing on deletion mechanics

---

#### 2. ✅ centroid_distance_only_mode (PASSED)
**Configuration**:
```yaml
simple_delete_only_mode: false
centroid_distance_only_mode: true
```

**Active Components**:
- ✓ `graph_reward`: Distance signal to goal
- ✗ All other components disabled

**Reward Mechanism**:
- Uses delta distance shaping (potential-based)
- Formula: `reward = scale × (cx_t - cx_{t-1}) / goal_x`
- Positive when moving right, negative when moving left

**Purpose**: Learn rightward migration through distance feedback only  
**Use Case**: Ablation studies focusing on spatial navigation

---

#### 3. ✅ Combined Mode (PASSED)
**Configuration**:
```yaml
simple_delete_only_mode: true
centroid_distance_only_mode: true
```

**Active Components**:
- ✓ `graph_reward`: Distance signal + delete penalties
- ✓ `delete_reward`: Delete penalties (for logging)
- ✗ All other components disabled

**Reward Composition**:
```
total_reward = distance_signal + delete_penalties_total

distance_signal = scale × (cx_t - cx_{t-1}) / goal_x  # Rightward migration
delete_penalties = growth_penalty + delete_penalty    # Efficient node management
```

**Purpose**: Learn rightward migration + efficient deletion simultaneously  
**Use Case**: Combined sparse reward shaping

---

#### 4. ✅ Normal Mode (PASSED)
**Configuration**:
```yaml
simple_delete_only_mode: false
centroid_distance_only_mode: false
```

**Active Components**:
- ✓ `graph_reward`: Always active
- ✓ `delete_reward`: Usually active
- ✓ `total_node_reward`: Usually active
- ✓ `survival_reward`: Usually active
- ~ `spawn_reward`: Conditional (may not trigger every step)
- ~ `efficiency_reward`: Conditional
- ~ `milestone_reward`: Conditional (triggers at distance thresholds)

**Purpose**: Full reward system with all components  
**Use Case**: Standard training with rich feedback

---

## Delete Ratio Compatibility

All 4 modes work correctly with the delete ratio architecture:

### Action Space
- **Type**: 5D continuous
- **Components**: `[delete_ratio, gamma, alpha, noise, theta]`
- **Delete Ratio**: ∈ [0.0, 0.5] controls fraction of nodes to delete
- **Spawn Parameters**: Single global parameters for all nodes

### Verification
```bash
✓ No discrete action bugs
✓ Environment uses delete_ratio architecture (5D continuous from policy)
✓ All reward components calculated correctly
✓ Mode switching works without errors
```

---

## Test Script

Location: `tools/test_experimental_modes.py`

**Features**:
- Tests all 4 experimental modes
- Verifies reward component activation/deactivation
- Checks for discrete action bugs
- Validates delete ratio architecture compatibility

**Usage**:
```bash
cd /home/arl_eifer/github/rl-durotaxis
python tools/test_experimental_modes.py
```

**Expected Output**:
```
############################################################
# TEST SUMMARY
############################################################

Discrete Action Check: ✅ PASSED

Experimental Mode Tests:
  Mode 1 (simple_delete_only)    ✅ PASSED
  Mode 2 (centroid_distance_only) ✅ PASSED
  Mode 3 (combined)              ✅ PASSED
  Mode 4 (normal)                ✅ PASSED

============================================================
✅ ALL TESTS PASSED!
============================================================

The delete ratio codebase is compatible with all 4 experimental modes:
  1. ✓ simple_delete_only_mode
  2. ✓ centroid_distance_only_mode
  3. ✓ Combined mode
  4. ✓ Normal mode

You can safely train with any of these modes.
```

---

## Configuration Examples

### Mode 1: simple_delete_only_mode
```yaml
environment:
  simple_delete_only_mode: true
  centroid_distance_only_mode: false
  include_termination_rewards: false
```

### Mode 2: centroid_distance_only_mode
```yaml
environment:
  simple_delete_only_mode: false
  centroid_distance_only_mode: true
  include_termination_rewards: false
  distance_mode:
    use_delta_distance: true
    distance_reward_scale: 5.0
```

### Mode 3: Combined
```yaml
environment:
  simple_delete_only_mode: true
  centroid_distance_only_mode: true
  include_termination_rewards: false
  distance_mode:
    use_delta_distance: true
    distance_reward_scale: 5.0
    delete_penalty_scale: 1.0
```

### Mode 4: Normal
```yaml
environment:
  simple_delete_only_mode: false
  centroid_distance_only_mode: false
  include_termination_rewards: false
```

---

## Reward Component Reference

### Always Active (in some modes)
- `graph_reward`: Base graph-level reward
- `delete_reward`: Penalties for improper deletion
- `total_node_reward`: Sum of node-level rewards
- `survival_reward`: Per-step survival bonus

### Conditional
- `spawn_reward`: Only when nodes spawn successfully
- `efficiency_reward`: Only when deletion efficiency meets threshold
- `milestone_reward`: Only when reaching distance milestones

---

## Implementation Details

### Mode Detection (durotaxis_env.py)
```python
# Line 1002-1004
is_normal_mode = not self.simple_delete_only_mode and not self.centroid_distance_only_mode
is_combined_mode = self.simple_delete_only_mode and self.centroid_distance_only_mode
is_special_mode_with_termination = (self.simple_delete_only_mode or self.centroid_distance_only_mode) and self.include_termination_rewards
```

### Reward Computation (durotaxis_env.py)
```python
# Lines 1312-1450: Mode-specific reward calculation
if self.simple_delete_only_mode and self.centroid_distance_only_mode:
    # Combined mode: distance + delete
    total_reward = distance_signal + delete_penalties_total
    
elif self.simple_delete_only_mode:
    # Simple delete mode: penalties only
    total_reward = growth_penalty_only + delete_penalty_only
    
elif self.centroid_distance_only_mode:
    # Distance mode: distance signal only
    total_reward = distance_signal
    
# else: normal mode (all components active)
```

---

## Conclusion

✅ **All 4 experimental modes are fully functional with the delete ratio architecture**

The codebase is ready for:
1. Training with any experimental mode
2. Ablation studies comparing reward configurations
3. Research experiments on reward shaping strategies

No further modifications needed for mode compatibility.

---

## Related Documentation
- `notes/SIMPLE_DELETE_MODE_GUIDE.md` - Details on Mode 1
- `notes/DISTANCE_MODE_OPTIMIZATION.md` - Details on Mode 2
- `notes/COMBINED_MODE_SETUP.md` - Details on Mode 3
- `notes/REWARD_SYSTEM_IMPROVEMENTS.md` - Normal mode details
- `notes/ROBUST_REWARD_PROCESSING.md` - Reward computation architecture
