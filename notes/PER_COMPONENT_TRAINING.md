# Per-Component Training Implementation

**Date**: 2025-11-01  
**Status**: ✅ Implemented and Tested

## Overview

Implemented **per-component training** so the agent learns separate value estimates for each reward component (spawn_reward, delete_reward, etc.) and can improve on specific components independently.

## Problem Statement

Previously:
- GAE computed returns/advantages using only `total_reward` (scalar)
- All critic value heads trained on the same total-return target
- Agent couldn't differentiate which component needed improvement
- Spawn reward influenced learning only as part of the total, not as a dedicated signal

## Solution

Modified the training pipeline to compute **component-specific returns and advantages**:

1. **Separate GAE per component**: Each reward component gets its own TD error and GAE computation
2. **Component-specific value targets**: Critic's `spawn_value` head trained on spawn-only returns, `delete_value` on delete-only returns, etc.
3. **Independent learning signals**: Agent can now identify and improve on weak components

## Changes Made

### 1. `TrajectoryBuffer.compute_returns_and_advantages_for_all_episodes()`
**Location**: `train.py` lines ~207-290

**Before**:
```python
# Computed scalar total_reward GAE
rewards_tensor = torch.tensor([r.get('total_reward', 0.0) for r in rewards])
# Single GAE loop
returns = ...  # List[Tensor]
advantages = ...  # List[Tensor]
```

**After**:
```python
# Compute GAE separately for each component
for component in component_names:
    component_rewards = [r.get(component, 0.0) for r in rewards]
    component_values = [v.get(component, 0.0) for v in values]
    # Separate GAE loop per component
    episode['returns'][component] = [...]  # Dict[component, List[Tensor]]
    episode['advantages'][component] = [...]
```

**Signature changed**:
```python
def compute_returns_and_advantages_for_all_episodes(
    self, gamma: float, gae_lambda: float, 
    component_names: List[str] = None  # NEW
)
```

### 2. `TrajectoryBuffer.get_batch_data()`
**Location**: `train.py` lines ~148-190

**Before**:
```python
all_returns = []  # List[Tensor]
all_advantages = []  # List[Tensor]
```

**After**:
```python
all_returns = {}  # Dict[component, List[Tensor]]
all_advantages = {}  # Dict[component, List[Tensor]]

# Concatenate per-component lists across episodes
for component in episode['returns']:
    all_returns[component].extend(episode['returns'][component])
```

### 3. `TrajectoryBuffer.create_minibatches()`
**Location**: `train.py` lines ~192-215

**Before**:
```python
for key, data in batch_data.items():
    minibatch[key] = [data[idx] for idx in batch_indices]
```

**After**:
```python
if key in ['returns', 'advantages']:
    # Handle per-component dict structure
    minibatch[key] = {}
    for component, component_data in data.items():
        minibatch[key][component] = [component_data[idx] for idx in batch_indices]
else:
    minibatch[key] = [data[idx] for idx in batch_indices]
```

### 4. `update_policy_minibatch()`
**Location**: `train.py` lines ~3056-3092

**Before**:
```python
# Duplicated scalar total return to all components
for component in self.component_names:
    returns_dict[component].append(ret_val)  # Same value!
    advantages_dict[component].append(adv_val)  # Same value!
```

**After**:
```python
# Use actual component-specific returns
for component in self.component_names:
    if component in returns and returns[component]:
        returns_dict[component] = torch.stack(returns[component])
    if component in advantages and advantages[component]:
        advantages_dict[component] = torch.stack(advantages[component])
```

**Signature changed**:
```python
def update_policy_minibatch(
    self, states: List[Dict], actions: List[Dict],
    returns: Dict[str, List[torch.Tensor]],  # Was: List[torch.Tensor]
    advantages: Dict[str, List[torch.Tensor]],  # Was: List[torch.Tensor]
    old_log_probs: List[Dict], old_values: List[Dict], episode: int = 0
)
```

### 5. Training loop call site
**Location**: `train.py` line ~3692

**Before**:
```python
self.trajectory_buffer.compute_returns_and_advantages_for_all_episodes(gamma, gae_lambda)
```

**After**:
```python
self.trajectory_buffer.compute_returns_and_advantages_for_all_episodes(gamma, gae_lambda, self.component_names)
```

## Testing

**Test file**: `tools/test_per_component_training.py`

Verified:
- ✅ Returns and advantages are dicts with per-component structure
- ✅ All components (total_reward, spawn_reward, delete_reward, etc.) have returns/advantages
- ✅ Component-specific returns are different (spawn ≠ delete)
- ✅ Batch data structure preserves per-component format
- ✅ Minibatch creation handles dict structure correctly

**Test output**:
```
Component-specific returns at step 0:
  Spawn return:  15.415
  Delete return: 11.021
✅ Component-specific returns are different (as expected)
```

## Benefits

### 1. **Component-Specific Learning**
The critic's value heads now learn what each component is worth:
- `spawn_value` head learns expected spawn rewards
- `delete_value` head learns expected delete rewards
- Each head gets its own training target

### 2. **Better Policy Improvement**
The agent can now:
- Identify which component is lacking (e.g., low delete_reward)
- Adjust actions to improve that specific component
- Balance multiple objectives more effectively

### 3. **More Informative Value Estimates**
Previously: All value heads predicted ~same total return  
Now: Each head specializes in its component, providing richer signal

### 4. **Advantage Weighting Still Works**
The `compute_enhanced_advantage_weights()` method still combines component advantages using learnable weights, so policy optimization benefits from all components while each critic head specializes.

## Backward Compatibility

The code handles legacy scalar returns gracefully:
```python
if isinstance(episode['returns'], dict):
    # Per-component structure
    ...
else:
    # Legacy scalar structure - convert to dict
    all_returns['total_reward'] = episode['returns']
```

## Example Training Scenario

**Scenario**: Agent is good at spawning but poor at deleting

**Before (scalar GAE)**:
- Total return: spawn(+20) + delete(-10) = +10
- All value heads predict: ~10
- Agent gets positive advantage → continues same behavior

**After (per-component GAE)**:
- spawn_value head predicts: +20 (accurate)
- delete_value head predicts: -10 (accurate)
- Agent sees delete component is negative
- Policy adjusts to improve deletion behavior

## Configuration

No config changes needed. The system automatically uses per-component training if `component_names` is defined in the trainer (which it is by default).

Components used:
```python
self.component_names = [
    'total_reward',
    'graph_reward',
    'spawn_reward',
    'delete_reward',
    'edge_reward',
    'total_node_reward'
]
```

## Future Enhancements

Possible extensions:
1. **Component prioritization**: Weight advantages by component importance
2. **Curriculum learning**: Start with total reward, gradually add component-specific signals
3. **Component-specific entropy**: Different exploration levels per component
4. **Diagnostic logging**: Track which components improve over training

## Summary

✅ **Implemented**: Per-component GAE computation  
✅ **Tested**: All data structures work correctly  
✅ **Benefits**: Agent can now improve on specific reward components  
✅ **Backward compatible**: Handles both dict and scalar returns

The agent will now learn to identify and improve on weak components (e.g., if spawn rewards are lacking, it will adjust spawn-related actions specifically).
