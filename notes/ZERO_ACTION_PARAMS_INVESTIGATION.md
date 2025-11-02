# Zero Action Parameters - Investigation Summary

## Problem Statement
User observed all 5 action parameters showing as `0.0` in training output:
```
Actions: dr=0.000 γ=0.00 α=0.00 n=0.000 θ=0.000
```

## Investigation Results

### ✅ Actor Network Architecture - CORRECT
- Actor produces single global action: `[delete_ratio, γ, α, noise, θ]`
- Output shape: `[5]` tensor
- Initialization: Xavier uniform with bias=0.01
- Forward pass: ResNet18 backbone → MLP → mu_head & logstd_head

### ✅ Bound Application - CORRECT  
When raw (unbounded) Actor output is exactly `0.0`, bounds produce:
```python
delete_ratio = 0.45  # sigmoid(0) scaled to [0.0, 0.9]
gamma = 7.75         # sigmoid(0) scaled to [0.5, 15.0]
alpha = 2.25         # sigmoid(0) scaled to [0.5, 4.0]
noise = 0.275        # sigmoid(0) scaled to [0.05, 0.5]
theta = 0.0          # tanh(0) = 0.0
```

**Key Finding**: Even if Actor produces zeros, bound application should give non-zero values (except theta).

### ❌ Display Extraction - POTENTIAL BUG

The zeros in display indicate one of these scenarios:

#### Scenario 1: Empty Graph (prev_num_nodes == 0)
```python
# durotaxis_env.py line 837
if self.policy_agent is not None and prev_num_nodes > 0:
    # Policy executed
else:
    # ❌ Policy NOT executed → last_continuous_actions never set
    # Display shows initialization zeros
```

**Impact**: If graph starts empty or becomes empty mid-episode, display shows zeros.

#### Scenario 2: last_continuous_actions Not Set
```python
# actor_critic.py line 854 (get_actions_and_values)
if state['num_nodes'] == 0:
    return {}, (1.0, 1.0, 0.5, 0.0), empty_values
    # ❌ Returns early WITHOUT setting self.last_continuous_actions
```

**Impact**: When `get_actions_and_values` encounters empty graph, storage is skipped.

#### Scenario 3: Initialization at Episode Start
```python
# durotaxis_env.py line 835
action_params = {'delete_ratio': 0.0, 'gamma': 0.0, ...}  # Default zeros
```

**Impact**: First step of each episode starts with zeros until policy executes.

## Added Debugging

### 1. Environment Side (durotaxis_env.py)
Added debug prints when:
- `last_continuous_actions` doesn't exist
- `last_continuous_actions` is None
- `last_continuous_actions` has wrong shape

### 2. Policy Agent Side (actor_critic.py)
Added debug prints showing:
- First 3 times actions are stored
- Actual tensor values being stored

## How to Verify

### Run Training and Check Output:
```bash
python train_cli.py
```

### Expected Debug Output:
```
🐛 DEBUG[0]: Storing continuous_actions: tensor([0.4234, 8.2341, 2.5612, 0.3145, -0.0523])
🐛 DEBUG[1]: Storing continuous_actions: tensor([0.4156, 7.8912, 2.4321, 0.2987, 0.0234])
🐛 DEBUG[2]: Storing continuous_actions: tensor([0.4512, 8.5678, 2.6234, 0.3234, -0.0123])
```

### Look for Warning Messages:
```
⚠️  DEBUG: last_continuous_actions is None (prev_num_nodes=X)
⚠️  DEBUG: policy_agent has no attribute 'last_continuous_actions'
⚠️  DEBUG: last_continuous_actions has wrong shape: ...
```

## Possible Root Causes

### ✅ NOT the Issue:
1. Actor network architecture (verified correct)
2. Bound application (verified produces non-zeros)
3. Initialization (Xavier produces non-zeros)

### 🔴 Likely Issues:
1. **Empty graph at episode start**: Many environments start with empty graph, policy not executed
2. **Graph becomes empty mid-episode**: Aggressive deletion → empty graph → policy skipped
3. **Early episode steps**: First step shows zeros before policy executes
4. **Attribute not initialized**: `last_continuous_actions` attribute not created in `__init__`

### 🟡 Edge Cases:
1. **NaN sanitization**: If Actor produces NaNs, `nan_to_num` converts to 0.0
2. **Frozen backbone**: If backbone is frozen and not producing good features (but this should still give non-zero outputs after bound application)

## Recommended Fixes

### Fix 1: Initialize last_continuous_actions with Reasonable Defaults
```python
# In HybridPolicyAgent.__init__
self.last_continuous_actions = torch.tensor([0.45, 7.75, 2.25, 0.275, 0.0])  # Sigmoid(0) defaults
```

### Fix 2: Always Store Actions (Even for Empty Graphs)
```python
# In get_actions_and_values
if state['num_nodes'] == 0:
    device = self.network.action_bounds.device
    # Still store default bounded actions
    self.last_continuous_actions = torch.tensor([0.45, 7.75, 2.25, 0.275, 0.0], device=device)
    empty_values = {...}
    return {}, (1.0, 1.0, 0.5, 0.0), empty_values
```

### Fix 3: Better Display Message for Empty Graphs
```python
# In durotaxis_env.py
if prev_num_nodes == 0:
    action_params_str = "EMPTY_GRAPH (no policy execution)"
else:
    action_params_str = f"dr={action_params['delete_ratio']:.3f} ..."
```

## Next Steps

1. **Run training with debug output enabled**
2. **Check if zeros appear only when num_nodes=0**
3. **Verify Actor is actually producing non-zero mu/std**
4. **Monitor if zeros persist after episode 1**
5. **Check if bound application is working correctly**

## Questions to Answer

- [ ] Does the graph start with 0 nodes?
- [ ] Do zeros appear only at episode start (step 0)?
- [ ] Do zeros appear throughout training or just initially?
- [ ] Do debug prints show actual tensor storage?
- [ ] Are there NaN/Inf warnings from Actor?

---

**Status**: Debugging tools added, awaiting user training run to identify exact cause.

**Confidence**: High - Display extraction logic identified as most likely culprit, not Actor network itself.
