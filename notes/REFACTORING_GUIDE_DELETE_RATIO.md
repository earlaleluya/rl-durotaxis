# Delete Ratio Action Space Refactoring Guide

## ✅ **PHASE 1: COMPLETED - Config File Updated**

### Changes Made to `config.yaml`:
1. ✅ Updated `continuous_dim: 5` (was 4) - Now includes delete_ratio
2. ✅ Updated `num_discrete_actions: 0` (was 2) - No more discrete actions
3. ✅ Renamed `spawn_parameter_bounds` → `action_parameter_bounds`
4. ✅ Added `delete_ratio: [0.0, 0.5]` to action bounds
5. ✅ Updated two_stage_curriculum comments to reflect new architecture
6. ✅ Changed Stage 1 fixed params: gamma=0.5, alpha=0.2, noise=0.1, theta=0.0

---

## ✅ **PHASE 2: COMPLETED - Actor-Critic Network Updated**

### Changes Made to `actor_critic.py`:

#### **Actor Class** (lines 79-225):
1. ✅ Removed `num_discrete_actions` parameter from `__init__`
2. ✅ Removed `spawn_bias_init` parameter
3. ✅ Removed `discrete_head` and `discrete_bias`
4. ✅ Modified forward to:
   - Aggregate node features via mean pooling → single global representation
   - Output single continuous action vector: `[delete_ratio, gamma, alpha, noise, theta]`
   - Return `(continuous_mu, continuous_logstd)` instead of `(discrete_logits, continuous_mu, continuous_logstd)`

#### **HybridActorCritic Class**:
1. ✅ Updated `continuous_dim` default to 5
2. ✅ Removed `spawn_bias_init` handling
3. ✅ Disabled WSA compatibility (not compatible with new architecture)
4. ✅ Updated action_bounds registration:
   - Changed from `spawn_parameter_bounds` to `action_parameter_bounds`
   - Added `delete_ratio: [0.0, 0.5]` as first parameter

5. ✅ **Updated `forward` method**:
   - Removed all discrete action handling
   - Only processes continuous actions now
   - Returns single global action vector

6. ✅ **Updated `_apply_bounds` method**:
   - Index 0: delete_ratio (sigmoid scaling [0.0, 0.5])
   - Indices 1-4: gamma, alpha, noise, theta (sigmoid/tanh scaling)

7. ✅ **Rewrote `get_topology_actions` method**:
   ```python
   def get_topology_actions(self, output, node_positions) -> Dict[int, str]:
       delete_ratio = output['continuous_actions'][0].item()
       num_nodes = len(node_positions)
       num_to_delete = int(delete_ratio * num_nodes)
       
       actions = {}
       for i, (node_id, x_pos) in enumerate(node_positions):
           if i < num_to_delete:
               actions[node_id] = 'delete'  # Leftmost nodes
           else:
               actions[node_id] = 'spawn'   # Rest spawn
       return actions
   ```

8. ✅ **Rewrote `get_spawn_parameters` method**:
   ```python
   def get_spawn_parameters(self, output) -> Tuple[float, float, float, float]:
       params = output['continuous_actions']
       # Extract indices [1, 2, 3, 4] = [gamma, alpha, noise, theta]
       return (params[1].item(), params[2].item(), params[3].item(), params[4].item())
   ```

9. ✅ **Updated HybridPolicyAgent**:
   - `get_actions_and_values` now returns single spawn_params tuple (not per-node dict)
   - Sorts nodes by x-position before calling `get_topology_actions`
   - All spawning nodes use same global parameters

---

## ⏳ **PHASE 3: IN PROGRESS - Train.py Updates**

### Required Changes in `train.py`:

#### **1. collect_episode Method** (lines 2295-2555):

**Current Issues**:
- Lines 2405-2420: Stores per-node `discrete_actions` and `discrete_log_probs`
- Lines 2423-2438: Uses per-node `get_spawn_parameters(output, node_id)`
- Needs to handle single global action vector

**Required Changes**:

```python
# REPLACE lines 2405-2450 with:

# Extract SINGLE GLOBAL continuous action
continuous_actions = output['continuous_actions']  # Shape: [5]
log_probs = output['continuous_log_probs']  # Shape: scalar

actions_taken.append({
    'continuous': continuous_actions,
    'mask': action_mask
})

log_probs_list.append({
    'continuous': log_probs,
    'total': log_probs
})

# Get node positions for delete ratio strategy
state = self.state_extractor.get_state_features(include_substrate=False)
node_features = state['node_features']
node_positions = [(i, node_features[i][0].item()) for i in range(state['num_nodes'])]
node_positions.sort(key=lambda x: x[1])  # Sort by x-position

# Execute actions using delete ratio
topology_actions = self.network.get_topology_actions(output, node_positions)

# Get single global spawn parameters
if self.training_stage == 1:
    # Stage 1: Use fixed parameters
    spawn_params = (
        self.stage_1_fixed_spawn_params['gamma'],
        self.stage_1_fixed_spawn_params['alpha'],
        self.stage_1_fixed_spawn_params['noise'],
        self.stage_1_fixed_spawn_params['theta']
    )
else:
    # Stage 2: Use network's learned parameters
    spawn_params = self.network.get_spawn_parameters(output)

gamma, alpha, noise, theta = spawn_params

# Execute spawn/delete actions
for node_id, action_type in topology_actions.items():
    try:
        if action_type == 'spawn':
            # Track spawn parameters (same for all nodes)
            self.current_episode_spawn_params['gamma'].append(gamma)
            self.current_episode_spawn_params['alpha'].append(alpha)
            self.current_episode_spawn_params['noise'].append(noise)
            self.current_episode_spawn_params['theta'].append(theta)
            
            self.env.topology.spawn(node_id, gamma=gamma, alpha=alpha, noise=noise, theta=theta)
        elif action_type == 'delete':
            self.env.topology.delete(node_id)
    except Exception as e:
        print(f"Failed to execute {action_type} on node {node_id}: {e}")

# Store executed actions for logging
previous_executed_actions = topology_actions
```

#### **2. Remove Discrete Action Processing**:

**Files to Update**:
- Line 1191: Remove `'discrete_actions'` from empty state handling
- Line 1193: Remove `'discrete_log_probs'` from empty state handling
- Line 1222: Remove `'discrete_actions'` and `'discrete_log_probs'` from key list
- Lines 2638-2646: Remove discrete log prob computation in PPO ratio
- Lines 2833-2853: Remove discrete action batching in `compute_policy_loss`

**Replace with**:
```python
# Only continuous actions now
for key in ['continuous_actions', 'continuous_log_probs', 'total_log_probs']:
    if key in output and output[key] is not None:
        # Ensure tensors are on device
        ...
```

#### **3. Update Loss Computation** (lines 2800-3000):

**Current Issue**: `compute_policy_loss` expects both discrete and continuous actions

**Required Changes**:
```python
def compute_policy_loss(self, states, actions, old_log_probs, advantages, ...):
    """Compute PPO policy loss for continuous-only action space"""
    
    # Batch continuous actions (now single per episode)
    all_continuous_actions = []
    for action_dict in actions:
        all_continuous_actions.append(action_dict['continuous'])
    
    batched_continuous_actions = torch.stack(all_continuous_actions, dim=0)  # [num_steps, 5]
    
    # Re-evaluate actions
    eval_outputs = []
    for state in states:
        output = self.network(state, deterministic=True)
        eval_outputs.append(output)
    
    # Compute new log probs
    new_continuous_log_probs = []
    for eval_output, action in zip(eval_outputs, all_continuous_actions):
        continuous_mu = eval_output['continuous_mu']
        continuous_std = eval_output['continuous_std']
        
        dist = torch.distributions.Normal(continuous_mu, continuous_std)
        log_prob = dist.log_prob(action).sum()
        new_continuous_log_probs.append(log_prob)
    
    new_log_probs = torch.stack(new_continuous_log_probs)
    old_log_probs_tensor = torch.stack([lp['total'] for lp in old_log_probs])
    
    # PPO ratio
    ratio = torch.exp(new_log_probs - old_log_probs_tensor)
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    return policy_loss, {
        'policy_loss': policy_loss.item(),
        'continuous_policy_loss': policy_loss.item(),
        'approx_kl': ((ratio - 1) - torch.log(ratio)).mean().item()
    }
```

#### **4. Update Action Masking** (if used):

**Current Issue**: Action mask expects per-node discrete actions

**Solution**: Remove action masking entirely OR adapt for delete ratio:
```python
def create_action_mask(self, state_dict):
    """Action mask not needed for delete ratio - return None"""
    return None
```

---

## 📊 **Expected Behavior After Refactoring**

### **Stage 1 Training** (Delete Ratio Only):
- Network outputs: `[delete_ratio, gamma, alpha, noise, theta]`
- Only `delete_ratio` is learned (backprop enabled)
- Spawn params are FIXED constants: `gamma=0.5, alpha=0.2, noise=0.1, theta=0.0`
- All spawning nodes use these same fixed parameters

### **Stage 2 Training** (Full Continuous):
- Network outputs: `[delete_ratio, gamma, alpha, noise, theta]`
- ALL 5 parameters are learned
- All spawning nodes still use the same global spawn parameters

### **Delete Ratio Strategy**:
```
Example with 10 nodes and delete_ratio=0.3:

Node positions (sorted): [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]
                         ^leftmost                         ^rightmost

Num to delete: 0.3 × 10 = 3

Actions: [delete, delete, delete, spawn, spawn, spawn, spawn, spawn, spawn, spawn]
```

---

## 🔥 **Breaking Changes**

1. **Old checkpoints WILL NOT WORK** - Network architecture changed
2. **Action space changed** from per-node to global
3. **No more discrete actions** - Pure continuous now
4. **Spawn parameters unified** - All nodes use same params per step

---

## ✅ **Testing Checklist**

After completing all changes:

1. [ ] Run syntax check: `python -m py_compile train.py`
2. [ ] Test Stage 1 training (delete ratio only)
3. [ ] Verify leftmost nodes are deleted
4. [ ] Verify all spawns use fixed params in Stage 1
5. [ ] Test Stage 2 training (full continuous)
6. [ ] Verify spawn params are learned in Stage 2
7. [ ] Check reward signals work correctly
8. [ ] Verify gradient flow to delete_ratio parameter

---

## 📝 **Summary of Changes**

| Component | Old Architecture | New Architecture |
|-----------|-----------------|------------------|
| **Action Space** | Per-node discrete + continuous | Single global continuous |
| **Discrete Actions** | spawn/delete per node | Removed |
| **Delete Strategy** | Per-node decision | Delete ratio (leftmost nodes) |
| **Spawn Params** | Per-node parameters | Single global parameters |
| **Stage 1** | Learn discrete only | Learn delete_ratio only |
| **Stage 2** | Learn continuous only | Learn delete_ratio + spawn params |
| **Network Output** | [N, 2] + [N, 4] | [5] = [ratio, γ, α, ν, θ] |

