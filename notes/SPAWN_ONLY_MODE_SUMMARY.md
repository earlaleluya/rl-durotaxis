# Simple Spawn-Only Mode Implementation Summary

## Overview
Successfully implemented `simple_spawn_only_mode` with Potential-Based Reward Shaping (PBRS) support, expanding the RL environment to support 8 mode combinations (2^3) for focused learning experiments.

## Implementation Details

### 1. Configuration (config.yaml)
**Added parameters** (lines ~341-347):
```yaml
simple_spawn_only_mode: false  # Enable spawn-only reward mode
spawn_reward: 2.0               # Single reward/penalty value
```

**Added PBRS configuration** (lines ~511-516):
```yaml
spawn_rewards:
  pbrs:
    enabled: false              # Enable PBRS for spawn rewards
    shaping_coeff: 0.0          # Shaping coefficient (try 0.05-0.2)
    phi_weight_spawnable: 1.0   # Weight for spawnable nodes
```

### 2. Environment Initialization (durotaxis_env.py)
**Mode flags** (lines ~254-257):
```python
self.simple_spawn_only_mode = config.get('simple_spawn_only_mode', False)
self.spawn_reward = float(config.get('spawn_reward', 2.0))
```

**PBRS parameters** (lines ~427-431):
```python
pbrs_spawn = self.spawn_rewards.get('pbrs', {})
self._pbrs_spawn_enabled = pbrs_spawn.get('enabled', False)
self._pbrs_spawn_coeff = float(pbrs_spawn.get('shaping_coeff', 0.0))
self._pbrs_spawn_w_spawnable = float(pbrs_spawn.get('phi_weight_spawnable', 1.0))
```

### 3. PBRS Potential Function (lines ~1750-1798)
**Spawn potential**: Φ_spawn(s) = w_spawnable * count(nodes with I ≥ delta_intensity)

```python
def _phi_spawn_potential(self, state):
    """
    Compute potential function Phi(s) for spawn reward shaping.
    Returns: w_spawnable * spawnable_nodes(s)
    """
    # Count nodes with high substrate intensity (spawnable candidates)
    spawnable_count = float((intensities_np >= self.delta_intensity).sum())
    phi = self._pbrs_spawn_w_spawnable * spawnable_count
    return float(phi)
```

**PBRS Theory**: F(s,a,s') = γ*Φ(s') - Φ(s)
- Increasing spawnable node count → positive shaping
- Preserves optimal policy (Markov property satisfied)

### 4. Spawn Reward Calculation (lines ~1556-1655)
**Dual mode support**:
- **Normal mode**: Uses spawn_success_reward/spawn_failure_penalty + boundary checks
- **Simple spawn-only mode**: Uses single spawn_reward value + NO boundary checks + PBRS shaping

```python
# Mode-dependent reward assignment
if self.simple_spawn_only_mode:
    if intensity_difference >= self.delta_intensity:
        spawn_reward += self.spawn_reward
    else:
        spawn_reward -= self.spawn_reward
else:
    # Normal mode logic with boundary checks
    ...

# Add PBRS shaping (spawn-only mode only)
if self.simple_spawn_only_mode and self._pbrs_spawn_enabled:
    phi_prev = self._phi_spawn_potential(prev_state)
    phi_new = self._phi_spawn_potential(new_state)
    pbrs_shaping = self._pbrs_gamma * phi_new - phi_prev
    spawn_reward += self._pbrs_spawn_coeff * pbrs_shaping
```

### 5. Mode Combination Logic (lines ~1204-1260)
**8 Mode Combinations** (2^3: delete/centroid/spawn):

| Delete | Centroid | Spawn | Reward Composition |
|--------|----------|-------|-------------------|
| False  | False    | False | Normal (all components) |
| True   | False    | False | R_delete only |
| False  | True     | False | R_distance only |
| False  | False    | True  | **R_spawn only** (NEW) |
| True   | True     | False | R_delete + R_distance |
| True   | False    | True  | **R_delete + R_spawn** (NEW) |
| False  | True     | True  | **R_distance + R_spawn** (NEW) |
| True   | True     | True  | **All three** (NEW) |

**Compositional reward formula**:
```python
if num_special_modes > 0:
    mode_reward = 0.0
    if has_delete_mode:
        mode_reward += delete_reward
    if has_centroid_mode:
        mode_reward += distance_signal
    if has_spawn_mode:
        mode_reward += spawn_reward
    
    total_reward = mode_reward
    # Zero out non-special-mode components
```

### 6. Termination Reward Handling (lines ~892-920)
**Updated logic** to handle all 8 combinations:
```python
has_delete_mode = self.simple_delete_only_mode
has_centroid_mode = self.centroid_distance_only_mode
has_spawn_mode = self.simple_spawn_only_mode
num_special_modes = sum([has_delete_mode, has_centroid_mode, has_spawn_mode])

is_normal_mode = num_special_modes == 0
is_special_mode_with_termination = num_special_modes > 0 and self.include_termination_rewards

if is_normal_mode or is_special_mode_with_termination:
    if has_centroid_mode:
        # Centroid mode(s): scaled+clipped termination
        scaled_termination = termination_reward * self.dm_term_scale
        if self.dm_term_clip:
            scaled_termination = max(-self.dm_term_clip_val, 
                                     min(self.dm_term_clip_val, scaled_termination))
        reward_components['total_reward'] += scaled_termination
    else:
        # Normal/delete/spawn modes: full termination reward
        reward_components['total_reward'] += termination_reward
```

### 7. Testing (tools/test_spawn_only_mode.py)
**Comprehensive test suite** with 5 tests (ALL PASSING ✅):

1. **Spawn-Only Mode Reward Composition**: Verifies ONLY spawn rewards are used
2. **Spawn Potential Function**: Tests Φ_spawn computation correctness
3. **PBRS Shaping Integration**: Validates F = γ*Φ(s') - Φ(s) is correctly added
4. **Mode Combinations**: Tests all 4 new combinations (spawn-only, spawn+delete, spawn+centroid, all three)
5. **Termination Rewards**: Validates termination handling with spawn mode

**Test Results**:
```
✓ Spawn-Only Mode Reward Composition: PASSED
✓ Spawn Potential Function: PASSED
✓ PBRS Shaping Integration: PASSED
✓ Mode Combinations: PASSED
✓ Termination Rewards with Spawn Mode: PASSED

5/5 tests passed
🎉 ALL TESTS PASSED!
```

### 8. Documentation Updates

**REWARD_EQUATIONS.md** (lines ~63-97, ~190-250):
- Added spawn PBRS potential function equation
- Added simplified spawn reward formula for simple_spawn_only_mode
- Expanded mode composition section with all 8 combinations
- Added formal mathematical notation for spawn PBRS

**Key Equations Added**:
- Spawn potential: Φ_spawn(s) = w_spawn · spawnable_nodes(s)
- PBRS shaping: F_spawn(s_t, s_{t+1}) = γ·Φ_spawn(s_{t+1}) - Φ_spawn(s_t)
- Applied as: R_spawn = R_spawn^base + shaping_coeff_spawn · F_spawn

## Usage Example

### Enable Simple Spawn-Only Mode
```yaml
# config.yaml
environment:
  simple_spawn_only_mode: true
  spawn_reward: 2.0
  
  spawn_rewards:
    pbrs:
      enabled: true
      shaping_coeff: 0.1  # Start with 0.05-0.2
      phi_weight_spawnable: 1.0
```

### Combine with Other Modes
```yaml
# Example: Spawn + Delete mode
environment:
  simple_spawn_only_mode: true
  simple_delete_only_mode: true
  centroid_distance_only_mode: false
```

## Key Benefits

1. **Focused Learning**: Agent learns ONLY spawn quality optimization
2. **PBRS Support**: Improves credit assignment without changing optimal policy
3. **Mode Flexibility**: Works with all combinations of delete/centroid/spawn modes
4. **Clean Implementation**: Compositional reward formula handles 8 combinations elegantly
5. **Fully Tested**: 5/5 tests passing, validates correctness
6. **Well Documented**: Formal equations + usage guide

## Design Principles

1. **Markov Property**: Φ_spawn depends only on current state (count of spawnable nodes)
2. **Policy Preservation**: PBRS guarantees optimal policy unchanged
3. **Compositional**: Mode combinations work via simple addition
4. **Backward Compatible**: Normal mode unchanged, special modes are additive
5. **Configurable**: PBRS can be enabled/disabled independently

## Files Modified

1. `config.yaml`: Added spawn mode flags and PBRS parameters
2. `durotaxis_env.py`: 
   - Initialization (lines ~254-257, ~427-431)
   - Potential function (lines ~1750-1798)
   - Spawn reward calculation (lines ~1556-1655)
   - Mode combination logic (lines ~1204-1260)
   - Termination handling (lines ~892-920)
3. `tools/test_spawn_only_mode.py`: Comprehensive test suite (NEW)
4. `notes/REWARD_EQUATIONS.md`: Updated equations and mode table

## Summary Statistics

- **Lines of code added**: ~150 (env) + ~380 (test)
- **Mode combinations**: 4 → 8 (4 new combinations)
- **PBRS potentials**: 2 → 3 (added spawn potential)
- **Test coverage**: 5/5 tests passing
- **Backward compatibility**: ✅ Maintained

## Future Work

- Experiment with different phi_weight_spawnable values
- Tune shaping_coeff for spawn PBRS (start 0.05-0.2)
- Compare learning curves: spawn-only vs normal mode
- Ablation studies on mode combinations
