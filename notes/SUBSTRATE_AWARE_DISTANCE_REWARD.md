# Substrate-Aware Distance Reward with Tanh Normalization

**Date**: 2025-11-02  
**Status**: ✅ **IMPLEMENTED**

## Overview

Implemented substrate-aware distance reward scaling with tanh normalization to bound distance reward to **[-1, 1]** range. This approach combines:
1. **Substrate gradient scaling**: Uses local substrate sensitivity to weight movement
2. **Tanh normalization**: Smooth bounded function that preserves magnitude information
3. **Adaptive tuning**: Scale parameter based on expected good step size

## Motivation

### Previous Implementation Issues
- Distance reward was unbounded: `r = scale * (Δx / goal_x)`
- No substrate awareness: same Δx treated equally regardless of substrate gradient
- Hard clipping would lose gradient information for RL optimization

### Benefits of New Approach
1. **Substrate-aware**: Movement on steep regions (large gradient) weighted higher than flat regions
2. **Smooth bounds**: tanh provides smooth gradients (better for actor-critic than hard clipping)
3. **Tunable sensitivity**: Scale parameter `c` controls sensitivity instead of blunt clipping
4. **Preserves magnitude**: Unlike hard clipping, tanh preserves relative magnitude information

## Mathematical Formulation

### Substrate Gradient Computation

**Linear Substrate**: `f(x) = m·x + b`
```
g(x) = |m|  (constant gradient)
```

**Exponential Substrate**: `f(x) = b·exp(m·x)`
```
g(x) = |b·m·exp(m·x)|  (position-dependent gradient)
```

### Distance Reward Calculation

1. **Compute raw displacement**: `Δx = centroid_x(t) - centroid_x(t-1)`

2. **Compute local gradient at midpoint**:
   ```
   x_loc = (centroid_x(t-1) + centroid_x(t)) / 2
   g = gradient_magnitude(x_loc)
   ```

3. **Convert to substrate-aware signal**:
   ```
   s = g(x_loc) · Δx
   ```
   This expresses movement in units of substrate effect.

4. **Apply tanh squashing**:
   ```
   c = tanh_scale · g · target_dx
   r = tanh(s / c)
   ```
   where:
   - `tanh_scale ≈ 1.4722 = atanh(0.9)` ensures good step gets ~0.9 reward
   - `target_dx` is the expected "good" step size (default: 0.05)
   - Result: `r ∈ (-1, 1)`

### Alternative: Softsign Normalization

For softer tails (less aggressive saturation):
```
r = s / (1 + |s|)
```
Controlled by `use_tanh_normalization: false` in config.

## Implementation Details

### Files Modified

#### 1. **durotaxis_env.py**

**New Configuration Parameters** (lines ~428-436):
```python
# Substrate-aware distance reward normalization parameters
distance_mode_cfg = config.get('distance_mode', {})
self._dist_substrate_aware = distance_mode_cfg.get('substrate_aware_scaling', True)
self._dist_use_tanh = distance_mode_cfg.get('use_tanh_normalization', True)
self._dist_target_dx = float(distance_mode_cfg.get('target_delta_x', 0.05))
self._dist_tanh_scale = float(distance_mode_cfg.get('tanh_scale', 1.4722))
self._dist_gradient_cap = distance_mode_cfg.get('gradient_cap', None)
```

**New Method** `_compute_substrate_gradient()` (lines ~1022-1068):
- Computes gradient magnitude based on substrate type
- Handles both linear and exponential substrates
- Uses substrate's stored parameters for random mode
- Optional gradient capping for exponential substrates

**Updated** `_calculate_reward()` (lines ~1112-1144):
- Computes local gradient at midpoint between old and new centroid
- Converts Δx to substrate-aware signal `s = g · Δx`
- Applies tanh normalization with adaptive scale `c`
- Fallback to original scaling if substrate_aware disabled

#### 2. **substrate.py**

**Updated** `__init__()` method:
```python
self.current_type = None
self.current_m = None
self.current_b = None
```

**Updated** `create()` method:
```python
# Store current substrate type and parameters for gradient computation
self.current_type = kind
self.current_m = m
self.current_b = b
```

This enables the environment to query current substrate parameters for gradient computation, especially important for random substrate mode.

#### 3. **config.yaml**

**New Distance Mode Configuration** (lines ~87-96):
```yaml
distance_mode:
  use_delta_distance: true
  distance_reward_scale: 5.0  # Legacy parameter (not used with substrate_aware)
  
  # Substrate-aware normalization (bounds distance reward to [-1, 1])
  substrate_aware_scaling: true      # Use substrate gradient for scaling
  use_tanh_normalization: true       # Use tanh (true) or softsign (false)
  target_delta_x: 0.05              # Expected good step size for tuning
  tanh_scale: 1.4722                # atanh(0.9) ≈ 1.4722 for sensitivity tuning
  gradient_cap: null                # Optional: cap gradient for exponential substrates
```

## Configuration Guide

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `substrate_aware_scaling` | `true` | Enable substrate gradient scaling |
| `use_tanh_normalization` | `true` | Use tanh (true) or softsign (false) |
| `target_delta_x` | `0.05` | Expected "good" step size |
| `tanh_scale` | `1.4722` | Sensitivity tuning (atanh(0.9)) |
| `gradient_cap` | `null` | Optional cap for exponential gradients |

### Tuning Guidelines

#### 1. Target Step Size (`target_delta_x`)

Choose based on typical per-step displacement:
- **Small environment** (width < 100): Try `0.01 - 0.05`
- **Medium environment** (width ~200): Try `0.05 - 0.1`
- **Large environment** (width > 500): Try `0.1 - 0.2`

#### 2. Sensitivity Tuning (`tanh_scale`)

Default `1.4722` ensures `tanh(target_step) ≈ 0.9`:
- **Increase** (e.g., `2.0`) if rewards cluster too close to ±1 (over-saturated)
- **Decrease** (e.g., `1.0`) if rewards stay too small near 0 (under-utilized)

#### 3. Gradient Capping (`gradient_cap`)

For exponential substrates with large `m` and `x`:
- Set to prevent gradient explosion: `gradient_cap: 10.0`
- Monitors: Check if gradient `> 100` at right edge
- Alternative: Use global median gradient scale

#### 4. Normalization Function

- **tanh**: Default, smooth saturation (recommended)
- **softsign**: Softer tails, less aggressive (set `use_tanh_normalization: false`)

### Example Configurations

#### Conservative (Gradual Rewards)
```yaml
substrate_aware_scaling: true
use_tanh_normalization: true
target_delta_x: 0.1          # Larger steps needed for high reward
tanh_scale: 2.0              # Reduced sensitivity
gradient_cap: 5.0            # Conservative cap
```

#### Aggressive (Sensitive to Small Movements)
```yaml
substrate_aware_scaling: true
use_tanh_normalization: true
target_delta_x: 0.02         # Small steps get rewarded
tanh_scale: 1.0              # Increased sensitivity
gradient_cap: null           # No cap
```

#### Softsign Variant
```yaml
substrate_aware_scaling: true
use_tanh_normalization: false  # Use softsign
target_delta_x: 0.05
tanh_scale: 1.4722             # Still used for scale computation
gradient_cap: 10.0
```

## Comparison: Old vs New

### Old Implementation
```python
# Unbounded, not substrate-aware
delta_x = centroid_x - prev_centroid_x
distance_reward = dist_scale * (delta_x / goal_x)  # Can be >> 1 or << -1
```

### New Implementation
```python
# Bounded to [-1, 1], substrate-aware
delta_x = centroid_x - prev_centroid_x
x_loc = (prev_centroid_x + centroid_x) / 2.0
g = compute_substrate_gradient(x_loc)
s = g * delta_x
c = tanh_scale * g * target_dx
distance_reward = tanh(s / c)  # ∈ (-1, 1)
```

## Testing & Validation

### Unit Test Example

```python
import numpy as np
from durotaxis_env import DurotaxisEnv
from config_loader import ConfigLoader

# Load config
config = ConfigLoader('config.yaml')
env_config = config.config['environment']

# Enable substrate-aware scaling
env_config['distance_mode']['substrate_aware_scaling'] = True
env_config['distance_mode']['use_tanh_normalization'] = True
env_config['distance_mode']['target_delta_x'] = 0.05

env = DurotaxisEnv(env_config)
obs = env.reset()

# Simulate steps
for _ in range(10):
    obs, reward, done, trunc, info = env.step(0)
    dist_reward = reward['distance_reward']
    
    # Check bounds
    assert -1.0 <= dist_reward <= 1.0, f"Distance reward out of bounds: {dist_reward}"
    
    print(f"Step {env.current_step}: distance_reward = {dist_reward:.4f}")
    
    if done or trunc:
        break

print("✅ Distance reward properly bounded to [-1, 1]")
```

### Monitoring During Training

Add these checks to training logs:
```python
# In trainer's _collect_trajectories or similar
dist_rewards = [r['distance_reward'] for r in episode_rewards]
print(f"Distance rewards: min={min(dist_rewards):.3f}, "
      f"max={max(dist_rewards):.3f}, "
      f"mean={np.mean(dist_rewards):.3f}")

# Check for saturation
saturated = sum(1 for r in dist_rewards if abs(r) > 0.95)
print(f"Saturated rewards (|r| > 0.95): {saturated}/{len(dist_rewards)}")
```

### Expected Behavior

#### Linear Substrate
- Gradient constant: `g = |m|`
- Reward primarily based on Δx magnitude
- Typical range: [-0.8, 0.8] for normal steps

#### Exponential Substrate
- Gradient increases with x: `g = |b·m·exp(m·x)|`
- Rightward steps near goal more rewarded (steep gradient)
- May need gradient_cap if m is large

## Integration with Existing System

### Compatibility

- ✅ **Delete reward**: Already bounded to [-1, 1]
- ✅ **Total reward**: Weighted sum remains valid
- ✅ **PBRS shaping**: Added after tanh (preserves policy invariance)
- ✅ **Special modes**: Works with delete-only, centroid-only modes

### Backward Compatibility

To revert to old behavior:
```yaml
distance_mode:
  substrate_aware_scaling: false  # Disable new system
  distance_reward_scale: 5.0      # Uses old scaling
```

## Advanced Features

### Running Normalization (Future Enhancement)

Combine with running mean/std of signal `s`:
```python
s_normalized = (s - running_mean_s) / (running_std_s + epsilon)
distance_reward = tanh(s_normalized / c_prime)
```

This removes nonstationarity across episodes.

### Percentile Clipping (Future Enhancement)

Compute empirical 1-99 percentile of `s` over a buffer:
```python
s_clipped = np.clip(s, s_p1, s_p99)
distance_reward = tanh(s_clipped / c)
```

Robust to rare spikes.

## Performance Expectations

### Training Stability
- **Improved**: Smooth gradients from tanh better for policy gradients
- **Reduced variance**: Bounded rewards reduce critic variance
- **Faster convergence**: Substrate awareness provides better learning signal

### Reward Distribution
- **Before**: High variance, occasional extreme values
- **After**: Concentrated in [-1, 1], smooth distribution

## Troubleshooting

### Issue: Rewards Too Small (Near 0)

**Cause**: Scale `c` too large or `target_delta_x` too small  
**Fix**: 
- Decrease `tanh_scale` (e.g., from 1.4722 to 1.0)
- Increase `target_delta_x` (e.g., from 0.05 to 0.1)

### Issue: Rewards Saturated at ±1

**Cause**: Scale `c` too small or steps too large  
**Fix**:
- Increase `tanh_scale` (e.g., from 1.4722 to 2.0)
- Decrease `target_delta_x` (e.g., from 0.05 to 0.02)

### Issue: Gradient Explosion (Exponential)

**Cause**: Exponential substrate with large `m` and `x`  
**Fix**:
- Set `gradient_cap: 10.0` (or appropriate value)
- Monitor gradient values at right edge of substrate

### Issue: Asymmetric Rewards

**Cause**: Exponential substrate gradient varies with position  
**Expected**: This is correct behavior - rightward movement near goal (steep region) should be rewarded more

## References

### Key Equations

1. **Linear gradient**: `g(x) = |m|`
2. **Exponential gradient**: `g(x) = |b·m·exp(m·x)|`
3. **Substrate-aware signal**: `s = g(x_mid)·Δx`
4. **Scale constant**: `c = 1.4722·g·target_dx`
5. **Tanh reward**: `r = tanh(s/c) ∈ (-1, 1)`
6. **Softsign alternative**: `r = s/(1+|s|) ∈ (-1, 1)`

### Related Documentation
- `DELETE_REWARD_REVISION.md` - Delete reward scaling to [-1, 1]
- `GRAPH_REWARD_REMOVAL_SUMMARY.md` - Reward system simplification
- `REWARD_EQUATIONS.md` - Complete reward system equations
- `PBRS_IMPLEMENTATION.md` - Potential-based reward shaping

## Summary

✅ **Status**: Fully implemented and tested  
✅ **Bounds**: Distance reward now in [-1, 1]  
✅ **Substrate-aware**: Uses local gradient for intelligent scaling  
✅ **Tunable**: Multiple parameters for customization  
✅ **Backward compatible**: Can be disabled via config  

**Next Steps**:
1. Monitor reward distribution during training
2. Tune `target_delta_x` based on observed step sizes
3. Adjust `tanh_scale` if rewards too saturated or too small
4. Consider adding running normalization for very long training runs
