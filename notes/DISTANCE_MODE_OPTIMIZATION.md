# Distance Mode Optimization Implementation

## Overview

This document describes the comprehensive 6-part optimization strategy implemented to fix slow convergence in `centroid_distance_only_mode`. The core problem was a **scale mismatch** between dense per-step distance signals (~[-1, 0]) and sparse termination rewards (~[-100, +500]), leading to poor credit assignment and slow rightward learning.

## Problem Diagnosis

### Original Behavior
- **Per-step distance penalty**: `-(goal_x - cx) / goal_x` produces values ~[-1, 0] per step
- **Termination rewards**: +100 (success), -30 (failure) dominate the signal
- **Result**: Even with robust advantage normalization, the scale mismatch causes the agent to focus on avoiding failures rather than learning dense rightward movement signals

### Root Cause
The static distance penalty doesn't provide **directional feedback** - it's the same whether the agent moves left or right, as long as distance to goal doesn't change. This creates ambiguity and slows learning.

---

## Implementation Summary

### 6-Part Optimization Strategy

#### ✅ Step 1: Configuration Knobs (config.yaml)
**File**: `config.yaml` (lines 289-310)

Added a new `distance_mode` section with 10 configuration parameters:

```yaml
distance_mode:
  use_delta_distance: true           # Enable potential-based delta distance shaping
  distance_reward_scale: 5.0          # Amplify dense distance signal (5x)
  terminal_reward_scale: 0.02         # Downscale sparse termination rewards (0.02x = 2%)
  clip_terminal_rewards: true         # Enable clipping of scaled termination
  terminal_reward_clip_value: 10.0    # Clip scaled termination to ±10.0
  
  # Adaptive scheduler parameters
  scheduler_enabled: true
  scheduler_window_size: 5            # Evaluate progress over 5 episodes
  scheduler_progress_threshold: 0.6   # 60% of episodes must show rightward progress
  scheduler_consecutive_windows: 3    # 3 consecutive good windows trigger decay
  scheduler_decay_rate: 0.9           # Multiply scale by 0.9 each trigger
  scheduler_min_scale: 0.005          # Floor at 0.5% of original termination scale
```

#### ✅ Step 2: Delta Distance Shaping (durotaxis_env.py)
**File**: `durotaxis_env.py`

**Changes in `__init__` (lines ~259-270)**:
- Load `distance_mode` configuration parameters
- Initialize `_prev_centroid_x = None` for tracking previous centroid position

**Changes in `reset()` (lines ~2422-2424)**:
- Reset `_prev_centroid_x = None` at episode start

**Changes in reward calculation (lines ~1323-1375)**:
```python
# Delta distance shaping (potential-based)
if self.dm_use_delta_distance and self._prev_centroid_x is not None and self.goal_x > 0:
    # Delta: reward = scale × (cx_t - cx_{t-1}) / goal_x
    # Positive when moving right, negative when moving left
    delta_x = centroid_x - self._prev_centroid_x
    distance_signal = self.dm_distance_reward_scale * (delta_x / self.goal_x)
else:
    # Fallback: Static distance penalty (original)
    if self.goal_x > 0:
        distance_signal = -(self.goal_x - centroid_x) / self.goal_x
    else:
        distance_signal = 0.0

# Update previous centroid for next step
self._prev_centroid_x = centroid_x
```

**Mathematical Foundation (Step 3: Potential-Based Shaping)**:
The delta distance formula is **potential-based shaping**:
- Define potential function: `Φ(s) = centroid_x / goal_x`
- Shaped reward: `r'(s,a,s') = r(s,a,s') + γΦ(s') - Φ(s)`
- For our case: `r' = scale × (cx_t - cx_{t-1}) / goal_x`
- **Property**: Preserves optimal policy (proven by Ng et al., 1999)

**Changes in termination handling (lines ~1009-1030)**:
```python
elif self.centroid_distance_only_mode:
    # Apply scaled and clipped termination
    scaled_termination = termination_reward * self.dm_terminal_reward_scale
    if self.dm_clip_terminal_rewards:
        scaled_termination = max(-self.dm_terminal_reward_clip_value, 
                                 min(self.dm_terminal_reward_clip_value, scaled_termination))
    reward_components['total_reward'] += scaled_termination
    # Store the scaled version for logging
    reward_components['termination_reward_scaled'] = scaled_termination
```

**Impact**:
- Dense rightward movement: +0.01 × 5.0 = +0.05 per step (amplified)
- Success termination: +100 × 0.02 = +2.0 (clipped to +10.0 max)
- Failure termination: -30 × 0.02 = -0.6 (clipped to -10.0 min)
- **Result**: Dense signals now comparable in magnitude to sparse signals

#### ✅ Step 4: Entropy Tuning (config.yaml)
**File**: `config.yaml` (lines 263-269)

Reduced entropy coefficients for faster policy commitment:

```yaml
entropy_regularization:
  entropy_coeff_start: 0.25    # Was 0.8 (3.2x reduction)
  entropy_coeff_end: 0.05      # Was 0.15 (3x reduction)
  entropy_decay_episodes: 300  # Was 500 (faster decay)
  discrete_entropy_weight: 1.5  # Was 3.0 (2x reduction)
  continuous_entropy_weight: 1.0 # Was 2.0 (2x reduction)
```

**Rationale**: Lower entropy encourages the agent to commit to actions faster, speeding up convergence once the reward signal is clear.

#### ✅ Step 5: Stability (Already Done)
The robust reward-to-loss pipeline implemented in previous sessions already handles stability:
- Safe standardization with zero-variance masking
- PPO ratio guards (3-layer protection)
- NaN/Inf guards in all loss calculations
- Per-component GAE with safe normalization

These ensure stability even with the new scaled reward signals.

#### ✅ Step 6: Adaptive Scheduler (train.py)
**File**: `train.py`

**Scheduler State Initialization (lines ~664-678)**:
```python
# Adaptive terminal scale scheduler (for distance mode optimization)
dm_config = self.config.get('distance_mode', {})
self.dm_scheduler_enabled = bool(dm_config.get('scheduler_enabled', True))
self.dm_scheduler_window_size = int(dm_config.get('scheduler_window_size', 5))
self.dm_scheduler_progress_threshold = float(dm_config.get('scheduler_progress_threshold', 0.6))
self.dm_scheduler_consecutive_windows = int(dm_config.get('scheduler_consecutive_windows', 3))
self.dm_scheduler_decay_rate = float(dm_config.get('scheduler_decay_rate', 0.9))
self.dm_scheduler_min_scale = float(dm_config.get('scheduler_min_scale', 0.005))
# Scheduler state
self._dm_rightward_progress_history = []
self._dm_consecutive_good_windows = 0
self._dm_terminal_scale_history = []
```

**Helper Methods (lines ~3183-3247)**:
1. `_compute_rightward_progress_rate(episode_count)`:
   - Computes fraction of recent episodes with positive rewards (proxy for rightward progress)
   - Uses sliding window of `scheduler_window_size` episodes
   - Returns progress rate ∈ [0.0, 1.0]

2. `_update_terminal_scale_scheduler(episode_count)`:
   - Called after each episode
   - Checks if progress_rate ≥ threshold for consecutive windows
   - Triggers scale reduction: `new_scale = max(current × decay_rate, min_scale)`
   - Logs scale changes for transparency
   - Resets consecutive counter after applying decay

**Integration (line ~3274)**:
```python
self._update_terminal_scale_scheduler(episode_count)
```

**Metrics Tracking (lines ~3924-3927)**:
```python
'dm_terminal_scale': self.env.dm_terminal_reward_scale if hasattr(self.env, 'dm_terminal_reward_scale') else None,
'dm_progress_rate': self._dm_rightward_progress_history[-1][1] if self._dm_rightward_progress_history else None
```

**Scheduler Logic**:
```
Episode N: Compute progress_rate over last 5 episodes
  ├─ If progress_rate ≥ 0.6:
  │   └─ Increment consecutive_good_windows
  └─ Else:
      └─ Reset consecutive_good_windows = 0

If consecutive_good_windows == 3:
  ├─ Reduce scale: terminal_scale *= 0.9
  ├─ Floor at min_scale = 0.005
  ├─ Log change to console
  └─ Reset consecutive_good_windows = 0
```

---

## Expected Behavior

### Phase 1: Early Training (Episodes 1-50)
- **Dense signal dominates**: Delta distance provides clear directional feedback
- **Termination downscaled**: Success (+2.0) and failures (-0.6 to -10.0) are less dominant
- **Agent learns**: Rightward movement consistently rewarded per-step
- **Entropy high**: Exploration encouraged

### Phase 2: Consistent Progress (Episodes 50-100)
- **Scheduler activates**: As progress_rate exceeds 60% for 3 consecutive windows
- **Scale reduction**: terminal_reward_scale reduced from 0.02 → 0.018 → 0.0162 → ...
- **Dense signal takes over**: Termination becomes negligible
- **Entropy decays**: Policy commits to learned rightward strategy

### Phase 3: Convergence (Episodes 100+)
- **Minimal termination influence**: Scale floors at 0.005 (0.5% of original)
- **Pure delta shaping**: Agent maximizes rightward movement per step
- **Stable learning**: Robust pipeline prevents instabilities

---

## Key Formulas

### Delta Distance Reward
```
r_distance = distance_reward_scale × (centroid_x_t - centroid_x_{t-1}) / goal_x
           = 5.0 × Δcx / goal_x
```

### Scaled Termination Reward
```
r_term_scaled = clip(termination_reward × terminal_reward_scale, -clip_value, +clip_value)
              = clip(r_term × 0.02, -10.0, +10.0)
```

### Total Reward (Distance Mode)
```
r_total = r_distance + r_term_scaled  (if include_termination_rewards=true)
        = r_distance                   (if include_termination_rewards=false)
```

### Scheduler Update Rule
```
progress_rate = (# episodes with positive reward) / window_size

If progress_rate ≥ 0.6 for 3 consecutive windows:
    terminal_reward_scale = max(terminal_reward_scale × 0.9, 0.005)
```

---

## Testing & Validation

### Manual Testing Steps
1. **Run training with distance mode**:
   ```bash
   python train.py --config config.yaml
   ```
   - Verify `centroid_distance_only_mode: true` in config
   - Check console logs for "🔧 Adaptive Scheduler" messages
   - Monitor reward_components_stats.json for `dm_terminal_scale` and `dm_progress_rate`

2. **Check reward scales**:
   - Early episodes: distance rewards ~[-0.05, +0.05], termination ~[-10, +10]
   - Later episodes: distance rewards similar, termination ~[-0.5, +0.5] (after scheduler)

3. **Verify rightward learning**:
   - Plot centroid_x over episodes (should increase consistently)
   - Check success rate improvement
   - Compare convergence speed to original static penalty

### Expected Logs
```
Episode    5: R=  -2.345 (Smooth=-2.45) | MB= 0.0 | Steps=100 | Success=False | Loss= 0.0234
...
Episode   50: R=   1.234 (Smooth= 1.12) | MB= 0.0 | Steps=120 | Success=False | Loss= 0.0189
🔧 Adaptive Scheduler: Reduced terminal_reward_scale: 0.0200 → 0.0180 (Progress: 0.65)
...
Episode  100: R=   5.678 (Smooth= 5.34) | MB= 0.0 | Steps=150 | Success=True  | Loss= 0.0145
🔧 Adaptive Scheduler: Reduced terminal_reward_scale: 0.0162 → 0.0146 (Progress: 0.72)
```

---

## Configuration Reference

### Full Distance Mode Configuration (config.yaml)
```yaml
# Environment settings
centroid_distance_only_mode: true
include_termination_rewards: false  # Set to true to include scaled termination

# Distance mode optimization
distance_mode:
  use_delta_distance: true
  distance_reward_scale: 5.0
  terminal_reward_scale: 0.02
  clip_terminal_rewards: true
  terminal_reward_clip_value: 10.0
  scheduler_enabled: true
  scheduler_window_size: 5
  scheduler_progress_threshold: 0.6
  scheduler_consecutive_windows: 3
  scheduler_decay_rate: 0.9
  scheduler_min_scale: 0.005

# Entropy tuning
entropy_regularization:
  entropy_coeff_start: 0.25
  entropy_coeff_end: 0.05
  entropy_decay_episodes: 300
  discrete_entropy_weight: 1.5
  continuous_entropy_weight: 1.0
```

---

## Files Modified

### 1. config.yaml
- **Lines 289-310**: Added `distance_mode` configuration section
- **Lines 263-269**: Updated `entropy_regularization` settings

### 2. durotaxis_env.py (2650 lines)
- **Lines 259-270** (`__init__`): Load distance_mode parameters, initialize `_prev_centroid_x`
- **Lines 1323-1375** (`_calculate_reward`): Implement delta distance shaping
- **Lines 1009-1030** (`step`): Apply scaled/clipped termination rewards
- **Lines 2422-2424** (`reset`): Reset `_prev_centroid_x`

### 3. train.py (4273 lines)
- **Lines 664-678** (`__init__`): Initialize scheduler state
- **Lines 3183-3247**: Added `_compute_rightward_progress_rate()` and `_update_terminal_scale_scheduler()`
- **Line 3274** (`_collect_and_process_episode`): Call scheduler after each episode
- **Lines 3924-3927** (`save_reward_statistics`): Track scheduler metrics in logs

---

## Troubleshooting

### Issue: Scheduler not activating
- **Check**: `scheduler_enabled: true` in config
- **Check**: `centroid_distance_only_mode: true`
- **Verify**: Progress rate logs in reward_components_stats.json

### Issue: Rewards still dominated by termination
- **Reduce**: `terminal_reward_scale` to 0.01 or lower
- **Increase**: `distance_reward_scale` to 10.0
- **Check**: `include_termination_rewards: true` is set

### Issue: Learning unstable
- **Verify**: Robust reward processing is active (check for NaN/Inf warnings)
- **Reduce**: `distance_reward_scale` to 3.0
- **Increase**: `terminal_reward_clip_value` to 20.0
- **Check**: PPO ratio guards are working (check logs for "Clamped PPO ratios")

### Issue: No rightward progress
- **Verify**: Delta distance is enabled: `use_delta_distance: true`
- **Check**: `_prev_centroid_x` is being updated (add debug prints)
- **Inspect**: reward_breakdown in info dict for positive distance signals
- **Increase**: `distance_reward_scale` for stronger signal

---

## References

- **Potential-Based Shaping**: Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.
- **PPO**: Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- **Reward Scaling**: Henderson, P., et al. (2018). *Deep Reinforcement Learning that Matters.* AAAI.

---

## Summary

This optimization addresses the scale mismatch problem in `centroid_distance_only_mode` through:
1. **Delta distance shaping** (5x amplification) provides dense directional feedback
2. **Scaled termination** (0.02x downscaling, clipped to ±10) reduces sparse signal dominance
3. **Adaptive scheduler** gradually reduces termination influence as learning progresses
4. **Entropy tuning** encourages faster policy commitment
5. **Robust stability** through existing reward-to-loss safeguards

Expected outcome: **Faster convergence** to rightward movement strategy, with **clearer credit assignment** and **stable learning dynamics**.
