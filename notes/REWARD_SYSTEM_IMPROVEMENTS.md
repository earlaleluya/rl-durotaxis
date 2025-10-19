# Reward System Improvements for Successful Migration

## Problem Analysis
The agent was experiencing early termination around 50 steps and failing to reach the rightmost substrate (migration goal). Key issues identified:
1. **Insufficient survival incentives** - Agent wasn't rewarded enough for staying alive longer
2. **Weak migration signal** - Rightward movement rewards were too small compared to other rewards
3. **Overly harsh penalties** - Agent was discouraged from exploration
4. **No intermediate goals** - Agent had no guidance toward the final goal
5. **Leftward drift** - No explicit penalty for moving in the wrong direction

## Comprehensive Solution Implemented

### 1. Environment Expansion (config.yaml)
```yaml
substrate_size: [600, 400]  # 3x larger width (was [200, 200])
max_steps: 1000             # 5x more time (was 200)
init_num_nodes: 3           # More stable start (was 1)
max_critical_nodes: 75      # More capacity (was 50)
consecutive_left_moves_limit: 30  # More forgiving (was 6)
```

**Impact**: Gives agent much more space and time to learn migration behavior.

### 2. Strengthened Survival Rewards
```yaml
survival_reward_config:
  enabled: true
  base_reward: 0.02            # Base reward per step
  bonus_threshold: 100         # Bonus after 100 steps
  bonus_reward: 0.05           # Additional bonus
  max_step_factor: 0.8         # Progressive scaling
```

**Impact**: Agent receives increasing rewards for longer episodes (up to 2x multiplier), directly incentivizing survival.

### 3. Enhanced Migration Incentives

#### A. Individual Node Movement
```yaml
movement_reward: 2.0          # 20x increase (was 0.1)
leftward_penalty: 1.0         # NEW: Explicit penalty for wrong direction
```

#### B. Collective Centroid Movement (NEW)
```yaml
centroid_movement_reward: 2.0  # Strong reward for group migration
```

**Impact**: Rewards BOTH individual cells moving right AND the collective group migrating right.

### 4. Progressive Milestone Rewards (NEW)
```yaml
milestone_rewards:
  enabled: true
  distance_25_percent: 25.0    # Reward at 25% progress
  distance_50_percent: 50.0    # Reward at 50% progress
  distance_75_percent: 100.0   # Reward at 75% progress
  distance_90_percent: 200.0   # Huge reward near goal
```

**Impact**: Creates intermediate goals that guide the agent step-by-step toward the final target.

### 5. Balanced Penalties
```yaml
# Reduced exploration penalties
intensity_penalty: 0.5        # Was 2.0 (75% reduction)
intensity_bonus: 1.5          # Was 0.1 (15x increase)

# Severe penalties for bad behaviors
out_of_bounds_penalty: -100.0 # Was -15.0 (discourage boundary violations)
no_nodes_penalty: -100.0      # Was -15.0 (discourage losing all nodes)

# Removed timeout penalty
timeout_penalty: 0.0          # Was -5.0 (exploration is good!)
```

**Impact**: Encourages exploration while strongly discouraging catastrophic failures.

### 6. Massive Success Reward
```yaml
success_reward: 500.0         # 5x increase (was 100.0)
```

**Impact**: Makes the final goal extremely attractive.

## New Reward Components Implemented

### 1. Centroid Movement Reward (`_calculate_centroid_movement_reward`)
```python
# Rewards the entire colony for moving rightward as a group
centroid_movement = curr_centroid_x - prev_centroid_x
reward = centroid_movement * self.centroid_movement_reward
```

### 2. Milestone Rewards (`_calculate_milestone_reward`)
```python
# Provides progressive rewards at 25%, 50%, 75%, 90% of substrate width
# Tracks milestones per episode to avoid double-rewarding
```

### 3. Enhanced Survival Reward (`get_survival_reward`)
```python
# Progressive scaling: reward increases with episode progress
progress_factor = min(step_count / (max_steps * 0.8), 1.0)
reward *= (1.0 + progress_factor)  # Up to 2x multiplier
```

### 4. Directional Movement Penalty
```python
# Explicit penalty for leftward movement
movement_rewards = np.where(
    x_movement > 0,
    x_movement * movement_reward,    # Rightward: reward
    x_movement * leftward_penalty    # Leftward: penalty
)
```

## Expected Outcomes

With these improvements, the agent should:

1. ✅ **Survive Longer**: Progressive survival rewards + removed timeout penalty → episodes lasting 200+ steps
2. ✅ **Move Rightward**: Strong movement rewards + centroid rewards + leftward penalties → consistent rightward progress
3. ✅ **Reach Milestones**: Progressive milestone rewards → agent learns incremental progress
4. ✅ **Avoid Catastrophic Failures**: Severe penalties for boundaries/node loss → more stable behavior
5. ✅ **Achieve Goal**: Massive success reward (500) + milestone rewards (375 total) → clear path to goal

## Training Recommendations

1. **Monitor Centroid Position**: Watch for consistent rightward movement (→) instead of leftward (←)
2. **Track Episode Length**: Should increase from ~50 steps to 200+ steps
3. **Observe Milestones**: Agent should trigger 25%, 50%, 75%, 90% milestones as it learns
4. **Check Success Rate**: After 200-300 episodes, agent should start occasionally reaching the goal
5. **Patience**: Migration is a long-horizon task; expect 500-1000 episodes for consistent success

## Testing the Improvements

Run training and look for:
```
🎯 MILESTONE REACHED! 25% of substrate width! Reward: +25.0
🎯 MILESTONE REACHED! 50% of substrate width! Reward: +50.0
🎯 MILESTONE REACHED! 75% of substrate width! Reward: +100.0
🎯 MILESTONE REACHED! 90% of substrate width! Reward: +200.0
🎯 Episode terminated: Node X reached rightmost location (x=599.x >= 599) - SUCCESS!
```

## Files Modified

1. `config.yaml` - All reward parameters and environment settings
2. `durotaxis_env.py` - New reward calculation methods and enhanced logic

## Debugging Tips

If agent still struggles:
1. Increase `centroid_movement_reward` to 5.0
2. Increase `movement_reward` to 5.0
3. Reduce `consecutive_left_moves_limit` to 50
4. Increase milestone rewards by 2x
5. Set `enable_visualization: true` to watch agent behavior

## Summary

These improvements create a **multi-layered reward structure** that:
- **Short-term**: Rewards survival and rightward steps
- **Medium-term**: Rewards reaching distance milestones
- **Long-term**: Massive reward for goal achievement

This gives the agent clear guidance at every timescale, making successful migration significantly more achievable.
