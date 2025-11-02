# PBRS Quick Reference Guide

## Quick Start

### Enable PBRS for Delete Reward (simple_delete_only_mode)

```yaml
# config.yaml
environment:
  simple_delete_only_mode: true
  
  delete_reward:
    proper_deletion: 2.0
    persistence_penalty: 2.0
    improper_deletion_penalty: 2.0
    
    pbrs:
      enabled: true              # ← Enable PBRS
      shaping_coeff: 0.1         # ← Start with 0.05-0.2
      phi_weight_pending_marked: 1.0
      phi_weight_safe_unmarked: 0.25
```

### Enable PBRS for Centroid Distance (centroid_distance_only_mode)

```yaml
# config.yaml
environment:
  centroid_distance_only_mode: true
  
  graph_rewards:
    centroid_movement_reward: 5.0
    
    pbrs_centroid:
      enabled: true              # ← Enable PBRS
      shaping_coeff: 0.1         # ← Start with 0.05-0.2
      phi_distance_scale: 1.0
```

---

## What PBRS Does

**In simple terms**: Adds a small bonus/penalty based on how "close" you are to the goal state.

**For delete reward**: 
- More marked nodes to delete → Lower potential → Negative bias
- Fewer marked nodes → Higher potential → Positive bias
- **Effect**: Encourages deleting marked nodes faster

**For centroid distance**:
- Farther from goal → Lower potential → Negative bias
- Closer to goal → Higher potential → Positive bias
- **Effect**: Encourages moving toward goal

---

## Key Parameters

### shaping_coeff

**What it does**: Controls strength of PBRS bias

**Recommended values**:
- `0.0`: Disabled (default)
- `0.05`: Weak bias (subtle guidance)
- `0.1`: Moderate bias (recommended starting point)
- `0.2`: Strong bias (more aggressive guidance)

**When to increase**: If learning is too slow
**When to decrease**: If PBRS dominates base rewards

### phi_weight_pending_marked (delete only)

**What it does**: Weight for nodes marked `to_delete=1`

**Default**: `1.0`

**Higher value**: Stronger penalty for keeping marked nodes

### phi_weight_safe_unmarked (delete only)

**What it does**: Weight for nodes marked `to_delete=0`

**Default**: `0.25`

**Higher value**: Stronger reward for keeping safe nodes

### phi_distance_scale (centroid only)

**What it does**: Scales distance-to-goal in potential

**Default**: `1.0`

**Adjust when**: Substrate size changes significantly

---

## Testing

### Run Full Test Suite

```bash
python tools/test_pbrs_implementation.py
```

**Expected**: 4/4 tests pass

### Quick Test

```python
from durotaxis_env import DurotaxisEnv

env = DurotaxisEnv('config.yaml')
print(f"Delete PBRS: {env._pbrs_delete_enabled}")
print(f"Centroid PBRS: {env._pbrs_centroid_enabled}")

# Run a few steps
obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
```

---

## Troubleshooting

### PBRS not working?

**Check 1**: Is `enabled: true` in config?
```python
env = DurotaxisEnv('config.yaml')
print(env._pbrs_delete_enabled)  # Should be True
print(env._pbrs_centroid_enabled)  # Should be True
```

**Check 2**: Is `shaping_coeff` > 0?
```python
print(env._pbrs_delete_coeff)  # Should be > 0
print(env._pbrs_centroid_coeff)  # Should be > 0
```

**Check 3**: Is gamma loaded correctly?
```python
print(env._pbrs_gamma)  # Should be 0.99 (from algorithm.gamma)
```

### PBRS too strong?

**Symptom**: Agent ignores base rewards, only follows PBRS gradient

**Solution**: Reduce `shaping_coeff`
```yaml
pbrs:
  shaping_coeff: 0.05  # Reduce from 0.1
```

### PBRS too weak?

**Symptom**: No noticeable improvement in learning

**Solution**: Increase `shaping_coeff`
```yaml
pbrs:
  shaping_coeff: 0.2  # Increase from 0.1
```

---

## Mathematical Guarantee

**PBRS preserves optimal policy**: The optimal policy under PBRS-shaped rewards is mathematically guaranteed to be the same as under base rewards (Ng et al., 1999).

**What this means**: You can safely tune PBRS parameters without worrying about changing the optimal solution.

---

## Best Practices

1. **Start disabled**: Begin training with PBRS disabled to establish baseline
2. **Enable gradually**: Add PBRS with low coefficient (0.05)
3. **Monitor carefully**: Track reward components during training
4. **Tune iteratively**: Adjust coefficient based on learning progress
5. **Compare results**: Compare with/without PBRS on same task

---

## Files Modified

- `durotaxis_env.py`: PBRS implementation
- `config.yaml`: PBRS configuration parameters
- `tools/test_pbrs_implementation.py`: Test suite
- `notes/PBRS_IMPLEMENTATION.md`: Detailed documentation
- `notes/PBRS_QUICK_REFERENCE.md`: This file

---

## Status

✅ **Production-ready**
✅ **Tested** (4/4 tests passing)
✅ **Documented**
✅ **Optional** (disabled by default)
