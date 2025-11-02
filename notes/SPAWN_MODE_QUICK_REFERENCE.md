# Simple Spawn-Only Mode Quick Reference

## Quick Start

### Enable Spawn-Only Mode
```yaml
# config.yaml
environment:
  simple_spawn_only_mode: true
  spawn_reward: 2.0
```

### Enable PBRS for Spawn
```yaml
environment:
  spawn_rewards:
    pbrs:
      enabled: true
      shaping_coeff: 0.1
      phi_weight_spawnable: 1.0
```

## What It Does

**Spawn-Only Mode** provides ONLY spawn-based rewards:
- **+spawn_reward** if ΔI ≥ delta_intensity (successful spawn on good substrate)
- **-spawn_reward** if ΔI < delta_intensity (failed to improve intensity)
- **NO** boundary checking penalties (unlike normal mode)
- **ALL** other rewards disabled (movement, delete, milestones, etc.)

## PBRS for Spawn

**Potential Function**: Φ_spawn(s) = w_spawnable × count(nodes with I ≥ delta_intensity)

**Shaping Term**: F = γ × Φ(s') - Φ(s)

**Effect**:
- More spawnable nodes → positive shaping (good!)
- Fewer spawnable nodes → negative shaping (bad!)
- Preserves optimal policy (proven, Ng et al. 1999)

## Mode Combinations

| Delete | Centroid | Spawn | Reward Formula |
|--------|----------|-------|----------------|
| ✗      | ✗        | ✗     | Normal (all components) |
| ✓      | ✗        | ✗     | R_delete |
| ✗      | ✓        | ✗     | R_distance |
| ✗      | ✗        | ✓     | R_spawn |
| ✓      | ✓        | ✗     | R_delete + R_distance |
| ✓      | ✗        | ✓     | R_delete + R_spawn |
| ✗      | ✓        | ✓     | R_distance + R_spawn |
| ✓      | ✓        | ✓     | R_delete + R_distance + R_spawn |

## Configuration Parameters

### Mode Flag
- **simple_spawn_only_mode** (bool, default: False)
  - Set to `true` to enable spawn-only reward mode

### Reward Value
- **spawn_reward** (float, default: 2.0)
  - Single value used for both success (+) and failure (-)
  - Range: 1.0 - 5.0 recommended

### PBRS Parameters

- **pbrs.enabled** (bool, default: False)
  - Enable PBRS for spawn rewards

- **pbrs.shaping_coeff** (float, default: 0.0)
  - Weight for PBRS shaping term
  - Range: 0.0 - 0.2 recommended
  - Start with: 0.05 - 0.1

- **pbrs.phi_weight_spawnable** (float, default: 1.0)
  - Weight for spawnable node count in potential
  - Typically keep at 1.0

## Termination Rewards

Control with `include_termination_rewards` flag:

```yaml
environment:
  simple_spawn_only_mode: true
  include_termination_rewards: false  # Pure spawn learning
```

- **False**: Focus purely on spawn quality (recommended for initial training)
- **True**: Include termination rewards (success/failure bonuses)

**Scaling** (if centroid mode also enabled):
- Termination rewards are scaled by `dm_term_scale` (default: 0.02)
- Clipped to [-dm_term_clip_val, +dm_term_clip_val] (default: ±10.0)

## Testing

Run the test suite:
```bash
python tools/test_spawn_only_mode.py
```

**Tests**:
1. ✓ Spawn-only mode reward composition
2. ✓ Spawn potential function correctness
3. ✓ PBRS shaping integration
4. ✓ Mode combinations (4 new modes)
5. ✓ Termination reward handling

## Differences from Normal Mode

| Feature | Normal Mode | Simple Spawn-Only Mode |
|---------|------------|----------------------|
| Reward success | spawn_success_reward (2.5) | spawn_reward (2.0) |
| Reward failure | spawn_failure_penalty (1.0) | spawn_reward (2.0) |
| Boundary checks | ✓ YES | ✗ NO |
| PBRS shaping | ✗ NO | ✓ YES (optional) |
| Other rewards | ✓ ALL | ✗ NONE |

## Use Cases

1. **Pure Spawn Learning**: Learn spawn quality optimization in isolation
2. **Spawn + Delete**: Learn node lifecycle management
3. **Spawn + Centroid**: Learn spawn placement for rightward movement
4. **All Three**: Learn comprehensive durotaxis strategy

## Tuning Guide

### Start Conservative
```yaml
spawn_reward: 2.0
pbrs:
  enabled: true
  shaping_coeff: 0.05  # Start small
```

### If Learning Too Slow
```yaml
spawn_reward: 3.0  # Increase reward magnitude
pbrs:
  shaping_coeff: 0.1  # Increase shaping
```

### If Training Unstable
```yaml
spawn_reward: 1.5  # Decrease reward magnitude
pbrs:
  shaping_coeff: 0.02  # Decrease shaping
```

## Monitoring

Watch these metrics:
- **spawn_reward**: Should increase over time (more successful spawns)
- **num_nodes**: Should grow appropriately (spawning happening)
- **avg_intensity**: Should increase (better substrate selection)

## Common Issues

### Problem: Agent not spawning
**Solution**: Check that spawn_reward is large enough (try 3.0-5.0)

### Problem: Too much spawning (graph explosion)
**Solution**: Combine with delete mode or reduce spawn_reward

### Problem: PBRS not helping
**Solution**: Adjust shaping_coeff (try 0.1-0.2) or disable PBRS

## Documentation

- **Implementation**: `notes/SPAWN_ONLY_MODE_SUMMARY.md`
- **Equations**: `notes/REWARD_EQUATIONS.md` (section 4 & mode table)
- **PBRS Theory**: `notes/PBRS_IMPLEMENTATION.md`
- **General PBRS**: `notes/PBRS_QUICK_REFERENCE.md`

## Example Configs

### Pure Spawn Learning (Recommended Start)
```yaml
environment:
  simple_spawn_only_mode: true
  spawn_reward: 2.0
  include_termination_rewards: false
  
  spawn_rewards:
    pbrs:
      enabled: true
      shaping_coeff: 0.1
```

### Spawn + Delete (Node Lifecycle)
```yaml
environment:
  simple_spawn_only_mode: true
  simple_delete_only_mode: true
  spawn_reward: 2.0
  
  spawn_rewards:
    pbrs:
      enabled: true
      shaping_coeff: 0.1
  
  delete_reward:
    pbrs:
      enabled: true
      shaping_coeff: 0.1
```

### All Three (Comprehensive)
```yaml
environment:
  simple_spawn_only_mode: true
  simple_delete_only_mode: true
  centroid_distance_only_mode: true
  spawn_reward: 2.0
  include_termination_rewards: true
```

## Next Steps

1. Start with spawn-only mode + PBRS
2. Monitor learning curves
3. Try mode combinations
4. Tune shaping_coeff based on results
5. Compare with normal mode baseline
