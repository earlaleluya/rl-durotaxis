# Simple Delete-Only Reward Mode

## Overview

This feature provides a simplified reward system that focuses **exclusively** on teaching the agent proper node deletion behavior through negative feedback (penalties only).

## Purpose

The standard reward system includes multiple components:
- Graph rewards (connectivity, growth, survival)
- Node rewards (movement, substrate, boundaries)
- Spawn rewards (successful/failed spawning)
- Delete rewards (proper/improper deletion)
- Edge rewards (directional bias)
- Milestone rewards (progress tracking)

This can be complex for initial experiments. The **simple delete-only mode** strips away all of this complexity and provides:

**ONLY TWO PENALTIES:**
1. **Rule 1 (Persistence Penalty)**: When a node marked `to_delete=1` still exists in the next state
2. **Rule 2 (Improper Deletion Penalty)**: When a node NOT marked `to_delete=0` is deleted anyway

**NO POSITIVE REWARDS** - Not even for proper deletions (proper deletions give 0.0)

## Configuration

### Enable the Mode

In `config.yaml`, find the `environment` section and set:

```yaml
environment:
  # EXPERIMENTAL: Simple Delete-Only Reward Mode
  simple_delete_only_mode: true  # Set to true to enable
```

### Disable the Mode (Default)

```yaml
environment:
  simple_delete_only_mode: false  # Standard multi-component rewards
```

## How It Works

### When `simple_delete_only_mode: true`

1. **All reward components are zeroed out:**
   - `graph_reward = 0.0` (except delete penalty)
   - `spawn_reward = 0.0`
   - `edge_reward = 0.0`
   - `centroid_reward = 0.0`
   - `milestone_reward = 0.0`
   - `survival_reward = 0.0`
   - `total_node_reward = 0.0`

2. **Only delete penalties are computed:**
   - If a node marked for deletion persists: `-persistence_penalty`
   - If an unmarked node is deleted: `-improper_deletion_penalty`
   - If a marked node is properly deleted: `0.0` (no reward!)

3. **Total reward = delete penalties only**

### Penalty Values

Configure in `config.yaml`:

```yaml
environment:
  delete_reward:
    proper_deletion: 2.0              # NOT used in simple mode
    persistence_penalty: 2.0          # RULE 1 penalty
    improper_deletion_penalty: 2.0    # RULE 2 penalty
```

In simple mode:
- `proper_deletion` is ignored (no positive rewards)
- `persistence_penalty` penalizes keeping marked nodes
- `improper_deletion_penalty` penalizes deleting unmarked nodes

## Expected Agent Behavior

With only penalties and no positive rewards, the agent should learn to:

1. **Avoid deleting nodes unnecessarily** (to avoid improper deletion penalty)
2. **Delete nodes when marked** (to avoid persistence penalty)
3. **Conservative deletion strategy** (bias toward keeping nodes to minimize penalties)

The hypothesis is that the agent will develop a cautious deletion policy, only acting when it's confident the node should be removed.

## Use Cases

### Experiment 1: Minimalist Learning
Test whether an agent can learn purely from negative feedback without any positive reinforcement.

### Experiment 2: Deletion Policy Isolation
Isolate the deletion learning problem from other objectives (movement, spawning, etc.).

### Experiment 3: Baseline Comparison
Compare training time and success rate against the full reward system.

### Experiment 4: Transfer Learning
Pre-train on simple delete-only mode, then fine-tune with full rewards.

## Example Training Command

```powershell
# Standard training with simple delete-only mode enabled
python train.py

# Or with custom config
python train_cli.py --config custom_config.yaml
```

Make sure `simple_delete_only_mode: true` is set in your config file.

## Testing the Feature

Run the dedicated test script:

```powershell
python .\tools\test_simple_delete_mode.py
```

This will verify:
- Normal mode behaves as expected
- Simple mode zeros out all non-delete rewards
- Rule 1 (persistence penalty) works correctly
- Rule 2 (improper deletion penalty) works correctly
- No positive rewards are given in simple mode

## Monitoring During Training

### Reward Components to Watch

In your training logs or `reward_components_stats.json`:

```json
{
  "delete_reward": {
    "mean": -1.234,  // Should be ≤ 0 in simple mode
    "std": 0.456
  },
  "spawn_reward": {
    "mean": 0.0,     // Should be exactly 0
    "std": 0.0
  },
  // ... all others should be 0
}
```

### Expected Patterns

- **Total reward ≤ 0**: Only penalties exist
- **Delete reward ≤ 0**: No positive delete rewards
- **All other rewards = 0**: Completely zeroed out

### Convergence Signs

- Delete reward increasing toward 0 (fewer penalties)
- Agent learns to delete marked nodes
- Agent avoids deleting unmarked nodes

## Switching Back to Normal Mode

Simply change the config:

```yaml
environment:
  simple_delete_only_mode: false
```

All reward components will be restored to their standard behavior.

## Implementation Details

### Code Changes

1. **Config Loading** (`durotaxis_env.py` line ~256):
   ```python
   self.simple_delete_only_mode = config.get('simple_delete_only_mode', False)
   ```

2. **Reward Calculation** (`durotaxis_env.py` line ~1240):
   ```python
   if self.simple_delete_only_mode:
       # Zero out everything except delete penalties
       delete_penalty_only = delete_reward if delete_reward < 0 else 0.0
       total_reward = delete_penalty_only
       # ... zero out all other components
   ```

3. **Delete Reward Logic** (`durotaxis_env.py` line ~1582):
   ```python
   if node_was_deleted:
       if not self.simple_delete_only_mode:
           delete_reward += self.delete_proper_reward
   ```

### Edge Cases Handled

- Empty graphs still trigger safety mechanisms
- Termination conditions unchanged
- Episode length limits still apply
- Visualization still works (if enabled)

## Troubleshooting

### Issue: Total reward is not ≤ 0

**Cause**: Mode not properly enabled or code changes not applied.

**Solution**: 
- Verify `simple_delete_only_mode: true` in config
- Check environment initialization: `env.simple_delete_only_mode` should be `True`

### Issue: Agent doesn't learn anything

**Cause**: Penalties might be too weak, or agent isn't triggering delete actions.

**Solution**:
- Increase `persistence_penalty` and `improper_deletion_penalty`
- Check if delete actions are being taken in your policy
- Verify the heuristic system is marking nodes with `to_delete`

### Issue: Other rewards are still non-zero

**Cause**: Old cached environment or configuration not reloaded.

**Solution**:
- Restart your training script
- Clear `__pycache__` directories
- Verify config is being loaded correctly

## Advanced: Combining with Curriculum Learning

You can enable simple mode for specific curriculum stages:

```yaml
curriculum_learning:
  stage_2_management:
    # ... other settings
    enable_simple_delete_mode: true  # Only for this stage
```

This would require additional code modifications to dynamically toggle the mode.

## Theoretical Foundation

This feature is inspired by:

1. **Sparse Reward RL**: Learning from minimal feedback signals
2. **Constraint Satisfaction**: Agent must satisfy constraints (don't break rules) rather than maximize rewards
3. **Safe RL**: Conservative policies through penalty-based learning

## References

- `durotaxis_env.py`: Main implementation
- `config.yaml`: Configuration flag
- `tools/test_simple_delete_mode.py`: Comprehensive tests
- `tools/test_delete_reward.py`: Original delete reward tests

## Future Extensions

Possible enhancements:
- Dynamic penalty scaling based on episode progress
- Curriculum learning integration
- Adaptive penalty weights
- Penalty normalization strategies
