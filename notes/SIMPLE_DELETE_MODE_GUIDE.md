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

**ONLY THREE PENALTIES:**
0. **Rule 0 (Growth Penalty)**: When `num_nodes > max_critical_nodes`
1. **Rule 1 (Persistence Penalty)**: When a node marked `to_delete=1` still exists in the next state
2. **Rule 2 (Improper Deletion Penalty)**: When a node NOT marked `to_delete=0` is deleted anyway

**PLUS TERMINATION REWARDS/PENALTIES:**
- **Success Reward**: When agent reaches rightmost substrate (goal completion)
- **Critical Nodes Penalty**: When `num_nodes > threshold_critical_nodes`
- **No Nodes Penalty**: When all nodes are lost
- **Out of Bounds Penalty**: When any node exits substrate boundaries
- **Leftward Drift Penalty**: When centroid moves left consecutively
- **Timeout Penalty**: When max steps reached without success/failure

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
   - `graph_reward = 0.0` (except growth penalty and delete penalties)
   - `spawn_reward = 0.0`
   - `edge_reward = 0.0`
   - `centroid_reward = 0.0`
   - `milestone_reward = 0.0`
   - `survival_reward = 0.0`
   - `total_node_reward = 0.0`

2. **Only three penalties are computed:**
   - **Rule 0**: If `num_nodes > max_critical_nodes`: `-growth_penalty * (1 + excess_nodes / max_critical_nodes)`
   - **Rule 1**: If a node marked for deletion persists: `-persistence_penalty`
   - **Rule 2**: If an unmarked node is deleted: `-improper_deletion_penalty`
   - If a marked node is properly deleted: `0.0` (no reward!)

3. **Termination rewards/penalties are included:**
   - Success (reach rightmost): `+success_reward`
   - Critical nodes exceeded: `+critical_nodes_penalty` (negative)
   - All nodes lost: `+no_nodes_penalty` (negative)
   - Out of bounds: `+out_of_bounds_penalty` (negative)
   - Leftward drift: `+leftward_drift_penalty` (negative)
   - Timeout: `+timeout_penalty` (negative)

4. **Total reward = growth_penalty + delete_penalties + termination_reward**

### Penalty Values

Configure in `config.yaml`:

```yaml
environment:
  max_critical_nodes: 75              # Rule 0 threshold
  threshold_critical_nodes: 500       # Termination threshold (higher than max_critical)
  
  graph_rewards:
    growth_penalty: 3.0               # Rule 0 penalty (scaled by excess)
  
  delete_reward:
    proper_deletion: 2.0              # NOT used in simple mode
    persistence_penalty: 2.0          # RULE 1 penalty
    improper_deletion_penalty: 2.0    # RULE 2 penalty
  
  termination_rewards:
    success_reward: 100.0             # Positive reward for goal completion
    critical_nodes_penalty: -25.0     # Penalty for exceeding critical threshold
    no_nodes_penalty: -30.0           # Penalty for losing all nodes
    out_of_bounds_penalty: -30.0      # Penalty for nodes going out of bounds
    leftward_drift_penalty: -30.0     # Penalty for persistent leftward drift
    timeout_penalty: -10.0            # Penalty for timeout without completion
```

In simple mode:
- `growth_penalty` penalizes spawning too many nodes (Rule 0)
- `proper_deletion` is ignored (no positive rewards for deletion)
- `persistence_penalty` penalizes keeping marked nodes (Rule 1)
- `improper_deletion_penalty` penalizes deleting unmarked nodes (Rule 2)
- All `termination_rewards` are included to provide outcome feedback

## Expected Agent Behavior

With only penalties and termination feedback (no step-wise positive rewards), the agent should learn to:

1. **Avoid spawning too many nodes** (to avoid growth penalty - Rule 0)
2. **Avoid deleting nodes unnecessarily** (to avoid improper deletion penalty - Rule 2)
3. **Delete nodes when marked** (to avoid persistence penalty - Rule 1)
4. **Stay within substrate bounds** (to avoid out-of-bounds termination)
5. **Maintain forward progress** (to avoid leftward drift termination)
6. **Complete the task efficiently** (to earn success reward and avoid timeout)

The hypothesis is that the agent will develop a cautious policy that:
- Carefully manages node count to stay below `max_critical_nodes`
- Only deletes nodes when they're marked with high confidence
- Learns the relationship between spawning and the need to delete marked nodes
- Understands episode outcomes (success vs. various failure modes)
- Optimizes for task completion to earn the success reward

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

### Implementation Details

### Code Changes

1. **Config Loading** (`durotaxis_env.py` line ~256):
   ```python
   self.simple_delete_only_mode = config.get('simple_delete_only_mode', False)
   ```

2. **Reward Calculation** (`durotaxis_env.py` line ~1240):
   ```python
   if self.simple_delete_only_mode:
       # Rule 0: Growth penalty
       growth_penalty_only = 0.0
       if num_nodes > self.max_critical_nodes:
           excess_nodes = num_nodes - self.max_critical_nodes
           growth_penalty_only = -self.growth_penalty * (1 + excess_nodes / self.max_critical_nodes)
       
       # Rules 1 & 2: Delete penalties
       delete_penalty_only = delete_reward if delete_reward < 0 else 0.0
       
       # Combine penalties (termination rewards handled separately)
       total_reward = growth_penalty_only + delete_penalty_only
       # ... zero out all other components
   ```

3. **Termination Reward Handling** (`durotaxis_env.py` line ~965):
   ```python
   if terminated:
       if self.simple_delete_only_mode:
           # Add termination reward to the penalties
           reward_components['total_reward'] = (
               reward_components.get('graph_reward', 0.0) + termination_reward
           )
       else:
           # Normal mode
           reward_components['total_reward'] += termination_reward
   ```

4. **Delete Reward Logic** (`durotaxis_env.py` line ~1582):
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
