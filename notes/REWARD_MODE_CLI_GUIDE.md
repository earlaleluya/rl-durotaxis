# Reward Mode CLI Guide

Quick reference for using the new reward mode flags in `train_cli.py`.

## New Command-Line Flags

### Reward Mode Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--simple-delete-only` | Enable simple delete-only mode (Rule 0, 1, 2 only) | False |
| `--no-simple-delete-only` | Disable simple delete-only mode | - |
| `--centroid-distance-only` | Enable centroid distance-only mode | False |
| `--no-centroid-distance-only` | Disable centroid distance-only mode | - |
| `--include-termination-rewards` | Include termination rewards in special modes | False |
| `--no-include-termination-rewards` | Exclude termination rewards in special modes | - |

## Usage Examples

### 1. Normal Training (Full Rewards)
```bash
python train_cli.py --experiment normal_full_rewards
```
- All reward components active
- Termination rewards always included
- Default behavior

### 2. Simple Delete-Only Mode (No Termination)
```bash
python train_cli.py --simple-delete-only --experiment delete_only_pure
```
- **Included**: Rule 0 (growth), Rule 1 (persistence), Rule 2 (improper deletion)
- **Excluded**: Spawn, movement, milestones, termination rewards
- Pure deletion learning

### 3. Simple Delete-Only Mode (With Termination)
```bash
python train_cli.py --simple-delete-only --include-termination-rewards --experiment delete_only_with_outcome
```
- **Included**: Rule 0, 1, 2 + success/failure rewards
- **Excluded**: Spawn, movement, milestones
- Deletion learning + outcome feedback

### 4. Centroid Distance-Only Mode (No Termination)
```bash
python train_cli.py --centroid-distance-only --experiment distance_pure
```
- **Included**: Distance penalty only: `-(goal_x - centroid_x) / goal_x`
- **Excluded**: All other rewards including termination
- Pure goal-seeking via distance gradient

### 5. Centroid Distance-Only Mode (With Termination)
```bash
python train_cli.py --centroid-distance-only --include-termination-rewards --experiment distance_with_success
```
- **Included**: Distance penalty + success/failure rewards
- **Excluded**: Movement, spawn, milestones
- Goal-seeking + outcome feedback

### 6. Override Config.yaml Settings
```bash
# Disable centroid mode if enabled in config.yaml
python train_cli.py --no-centroid-distance-only --experiment normal_override

# Enable simple delete mode with termination
python train_cli.py --simple-delete-only --include-termination-rewards --seed 42
```

## Mode Comparison

| Mode | Rewards Included | Use Case |
|------|------------------|----------|
| **Normal** | All components | Standard training |
| **Simple Delete (no term)** | Rule 0, 1, 2 only | Pure deletion learning |
| **Simple Delete (with term)** | Rule 0, 1, 2 + termination | Deletion + outcomes |
| **Distance (no term)** | Distance penalty only | Pure distance learning |
| **Distance (with term)** | Distance + termination | Distance + outcomes |

## Reward Breakdown by Mode

### Normal Mode
```
✅ Spawn rewards
✅ Delete rewards (Rule 1, 2)
✅ Growth penalty (Rule 0)
✅ Movement rewards
✅ Milestone rewards
✅ Edge rewards
✅ Survival rewards
✅ Termination rewards (success, failure, etc.)
```

### Simple Delete-Only Mode (no termination)
```
❌ Spawn rewards
✅ Delete rewards (Rule 1, 2) - penalties only
✅ Growth penalty (Rule 0)
❌ Movement rewards
❌ Milestone rewards
❌ Edge rewards
❌ Survival rewards
❌ Termination rewards
```

### Simple Delete-Only Mode (with termination)
```
❌ Spawn rewards
✅ Delete rewards (Rule 1, 2) - penalties only
✅ Growth penalty (Rule 0)
❌ Movement rewards
❌ Milestone rewards
❌ Edge rewards
❌ Survival rewards
✅ Termination rewards (success +500, failures -100, etc.)
```

### Centroid Distance-Only Mode (no termination)
```
❌ All standard rewards
✅ Distance penalty: -(goal_x - centroid_x) / goal_x
❌ Termination rewards
```

### Centroid Distance-Only Mode (with termination)
```
❌ All standard rewards
✅ Distance penalty: -(goal_x - centroid_x) / goal_x
✅ Termination rewards (success +500, failures -100, etc.)
```

## Complete Example Workflows

### Ablation Study: Compare Reward Modes

```bash
# Baseline: Full rewards
python train_cli.py --experiment ablation_full --seed 1

# Ablation 1: Distance only (no outcome)
python train_cli.py --centroid-distance-only \
    --experiment ablation_distance_pure --seed 1

# Ablation 2: Distance + outcome
python train_cli.py --centroid-distance-only --include-termination-rewards \
    --experiment ablation_distance_outcome --seed 1

# Ablation 3: Delete only (no outcome)
python train_cli.py --simple-delete-only \
    --experiment ablation_delete_pure --seed 1

# Ablation 4: Delete + outcome
python train_cli.py --simple-delete-only --include-termination-rewards \
    --experiment ablation_delete_outcome --seed 1
```

### Quick Testing

```bash
# Test distance mode with success rewards for 500 episodes
python train_cli.py --centroid-distance-only --include-termination-rewards \
    --total-episodes 500 --experiment quick_distance_test

# Test delete mode without termination for 500 episodes
python train_cli.py --simple-delete-only \
    --total-episodes 500 --experiment quick_delete_test
```

## Notes

1. **Flag Precedence**: CLI flags override `config.yaml` settings
2. **Normal Mode**: When both special modes are False, termination rewards are ALWAYS included (flag ignored)
3. **Special Modes**: When either special mode is True, termination rewards are EXCLUDED by default unless `--include-termination-rewards` is set
4. **Combining Modes**: Don't enable both `--simple-delete-only` and `--centroid-distance-only` simultaneously (last one wins)

## Configuration File Equivalents

### CLI vs Config.yaml

```bash
# CLI
python train_cli.py --centroid-distance-only --include-termination-rewards
```

```yaml
# config.yaml
environment:
  centroid_distance_only_mode: true
  include_termination_rewards: true
```

---

**Tip**: Use `python train_cli.py --help` to see all available options and examples!
