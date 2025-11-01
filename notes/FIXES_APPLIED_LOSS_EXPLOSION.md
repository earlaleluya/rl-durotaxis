# Critical Bug Fixes for Loss Explosion

## Applied Fixes

### 1. ✅ Relative Value Clipping (Lines 593-610)
Added support for relative value clipping that scales with return magnitude:
- New config: `use_relative_value_clip: true` (default)
- Clips value changes as % of old value instead of absolute amount
- Prevents tiny clips when returns are large (1000+)

### 2. ✅ Return Normalization (Lines 2765-2771)  
Added return normalization to prevent value targets from exploding:
- New config: `normalize_returns: true` (default)
- New config: `return_scale: 10.0` (normalize then scale to this magnitude)
- Prevents value loss from computing (prediction - 1500)² type errors

### 3. ⚠️ NEEDS MANUAL EDIT: Value Loss Weight
**Lines to change**: 3083 and 3407

Change:
```python
total_loss = avg_total_policy_loss + 0.5 * total_value_loss + avg_entropy_loss + entropy_bonus
```

To:
```python
value_loss_weight = 0.25  # Reduced from 0.5 to prevent value loss domination
total_loss = avg_total_policy_loss + value_loss_weight * total_value_loss + avg_entropy_loss + entropy_bonus
losses['value_loss_weight'] = value_loss_weight
```

**Reason**: Reduces impact of large value losses on total loss.

## Config.yaml Additions

Add these to your `config.yaml` under the `ppo:` section:

```yaml
ppo:
  # ... existing config ...
  
  # Value clipping improvements (prevents loss explosion)
  enable_value_clipping: true
  value_clip_epsilon: 0.2  # Interpreted as relative when use_relative_value_clip=true
  use_relative_value_clip: true  # Clip as % of old value, not absolute
  
  # Return normalization (prevents value targets from exploding)
  normalize_returns: true
  return_scale: 10.0  # Scale normalized returns to this magnitude
```

## Testing

After applying fixes, monitor:

1. **Loss metrics**: Should stay < 100k and trend downward
2. **Rewards**: Should improve or stay positive  
3. **Value losses**: Each component < 10,000
4. **Warnings**: Check for value clipping warnings in logs

## Verification Commands

```bash
# Check if fixes are applied
grep -n "use_relative_value_clip" train.py
grep -n "normalize_returns" train.py
grep -n "value_loss_weight = 0.25" train.py  # Should find 2 instances

# Start training and monitor
python train_cli.py --mode single_batch --verbose

# Check loss trends
tail -f training_results/run*/loss_metrics.json
```

## Expected Improvements

Before fixes:
- Loss: 36k → 200k (exploding)
- Rewards: 1900 → -400 (collapsing)

After fixes:
- Loss: Should stay < 100k, ideally decreasing
- Rewards: Should improve over time
- Training: More stable, no explosive growth

## Rollback

If issues occur, set in config.yaml:
```yaml
ppo:
  use_relative_value_clip: false
  normalize_returns: false
  value_clip_epsilon: 200.0  # Use large absolute clip as fallback
```

## Additional Notes

- The root cause was value_clip_epsilon=0.2 being too small for returns of 1000+
- Relative clipping interprets 0.2 as "20% of old value" which scales appropriately
- Return normalization ensures value network doesn't have to learn extreme scales
- Reduced value loss weight prevents value learning from destabilizing policy
