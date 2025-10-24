# PPO Health Metrics - Quick Reference Guide

## How to Use PPO Metrics for Debugging Training Issues

### Problem: Training Loss Explodes or NaN

**Check these metrics:**
- KL Divergence: Should be < 0.1 (if much higher, policy is changing too fast)
- Ratio: Should be between 0.5 and 2.0 (extreme values indicate instability)
- Clip Fraction: If > 0.8, try reducing learning rate

**Solutions:**
```yaml
# In config.yaml
trainer:
  learning_rate: 0.0001  # Reduce from 0.0003

algorithm:
  clip_epsilon: 0.1      # Reduce from 0.2 (more conservative clipping)
  target_kl: 0.01        # Reduce from 0.03 (stricter early stopping)
```

---

### Problem: Policy Stops Learning (Flat Rewards)

**Check these metrics:**
- Clip Fraction: If < 0.05, policy updates are too small
- KL Divergence: If < 0.001, policy is barely changing
- Entropy: If < 0.1, policy has collapsed (no exploration)

**Solutions:**
```yaml
# In config.yaml
trainer:
  learning_rate: 0.001        # Increase from 0.0003
  update_epochs: 8            # Increase from 4 (more gradient steps)

algorithm:
  clip_epsilon: 0.3           # Increase from 0.2 (allow larger updates)
  
entropy_regularization:
  entropy_coeff_start: 0.3    # Increase from 0.2 (more exploration)
```

---

### Problem: Value Function Not Learning

**Check these metrics:**
- Explained Variance: Should be > 0.3 (if negative, value function is broken)

**Solutions:**
```yaml
# In config.yaml
trainer:
  # Value loss weight (in code, check policy_loss_weights)
  # Increase value_loss coefficient relative to policy_loss

# Also check:
# 1. Reward scaling (are rewards too large/small?)
# 2. GAE lambda (try 0.95-0.99)
# 3. Gamma (try 0.99-0.999)
```

---

### Problem: CPU vs GPU Training Differs

**Check these metrics on both devices:**
- Compare KL, Ratio, Clip Fraction between runs
- If significantly different: numerical instability issue

**Solutions:**
1. Ensure all tensors created with explicit device
2. Check for CPU-only operations in forward pass
3. Verify batch norm behavior (use eval mode for deterministic inference)
4. Set random seeds: `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`

---

## Optimal Metric Ranges (Rules of Thumb)

| Metric | Good Range | Warning Signs |
|--------|-----------|---------------|
| KL Divergence | 0.01 - 0.05 | > 0.1 (too aggressive), < 0.001 (too conservative) |
| Clip Fraction (Discrete) | 0.1 - 0.4 | > 0.7 (clipping too much), < 0.05 (not learning) |
| Clip Fraction (Continuous) | 0.1 - 0.4 | > 0.7 (clipping too much), < 0.05 (not learning) |
| Ratio | 0.8 - 1.2 | > 2.0 or < 0.5 (policy unstable) |
| Explained Variance | 0.3 - 0.9 | < 0.0 (value broken), > 0.95 (possible overfitting) |
| Entropy | 0.3 - 1.5 | < 0.1 (no exploration), > 2.0 (random policy) |

---

## Common Patterns and What They Mean

### Pattern 1: High KL → Early Stopping → Low Clip Fraction
**Meaning**: Policy trying to change too fast, but early stopping prevents it

**Action**: This is working as intended! PPO is protecting you from policy collapse.
- If happening every batch: increase target_kl slightly (0.03 → 0.05)
- If happening rarely: leave as is

---

### Pattern 2: Low KL + Low Clip Fraction + Flat Rewards
**Meaning**: Learning rate too low or policy stuck in local optimum

**Action**: 
1. Increase learning rate
2. Increase entropy coefficient (boost exploration)
3. Check if reward signal is too sparse

---

### Pattern 3: Explained Variance Negative or Decreasing
**Meaning**: Value function learning is broken

**Action**:
1. Check reward normalization (values too large/small?)
2. Increase value loss weight
3. Verify GAE computation is correct
4. Check for NaN in advantages/returns

---

### Pattern 4: Discrete Metrics Good, Continuous Metrics Bad (or vice versa)
**Meaning**: Imbalanced learning between action spaces

**Action**:
```yaml
# Adjust policy loss weights
trainer:
  policy_loss_weights:
    discrete_weight: 0.7    # Topology actions
    continuous_weight: 0.3  # Spawn parameters
```

Or enable adaptive gradient scaling:
```yaml
gradient_scaling:
  enable_adaptive_scaling: true  # Auto-balance gradients
```

---

## Quick Diagnostic Commands

### View Current Metrics in Training Log
```bash
# Watch training in real-time
tail -f training_results/run####/training.log | grep "PPO Health"

# Count early stopping events
grep "Early stopping" training_results/run####/training.log | wc -l
```

### Extract Metrics for Plotting
```bash
# Extract KL divergence values
grep "KL:" training_results/run####/training.log | \
  awk '{print $3}' > kl_values.txt

# Extract clip fractions
grep "Clip Frac:" training_results/run####/training.log | \
  sed 's/.*D=\([0-9.]*\) C=\([0-9.]*\).*/\1 \2/' > clip_fractions.txt
```

---

## When to Intervene vs When to Wait

### Intervene Immediately If:
- KL > 0.5 (policy diverging)
- Explained Variance < -0.5 (value function catastrophically bad)
- Ratio > 5.0 or < 0.2 (numerical instability)
- Loss = NaN or Inf

### Wait and Monitor If:
- KL consistently 0.05-0.1 (acceptable, just watch it)
- Clip Fraction 0.4-0.6 (high but not dangerous)
- Explained Variance slowly improving from 0.0 → 0.3 (value function learning)
- Early stopping 1-2 times per batch (PPO working as designed)

### Normal Patterns:
- Early training: High KL, high clip fraction, low explained variance
- Mid training: Moderate KL (0.02-0.05), moderate clip fraction (0.2-0.4), rising explained variance
- Late training: Low KL (0.01-0.02), low clip fraction (0.1-0.2), high explained variance (> 0.6)

---

## Advanced: Customizing Metric Thresholds

Edit `train.py` to adjust warnings and early stopping behavior:

```python
# Around line 3180 in train.py
print(f"   📊 PPO Health Metrics:")
print(f"      KL: {approx_kl:.4f} (target: {self.target_kl:.4f}) "
      f"{'⚠️ HIGH' if approx_kl > self.target_kl else '✓'}")

# Change the target_kl check to be more/less strict:
# More strict: approx_kl > 0.02
# More lenient: approx_kl > 0.1
```

---

## Related Documentation

- [PPO_METRICS_VERIFICATION.md](./PPO_METRICS_VERIFICATION.md) - Verification results
- [REWARD_SYSTEM_IMPROVEMENTS.md](./REWARD_SYSTEM_IMPROVEMENTS.md) - Reward structure details
- [ABLATION_STUDY_QUICKSTART.md](./ABLATION_STUDY_QUICKSTART.md) - Hyperparameter tuning guide
