# Ablation Study Quick Reference Card

## ✅ STATUS: ALL 8 CONFIGURATIONS READY

---

## The 8 Configurations

```
┌─────────────────────────────────────────────────────────────────┐
│  GROUP (a): BASELINE - No WSA + No SEM                          │
├─────────────────────────────────────────────────────────────────┤
│  a1: imagenet + No WSA + No SEM  ← PRIMARY BASELINE             │
│  a2: random   + No WSA + No SEM                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  GROUP (b): WSA ENHANCEMENT - WSA + No SEM                      │
├─────────────────────────────────────────────────────────────────┤
│  b1: imagenet + WSA + No SEM                                    │
│  b2: random   + WSA + No SEM                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  GROUP (c): SEM ENHANCEMENT - No WSA + SEM                      │
├─────────────────────────────────────────────────────────────────┤
│  c1: imagenet + No WSA + SEM                                    │
│  c2: random   + No WSA + SEM                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  GROUP (d): FULL STACK - WSA + SEM                              │
├─────────────────────────────────────────────────────────────────┤
│  d1: imagenet + WSA + SEM  ← EXPECTED BEST                      │
│  d2: random   + WSA + SEM                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Template

### Edit `config.yaml` for each run:

```yaml
actor_critic:
  # Set to 'imagenet' or 'random'
  pretrained_weights: 'imagenet'
  
  wsa:
    # Set to true or false
    enabled: false
  
  simplicial_embedding:
    # Set to true or false
    enabled: false
```

---

## Example Configurations

### (a1) Baseline - ImageNet + No WSA + No SEM
```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: false
  simplicial_embedding:
    enabled: false
```

### (b1) WSA - ImageNet + WSA + No SEM
```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: true
  simplicial_embedding:
    enabled: false
```

### (c1) SEM - ImageNet + No WSA + SEM
```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: false
  simplicial_embedding:
    enabled: true
```

### (d1) Full Stack - ImageNet + WSA + SEM
```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: true
  simplicial_embedding:
    enabled: true
```

---

## Running Experiments

### Test All Configurations
```bash
python test_ablation_configurations.py
```

### Run Single Configuration (5 seeds)
```bash
for seed in 1 2 3 4 5; do
    python train.py \
        --config config.yaml \
        --experiment <config_name>_seed${seed} \
        --seed ${seed}
done
```

### Monitor Training
```bash
tensorboard --logdir runs/
```

---

## Key Comparisons

| Comparison | Tests | Expected Result |
|------------|-------|-----------------|
| **a1 vs a2** | Pretrained weights value | a1 > a2 |
| **b1 vs a1** | WSA enhancement | b1 > a1 |
| **c1 vs a1** | SEM enhancement | c1 > a1 |
| **d1 vs all** | Best combination | d1 > all others |

---

## Metrics to Track

- ✅ Episode Reward (primary)
- ✅ Success Rate
- ✅ Episode Length
- ✅ Convergence Speed
- ✅ Training Time
- ✅ Memory Usage

---

## Expected Performance Ranking

```
Best  → d1 (imagenet + WSA + SEM)
      ↓ b1 or c1 (depends on WSA vs SEM strength)
      ↓ d2 (random + WSA + SEM)
      ↓ b2 or c2
      ↓ a1 (imagenet baseline)
Worst → a2 (random baseline)
```

---

## Files You Need

### Test Script
`test_ablation_configurations.py`

### Guides
- `ABLATION_READINESS_SUMMARY.md` - This summary
- `COMPREHENSIVE_ABLATION_GUIDE.md` - Detailed guide
- `ABLATION_STUDY_QUICKSTART.md` - Quick start

### Configuration
`config.yaml` - Edit for each run

---

## Quick Checks

### ✅ Everything Working?
```bash
python test_ablation_configurations.py
# Expected: "✅ ALL ABLATION CONFIGURATIONS ARE READY!"
```

### ✅ Config Correct?
Check these lines in `config.yaml`:
- `actor_critic.pretrained_weights`
- `actor_critic.wsa.enabled`
- `actor_critic.simplicial_embedding.enabled`

### ✅ GPU Available?
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Time Estimates

| Item | Time |
|------|------|
| Per seed | 2-4 hours |
| Per config (5 seeds) | 10-20 hours |
| All 8 configs | 80-160 hours |
| With 4 GPUs (parallel) | 20-40 hours |

---

## Remember

- Run **5 seeds** per configuration for statistical validity
- Use **same environment seed** across configs
- Vary **network initialization seed**
- Save all checkpoints and logs
- Monitor with TensorBoard

---

**Last Updated**: December 2024  
**Status**: ✅ READY  
**Configs Tested**: 8/8 PASSING
