# Ablation Study Quick Start Guide

## Overview
This guide provides quick instructions for conducting ablation studies comparing standard Actor vs WSA-Enhanced Actor.

## Quick Toggle

### Option 1: Baseline (Standard Actor)
```yaml
# config.yaml
actor_critic:
  wsa:
    enabled: false  # ← Set to false
```

### Option 2: Enhanced (WSA with Multi-PTM)
```yaml
# config.yaml
actor_critic:
  wsa:
    enabled: true  # ← Set to true
```

## Running Experiments

### 1. Baseline Run (Standard Actor + ResNet18)
```bash
# Edit config.yaml: wsa.enabled = false
python train.py --config config.yaml --experiment baseline_resnet18
```

### 2. WSA Run (Multi-PTM with Attention)
```bash
# Edit config.yaml: wsa.enabled = true
python train.py --config config.yaml --experiment wsa_multi_ptm
```

## Custom PTM Configurations

### Example 1: ResNet Ensemble
```yaml
pretrained_models:
  - name: resnet18_imagenet
    model_type: resnet18
    weights: imagenet
    freeze_backbone: true
  
  - name: resnet34_imagenet
    model_type: resnet34
    weights: imagenet
    freeze_backbone: true
  
  - name: resnet50_random
    model_type: resnet50
    weights: random
    freeze_backbone: false
```

### Example 2: Mixed Architecture
```yaml
pretrained_models:
  - name: resnet18_imagenet
    model_type: resnet18
    weights: imagenet
    freeze_backbone: true
  
  - name: graph_cnn_custom
    model_type: graph_cnn
    weights: null
    freeze_backbone: false
```

### Example 3: All Random (For Fair Comparison)
```yaml
pretrained_models:
  - name: resnet18_random_1
    model_type: resnet18
    weights: random
    freeze_backbone: false
  
  - name: resnet18_random_2
    model_type: resnet18
    weights: random
    freeze_backbone: false
  
  - name: resnet18_random_3
    model_type: resnet18
    weights: random
    freeze_backbone: false
```

## Monitoring WSA Performance

### Key Metrics to Track

1. **Attention Weights Distribution**
   - Enable logging: `wsa.log_attention_weights: true`
   - Check which PTMs are emphasized during training
   - Look for patterns: Does one PTM dominate? Do weights balance?

2. **Performance Metrics**
   - Episode reward
   - Success rate
   - Convergence speed
   - Final performance plateau

3. **Feature Quality**
   - Enable feature logging: `wsa.log_features: true`
   - Analyze diversity of features from different PTMs
   - Check for complementary vs redundant features

### TensorBoard Monitoring
```bash
tensorboard --logdir runs/
```

Look for:
- `wsa/attention_weights_*` - Attention distribution over PTMs
- `wsa/feature_norms_*` - Feature magnitude from each PTM
- `wsa/attention_entropy` - Diversity of attention (high = balanced, low = focused)

## Ablation Study Design

### Recommended Experiments

#### Phase 1: Baseline vs WSA
1. **Baseline**: Standard Actor with ResNet18 (ImageNet)
2. **WSA-3**: ResNet18 (ImageNet) + Graph-CNN + ResNet18 (Random)

**Hypothesis**: WSA with multiple PTMs should outperform single ResNet18

#### Phase 2: PTM Type Ablation
1. **WSA-ImageNet**: 3x ResNet18 (all ImageNet)
2. **WSA-Random**: 3x ResNet18 (all Random)
3. **WSA-Mixed**: 1x ImageNet + 2x Random

**Hypothesis**: Mixed weights provide best diversity

#### Phase 3: Architecture Ablation
1. **WSA-ResNet18**: 3x ResNet18
2. **WSA-ResNet-Mix**: ResNet18 + ResNet34 + ResNet50
3. **WSA-Hybrid**: ResNet + Graph-CNN

**Hypothesis**: Architecture diversity improves robustness

#### Phase 4: Attention Mechanism Ablation
1. **WSA**: Full attention mechanism
2. **Equal-Weight**: Disable attention, average all features equally
3. **Single-Best**: Use only best-performing PTM (oracle)

**Hypothesis**: Learned attention outperforms static weighting

### Statistical Analysis

Run each configuration **5 times** with different seeds:
```bash
for seed in 1 2 3 4 5; do
    python train.py --config config.yaml \
                    --experiment baseline_resnet18 \
                    --seed $seed
done
```

Compare using:
- Mean ± Std of final performance
- Learning curves with confidence intervals
- Statistical significance tests (t-test, Mann-Whitney U)

## Expected Outcomes

### Success Indicators for WSA
✅ **Higher Final Performance**: WSA achieves better asymptotic reward
✅ **Faster Convergence**: WSA reaches target performance sooner
✅ **Better Sample Efficiency**: WSA needs fewer episodes
✅ **Dynamic Attention**: Attention weights change meaningfully during training
✅ **Complementary Features**: Different PTMs capture different aspects

### Warning Signs
⚠️ **Attention Collapse**: One PTM gets >90% attention consistently
⚠️ **Slower Convergence**: WSA takes longer to learn
⚠️ **Higher Variance**: WSA performance unstable across runs
⚠️ **No Improvement**: WSA performs same as baseline

## Troubleshooting

### Issue: WSA underperforming baseline
**Possible Causes**:
- PTMs not diverse enough (all using same architecture/weights)
- Attention mechanism not learning (check gradients)
- Increased model complexity without benefit (more parameters to learn)

**Solutions**:
- Try more diverse PTM configurations
- Adjust attention learning rate separately
- Start with frozen PTMs, gradually unfreeze

### Issue: Training too slow
**Possible Causes**:
- Too many PTMs (computational overhead)
- All PTMs trainable (too many parameters)

**Solutions**:
- Reduce number of PTMs to 2-3
- Freeze more backbones
- Use smaller ResNet variants (ResNet18 instead of ResNet50)

### Issue: Attention weights not changing
**Possible Causes**:
- Attention learning rate too low
- Features too similar (PTMs producing same representations)

**Solutions**:
- Increase attention learning rate
- Use more diverse PTM configurations
- Check feature visualization to ensure diversity

## Quick Commands

### Test Configuration
```bash
python test_wsa_config.py
```

### Train Baseline
```bash
# config.yaml: wsa.enabled = false
python train.py
```

### Train WSA
```bash
# config.yaml: wsa.enabled = true
python train.py
```

### Monitor Training
```bash
tensorboard --logdir runs/
```

### Compare Results
```bash
# Plot learning curves
python plotter.py --experiments baseline_resnet18 wsa_multi_ptm
```

## Configuration Checklist

Before starting ablation study:

- [ ] Set unique experiment name for each run
- [ ] Configure random seed for reproducibility
- [ ] Enable WSA logging (`log_attention_weights`, `log_features`)
- [ ] Set appropriate learning rates
- [ ] Plan number of training episodes
- [ ] Prepare result tracking (TensorBoard, CSV logs)
- [ ] Document PTM configurations in experiment notes

## Results Organization

Suggested directory structure:
```
results/
├── baseline/
│   ├── seed_1/
│   ├── seed_2/
│   └── ...
├── wsa_3ptm/
│   ├── seed_1/
│   ├── seed_2/
│   └── ...
└── analysis/
    ├── learning_curves.png
    ├── attention_evolution.png
    └── statistical_tests.csv
```

## Next Steps

1. **Run baseline** with WSA disabled
2. **Run WSA** with default 3-PTM configuration
3. **Compare performance** using learning curves
4. **Analyze attention** weights to understand PTM contributions
5. **Iterate** with different PTM configurations if needed
6. **Document findings** for publication/presentation

Good luck with your ablation study! 🚀
