# Two-Stage Training Curriculum Guide

## Overview

This guide explains how to use the **two-stage training curriculum** to intelligently train your durotaxis agent. This approach breaks down the complex learning problem into two manageable stages:

1. **Stage 1 (Discrete Actions)**: Train the agent to master migration using only spawn/delete decisions with fixed spawn parameters
2. **Stage 2 (Fine-Tuning)**: Fine-tune the agent to optimize the continuous spawn parameters (gamma, alpha, noise, theta)

### Why Use Two-Stage Training?

**Benefits:**
- **Reduced Complexity**: Stage 1 focuses on "what to do" and "where to do it" without worrying about "how to do it optimally"
- **Stable Learning**: The agent first learns the fundamentals of survival and migration
- **Better Final Performance**: Fine-tuning continuous parameters on top of a solid discrete policy often yields better results than learning everything simultaneously
- **Prevents Local Minima**: Avoids getting stuck in poor local optima where bad spawn parameters prevent the agent from learning good topology decisions

**When to Use This:**
- When training from scratch on a complex environment
- When the agent struggles to learn with full hybrid action space
- When you want to guarantee the agent learns basic migration first
- When you want to understand the contribution of discrete vs continuous actions

---

## Configuration

All two-stage curriculum settings are located in `config.yaml` under the `algorithm` section:

```yaml
algorithm:
  # ... other algorithm settings ...
  
  two_stage_curriculum:
    stage: 1  # Set to 1 for discrete-only training, 2 for fine-tuning
    
    # Fixed spawn parameters used during Stage 1
    # These should be reasonable defaults that encourage stable growth
    stage_1_fixed_spawn_params:
      gamma: 5.0    # Growth rate (higher = faster spawning)
      alpha: 1.0    # Angle spread (higher = wider spread)
      noise: 0.1    # Randomness in spawn direction
      theta: 0.0    # Base angle offset
```

### Configuration Parameters

| Parameter | Description | Stage 1 Default | Notes |
|-----------|-------------|-----------------|-------|
| `stage` | Training stage (1 or 2) | 1 | Controls which training mode is active |
| `gamma` | Growth rate for spawning | 5.0 | Higher values = faster node generation |
| `alpha` | Angular spread parameter | 1.0 | Controls how wide new nodes spread |
| `noise` | Random noise in spawn direction | 0.1 | Small value = more deterministic |
| `theta` | Base angle offset | 0.0 | Rotation offset for spawn direction |

### Choosing Good Fixed Parameters

The fixed parameters should be:
- **Stable**: Don't cause immediate catastrophic failures
- **Reasonable**: Allow the environment to demonstrate its dynamics clearly
- **Not Optimal**: If they're too good, Stage 2 won't have room to improve

**Recommended starting values:**
```yaml
stage_1_fixed_spawn_params:
  gamma: 5.0   # Moderate growth rate
  alpha: 1.0   # Moderate spread
  noise: 0.1   # Small randomness
  theta: 0.0   # No rotation bias
```

**For different substrate types:**
```yaml
# Linear substrates (simple gradients)
stage_1_fixed_spawn_params:
  gamma: 5.0
  alpha: 1.0
  noise: 0.05  # Less noise for clearer gradient
  theta: 0.0

# Exponential substrates (steep gradients)
stage_1_fixed_spawn_params:
  gamma: 3.0   # Lower gamma for more control
  alpha: 1.5   # Wider spread to explore
  noise: 0.15  # More noise for exploration
  theta: 0.0
```

---

## Stage 1: Discrete Action Training

### Goal
Train the agent to successfully migrate from left to right using only topology decisions (spawn vs delete, and which node to act on).

### What Happens in Stage 1

1. **Fixed Spawn Parameters**: All spawn actions use the fixed parameters from config
2. **Discrete Policy Learning**: The network learns which nodes to spawn from and which to delete
3. **No Continuous Loss**: The continuous action head receives no gradient signal
4. **Full Reward Signal**: The agent receives all reward components based on its discrete decisions

### Setup for Stage 1

1. **Update `config.yaml`:**
   ```yaml
   algorithm:
     two_stage_curriculum:
       stage: 1  # Discrete-only training
   ```

2. **Set Training Duration:**
   ```yaml
   trainer:
     total_episodes: 1000  # Or more, until agent masters migration
   ```

3. **Monitor Training Progress:**
   - Watch episode rewards increasing
   - Check success rate (reaching rightmost substrate)
   - Monitor average episode length
   - Look for consistent rightward migration

### Running Stage 1 Training

```bash
# Activate your environment
conda activate durotaxis

# Run training
python train.py --config config.yaml

# Or use the CLI
python train_cli.py --total-episodes 1000
```

### Stage 1 Success Criteria

Stop Stage 1 training when the agent demonstrates:
- ✓ **Consistent migration**: 50%+ episodes reach the rightmost region
- ✓ **Stable topology**: Maintains 5-30 nodes without collapsing
- ✓ **Long episodes**: Average episode length > 100 steps
- ✓ **Positive rewards**: Moving average reward > -50

**Typical Stage 1 training time:**
- Simple substrates (linear): 500-1000 episodes
- Complex substrates (exponential): 1000-2000 episodes

---

## Stage 2: Continuous Parameter Fine-Tuning

### Goal
Fine-tune the continuous spawn parameters to optimize the migration strategy learned in Stage 1.

### What Happens in Stage 2

1. **Network Output Used**: Spawn parameters now come from the network's continuous action head
2. **Full Hybrid Loss**: Both discrete and continuous policy losses contribute to training
3. **Preserved Discrete Policy**: The strong discrete policy from Stage 1 is retained
4. **Parameter Optimization**: The network learns optimal gamma, alpha, noise, theta for each spawn decision

### Setup for Stage 2

1. **Save Your Stage 1 Model:**
   
   After Stage 1 completes, note the path to the best model:
   ```
   training_results/run0001/best_model_episode_850.pt
   ```

2. **Update `config.yaml` for Stage 2:**
   ```yaml
   trainer:
     total_episodes: 1000  # Additional episodes for fine-tuning
     learning_rate: 5e-5   # Lower learning rate to preserve Stage 1 learning
     
     resume_training:
       enabled: true
       checkpoint_path: "training_results/run0001/best_model_episode_850.pt"
       reset_optimizer: true      # Fresh optimizer for fine-tuning
       reset_episode_count: true  # Start episode counter from 0
   
   algorithm:
     two_stage_curriculum:
       stage: 2  # Enable continuous action learning
   ```

3. **Important Stage 2 Configuration:**
   - **Lower Learning Rate**: Use 5e-5 to 1e-4 (10-20x smaller than Stage 1)
   - **Reset Optimizer**: Start with fresh momentum for the new objective
   - **Monitor Carefully**: Watch that discrete performance doesn't degrade

### Running Stage 2 Training

```bash
# Activate your environment
conda activate durotaxis

# Run Stage 2 fine-tuning
python train.py --config config.yaml

# Or use CLI with resume
python train_cli.py --resume training_results/run0001/best_model_episode_850.pt
```

### Stage 2 Expected Outcomes

**Good Fine-Tuning (Stage 2 is working):**
- ✓ Gradual reward improvement (5-15% over Stage 1)
- ✓ Maintained or improved success rate
- ✓ More efficient node usage (fewer nodes for same migration)
- ✓ Smoother trajectories (less zig-zagging)

**Warning Signs (Stage 2 may be hurting):**
- ⚠️ Sudden drop in success rate
- ⚠️ Rewards decreasing instead of improving
- ⚠️ Unstable topology (node counts fluctuating wildly)

**If Stage 2 is hurting performance:**
1. **Reduce learning rate** (try 1e-5)
2. **Reduce update epochs** (from 4 to 2)
3. **Increase target_kl** (from 0.03 to 0.01 for more conservative updates)

---

## Complete Training Workflow

### Step-by-Step Process

#### 1. Initial Setup
```bash
# Clone or navigate to repository
cd rl-durotaxis

# Activate environment
conda activate durotaxis

# Verify configuration
python show_default_config.py
```

#### 2. Stage 1: Train Discrete Policy
```yaml
# config.yaml
algorithm:
  two_stage_curriculum:
    stage: 1
    stage_1_fixed_spawn_params:
      gamma: 5.0
      alpha: 1.0
      noise: 0.1
      theta: 0.0

trainer:
  total_episodes: 1000
  learning_rate: 0.0003
```

```bash
# Run Stage 1 training
python train.py --config config.yaml

# Monitor progress
tail -f training_results/run0001/training.log
```

**Wait for Stage 1 to complete** (look for consistent success)

#### 3. Transition to Stage 2

```yaml
# config.yaml - UPDATE THESE SECTIONS
trainer:
  total_episodes: 1000
  learning_rate: 5e-5  # REDUCED
  
  resume_training:
    enabled: true
    checkpoint_path: "training_results/run0001/best_model_episode_XXX.pt"  # Use your best model
    reset_optimizer: true
    reset_episode_count: true

algorithm:
  two_stage_curriculum:
    stage: 2  # CHANGED FROM 1
```

#### 4. Stage 2: Fine-Tune Continuous Parameters
```bash
# Run Stage 2 training
python train.py --config config.yaml

# Monitor fine-tuning progress
tail -f training_results/run0002/training.log
```

#### 5. Evaluation
```bash
# Deploy and test final agent
python deploy.py --model training_results/run0002/best_model_final.pt --episodes 100
```

---

## Monitoring and Diagnostics

### Key Metrics to Watch

#### Stage 1 Metrics
| Metric | Good Range | Warning |
|--------|-----------|---------|
| Success Rate | 40-80% | < 20% (not learning) |
| Avg Episode Length | 80-200 steps | < 50 steps (dying too fast) |
| Avg Reward | -50 to +50 | < -100 (struggling) |
| Avg Node Count | 5-30 nodes | < 3 or > 50 (unstable) |

#### Stage 2 Metrics
| Metric | Expected Change | Warning |
|--------|----------------|---------|
| Success Rate | +0% to +20% | -10% or worse (degrading) |
| Avg Reward | +5% to +15% | -5% or worse (hurting) |
| Avg Nodes | -10% to +0% (more efficient) | +30% (wasteful) |

### Logging Output

#### Stage 1 Training Log
```
⭐️ Training Mode: Stage 1 (Discrete Actions Only)
Episode 100: R=+25.3 | Steps=85 | Success=True | Loss=0.234
  └─ Spawn params: γ=5.00 α=1.00 ν=0.10 θ=0.00 (Fixed)
```

#### Stage 2 Training Log
```
⭐️ Training Mode: Stage 2 (Fine-tuning Continuous Actions)
Episode 100: R=+31.7 | Steps=82 | Success=True | Loss=0.156
  └─ Spawn params: γ=4.32 α=1.45 ν=0.08 θ=0.12 (Network)
```

### Diagnostic Commands

```bash
# View training stage from logs
grep "Training Mode" training_results/run0001/training.log

# Check spawn parameter usage
grep "Spawn params" training_results/run0001/training.log | tail -20

# Monitor PPO health metrics
grep "PPO Health" training_results/run0001/training.log | tail -10

# Compare Stage 1 vs Stage 2 rewards
grep "Episode.*R=" training_results/run0001/training.log | awk '{print $3}' | tail -100
grep "Episode.*R=" training_results/run0002/training.log | awk '{print $3}' | tail -100
```

---

## Advanced Usage

### Using Different Fixed Parameters for Different Substrates

You can create multiple config files:

**config_stage1_linear.yaml:**
```yaml
algorithm:
  two_stage_curriculum:
    stage: 1
    stage_1_fixed_spawn_params:
      gamma: 5.0
      alpha: 1.0
      noise: 0.05
      theta: 0.0

environment:
  substrate_type: linear
```

**config_stage1_exponential.yaml:**
```yaml
algorithm:
  two_stage_curriculum:
    stage: 1
    stage_1_fixed_spawn_params:
      gamma: 3.0
      alpha: 1.5
      noise: 0.15
      theta: 0.0

environment:
  substrate_type: exponential
```

### Gradual Transition (Optional Intermediate Stage)

For very challenging environments, you can add an intermediate stage:

1. **Stage 1**: Pure discrete with fixed parameters
2. **Stage 1.5**: Discrete with slightly noisy parameters
3. **Stage 2**: Full fine-tuning

```yaml
# Stage 1.5 config (manual implementation)
algorithm:
  two_stage_curriculum:
    stage: 1
    stage_1_fixed_spawn_params:
      gamma: 5.0
      alpha: 1.0
      noise: 0.2  # Increased noise to add variability
      theta: 0.0
```

Run Stage 1.5 for 200-300 episodes, then proceed to Stage 2.

### Freezing the Discrete Policy During Stage 2

If you want to ensure the discrete policy doesn't change during fine-tuning, you can modify the code to freeze those parameters. This requires custom implementation:

```python
# In train.py, add to Stage 2 setup:
if self.training_stage == 2:
    # Freeze discrete action head parameters
    for name, param in self.network.named_parameters():
        if 'discrete' in name or 'topology' in name:
            param.requires_grad = False
```

**Note:** This is not currently implemented by default but can be added if needed.

---

## Troubleshooting

### Stage 1 Issues

#### Problem: Agent not learning to migrate
**Symptoms:** Success rate stays near 0%, rewards stay very negative

**Solutions:**
1. Check fixed spawn parameters aren't too aggressive:
   ```yaml
   stage_1_fixed_spawn_params:
     gamma: 3.0  # Reduce from 5.0
   ```

2. Increase exploration:
   ```yaml
   entropy_regularization:
     entropy_coeff_start: 0.3  # Increase from 0.2
   ```

3. Reduce learning rate:
   ```yaml
   trainer:
     learning_rate: 0.0001  # Reduce from 0.0003
   ```

#### Problem: Agent learns but topology becomes unstable
**Symptoms:** Node counts fluctuate wildly (0 to 50+)

**Solutions:**
1. Enable stricter empty graph recovery:
   ```yaml
   environment:
     empty_graph_recovery_enabled: true
     empty_graph_recovery_nodes: 5
   ```

2. Adjust reward penalties:
   ```yaml
   rewards:
     boundary_penalty_weight: 3.0  # Increase to discourage risky moves
   ```

### Stage 2 Issues

#### Problem: Performance degrades after switching to Stage 2
**Symptoms:** Success rate drops, rewards decrease

**Solutions:**
1. **Reduce learning rate** (most important):
   ```yaml
   trainer:
     learning_rate: 1e-5  # Very conservative
   ```

2. **Reduce update epochs**:
   ```yaml
   trainer:
     update_epochs: 2  # From 4
   ```

3. **Increase KL threshold**:
   ```yaml
   algorithm:
     target_kl: 0.01  # From 0.03 (more conservative)
   ```

4. **Go back to Stage 1**: If degradation is severe, restart Stage 1 with different hyperparameters

#### Problem: No improvement in Stage 2
**Symptoms:** Rewards plateau, continuous parameters don't seem to matter

**Solutions:**
1. **Verify Stage 2 is actually running**:
   ```bash
   grep "Training Mode" training_results/run0002/training.log
   # Should show "Stage 2 (Fine-tuning Continuous Actions)"
   ```

2. **Check continuous loss is non-zero**:
   ```bash
   grep "continuous_loss" training_results/run0002/training.log
   ```

3. **Increase learning rate slightly**:
   ```yaml
   trainer:
     learning_rate: 1e-4  # From 5e-5
   ```

4. **Verify you loaded the Stage 1 checkpoint**:
   ```bash
   head -20 training_results/run0002/training.log
   # Should show "Resuming from checkpoint..."
   ```

---

## Best Practices

### DO:
- ✓ Train Stage 1 until clearly successful (50%+ success rate)
- ✓ Use a much lower learning rate for Stage 2 (5-10x smaller)
- ✓ Reset optimizer state when transitioning to Stage 2
- ✓ Monitor both stages carefully with tensorboard or logs
- ✓ Save multiple checkpoints from Stage 1 (try different ones for Stage 2)
- ✓ Document which fixed parameters you used

### DON'T:
- ✗ Rush Stage 1 (it's the foundation)
- ✗ Use the same learning rate for both stages
- ✗ Skip monitoring - watch for performance degradation
- ✗ Expect huge improvements in Stage 2 (5-15% is good)
- ✗ Give up on Stage 2 after 100 episodes (it takes time)

---

## Example Training Session

Here's a complete example from start to finish:

```bash
# ===== STAGE 1: DISCRETE TRAINING =====
# Edit config.yaml to set stage: 1
vim config.yaml

# Run Stage 1
python train.py --config config.yaml

# Wait for training (monitor in another terminal)
tail -f training_results/run0001/training.log

# After ~800 episodes, success rate hits 60%, stop training (Ctrl+C)
# Best model: training_results/run0001/best_model_episode_752.pt

# ===== STAGE 2: FINE-TUNING =====
# Edit config.yaml:
# - Set stage: 2
# - Set resume checkpoint path
# - Reduce learning rate to 5e-5
vim config.yaml

# Run Stage 2
python train.py --config config.yaml

# Monitor fine-tuning
tail -f training_results/run0002/training.log

# After ~500 episodes, rewards improved 10%, success rate at 68%
# Best model: training_results/run0002/best_model_episode_456.pt

# ===== EVALUATION =====
python deploy.py --model training_results/run0002/best_model_episode_456.pt --episodes 100
```

---

## Related Documentation

- [CLI_USAGE_GUIDE.md](./CLI_USAGE_GUIDE.md) - Command-line training options
- [RESUME_TRAINING_GUIDE.md](./RESUME_TRAINING_GUIDE.md) - Checkpoint loading details
- [PPO_METRICS_GUIDE.md](./PPO_METRICS_GUIDE.md) - Monitoring PPO health
- [DEFAULT_CONFIGURATION.md](./DEFAULT_CONFIGURATION.md) - Full config reference

---

## FAQ

**Q: Can I skip Stage 1 and train directly with Stage 2?**
A: Yes, but it's much harder and often less successful. The full hybrid action space is very large. Stage 1 provides structure.

**Q: How long does each stage take?**
A: Stage 1: 500-2000 episodes (1-4 hours). Stage 2: 500-1000 episodes (1-2 hours). Depends on hardware and environment complexity.

**Q: What if my Stage 1 agent never succeeds?**
A: Check your fixed spawn parameters - they might be too aggressive. Also verify your reward structure is providing clear signals. Try training on a simpler substrate first.

**Q: Can I run multiple Stage 2 experiments from the same Stage 1 checkpoint?**
A: Absolutely! This is recommended. Try different learning rates, update frequencies, etc.

**Q: Is Stage 2 always better than Stage 1?**
A: No. Sometimes the fixed parameters are already near-optimal, or the continuous action space is too large to fine-tune effectively. Monitor carefully.

**Q: Can I go back to Stage 1 after Stage 2?**
A: Not recommended. The continuous policy will have been trained. If Stage 2 is failing, restart from a different Stage 1 checkpoint with different Stage 2 hyperparameters.

---

## Summary

The two-stage training curriculum is a powerful technique for training your durotaxis agent:

1. **Stage 1** builds a strong foundation with discrete actions only
2. **Stage 2** fine-tunes continuous parameters on top of that foundation
3. Use lower learning rates and careful monitoring in Stage 2
4. Expect 5-15% improvement from fine-tuning, not miracles

Good luck with your training! 🚀
