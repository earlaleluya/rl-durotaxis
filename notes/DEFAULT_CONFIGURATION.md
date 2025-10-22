# Default Configuration Summary

## Current Default Configuration (config.yaml)

When you run `python train.py` without any modifications, here is your default configuration:

### Architecture Configuration

```yaml
actor_critic:
  pretrained_weights: 'imagenet'        # ✅ ImageNet pre-trained ResNet18
  
  wsa:
    enabled: false                      # ❌ WSA disabled (using standard Actor)
  
  simplicial_embedding:
    enabled: true                       # ✅ SEM enabled
```

**Your default is: Configuration (c1) - ImageNet + No WSA + SEM**

---

## Training Parameters

```yaml
trainer:
  total_episodes: 1000                  # Total training episodes
  max_steps: 1000                       # Max steps per episode
  learning_rate: 0.0001                 # Learning rate (1e-4)
  save_dir: "./training_results"        # Save directory
  
  log_every: 50                         # Log progress every 50 episodes
  progress_print_every: 5               # Print every 5 episodes
  checkpoint_every: null                # No automatic checkpoints (disabled)
  
  rollout_batch_size: 10                # Episodes per batch
  update_epochs: 4                      # Update epochs per batch
  minibatch_size: 64                    # Minibatch size for updates
```

---

## Environment Parameters

```yaml
environment:
  substrate_size: [600, 400]            # Large substrate (migration task)
  substrate_type: 'linear'              # Linear substrate (NOT random)
  
  substrate_params:
    m: 0.01                             # Slope
    b: 1.0                              # Intercept
  
  init_num_nodes: 5                     # Start with 5 nodes
  max_critical_nodes: 75                # Max 75 nodes allowed
  max_steps: 1000                       # 1000 steps per episode
  
  delta_time: 3                         # Simulation delta
  delta_intensity: 2.50                 # Intensity factor
```

---

## Encoder Configuration

```yaml
encoder:
  out_dim: 64                           # Output dimension
  num_layers: 4                         # Transformer layers
```

---

## Algorithm Parameters

```yaml
algorithm:
  gamma: 0.99                           # Discount factor
  gae_lambda: 0.95                      # GAE lambda
  ppo_epochs: 4                         # PPO epochs
  clip_epsilon: 0.1                     # PPO clip
  value_loss_coeff: 0.5                 # Value loss coefficient
  max_grad_norm: 0.5                    # Gradient clipping
```

---

## System Configuration

```yaml
system:
  device: 'auto'                        # Auto-detect CUDA/CPU
  num_workers: 1                        # Single worker
  seed: null                            # No fixed seed (random)
```

---

## Command-Line Overrides

### ⚠️ Current Status: Limited Command-Line Support

The original `train.py` does NOT have built-in command-line argument parsing.

However, you have **two options**:

### Option 1: Use `train_cli.py` (NEW - Just Created)

```bash
# Use the new CLI wrapper
python train_cli.py --help

# Examples:
python train_cli.py --total-episodes 2000 --learning-rate 0.0005
python train_cli.py --seed 42 --experiment my_test
python train_cli.py --pretrained-weights random
```

**Note**: Architecture parameters (--pretrained-weights, --wsa-enabled, --no-sem) 
require editing config.yaml first, as the network is initialized during trainer creation.

### Option 2: Edit config.yaml (Recommended for Ablation Studies)

For ablation studies, it's cleaner to create separate config files:

```bash
# Create config files for each ablation
cp config.yaml config_a1_imagenet_nowsa_nosem.yaml
cp config.yaml config_b1_imagenet_wsa_nosem.yaml
# ... etc

# Edit each file
# Then run:
python train.py  # Uses config.yaml by default
```

---

## Quick Configuration Changes

### To change to Baseline (a1): ImageNet + No WSA + No SEM

```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: false
  simplicial_embedding:
    enabled: false        # ← Change this to false
```

### To change to WSA (b1): ImageNet + WSA + No SEM

```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: true         # ← Change this to true
  simplicial_embedding:
    enabled: false        # ← Change this to false
```

### To change to Full Stack (d1): ImageNet + WSA + SEM

```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: true         # ← Change this to true
  simplicial_embedding:
    enabled: true
```

### To change pretrained weights to random (any config)

```yaml
actor_critic:
  pretrained_weights: 'random'   # ← Change from 'imagenet' to 'random'
```

---

## Programmatic Overrides (in Python)

If you want to modify `train.py` directly, you can pass overrides:

```python
# In train.py main() function
trainer = DurotaxisTrainer(
    config_path="config.yaml",
    total_episodes=2000,           # Override total episodes
    learning_rate=0.0005,          # Override learning rate
    substrate_type='exponential',  # Override substrate type
    log_every=25,                  # Override logging frequency
)
```

**Limitation**: This works for top-level trainer config, but NOT for nested configs like:
- `actor_critic.pretrained_weights`
- `actor_critic.wsa.enabled`
- `actor_critic.simplicial_embedding.enabled`

For those, you must edit config.yaml.

---

## Summary

### Your Current Default When Running `python train.py`:

| Parameter | Value | Group |
|-----------|-------|-------|
| **Pretrained Weights** | `imagenet` | ✅ |
| **WSA** | `disabled` (false) | ❌ |
| **SEM** | `enabled` (true) | ✅ |
| **Configuration** | **(c1)** | ImageNet + No WSA + SEM |

### Command-Line Override Status:

- ✅ **Working**: Training parameters (episodes, lr, substrate_type, etc.) via `DurotaxisTrainer` kwargs
- ⚠️ **Limited**: Architecture parameters require editing config.yaml
- 🆕 **New CLI**: `train_cli.py` provides better argument parsing with warnings

### Recommended Workflow for Ablation Studies:

1. **Create config files** for each ablation configuration
2. **Use separate config files** instead of command-line overrides
3. **Run with**: `python train.py` (after editing config.yaml)
4. **Track experiments** by saving different config.yaml versions

### Alternative: Use train_cli.py

```bash
# After editing config.yaml to set architecture
python train_cli.py --seed 42 --experiment my_test --total-episodes 500
```

---

**Last Updated**: December 2024
