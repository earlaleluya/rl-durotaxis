# Configuration and Command-Line Usage Guide

## Your Default Configuration

When you run `python train.py` without any modifications:

**Configuration: c1 - SEM Enhancement**
- ✅ Pretrained Weights: `imagenet`
- ❌ WSA: `disabled`
- ✅ SEM: `enabled`

This is **NOT** the baseline configuration for ablation studies!

### To Run Baseline (a1) Instead:

Edit `config.yaml`:
```yaml
actor_critic:
  simplicial_embedding:
    enabled: false   # ← Change from true to false
```

---

## Command-Line Override Support

### Current Status: ⚠️ Partially Supported

The original `train.py` does NOT have built-in argument parsing, but the `DurotaxisTrainer` class accepts keyword arguments for **some** overrides.

### What Works ✅

These parameters CAN be overridden programmatically (edit `train.py` main function):

```python
# In train.py, modify the main() function:
trainer = DurotaxisTrainer(
    config_path="config.yaml",
    
    # Training parameters - THESE WORK
    total_episodes=2000,
    max_steps=500,
    learning_rate=0.0005,
    
    # Logging parameters - THESE WORK
    log_every=25,
    checkpoint_every=100,
    save_dir="./my_experiment",
    
    # Environment parameters - THESE WORK  
    substrate_type='exponential',
)
```

### What Doesn't Work ❌

These parameters CANNOT be overridden via kwargs (must edit config.yaml):

- ❌ `actor_critic.pretrained_weights` (imagenet vs random)
- ❌ `actor_critic.wsa.enabled` (WSA on/off)
- ❌ `actor_critic.simplicial_embedding.enabled` (SEM on/off)
- ❌ `environment.init_num_nodes` (initial nodes)
- ❌ `environment.max_critical_nodes` (max nodes)

**Reason**: These are read from nested config sections and used during network initialization, before the trainer can apply overrides.

---

## Solutions for Command-Line Usage

### Option 1: Use `train_cli.py` (NEW - Recommended for Quick Tests)

I created a new CLI wrapper that handles arguments:

```bash
# Show help
python train_cli.py --help

# Run with overrides
python train_cli.py --total-episodes 2000 --learning-rate 0.0005 --seed 42

# Architecture changes (requires confirmation)
python train_cli.py --pretrained-weights random --experiment test_random
```

**Note**: Architecture parameters require editing config.yaml first.

### Option 2: Create Config Files (Recommended for Ablation Studies)

Best practice: Create separate config files for each ablation:

```bash
# Create configs
cp config.yaml config_a1_baseline.yaml
cp config.yaml config_b1_wsa.yaml
cp config.yaml config_c1_sem.yaml
cp config.yaml config_d1_fullstack.yaml

# Edit each file for the specific ablation
# Then run (edit train.py to use specific config):
python train.py
```

### Option 3: Edit train.py Directly

Simplest for quick tests:

```python
# In train.py main() function, change:
trainer = DurotaxisTrainer(
    config_path="config.yaml",
    total_episodes=500,        # Your override here
    learning_rate=0.0003,      # Your override here
)
```

---

## Quick Reference: How to Change Configurations

### Show Current Default

```bash
python show_default_config.py
```

### Change to Baseline (a1)

**Edit config.yaml:**
```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: false
  simplicial_embedding:
    enabled: false    # ← Change this
```

### Change to WSA (b1)

**Edit config.yaml:**
```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: true     # ← Change this
  simplicial_embedding:
    enabled: false    # ← Change this
```

### Change to Full Stack (d1)

**Edit config.yaml:**
```yaml
actor_critic:
  pretrained_weights: 'imagenet'
  wsa:
    enabled: true     # ← Change this
  simplicial_embedding:
    enabled: true     # Already true by default
```

### Change Pretrained Weights to Random

**Edit config.yaml:**
```yaml
actor_critic:
  pretrained_weights: 'random'   # ← Change from 'imagenet'
```

---

## Recommended Ablation Study Workflow

### Step 1: Create Config Files

```bash
# Configuration a1: imagenet + No WSA + No SEM
cp config.yaml configs/config_a1.yaml
# Edit: pretrained_weights='imagenet', wsa.enabled=false, sem.enabled=false

# Configuration b1: imagenet + WSA + No SEM  
cp config.yaml configs/config_b1.yaml
# Edit: pretrained_weights='imagenet', wsa.enabled=true, sem.enabled=false

# Configuration c1: imagenet + No WSA + SEM
cp config.yaml configs/config_c1.yaml
# Edit: pretrained_weights='imagenet', wsa.enabled=false, sem.enabled=true

# Configuration d1: imagenet + WSA + SEM
cp config.yaml configs/config_d1.yaml
# Edit: pretrained_weights='imagenet', wsa.enabled=true, sem.enabled=true

# Repeat for 'random' variants (a2, b2, c2, d2)
```

### Step 2: Run Each Configuration

```bash
# For each config, run with multiple seeds
for config in configs/config_*.yaml; do
    for seed in 1 2 3 4 5; do
        # Copy config to main location
        cp $config config.yaml
        
        # Run training (modify train.py to accept seed)
        python train.py
        
        # Or use train_cli.py
        python train_cli.py --config $config --seed $seed --experiment $(basename $config .yaml)_seed${seed}
    done
done
```

---

## Summary

### Your Default Configuration (c1)
- Pretrained: ImageNet ✅
- WSA: Disabled ❌
- SEM: Enabled ✅

### Command-Line Override Status
- ✅ **Working**: Basic training parameters (episodes, lr, substrate_type)
- ❌ **Not Working**: Architecture parameters (pretrained_weights, wsa, sem)
- 🆕 **New**: `train_cli.py` provides better CLI with warnings

### Best Practice for Ablation Studies
1. Create separate config YAML files for each ablation
2. Edit architecture parameters in config files (not command-line)
3. Use command-line/programmatic overrides for training parameters only
4. Run `python show_default_config.py` to verify current settings

### Quick Commands

```bash
# Check current configuration
python show_default_config.py

# Test all ablation configs
python test_ablation_configurations.py

# Run training (default config)
python train.py

# Run with CLI (after editing config.yaml)
python train_cli.py --seed 42 --experiment my_test
```

---

**Files Created:**
- `train_cli.py` - CLI wrapper with argument parsing
- `show_default_config.py` - Display current configuration
- `notes/DEFAULT_CONFIGURATION.md` - Detailed configuration guide
