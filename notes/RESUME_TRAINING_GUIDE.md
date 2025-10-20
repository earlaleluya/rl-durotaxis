# Resume Training Guide

This guide explains how to resume training from a previously saved checkpoint.

## Configuration

Add the following section to your `config.yaml`:

```yaml
resume_training:
  enabled: false                    # Set to true to enable resume
  checkpoint_path: null             # Path to checkpoint file (required when enabled)
  resume_from_best: false          # If true, loads best_model*.pt instead
  reset_optimizer: false           # If true, resets optimizer state (fresh optimization)
  reset_episode_count: false       # If true, starts from episode 0 (but keeps weights)
```

## Usage Examples

### 1. Resume from Last Checkpoint

Resume training from a specific checkpoint file:

```yaml
resume_training:
  enabled: true
  checkpoint_path: "./training_results/run0001/checkpoint_batch10.pt"
  resume_from_best: false
  reset_optimizer: false
  reset_episode_count: false
```

This will:
- ✅ Load network weights
- ✅ Restore optimizer state
- ✅ Continue from the saved episode count
- ✅ Preserve training history (rewards, losses, best reward)

### 2. Resume from Best Model

Automatically load the best performing model:

```yaml
resume_training:
  enabled: true
  checkpoint_path: "./training_results/run0001/"  # Directory path
  resume_from_best: true
  reset_optimizer: false
  reset_episode_count: false
```

The system will find the most recent `best_model_batch*.pt` file in the run directory.

### 3. Fine-tune with Fresh Optimizer

Load trained weights but reset optimizer (useful for fine-tuning):

```yaml
resume_training:
  enabled: true
  checkpoint_path: "./training_results/run0001/best_model_batch5.pt"
  resume_from_best: false
  reset_optimizer: true          # Start with fresh optimizer
  reset_episode_count: true      # Start counting from episode 0
```

This is useful when:
- You want to fine-tune a model with different learning rate
- Previous training was stuck in local optimum
- You want to adapt to different environment parameters

### 4. Transfer Learning

Load weights but start fresh training:

```yaml
resume_training:
  enabled: true
  checkpoint_path: "./training_results/run0001/final_model.pt"
  resume_from_best: false
  reset_optimizer: true
  reset_episode_count: true
```

## Checkpoint Contents

Each checkpoint file contains:

```python
{
    'network_state_dict': ...,      # Network weights
    'optimizer_state_dict': ...,    # Optimizer state (momentum, etc.)
    'scheduler_state_dict': ...,    # Learning rate scheduler state
    'episode_rewards': {...},       # Training reward history
    'losses': {...},                # Loss history
    'best_reward': float,           # Best reward achieved
    'component_weights': {...},     # Reward component weights
    'run_number': int,              # Run number
    'episode_count': int,           # Episodes completed
    'smoothed_rewards': [...],      # Moving average rewards
    'smoothed_losses': [...]        # Moving average losses
}
```

## Training Output

When resume is enabled, you'll see:

```
🔄 Loading checkpoint for resume: ./training_results/run0001/checkpoint_batch10.pt
   ✅ Loaded network weights
   ✅ Loaded optimizer state
   ✅ Loaded scheduler state
   ✅ Resuming from episode 100
   ✅ Loaded 100 episode reward history
   ✅ Loaded loss history
   ✅ Best reward so far: -15.42
   ✅ Loaded component weights
   ✅ Loaded smoothed rewards
   ✅ Loaded smoothed losses
✅ Resume checkpoint loaded successfully!
🏋️ Starting training for 1000 episodes (Run #0002)
🔄 Resuming from episode 100
```

## Checkpoint Saving

During training, checkpoints are automatically saved:

1. **Best Model**: `best_model_batch{N}.pt` - Saved when a new best reward is achieved
2. **Periodic**: `checkpoint_batch{N}.pt` - Saved at intervals (see `checkpoint_every` config)
3. **Final**: `final_model.pt` - Saved at end of training

All checkpoints include full training state for resume.

## Tips

1. **Backup checkpoints**: Always keep copies of important checkpoints before resuming
2. **Episode count**: The total_episodes in config is the target, not additional episodes
3. **Run directory**: Resumed training uses a new run directory to avoid overwriting
4. **Best reward tracking**: The best_reward is preserved across resume sessions
5. **Relative paths**: checkpoint_path can be relative to current working directory

## Troubleshooting

### Checkpoint not found
```
❌ Checkpoint file not found: ./path/to/checkpoint.pt
```
- Check the path is correct
- Use absolute path if relative path doesn't work
- Verify file exists: `ls ./training_results/run0001/*.pt`

### No checkpoint_path specified
```
❌ Resume training enabled but no checkpoint_path specified
```
- Add `checkpoint_path` to resume_training config
- Or enable `resume_from_best: true` to auto-find best model

### Version mismatch
If network architecture changed between training sessions, you may see errors loading state dict. In this case:
- Use the same code version that created the checkpoint
- Or manually modify the checkpoint to match new architecture
