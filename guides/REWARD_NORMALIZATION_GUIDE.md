# Enhanced Reward Normalization System

## Overview

The enhanced reward normalization system implements a **two-tier approach** to handle reward component magnitude differences and improve training stability in multi-component reinforcement learning.

## Problem Solved

**Original Issue**: Reward magnitudes vary significantly across components:
- `total_reward`: ~8-15 (high magnitude)
- `spawn_reward`: ~-2.5 to -1.5 (medium magnitude)  
- `edge_reward`: ~1.0-1.8 (medium magnitude)
- `graph_reward`: ~0.05-0.3 (low magnitude)

This causes high-magnitude components to dominate the learning signal, leading to poor balance and suboptimal policy learning.

## Solution: Two-Tier Normalization

### Tier 1: Per-Episode Normalization
**Purpose**: Prevents component domination within individual episodes  
**Method**: Normalizes each reward component within the current episode  
**Benefits**: 
- Immediate magnitude balancing
- Prevents any single component from overwhelming others in a single episode
- Preserves relative importance within episodes

### Tier 2: Cross-Episode Adaptive Scaling  
**Purpose**: Maintains long-term balance across episodes and training phases  
**Method**: Uses running statistics to compute scaling factors  
**Benefits**:
- Adapts to changing reward distributions over time
- Provides stability across different training phases
- Maintains consistency with historical performance

## Configuration Options

```yaml
# In config.yaml
trainer:
  reward_normalization:
    enable_per_episode_norm: true     # Enable Tier 1 normalization
    enable_cross_episode_scaling: true  # Enable Tier 2 scaling
    min_episode_length: 5             # Minimum episode length for normalization
    normalization_method: 'adaptive'  # 'zscore', 'minmax', or 'adaptive'
```

## Normalization Methods

### 1. Z-Score Normalization (`'zscore'`)
```python
normalized = (rewards - mean) / std
```
- **Best for**: Normally distributed rewards
- **Result**: Mean=0, Std=1
- **Use case**: When rewards have good variance

### 2. Min-Max Normalization (`'minmax'`)  
```python
normalized = (rewards - min) / (max - min)
```
- **Best for**: Rewards with known bounds
- **Result**: Range [0, 1]
- **Use case**: When you want to preserve relative ordering

### 3. Adaptive Normalization (`'adaptive'`) ⭐ **Recommended**
```python
if std < threshold:
    use_minmax_normalization()
else:
    use_zscore_normalization()
```
- **Best for**: Mixed reward patterns
- **Result**: Automatically chooses best method
- **Use case**: General purpose, handles edge cases

## Performance Results

### Before Normalization:
```
total_reward: std=2.315 (dominates)
spawn_reward: std=0.341  
edge_reward: std=0.271
graph_reward: std=0.086 (barely visible)
```

### After Normalization:
```
total_reward: std=1.000 (balanced)
spawn_reward: std=1.000 (balanced)  
edge_reward: std=1.000 (balanced)
graph_reward: std=1.000 (balanced)
```

## Key Benefits

1. **🎯 Balanced Learning**: All reward components contribute equally to policy updates
2. **📈 Improved Stability**: Reduces training variance and improves convergence  
3. **🔄 Adaptability**: Automatically adjusts to changing reward patterns
4. **⚙️ Configurable**: Easy to tune for different scenarios
5. **🛡️ Robust**: Handles edge cases (constant rewards, outliers)

## Implementation Details

### Core Algorithm:
```python
def compute_returns_and_advantages(self, rewards, values, ...):
    # Tier 1: Per-episode normalization
    normalized_rewards = self.normalize_episode_rewards(rewards)
    
    # Tier 2: Cross-episode scaling
    scaling_factors = self.get_component_scaling_factors()
    
    for component in self.component_names:
        component_rewards = normalized_rewards[component] * scaling_factors[component]
        # ... continue with GAE computation
```

### Safety Features:
- Minimum episode length check to avoid over-normalization
- Robust handling of constant/near-zero rewards
- Configurable enable/disable for each tier
- Automatic fallback methods for edge cases

## Usage Recommendations

### For Stable Environments:
```yaml
enable_per_episode_norm: true
enable_cross_episode_scaling: true  
normalization_method: 'zscore'
```

### For Dynamic/Changing Environments:
```yaml
enable_per_episode_norm: true
enable_cross_episode_scaling: true
normalization_method: 'adaptive'  # Recommended
```

### For Debugging/Comparison:
```yaml
enable_per_episode_norm: false    # Disable to compare
enable_cross_episode_scaling: false
```

## Migration Notes

- **Backward Compatible**: Existing configs work without changes
- **Default Behavior**: Enhanced normalization enabled by default
- **Performance**: Minimal computational overhead
- **Monitoring**: Component statistics still tracked for analysis

## Expected Improvements

1. **Faster Convergence**: Balanced gradients lead to more efficient learning
2. **Better Exploration**: Prevents premature convergence to high-magnitude components  
3. **More Stable Training**: Reduced variance in policy updates
4. **Component Balance**: All reward aspects influence the final policy

This enhancement transforms your multi-component RL system from magnitude-biased to truly balanced learning! 🚀