# Visualization Guide for Delete Ratio Architecture

## Overview

The `test_visualize_rightward.py` script provides real-time visualization of the agent's rightward movement behavior using the delete ratio architecture.

## Features

### 4-Panel Visualization

1. **Graph Topology (Top-Left)**
   - Shows current node positions (blue dots)
   - Edge connections (gray lines)
   - Centroid position (red star)
   - Goal region (green circle)
   - Real-time node count and centroid position

2. **Stiffness Substrate (Top-Right)**
   - Substrate stiffness profile along x-axis
   - Goal position marker
   - Gradient visualization

3. **Centroid Trajectory (Bottom-Left)**
   - Centroid x-position over time
   - Goal line reference
   - Progress percentage

4. **Action Values (Bottom-Right)**
   - Delete ratio (fraction of nodes to delete)
   - Gamma (stiffness sensing parameter)
   - Alpha (directional influence)
   - Noise (exploration parameter)
   - Theta (preferred angle, normalized)

## Usage

### Basic Usage (Random Agent)

```bash
python tools/test_visualize_rightward.py
```

This runs with random actions to demonstrate the environment dynamics.

### With Trained Model

```bash
python tools/test_visualize_rightward.py --model_path training_results/run0007/best_model.pt
```

### Custom Episode Length

```bash
python tools/test_visualize_rightward.py --max_steps 500
```

### Different Substrate Types

```bash
# Linear gradient (default)
python tools/test_visualize_rightward.py --substrate_type linear --m 0.05 --b 1.0

# Sigmoid profile
python tools/test_visualize_rightward.py --substrate_type sigmoid --m 0.1 --b 2.0

# Exponential profile
python tools/test_visualize_rightward.py --substrate_type exponential --m 0.08

# Step function
python tools/test_visualize_rightward.py --substrate_type step --m 0.3
```

### Custom Initial Conditions

```bash
python tools/test_visualize_rightward.py --initial_nodes 30 --max_steps 300
```

### Complete Example

```bash
python tools/test_visualize_rightward.py \
    --model_path training_results/run0007/best_model.pt \
    --substrate_type linear \
    --m 0.05 \
    --b 1.0 \
    --initial_nodes 20 \
    --max_steps 200
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Path to trained model (optional) |
| `--config_path` | str | config.yaml | Path to configuration file |
| `--max_steps` | int | 200 | Maximum steps per episode |
| `--substrate_type` | str | linear | Stiffness profile type |
| `--m` | float | 0.05 | Substrate gradient parameter |
| `--b` | float | 1.0 | Substrate offset parameter |
| `--initial_nodes` | int | 20 | Initial number of nodes |

## Understanding the Visualization

### Delete Ratio Architecture

The visualization shows how the agent uses a **single global action** to control the graph:

**Action Space: [delete_ratio, gamma, alpha, noise, theta]**

- **delete_ratio** (0-1): Fraction of leftmost nodes to delete
  - 0.0 = delete nothing
  - 0.2 = delete leftmost 20%
  - 1.0 = delete all nodes
  
- **gamma** (0-1): Stiffness sensing range parameter
  - Higher = stronger preference for high stiffness regions
  
- **alpha** (0-1): Directional influence strength
  - Controls how much stiffness gradient affects movement direction
  
- **noise** (0-1): Exploration/randomness parameter
  - Higher = more random spawning
  
- **theta** (0-2π): Preferred spawning angle
  - 0 = rightward
  - π/2 = upward
  - π = leftward

### Node Deletion Strategy

1. **Sort** all nodes by x-position (left to right)
2. **Delete** leftmost `delete_ratio` fraction
3. **Spawn** from remaining nodes using global parameters

### Rightward Movement Indicators

**Success Indicators:**
- ✅ Centroid trajectory shows steady rightward movement
- ✅ Centroid reaches goal x-position (default: 600)
- ✅ Progress percentage reaches 100%

**Strategy Indicators:**
- Delete ratio ~0.1-0.3: Conservative pruning
- Delete ratio ~0.4-0.6: Aggressive pruning
- Theta ~0: Strong rightward bias
- Gamma high: Prioritizing high stiffness

## Typical Behavior Patterns

### Random Agent
- **Erratic trajectory**: Centroid wanders
- **Inconsistent deletion**: Delete ratio varies widely
- **No directional bias**: Movement in all directions
- **Low progress**: Rarely reaches goal

### Trained Agent
- **Smooth trajectory**: Steady rightward movement
- **Consistent deletion**: Delete ratio stabilizes (e.g., 0.2-0.3)
- **Rightward bias**: Theta near 0, spawning to the right
- **High progress**: Consistently reaches goal

## Interpreting the Plots

### Graph Topology
- **Node density**: Healthy agents maintain 15-40 nodes
- **Spatial distribution**: Nodes should cluster near high stiffness
- **Centroid position**: Should move steadily rightward

### Centroid Trajectory
- **Slope**: Positive slope = rightward progress
- **Smoothness**: Smooth curve = stable policy
- **Plateau**: Flattening = stagnation (bad)
- **Progress %**: >100% = reached goal

### Action Values
- **Delete ratio stability**: Should stabilize after initial exploration
- **Gamma consistency**: Typically high (0.7-0.9) for trained agents
- **Theta bias**: Should stay near 0 for rightward movement
- **Alpha & Noise**: Balance exploration vs exploitation

## Troubleshooting

### Visualization doesn't appear
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Try setting backend explicitly
export MPLBACKEND=TkAgg
python tools/test_visualize_rightward.py
```

### Slow visualization
```bash
# Reduce max_steps for faster execution
python tools/test_visualize_rightward.py --max_steps 100
```

### Model loading errors
```bash
# Verify model path exists
ls -lh training_results/run0007/best_model.pt

# Check model compatibility
python tools/test_visualize_rightward.py --config_path config.yaml
```

### Agent doesn't move rightward
- **Random agent**: Expected behavior without training
- **Trained agent**: Check model quality, may need more training
- **Wrong substrate**: Ensure substrate has rightward gradient

## Output Information

The script prints:
```
🖥️  Using device: cpu
📦 Loading model from: training_results/run0007/best_model.pt
✅ Model loaded successfully!

🌍 Creating environment...
   Substrate: linear (m=0.05, b=1.0)
   Initial nodes: 20

============================================================
🎮 Starting Visualization Episode
============================================================
Step  10 | Reward:   12.34 | Delete Ratio: 0.234 | Nodes: 22
Step  20 | Reward:   23.45 | Delete Ratio: 0.198 | Nodes: 25
...

============================================================
✅ Episode Complete!
   Total Steps: 147
   Total Reward: 2341.23
   Final Nodes: 28
   Final Centroid X: 612.3
   Goal X: 600.0
   Progress: 102.1%
   🎉 SUCCESS! Reached goal!
============================================================
```

## Integration with Training

### Quick Model Testing
After training, quickly verify learned behavior:
```bash
# Test latest model
python tools/test_visualize_rightward.py \
    --model_path training_results/run0007/best_model.pt

# Compare with random baseline
python tools/test_visualize_rightward.py
```

### Debugging Training Issues
If training rewards are low:
1. Visualize the trained model
2. Check if centroid moves rightward
3. Examine delete ratio values
4. Verify theta bias toward 0

### Model Comparison
Compare different checkpoints:
```bash
# Early training
python tools/test_visualize_rightward.py \
    --model_path training_results/run0007/best_model_batch1.pt

# Late training
python tools/test_visualize_rightward.py \
    --model_path training_results/run0007/best_model_batch11.pt
```

## Advanced Usage

### Save Visualization Data
Modify the script to export trajectory data:
```python
# After episode completes
import json
data = {
    'centroid_history': visualizer.centroid_history,
    'action_history': visualizer.action_history,
    'total_reward': total_reward
}
with open('trajectory_data.json', 'w') as f:
    json.dump(data, f)
```

### Create Video Recording
```bash
# Install ffmpeg
sudo apt-get install ffmpeg

# Modify script to save frames and create video
# (Implementation details left as exercise)
```

## See Also

- `deploy.py` - Full deployment script with multiple episodes
- `train.py` - Training script
- `notes/CLI_DEPLOYMENT_UPDATE.md` - Deployment documentation
- `notes/REWARD_MODE_CLI_GUIDE.md` - Experimental modes
