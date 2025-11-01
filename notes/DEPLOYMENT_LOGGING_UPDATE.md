# Deployment Logging and Visualization Update

## Summary

The `deploy.py` script has been updated with comprehensive logging functionalities that generate multiple JSON files for detailed analysis. The `plotter.py` script can be used to visualize these results.

## Changes to deploy.py

### 1. Substrate Size Configuration
- Added `--substrate_width` and `--substrate_height` arguments (default: 600x400 to match training)
- Substrate size is now fully configurable from command line
- Size information included in all output files

### 2. Enhanced Node/Edge Tracking
- Node and edge counts are now tracked at every step
- Added to `reward_components` dictionary for each step
- Enables detailed topology evolution analysis

### 3. Comprehensive JSON Output Files

When `--save_results` is enabled (default), the following files are generated:

#### a) `evaluation_[config]_[timestamp].json`
Main evaluation summary with:
- Substrate configuration (type, size, m, b)
- Evaluation parameters (episodes, steps, deterministic mode)
- Overall statistics (mean/std/min/max rewards, episode lengths, success rate)
- Complete episode details with per-step data

#### b) `detailed_nodes_all_episodes_[timestamp].json`
Node count evolution per episode:
```json
[
  {
    "episode": 0,
    "nodes_per_step": [
      {"step": 0, "nodes": 3},
      {"step": 1, "nodes": 5},
      ...
    ]
  },
  ...
]
```

#### c) `spawn_parameters_stats_[timestamp].json`
Spawn parameter statistics per episode:
```json
[
  {
    "episode": 0,
    "parameters": {
      "gamma": {"mean": 6.95, "std": 1.62, "min": 4.2, "max": 9.1},
      "alpha": {"mean": 0.45, "std": 0.12, "min": 0.2, "max": 0.7},
      "noise": {"mean": 0.05, "std": 0.02, "min": 0.01, "max": 0.09},
      "theta": {"mean": 1.57, "std": 0.3, "min": 1.0, "max": 2.0}
    }
  },
  ...
]
```

#### d) `reward_components_stats_[timestamp].json`
Reward component statistics per episode:
```json
[
  {
    "episode": 0,
    "reward_components": {
      "total_reward": {"mean": -12.1, "std": 14.3, "min": -45.2, "max": 15.6, "sum": -145.2},
      "graph_reward": {"mean": 5.2, "std": 2.1, ...},
      "spawn_reward": {...},
      "delete_reward": {...},
      "edge_reward": {...},
      "total_node_reward": {...}
    }
  },
  ...
]
```

#### e) `training_metrics_[timestamp].json`
Episode-level summary metrics:
```json
[
  {
    "episode": 0,
    "total_reward": -145.2,
    "episode_length": 12,
    "final_nodes": 45,
    "final_edges": 120,
    "terminated": true,
    "component_rewards": {
      "total_reward": -145.2,
      "graph_reward": 62.4,
      ...
    }
  },
  ...
]
```

#### f) `loss_metrics_[timestamp].json`
Placeholder for consistency with training format (no actual loss in evaluation):
```json
[
  {
    "episode": 0,
    "loss": 0.0,
    "smoothed_loss": 0.0,
    "note": "Evaluation mode - no loss computed"
  },
  ...
]
```

## Using plotter.py with Evaluation Results

### Current Plotter Capabilities

The existing `plotter.py` can visualize evaluation results:

```bash
# Find the latest evaluation files in a directory
cd training_results/run0018

# Plot spawn parameters from evaluation
python ../../plotter.py --input spawn_parameters_stats_20251101_143022.json --show

# Plot reward components
python ../../plotter.py --input spawn_parameters_stats_20251101_143022.json --rewards --show

# Generate all plots
python ../../plotter.py --input . --combined --rewards --show
```

### Recommended Workflow

1. **Run Evaluation:**
```bash
python deploy.py --model_path ./training_results/run0018/best_model_batch2.pt \
                 --substrate_type linear --m 0.05 --b 1.0 \
                 --substrate_width 200 --substrate_height 100 \
                 --deterministic --max_episodes 10 --max_steps 1000 \
                 --max_critical_nodes 50 --threshold_critical_nodes 200 \
                 --no_viz
```

2. **Generate Visualizations:**
```bash
# Navigate to the model directory
cd training_results/run0018

# Generate all available plots from the timestamped files
python ../../plotter.py --input . --combined --rewards --show
```

3. **Analyze Results:**
The plotter will automatically find the most recent files with timestamps and generate:
- `spawn_parameters_evolution_*.png` - Individual parameter trends
- `spawn_parameters_combined_*.png` - Normalized overlay comparison
- `reward_components_*.png` - Reward component evolution

## Example Usage

### Basic Evaluation with Custom Substrate
```bash
python deploy.py \
    --model_path ./training_results/run0018/best_model_batch2.pt \
    --substrate_type linear \
    --substrate_width 400 \
    --substrate_height 300 \
    --m 0.05 \
    --b 1.0 \
    --deterministic \
    --max_episodes 20 \
    --max_steps 500 \
    --no_viz
```

### Evaluation with Visualization
```bash
python deploy.py \
    --model_path ./training_results/run0018/best_model_batch2.pt \
    --substrate_type exponential \
    --substrate_width 600 \
    --substrate_height 400 \
    --m 0.02 \
    --b 1.0 \
    --max_episodes 5 \
    --max_steps 1000
```

### Quick Analysis
```bash
# After running evaluation, generate plots
cd training_results/run0018
python ../../plotter.py --input . --combined --rewards --loss --show
```

## File Naming Convention

All output files include timestamps in format `YYYYMMDD_HHMMSS` to:
- Prevent overwriting previous evaluations
- Enable temporal tracking of multiple runs
- Facilitate comparison across different configurations

Example filenames:
- `evaluation_linear_m0.05_b1.0_20251101_143022.json`
- `spawn_parameters_stats_20251101_143022.json`
- `reward_components_stats_20251101_143022.json`

## Benefits

1. **Comprehensive Tracking**: Every aspect of evaluation is logged
2. **Reproducibility**: All parameters and configurations saved
3. **Analysis-Ready**: JSON format compatible with plotting tools
4. **Timestamped**: Multiple evaluations don't interfere
5. **Consistent Format**: Matches training output structure
6. **Plotter Compatible**: Works seamlessly with existing visualization tools

## Notes

- All files are saved in the same directory as the model file
- Verbose output can be disabled with `--no_viz` flag
- The `loss_metrics.json` file is a placeholder in evaluation mode
- Node/edge tracking adds minimal computational overhead
- Spawn parameter extraction assumes delete ratio architecture (5D action space)
