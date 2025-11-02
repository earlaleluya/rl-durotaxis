# Device Management in RL-Durotaxis

This project is **device-agnostic** and automatically detects whether to use CPU or GPU. You can also explicitly control device selection.

## Automatic Device Selection

By default, the system will:
1. Use CUDA (GPU) if available via `torch.cuda.is_available()`
2. Fall back to CPU if CUDA is not available

No configuration needed—just run your scripts normally:

```bash
python train.py
python deploy.py --model_path ./training_results/run0001/best_model.pt
```

## Force CPU Mode

To explicitly force CPU execution (useful for debugging or running on machines with GPUs that you want to ignore):

### Option 1: Environment Variable

```bash
# Force CPU via environment variable
FORCE_CPU=1 python train.py
FORCE_CPU=true python deploy.py --model_path ./model.pt
```

### Option 2: DEVICE Environment Variable

```bash
# Explicitly set device
DEVICE=cpu python train.py
DEVICE=cuda python train.py      # Force CUDA
DEVICE=cuda:0 python train.py    # Specific GPU
```

### Option 3: Config File

Add to `config.yaml`:

```yaml
system:
  device: 'cpu'        # or 'cuda' or 'auto' (default)
```

## Device Utilities

The project provides a centralized `device.py` module with helpful utilities:

### `get_device()`

Returns the appropriate device based on environment variables and availability:

```python
from device import get_device

device = get_device()  # torch.device('cuda') or torch.device('cpu')
```

### `to_device(obj, device=None)`

Moves tensors, modules, or nested structures (dicts, lists) to a device:

```python
from device import to_device

# Move a single tensor
tensor = to_device(tensor)

# Move a module
model = to_device(model)

# Move nested structures
state_dict = {
    'features': torch.randn(10, 5),
    'labels': torch.randint(0, 2, (10,)),
    'nested': {
        'x': torch.ones(3, 3)
    }
}
state_dict = to_device(state_dict)  # All tensors moved to device
```

### `cpu_numpy(tensor)`

Safely converts PyTorch tensors to NumPy arrays on CPU (handles GPU tensors):

```python
from device import cpu_numpy

gpu_tensor = torch.randn(10, 5, device='cuda')
numpy_array = cpu_numpy(gpu_tensor)  # Automatically detaches, moves to CPU, converts
```

This is safer than calling `.detach().cpu().numpy()` manually because:
- Handles `None` gracefully
- Works with tensors already on CPU or GPU
- Can convert non-tensor array-likes

### `tensor(data, dtype=None, device=None)`

Wrapper for `torch.tensor()` that automatically uses the correct device:

```python
from device import tensor

# Automatically uses the configured device
x = tensor([1, 2, 3])  # Creates on CPU or CUDA based on get_device()

# Override device if needed
y = tensor([4, 5, 6], device='cuda')
```

## Architecture Integration

Device management is integrated throughout the codebase:

### Topology

```python
from topology import Topology
from substrate import Substrate

substrate = Substrate((400, 300))
substrate.create('linear', m=0.05, b=1.0)

# Topology automatically uses the appropriate device
topology = Topology(substrate=substrate, device=device)
```

All graph node data (`pos`, `persistent_id`, `to_delete`, etc.) is created on the specified device.

### Environment

```python
from durotaxis_env import DurotaxisEnv

# Device is automatically detected and propagated
env = DurotaxisEnv(config_path='config.yaml')

# Or explicitly set
env = DurotaxisEnv(config_path='config.yaml', device='cpu')
```

The environment propagates device to:
- Topology (graph data)
- Observation encoder (neural network)
- State tensors

### Training

```python
from train import DurotaxisTrainer

# Device is read from config or auto-detected
trainer = DurotaxisTrainer(config_path='config.yaml')
trainer.train()
```

Device configuration in `config.yaml`:

```yaml
system:
  device: 'auto'  # or 'cpu', 'cuda', 'cuda:0', etc.
```

### Deployment

```python
from deploy import DurotaxisDeployment

# Device is auto-detected
deployment = DurotaxisDeployment(
    model_path='./training_results/run0001/best_model.pt',
    config_path='config.yaml'
)

# Or explicitly set
deployment = DurotaxisDeployment(
    model_path='./model.pt',
    config_path='config.yaml',
    device='cpu'
)
```

## Common Scenarios

### Training on GPU, Deploying on CPU

```bash
# Train with GPU (automatic)
python train.py

# Deploy on CPU (explicit)
FORCE_CPU=1 python deploy.py --model_path ./training_results/run0001/best_model.pt
```

The trained model will work on any device—PyTorch automatically handles device mapping during `load_state_dict()`.

### Multi-GPU Setup

To use a specific GPU:

```bash
DEVICE=cuda:1 python train.py  # Use GPU 1
DEVICE=cuda:2 python deploy.py --model_path ./model.pt  # Use GPU 2
```

Or set in config:

```yaml
system:
  device: 'cuda:1'
```

### Debugging on CPU

Force CPU mode to avoid CUDA errors during debugging:

```bash
FORCE_CPU=1 python train.py --max_episodes 10
```

## Best Practices

1. **Let the system auto-detect** by default—no configuration needed for most cases
2. **Use environment variables** for temporary overrides (debugging, testing)
3. **Set config.yaml** for persistent preferences across runs
4. **Use device utilities** (`cpu_numpy`, `to_device`) when writing custom code to ensure device safety

## Testing Device-Agnostic Code

Test both CPU and GPU modes:

```bash
# Test on CPU
FORCE_CPU=1 python tools/verify_id_tracking.py

# Test on GPU (if available)
python tools/verify_id_tracking.py
```

All tests should pass regardless of device.

## Troubleshooting

### RuntimeError: Expected all tensors to be on the same device

This means you're mixing CPU and GPU tensors in an operation. Solutions:

1. Use `to_device()` to move data structures:
   ```python
   from device import to_device
   state_dict = to_device(state_dict)
   ```

2. Use `cpu_numpy()` before NumPy operations:
   ```python
   from device import cpu_numpy
   positions_np = cpu_numpy(positions)
   ```

3. Ensure constructors receive device:
   ```python
   topology = Topology(substrate=substrate, device=device)
   env = DurotaxisEnv(config_path='config.yaml', device=device)
   ```

### CUDA out of memory

Switch to CPU temporarily:

```bash
FORCE_CPU=1 python train.py
```

Or reduce batch size/max_critical_nodes in config.

### Model trained on GPU won't load on CPU

This should work automatically, but if issues persist:

```python
# Explicitly map to CPU during load
checkpoint = torch.load(model_path, map_location='cpu')
```

The codebase already does this in `deploy.py` and `train.py`.
