# rl-durotaxis
Implementation of Durotaxis via Reinforcement Learning


## Dependencies
### Create and Activate a Conda Environment
First, create a fresh conda environment (named `durotaxis`) to avoid any conflicts with existing packages. The supported Python versions of [Deep Graph Library (DGL)](https://www.dgl.ai/pages/start.html) are 3.8, 3.9, 3.10, 3.11, 3.12. 
```bash
conda create -n durotaxis python=3.12.11
```

### Activate the environment
```bash
conda activate durotaxis
```

### Install PyTorch 
You can install PyTorch either using CPU or GPU. The Deep Graph Library (DGL) only supports:
* CUDA 11.8 and 12.1 (Make sure to install first the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local).)
* [PyTorch](https://pytorch.org/get-started/locally/) 2.1.0+

If you wish to use CPU for PyTorch, use the command:
```bash
pip install torch==2.2.1
```

If your computer has NVIDIA CUDA support, use:
```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

### Install Deep Graph Library (DGL)
After PyTorch is installed, you can install DGL. Since 2024.06.27, DGL have stopped providing packages for Windows and MacOS. The latest version of available package is 2.2.1. For DGL installation instructions, go to [DGL page](https://www.dgl.ai/pages/start.html).
For CPU:
```bash
pip install  dgl -f https://data.dgl.ai/wheels/repo.html
```
For GPU:
```bash
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
```

### Install other packages
```bash
pip install -r requirements.txt
```

### Installation in WSL
There might be some issues with library compatability. If you are using Windows, I suggest that you create environment in WSL.
```bash
conda create -n durotaxis python=3.12.11
conda activate durotaxis
pip install torch==2.4.0 stable-baselines3 dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
pip install -r requirements.txt
```

# Action space
Investigate if we can use "topology".

# Success termination
If agent reaches the rightmost area.

# Training the Agent

## Basic Usage
```python
from durotaxis_sim import DurotaxisEnv
from stable_baselines3 import PPO

# Create environment with model saving enabled
env = DurotaxisEnv(
    substrate_size=(200, 150),
    max_nodes=50,
    max_steps=30,
    model_path="./saved_models",  # Directory to save models
    show_display=True
)

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Model Saving and Loading

### Automatic Model Saving
Models are automatically saved after each episode when `model_path` is specified in the environment constructor. The system supports both external RL algorithms (PPO, DQN, SAC, etc.) and internal graph transformer policies.

#### Key Features:
- **Automatic saving per episode**: No manual intervention required
- **Unique naming per run**: Each training run gets a timestamp-based identifier
- **Algorithm identification**: Model files include the algorithm name
- **Metadata storage**: Training information saved alongside models
- **Directory auto-creation**: Creates save directory if it doesn't exist

#### File Naming Convention:
```
{algorithm}_{timestamp}_ep{episode:05d}.zip
{algorithm}_{timestamp}_ep{episode:05d}_metadata.json
```

Example:
```
PPO_20251004_104530_ep00001.zip
PPO_20251004_104530_ep00001_metadata.json
PPO_20251004_104530_ep00002.zip
PPO_20251004_104530_ep00002_metadata.json
```

### Setting Up Model Saving
```python
# Initialize environment with model saving
env = DurotaxisEnv(
    substrate_size=(200, 150),
    max_nodes=50,
    max_steps=30,
    model_path="./my_models",  # Specify where to save models
    show_display=False
)

# Models will be automatically saved after each episode
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)  # Saves model every episode
```

### Manual Model Operations

#### Listing Saved Models
```python
# List all saved models
models_info = env.list_saved_models()
print("Available models:")
for category, files in models_info.items():
    print(f"{category}: {len(files)} files")
    for file in files[:5]:  # Show first 5
        print(f"  📄 {file}")
```

#### Loading a Saved Model
```python
# Load a specific model
model_filename = "PPO_20251004_104530_ep00100.zip"
loaded_model = env.load_model(model_filename)

# Use the loaded model for inference
obs = env.reset()
for _ in range(1000):
    action, _ = loaded_model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### Metadata Information
Each saved model includes a JSON metadata file containing:
```json
{
  "algorithm": "PPO",
  "episode": 100,
  "run_timestamp": "20251004_104530",
  "substrate_size": [200, 150],
  "max_nodes": 50,
  "max_steps": 30,
  "embedding_dim": 32,
  "hidden_dim": 64,
  "total_reward": -45.23,
  "save_timestamp": "2025-10-04T10:45:30.123456"
}
```

### Advanced Usage Examples

#### Training with Custom Parameters
```python
# Advanced training with model saving
env = DurotaxisEnv(
    substrate_size=(300, 200),
    max_nodes=100,
    max_steps=50,
    embedding_dim=64,
    hidden_dim=128,
    model_path="./advanced_models",
    show_display=True,
    substrate_only=False
)

# Train multiple algorithms
algorithms = {
    "PPO": PPO("MlpPolicy", env, verbose=1),
    "SAC": SAC("MlpPolicy", env, verbose=1),
    "DQN": DQN("MlpPolicy", env, verbose=1)
}

for name, model in algorithms.items():
    print(f"Training {name}...")
    model.learn(total_timesteps=20000)
    print(f"{name} training complete. Models saved automatically.")
```

#### Loading and Evaluating Models
```python
# Load and evaluate a trained model
env = DurotaxisEnv(
    substrate_size=(200, 150),
    max_nodes=50,
    max_steps=30,
    model_path="./saved_models",
    show_display=True
)

# Load best performing model
best_model = env.load_model("PPO_20251004_104530_ep00200.zip")

# Evaluate performance
total_rewards = []
for episode in range(10):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    
    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

print(f"Average reward: {sum(total_rewards)/len(total_rewards):.2f}")
```

#### Resuming Training from Saved Model
```python
# Resume training from a previously saved model
env = DurotaxisEnv(
    substrate_size=(200, 150),
    max_nodes=50,
    max_steps=30,
    model_path="./continued_training",
    show_display=False
)

# Load existing model
checkpoint_model = env.load_model("PPO_20251004_104530_ep00100.zip")

# Continue training
print("Resuming training from episode 100...")
checkpoint_model.learn(total_timesteps=50000)  # Additional training
print("Continued training complete!")
```

### Directory Structure
When using model saving, your directory structure will look like:
```
your_project/
├── durotaxis_sim.py
├── saved_models/
│   ├── PPO_20251004_104530_ep00001.zip
│   ├── PPO_20251004_104530_ep00001_metadata.json
│   ├── PPO_20251004_104530_ep00002.zip
│   ├── PPO_20251004_104530_ep00002_metadata.json
│   ├── ...
│   └── PPO_20251004_104530_ep00200.zip
└── your_training_script.py
```

## Deploying the Agent

### Production Deployment
```python
# Load trained model for production use
from durotaxis_sim import DurotaxisEnv

# Create environment (no model_path needed for inference only)
env = DurotaxisEnv(
    substrate_size=(200, 150),
    max_nodes=50,
    max_steps=30,
    show_display=False  # Disable visualization for faster inference
)

# Load your best model
trained_model = env.load_model("path/to/PPO_20251004_104530_ep00200.zip")

# Deploy for inference
def predict_action(observation):
    action, _ = trained_model.predict(observation, deterministic=True)
    return action

# Use in your application
obs = env.reset()
while True:
    action = predict_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```