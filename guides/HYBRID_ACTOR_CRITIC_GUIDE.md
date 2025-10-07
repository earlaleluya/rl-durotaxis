# HybridActorCritic for Durotaxis Environment

## Overview

The `HybridActorCritic` class provides a sophisticated neural network architecture specifically designed for your durotaxis environment. It combines:

- **Discrete Actions**: spawn/delete decisions per node
- **Continuous Actions**: spawn parameters (gamma, alpha, noise, theta)  
- **Multi-Component Value Estimation**: separate value predictions for different reward components
- **Graph Neural Networks**: integration with your existing GraphInputEncoder

## Key Features

### 🎭 **Hybrid Action Space**
```python
# Discrete actions per node
discrete_actions = [0, 1, 0, 1]  # 0=spawn, 1=delete for each node

# Continuous parameters for spawning
continuous_actions = [
    [gamma1, alpha1, noise1, theta1],  # Node 0 spawn params
    [gamma2, alpha2, noise2, theta2],  # Node 1 spawn params
    # ... for each node
]
```

### 📊 **Multi-Component Value Prediction**
The network can predict values for different reward components:
```python
value_predictions = {
    'total_value': 0.123,      # Overall expected return
    'graph_value': -0.045,     # Graph-level reward components
    'spawn_value': -0.234,     # Spawning behavior rewards
    'node_value': 0.067,       # Node-level rewards
    'edge_value': 0.089,       # Edge direction rewards
    'delete_value': 0.012      # Deletion compliance rewards
}
```

### 🧠 **Graph Neural Network Integration**
- Uses your existing `GraphInputEncoder` for state representation
- Handles variable graph sizes with proper batching
- Provides node-level action prediction with global graph context

## Architecture Details

```
Input: Graph State (nodes, edges, graph features)
  ↓
GraphInputEncoder (GraphTransformer)
  ↓
Split: Graph Token + Node Tokens
  ↓
┌─────────────────┬─────────────────┐
│ ACTOR HEADS     │ CRITIC HEAD     │
├─────────────────┼─────────────────┤
│ Discrete Head   │ Graph Value MLP │
│ (spawn/delete)  │ Multi-component │
│                 │ Value Heads     │
│ Continuous Head │                 │
│ (spawn params)  │                 │
└─────────────────┴─────────────────┘
```

## Usage Examples

### Basic Setup

```python
from actor_critic import HybridActorCritic, HybridPolicyAgent
from encoder import GraphInputEncoder
from durotaxis_sim import Durotaxis

# Create environment
env = Durotaxis(substrate_size=(300, 200), init_num_nodes=3)

# Create encoder
encoder = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=3)

# Create hybrid network
network = HybridActorCritic(
    encoder=encoder,
    hidden_dim=128,
    value_components=['total_value', 'graph_value', 'spawn_value', 'node_value', 'edge_value']
)

# Create agent wrapper
agent = HybridPolicyAgent(
    topology=env.topology,
    state_extractor=env.state_extractor,
    hybrid_network=network
)
```

### Action Selection and Execution

```python
# Get actions and values
actions, spawn_params, value_predictions = agent.get_actions_and_values(deterministic=False)

# Execute actions (automatically handles spawn/delete)
executed_actions = agent.act_with_policy(deterministic=False)

# Get environment step with reward components
obs, reward_dict, terminated, truncated, info = env.step(0)
```

### Training Loop Integration

```python
import torch.optim as optim

optimizer = optim.Adam(network.parameters(), lr=1e-3)

for episode in range(num_episodes):
    obs, info = env.reset()
    
    while not done:
        # Get state
        state = env.state_extractor.get_state_features(include_substrate=True)
        
        # Forward pass
        output = network(state, deterministic=False)
        
        # Execute step
        obs, reward_dict, terminated, truncated, info = env.step(0)
        
        # Compute losses using reward components
        total_loss = compute_loss(output, reward_dict, state)
        
        # Update network
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

## Key Methods

### `HybridActorCritic.forward(state_dict, deterministic=False)`
**Returns:** Dictionary with actions, probabilities, value predictions
```python
output = {
    'discrete_logits': torch.Tensor,      # [num_nodes, 2]
    'discrete_probs': torch.Tensor,       # [num_nodes, 2] 
    'continuous_mu': torch.Tensor,        # [num_nodes, 4]
    'continuous_std': torch.Tensor,       # [num_nodes, 4]
    'value_predictions': Dict[str, torch.Tensor],
    'discrete_actions': torch.Tensor,     # [num_nodes] (if sampled)
    'continuous_actions': torch.Tensor,   # [num_nodes, 4] (if sampled)
    'discrete_log_probs': torch.Tensor,   # [num_nodes] (if sampled)
    'continuous_log_probs': torch.Tensor, # [num_nodes] (if sampled)
}
```

### `HybridActorCritic.evaluate_actions(state_dict, discrete_actions, continuous_actions)`
**Purpose:** Evaluate actions for policy gradient updates
**Returns:** Log probabilities, value predictions, entropy

### `HybridPolicyAgent.get_actions_and_values(deterministic=False)`
**Purpose:** High-level interface for action selection
**Returns:** `(actions_dict, spawn_params_dict, value_predictions_dict)`

## Integration with Reward Components

The network is designed to work seamlessly with your environment's reward component dictionary:

```python
# Environment returns reward components
reward_dict = {
    'total_reward': -1.234,
    'graph_reward': -0.567,
    'spawn_reward': -1.000,
    'node_reward': 0.123,
    'edge_reward': 0.210,
    # ... other components
}

# Network predicts corresponding values
value_predictions = {
    'total_value': 0.123,    # → total_reward
    'graph_value': -0.045,   # → graph_reward
    'spawn_value': -0.234,   # → spawn_reward
    'node_value': 0.067,     # → node_reward
    'edge_value': 0.089,     # → edge_reward
}
```

## Advantages for Learning

1. **Component-Specific Learning**: Train separate value functions for different reward aspects
2. **Flexible Loss Design**: Weight different components differently in your loss function
3. **Better Credit Assignment**: Understand which actions contribute to which rewards
4. **Robust Training**: Multi-head value prediction provides more stable learning signals

## Next Steps

1. **Implement Your Learning Algorithm**: Use the network outputs to implement PPO, A2C, or custom algorithms
2. **Tune Value Components**: Experiment with different reward component weightings
3. **Add Regularization**: Implement entropy bonuses, value clipping, etc.
4. **Scaling**: The architecture handles variable graph sizes automatically

The `HybridActorCritic` provides a complete foundation for sophisticated RL algorithms tailored to your durotaxis environment! 🚀