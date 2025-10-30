# RL-Durotaxis Codebase Workflow

## Overview

This codebase implements a **Reinforcement Learning (RL) system** for simulating and optimizing **durotaxis** - the directed migration of cells in response to substrate stiffness gradients. The agent learns to control a dynamic graph topology representing a cellular network, making decisions about cell spawning, deletion, and migration to maximize rightward movement along substrate gradients.

### Key Technical Highlights

- **Graph Transformer Encoder:** PyTorch Geometric TransformerConv with edge-aware attention
- **Edge Features (`edge_attr`):** 3D edge properties [distance, direction_x, direction_y] used in attention mechanism
- **Continuous Action Space:** Delete-ratio architecture (5D: delete_ratio, gamma, alpha, noise, theta)
- **Simplicial Embedding:** Optional geometric constraint layer for ablation studies
- **Experimental Reward Modes:** Dense centroid-distance reward with adaptive terminal scaling

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING SYSTEM                              │
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      │
│  │   config.    │─────▶│  train.py    │─────▶│  Training    │      │
│  │   yaml       │      │              │      │  Results     │      │
│  └──────────────┘      │ DurotaxisT-  │      └──────────────┘      │
│                        │   rainer     │                             │
│                        └──────┬───────┘                             │
│                               │                                     │
│                               ▼                                     │
│                    ┌──────────────────────┐                         │
│                    │  durotaxis_env.py    │                         │
│                    │  DurotaxisEnv        │                         │
│                    │  (Gym Environment)   │                         │
│                    └──────────┬───────────┘                         │
│                               │                                     │
│              ┌────────────────┼────────────────┐                    │
│              ▼                ▼                ▼                    │
│      ┌──────────┐     ┌──────────┐    ┌──────────┐                │
│      │topology  │     │substrate │    │  state   │                │
│      │  .py     │     │  .py     │    │  .py     │                │
│      └──────────┘     └──────────┘    └──────────┘                │
│                               │                                     │
│                               ▼                                     │
│                    ┌──────────────────────┐                         │
│                    │  actor_critic.py     │                         │
│                    │  HybridActorCritic   │                         │
│                    │  (RL Agent)          │                         │
│                    └──────────┬───────────┘                         │
│                               │                                     │
│                               ▼                                     │
│                    ┌──────────────────────┐                         │
│                    │   encoder.py         │                         │
│                    │   GraphInputEncoder  │                         │
│                    │   (GNN + Attention)  │                         │
│                    └──────────────────────┘                         │
└───────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Workflow

### 1. **Configuration & Initialization**

```
config.yaml
    │
    ├─▶ trainer: Training parameters (episodes, learning rate, batch size)
    ├─▶ environment: Environment setup (substrate size, node limits, rewards)
    ├─▶ actor_critic: Network architecture (hidden dims, action spaces)
    ├─▶ curriculum_learning: Progressive difficulty stages
    ├─▶ algorithm: PPO parameters (gamma, GAE, clipping)
    └─▶ system: Device configuration (CUDA/CPU)
            │
            ▼
    config_loader.py
    (ConfigLoader)
            │
            ▼
    train.py
    (DurotaxisTrainer.__init__)
```

**Key Files:**
- `config.yaml`: Central configuration file
- `config_loader.py`: Parses YAML and provides structured config access
- `train_cli.py`: Command-line wrapper for overriding config parameters

---

### 2. **Training Loop Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                               │
│                    (train.py: train())                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  FOR each episode (1 to total_episodes):                         │
│                                                                   │
│    1. ┌──────────────────────────────────────┐                  │
│       │  EPISODE COLLECTION                   │                  │
│       │  - Reset environment                  │                  │
│       │  - Generate substrate (random/fixed)  │                  │
│       │  - Collect trajectory data            │                  │
│       └──────────────────────────────────────┘                  │
│                     │                                             │
│                     ▼                                             │
│    2. ┌──────────────────────────────────────┐                  │
│       │  STEP LOOP (max_steps=1000)          │                  │
│       │                                        │                  │
│       │  ┌─────────────────────────────────┐ │                  │
│       │  │ a) Get graph state               │ │                  │
│       │  │    - Node features (x, y, int.)  │ │                  │
│       │  │    - Edge features               │ │                  │
│       │  │    - Graph features              │ │                  │
│       │  └─────────────────────────────────┘ │                  │
│       │                │                       │                  │
│       │                ▼                       │                  │
│       │  ┌─────────────────────────────────┐ │                  │
│       │  │ b) Actor-Critic forward pass     │ │                  │
│       │  │    - GraphInputEncoder           │ │                  │
│       │  │    - Actor (discrete + cont.)    │ │                  │
│       │  │    - Critic (value predictions)  │ │                  │
│       │  └─────────────────────────────────┘ │                  │
│       │                │                       │                  │
│       │                ▼                       │                  │
│       │  ┌─────────────────────────────────┐ │                  │
│       │  │ c) Sample actions                │ │                  │
│       │  │    - Discrete: spawn/delete      │ │                  │
│       │  │    - Continuous: γ,α,noise,θ     │ │                  │
│       │  └─────────────────────────────────┘ │                  │
│       │                │                       │                  │
│       │                ▼                       │                  │
│       │  ┌─────────────────────────────────┐ │                  │
│       │  │ d) Environment step              │ │                  │
│       │  │    - Apply topology actions      │ │                  │
│       │  │    - Update node positions       │ │                  │
│       │  │    - Compute rewards             │ │                  │
│       │  │    - Check termination           │ │                  │
│       │  └─────────────────────────────────┘ │                  │
│       │                │                       │                  │
│       │                ▼                       │                  │
│       │  ┌─────────────────────────────────┐ │                  │
│       │  │ e) Store transition              │ │                  │
│       │  │    - State, action, reward       │ │                  │
│       │  │    - Value, log_prob             │ │                  │
│       │  └─────────────────────────────────┘ │                  │
│       │                                        │                  │
│       └──────────────────────────────────────┘                  │
│                     │                                             │
│                     ▼                                             │
│    3. ┌──────────────────────────────────────┐                  │
│       │  COMPUTE RETURNS & ADVANTAGES         │                  │
│       │  - GAE (Generalized Advantage Est.)   │                  │
│       │  - Multi-component value targets      │                  │
│       │  - Reward normalization               │                  │
│       └──────────────────────────────────────┘                  │
│                     │                                             │
│                     ▼                                             │
│    4. ┌──────────────────────────────────────┐                  │
│       │  BATCH TRAINING                       │                  │
│       │  (Every rollout_batch_size=10 eps)    │                  │
│       │                                        │                  │
│       │  - Create minibatches                 │                  │
│       │  - PPO policy updates                 │                  │
│       │  - Multi-component value loss         │                  │
│       │  - Gradient clipping                  │                  │
│       │  - Optimizer step                     │                  │
│       └──────────────────────────────────────┘                  │
│                     │                                             │
│                     ▼                                             │
│    5. ┌──────────────────────────────────────┐                  │
│       │  LOGGING & CHECKPOINTING              │                  │
│       │  - Episode metrics                    │                  │
│       │  - Reward components                  │                  │
│       │  - Model checkpoints                  │                  │
│       │  - Detailed node logs (JSON)          │                  │
│       └──────────────────────────────────────┘                  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

### 3. **Environment Workflow (DurotaxisEnv)**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUROTAXIS ENVIRONMENT                         │
│                  (durotaxis_env.py)                              │
└─────────────────────────────────────────────────────────────────┘

INITIALIZATION:
    ├─▶ Topology (topology.py)
    │   └─ Graph structure: nodes (cells), edges (connections)
    │
    ├─▶ Substrate (substrate.py)
    │   └─ Stiffness field: linear, exponential, or custom gradients
    │
    └─▶ TopologyState (state.py)
        └─ State representation: features, history, metrics

─────────────────────────────────────────────────────────────────

STEP FUNCTION WORKFLOW:

  env.step(actions)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. APPLY TOPOLOGY ACTIONS                                │
  │    - For each node:                                      │
  │      • If discrete_action == 'spawn':                    │
  │          - Check durotaxis conditions                    │
  │          - Spawn new node with spawn_params (γ,α,θ,n)    │
  │          - Update graph structure                        │
  │      • If discrete_action == 'delete':                   │
  │          - Remove node from topology                     │
  │          - Update connectivity                           │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. UPDATE NODE POSITIONS                                 │
  │    - Apply movement dynamics                             │
  │    - Compute substrate interactions                      │
  │    - Update spatial coordinates                          │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. COMPUTE REWARD COMPONENTS                             │
  │                                                           │
  │    **EXPERIMENTAL REWARD MODES** (mutually exclusive):   │
  │                                                           │
  │    A) NORMAL MODE (default):                             │
  │       All standard reward components enabled             │
  │                                                           │
  │    B) CENTROID DISTANCE ONLY MODE:                       │
  │       environment.centroid_distance_only_mode: true      │
  │       ├─▶ Dense delta-distance reward: scale × (cx_t - cx_{t-1}) / goal_x │
  │       ├─▶ Potential-based shaping (preserves optimal policy) │
  │       ├─▶ Terminal rewards scaled down (×0.02) to avoid domination │
  │       └─▶ Adaptive scheduler: reduces terminal scale as progress improves │
  │                                                           │
  │    C) SIMPLE DELETE-ONLY MODE (deprecated):              │
  │       environment.simple_delete_only_mode: true          │
  │       Only penalties + termination rewards               │
  │                                                           │
  │    ─────────────────────────────────────────────────────│
  │    NORMAL MODE REWARDS:                                  │
  │                                                           │
  │    Graph Rewards:                                        │
  │    ├─▶ connectivity_penalty (nodes < 2)                  │
  │    ├─▶ growth_penalty (nodes > max_critical_nodes)       │
  │    ├─▶ survival_reward (per step)                        │
  │    └─▶ centroid_movement_reward (rightward progress)     │
  │                                                           │
  │    Node Rewards:                                         │
  │    ├─▶ movement_reward (rightward movement per node)     │
  │    ├─▶ intensity_bonus/penalty (substrate interaction)   │
  │    ├─▶ leftward_penalty (discourage left movement)       │
  │    └─▶ position_rewards (boundary awareness + safe zones)│
  │                                                           │
  │    Edge Rewards:                                         │
  │    ├─▶ rightward_bonus (edges pointing right)            │
  │    └─▶ leftward_penalty (edges pointing left)            │
  │                                                           │
  │    Delete Rewards (delete_ratio action):                 │
  │    ├─▶ proper_deletion_reward (delete leftmost nodes)    │
  │    ├─▶ persistence_penalty (keeping flagged nodes)       │
  │    └─▶ improper_deletion_penalty (delete non-flagged)    │
  │                                                           │
  │    Spawn Rewards (via gamma/alpha/noise/theta):          │
  │    ├─▶ spawn_success_reward                              │
  │    ├─▶ spawn_failure_penalty                             │
  │    └─▶ spawn_near_boundary_penalty (prevent risky spawns)│
  │                                                           │
  │    Milestone Rewards:                                    │
  │    ├─▶ distance_25_percent: +50.0 (reached 25% of width) │
  │    ├─▶ distance_50_percent: +150.0 (reached 50%)         │
  │    ├─▶ distance_75_percent: +300.0 (reached 75%)         │
  │    └─▶ distance_90_percent: +500.0 (almost at goal)      │
  │                                                           │
  │    Termination Rewards:                                  │
  │    ├─▶ success_reward: +500.0 (reached goal!)            │
  │    ├─▶ out_of_bounds_penalty: -100.0 (boundary violation)│
  │    ├─▶ no_nodes_penalty: -100.0 (lost all nodes)         │
  │    ├─▶ leftward_drift_penalty: -50.0 (persistent left)   │
  │    └─▶ timeout_penalty: 0.0 (no penalty for exploration) │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. CHECK TERMINATION CONDITIONS                          │
  │                                                           │
  │    Success:                                              │
  │    └─▶ Any node reaches rightmost boundary (x ≥ width)   │
  │                                                           │
  │    Failure:                                              │
  │    ├─▶ Boundary violation (node out of bounds)           │
  │    ├─▶ No nodes left (graph empty)                       │
  │    ├─▶ Critical node count exceeded                      │
  │    └─▶ Persistent leftward drift                         │
  │                                                           │
  │    Timeout:                                              │
  │    └─▶ Max steps reached (1000 steps)                    │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 5. UPDATE STATE                                          │
  │    - Build new observation dict                          │
  │    - Update history buffer                               │
  │    - Extract graph features                              │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  Return: (observation, reward, terminated, truncated, info)
```

---

### 4. **Neural Network Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│              HYBRID ACTOR-CRITIC NETWORK                         │
│              (actor_critic.py)                                   │
└─────────────────────────────────────────────────────────────────┘

INPUT: Graph State
  ├─ node_features: [N, 11] - Per-cell properties:
  │   [node_x, node_y, substrate_intensity, in_degree, out_degree, 
  │    centrality, centroid_distance, is_boundary, new_node_flag, age, stagnation]
  ├─ edge_features: [E, 3] - Connection properties:
  │   [distances, direction_norm_x, direction_norm_y]
  ├─ graph_features: [19] - Global properties:
  │   [num_nodes, num_edges, density, centroid_x, centroid_y, 
  │    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, bbox_width, 
  │    bbox_height, bbox_area, hull_area, avg_degree, max_degree, 
  │    sum_degree, mean_intensity, max_intensity, sum_intensity]
  └─ edge_index: [2, E] (connectivity in COO format)

        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  GRAPH INPUT ENCODER (encoder.py)                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  1. Feature Projection (MLP with GELU + LayerNorm)        │ │
│  │     - Node: [11] → [hidden_dim=128]                       │ │
│  │     - Edge: [3] → [hidden_dim=128]                        │ │
│  │     - Graph: [19] → [hidden_dim=128] (virtual node)       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  2. Graph Transformer (PyTorch Geometric TransformerConv)  │ │
│  │     - 4 layers of multi-head attention (4 heads)           │ │
│  │     - Edge-aware message passing with edge_attr            │ │
│  │     - Residual connections (when dims match)               │ │
│  │     - Layer normalization + GELU activation                │ │
│  │     - Dropout (0.1) for regularization                     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  3. Simplicial Embedding (Optional, if SEM enabled)        │ │
│  │     - Group-wise softmax: out_dim → num_groups × group_size│ │
│  │     - Constrains embeddings to product of simplices        │ │
│  │     - Temperature-controlled softmax (τ=1.0 default)       │ │
│  │     - Config: actor_critic.simplicial_embedding.enabled    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  OUTPUT: [graph_token, node_tokens]                             │
│    - graph_token: [1, out_dim=128] (global context)            │
│    - node_tokens: [N, out_dim=128] (per-node embeddings)       │
└─────────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────┬─────────────────────────────┐
        ▼                      ▼                             ▼
┌─────────────────┐  ┌─────────────────┐    ┌─────────────────────┐
│     ACTOR       │  │     CRITIC      │    │   ARCHITECTURE      │
│                 │  │                 │    │   NOTES             │
│  node_tokens    │  │  graph_token    │    │                     │
│  graph_token    │  │                 │    │  SEM: Applied in    │
│       │         │  │       │         │    │  encoder output     │
│       ▼         │  │       ▼         │    │  before Actor/Critic│
│  ┌──────────┐  │  │  ┌──────────┐  │    │                     │
│  │ ResNet18 │  │  │  │ ResNet18 │  │    │  WSA: Deprecated    │
│  │ Backbone │  │  │  │ Backbone │  │    │  (not in current    │
│  │(ImageNet)│  │  │  │(ImageNet)│  │    │   config)           │
│  └────┬─────┘  │  │  └────┬─────┘  │    │                     │
│       │         │  │       │         │    │  ResNet Freezing:   │
│       ▼         │  │       ▼         │    │  - freeze_mode      │
│  ┌──────────┐  │  │  ┌──────────┐  │    │    options: none,   │
│  │   MLP    │  │  │  │   MLP    │  │    │    all, until_layer3│
│  │(hidden=128)│  │  │ (hidden=128)│  │    │    last_block      │
│  └────┬─────┘  │  │  └────┬─────┘  │    │                     │
│       │         │  │       │         │    │  Input Adapter:     │
│       ▼         │  │       ▼         │    │  - repeat3: RGB    │
│  ┌──────────┐  │  │  ┌──────────┐  │    │  - 1ch_conv: Conv1x1│
│  │ Outputs: │  │  │  │ Outputs: │  │    └─────────────────────┘
│  │          │  │  │  │          │  │
│  │continuous│  │  │  │  values  │  │
│  │   mu     │  │  │  │  (6 comp)│  │
│  │  [N, 5]  │  │  │  │          │  │
│  │          │  │  │  │ - total  │  │
│  │continuous│  │  │  │ - graph  │  │
│  │  logstd  │  │  │  │ - spawn  │  │
│  │  [N, 5]  │  │  │  │ - delete │  │
│  │          │  │  │  │ - edge   │  │
│  │          │  │  │  │ - node   │  │
│  │          │  │  │  │          │  │
│  └──────────┘  │  │  └──────────┘  │
└─────────────────┘  └─────────────────┘

ACTOR OUTPUT PROCESSING:
  ├─▶ continuous_mu, continuous_std → Normal() → Sample: [delete_ratio, γ, α, noise, θ]
  │   └─ delete_ratio: [0.0, 0.5] - fraction of leftmost nodes to delete
  │   └─ γ: [0.5, 15.0] - spawn growth rate (widened range)
  │   └─ α: [0.5, 4.0] - directional bias strength
  │   └─ noise: [0.05, 0.5] - stochastic noise level
  │   └─ θ: [-π/6, π/6] - migration angle (-30° to +30°)
  │
  └─▶ TWO-STAGE CURRICULUM (config: algorithm.two_stage_curriculum):
      ├─ Stage 1: Learn delete_ratio only (spawn params fixed)
      └─ Stage 2: Learn all 5 continuous actions (delete_ratio + spawn params)

CRITIC OUTPUT:
  └─▶ value_predictions: dict with 6 components
      - Used for multi-objective advantage estimation
```

---

### 5. **Policy Update (PPO with Value Clipping)**

```
┌─────────────────────────────────────────────────────────────────┐
│               PPO POLICY UPDATE WORKFLOW                         │
│           (train.py: update_policy_minibatch)                    │
└─────────────────────────────────────────────────────────────────┘

INPUT: Minibatch of transitions
  ├─ states: [B, ...]
  ├─ actions: [B, ...]
  ├─ old_log_probs: [B]
  ├─ advantages: [B]
  ├─ returns: [B, 6]  (6 reward components)
  └─ old_values: [B, 6]

        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. FORWARD PASS (Re-evaluate actions)                          │
│     network.evaluate_actions(states, actions)                   │
│                                                                  │
│     Returns:                                                     │
│     ├─ discrete_log_probs: [B]                                  │
│     ├─ continuous_log_probs: [B]                                │
│     ├─ total_log_probs: [B]                                     │
│     ├─ value_predictions: [B, 6]                                │
│     └─ entropy: scalar                                           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. COMPUTE POLICY LOSS (PPO Clipped Objective)                 │
│                                                                  │
│     ratio = exp(new_log_prob - old_log_prob)                    │
│                                                                  │
│     surr1 = ratio * advantages                                  │
│     surr2 = clamp(ratio, 1-ε, 1+ε) * advantages                 │
│                                                                  │
│     policy_loss = -min(surr1, surr2).mean()                     │
│                                                                  │
│     Hybrid weighting:                                            │
│     total_policy_loss = (discrete_weight * discrete_loss +       │
│                          continuous_weight * continuous_loss)    │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. COMPUTE VALUE LOSS (Multi-Component with Clipping)          │
│                                                                  │
│     For each reward component c in [total, graph, spawn, ...]:  │
│                                                                  │
│       # Unclipped value loss                                    │
│       value_loss_unclipped = (returns[c] - values[c])²          │
│                                                                  │
│       # Clipped value loss                                      │
│       values_clipped = old_values[c] +                          │
│                        clamp(values[c] - old_values[c],         │
│                              -value_clip_ε, +value_clip_ε)      │
│       value_loss_clipped = (returns[c] - values_clipped)²       │
│                                                                  │
│       # Take maximum (conservative update)                      │
│       component_loss[c] = max(value_loss_unclipped,             │
│                               value_loss_clipped)               │
│                                                                  │
│     # Weighted combination                                      │
│     value_loss = Σ (weight[c] * component_loss[c])              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. COMPUTE TOTAL LOSS                                          │
│                                                                  │
│     total_loss = policy_loss +                                  │
│                  value_loss_coeff * value_loss -                │
│                  entropy_coeff * entropy                         │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. BACKWARD PASS & OPTIMIZATION                                │
│                                                                  │
│     optimizer.zero_grad()                                       │
│     total_loss.backward()                                       │
│                                                                  │
│     # NaN/Inf safety check                                      │
│     if isnan(total_loss) or isinf(total_loss):                  │
│         skip update, return                                     │
│                                                                  │
│     # Gradient clipping                                         │
│     clip_grad_norm_(network.parameters(), max_norm=0.5)         │
│                                                                  │
│     # Check for NaN gradients                                   │
│     for param in network.parameters():                          │
│         if param.grad has NaN/Inf:                              │
│             zero out gradient                                   │
│             skip optimizer step                                 │
│                                                                  │
│     optimizer.step()                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

### 6. **Curriculum Learning System**

```
┌─────────────────────────────────────────────────────────────────┐
│            CURRICULUM LEARNING PROGRESSION                       │
│          (curriculum_learning.py: CurriculumManager)             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: SURVIVAL & BASIC MOVEMENT (Episodes 0-200)            │
│  ─────────────────────────────────────────────────────────────  │
│  Goal: Learn to stay alive and move rightward                   │
│                                                                  │
│  Environment:                                                    │
│  ├─ max_nodes_allowed: 30                                       │
│  ├─ simplified_actions: false                                   │
│  └─ boundary_buffer: 10.0 (safety zone)                         │
│                                                                  │
│  Reward Multipliers:                                             │
│  ├─ survival_reward: 3.0x                                       │
│  ├─ movement_reward: 2.0x                                       │
│  ├─ centroid_movement_reward: 2.0x                              │
│  ├─ boundary_penalty: 5.0x                                      │
│  └─ milestone_rewards: 2.0x                                     │
│                                                                  │
│  Success Criteria:                                               │
│  ├─ min_episode_length: 50 steps                                │
│  ├─ max_boundary_violations: 2                                  │
│  ├─ min_rightward_progress: 50.0 units                          │
│  └─ min_centroid_x: 100.0                                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: DISTANCE MILESTONES (Episodes 201-500)                │
│  ─────────────────────────────────────────────────────────────  │
│  Goal: Reach 25%, 50%, and 75% distance milestones              │
│                                                                  │
│  Environment:                                                    │
│  ├─ max_nodes_allowed: 50                                       │
│  ├─ unlock_advanced_actions: true                               │
│  └─ enable_deletion_training: true                              │
│                                                                  │
│  Reward Multipliers:                                             │
│  ├─ survival_reward: 2.0x                                       │
│  ├─ movement_reward: 1.5x                                       │
│  ├─ centroid_movement_reward: 2.5x                              │
│  ├─ milestone_rewards: 3.0x  ← PRIMARY FOCUS                    │
│  ├─ spawn_success_reward: 1.5x                                  │
│  ├─ delete_reward: 2.5x                                         │
│  └─ intensity_bonus: 2.0x                                       │
│                                                                  │
│  Success Criteria:                                               │
│  ├─ min_episode_length: 100 steps                               │
│  ├─ max_boundary_violations: 1                                  │
│  ├─ min_rightward_progress: 150.0 units (25%)                   │
│  ├─ milestone_25_percent: true                                  │
│  └─ milestone_50_percent: true                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: GOAL ACHIEVEMENT (Episodes 501-1000)                  │
│  ─────────────────────────────────────────────────────────────  │
│  Goal: Consistently reach rightmost substrate (100% success)    │
│                                                                  │
│  Environment:                                                    │
│  ├─ max_nodes_allowed: 75                                       │
│  ├─ enable_all_constraints: true                                │
│  └─ competition_mode: true                                      │
│                                                                  │
│  Reward Multipliers:                                             │
│  ├─ survival_reward: 1.5x                                       │
│  ├─ movement_reward: 1.0x                                       │
│  ├─ centroid_movement_reward: 2.0x                              │
│  ├─ milestone_rewards: 1.5x                                     │
│  ├─ success_reward: 2.0x  ← MAIN GOAL                           │
│  ├─ spawn_success_reward: 1.2x                                  │
│  ├─ delete_reward: 1.5x                                         │
│  └─ efficiency_bonus: 3.0x                                      │
│                                                                  │
│  Success Criteria:                                               │
│  ├─ min_episode_length: 200 steps                               │
│  ├─ max_boundary_violations: 0                                  │
│  ├─ min_rightward_progress: 450.0 units (75%+)                  │
│  ├─ completion_rate: 0.15 (15% success)                         │
│  ├─ milestone_90_percent: true                                  │
│  └─ efficiency_threshold: 0.7                                   │
└─────────────────────────────────────────────────────────────────┘

PROGRESSION LOGIC:
  ├─ auto_advance: true
  ├─ advancement_criteria: "mixed" (success rate + episode threshold)
  ├─ min_success_rate: 0.10 (10%)
  ├─ evaluation_window: 100 episodes
  ├─ allow_early_advance: true
  └─ force_advance_at_end: true (always advance at episode_end)
```

---

### 7. **Key Data Structures**

```python
# State Dictionary (observations)
state_dict = {
    'graph_features': torch.Tensor,     # [19] - Global properties (enriched with pooling stats)
    'node_features': torch.Tensor,      # [N, 11] - Per-node properties (incl. age, stagnation)
    'edge_features': torch.Tensor,      # [E, 3] - Per-edge properties (distance + direction)
    'edge_index': torch.Tensor,         # [2, E] - Connectivity in COO format
    'num_nodes': int                    # Number of nodes
}

# Action Dictionary (DELETE_RATIO ARCHITECTURE)
actions = {
    'continuous_actions': torch.Tensor, # [N, 5] (delete_ratio, γ, α, noise, θ)
    'continuous_log_probs': torch.Tensor, # [N] - Log probability of continuous actions
    'total_log_probs': torch.Tensor     # [N] - Same as continuous_log_probs (no discrete)
}

# Action bounds (from config.yaml: actor_critic.action_parameter_bounds)
action_bounds = {
    'delete_ratio': [0.0, 0.5],        # Fraction of leftmost nodes to delete
    'gamma': [0.5, 15.0],              # Spawn growth rate (widened from [2.0, 8.0])
    'alpha': [0.5, 4.0],               # Directional bias strength
    'noise': [0.05, 0.5],              # Stochastic noise level
    'theta': [-0.5236, 0.5236]         # Migration angle (-π/6 to π/6, -30° to +30°)
}

# Reward Dictionary (multi-component)
rewards = {
    'total_reward': float,
    'graph_reward': float,
    'spawn_reward': float,
    'delete_reward': float,
    'edge_reward': float,
    'total_node_reward': float
}

# Value Predictions (multi-component)
value_predictions = {
    'total_reward': torch.Tensor,       # [1]
    'graph_reward': torch.Tensor,       # [1]
    'spawn_reward': torch.Tensor,       # [1]
    'delete_reward': torch.Tensor,      # [1]
    'edge_reward': torch.Tensor,        # [1]
    'total_node_reward': torch.Tensor   # [1]
}
```

---

### 8. **File Organization & Responsibilities**

| File | Responsibility | Key Classes/Functions |
|------|---------------|----------------------|
| **train.py** | Main training loop, PPO updates, logging | `DurotaxisTrainer`, `TrajectoryBuffer` |
| **train_cli.py** | Command-line interface wrapper | CLI argument parsing |
| **durotaxis_env.py** | Gym environment, reward computation, termination | `DurotaxisEnv` |
| **actor_critic.py** | Neural network architecture | `HybridActorCritic`, `Actor`, `Critic` |
| **encoder.py** | Graph neural network encoder | `GraphInputEncoder`, `MyGraphTransformer`, `SimplicialEmbedding` |
| **topology.py** | Graph structure, node/edge management | `Topology` |
| **substrate.py** | Stiffness field generation | `Substrate` |
| **state.py** | State representation and history | `TopologyState` |
| **config.yaml** | Configuration parameters | N/A (YAML) |
| **config_loader.py** | Configuration parsing | `ConfigLoader` |
| **curriculum_learning.py** | Progressive difficulty stages | `CurriculumManager` |
| **pretrained_fusion.py** | WSA multi-PTM fusion (optional) | `WeightSharingAttention` |
| **plotter.py** | Visualization and plotting | Plotting utilities |

---

### 9. **Training Execution Flow**

```bash
# Command Line
$ python train.py

# Or with CLI overrides
$ python train_cli.py --total-episodes 2000 --learning-rate 0.0003
```

**Execution Steps:**

1. **Load Configuration**
   ```
   config_loader.py → Parse config.yaml → Build config dict
   ```

2. **Initialize Trainer**
   ```
   DurotaxisTrainer.__init__()
   ├─ Create device (CUDA/CPU)
   ├─ Initialize environment (DurotaxisEnv)
   ├─ Create network (HybridActorCritic)
   ├─ Setup optimizer (Adam)
   ├─ Initialize curriculum manager
   ├─ Create trajectory buffer
   └─ Setup logging directories
   ```

3. **Training Loop**
   ```
   for episode in range(total_episodes):
       ├─ Collect trajectory
       │   └─ for step in range(max_steps):
       │       ├─ network.forward(state)
       │       ├─ env.step(actions)
       │       └─ buffer.add_step()
       │
       ├─ Compute returns & advantages (GAE)
       │
       ├─ If batch full (every 10 episodes):
       │   └─ for epoch in range(4):
       │       └─ for minibatch in batches:
       │           ├─ update_policy_minibatch()
       │           └─ optimizer.step()
       │
       ├─ Update curriculum stage
       ├─ Log metrics
       └─ Save checkpoint
   ```

4. **Output**
   ```
   training_results/
   ├─ run0001/
   │   ├─ config.yaml (snapshot)
   │   ├─ best_model.pt
   │   ├─ checkpoint_episode_500.pt
   │   ├─ training_log.txt
   │   ├─ detailed_nodes_all_episodes.json
   │   ├─ reward_components_stats.json
   │   └─ spawn_parameters_stats.json
   ```

---

## Key Design Patterns

### 1. **Multi-Component Value Learning**
- Separate value heads for 6 reward components
- Adaptive weighting based on component variance
- Attention-based dynamic weighting
- Value clipping for stable PPO updates

### 2. **Delete-Ratio Action Architecture (Current)**
- **Continuous-only action space:** [delete_ratio, gamma, alpha, noise, theta]
- Delete leftmost nodes by fraction (delete_ratio ∈ [0, 0.5])
- Spawn new nodes using durotaxis parameters (gamma, alpha, noise, theta)
- Two-stage curriculum: Stage 1 (delete_ratio only), Stage 2 (all 5 params)

### 3. **Graph Neural Networks with Edge Features**
- **PyTorch Geometric TransformerConv:** Multi-head graph attention
- **Edge-aware message passing:** `edge_attr` parameter provides edge features to attention
- **Residual connections:** Applied when input/output dimensions match
- **Enriched features:** 19 graph-level, 11 node-level, 3 edge-level dimensions
- **Optional Simplicial Embedding (SEM):** Group-wise softmax for geometric constraints

### 4. **Experimental Reward Modes**
- **Normal Mode:** Full reward structure (graph, node, edge, spawn, delete, milestones)
- **Centroid Distance Only Mode:** Dense delta-distance reward with adaptive terminal scaling
  - Potential-based shaping preserves optimal policy
  - Adaptive scheduler reduces terminal reward scale as progress improves
  - Config: `environment.centroid_distance_only_mode: true`
- **Simple Delete-Only Mode (Deprecated):** Penalties + termination only

### 5. **Curriculum Learning**
- 3 progressive stages with adaptive transitions
- Stage-specific reward multipliers
- Success criteria for stage advancement
- Optional overlap between stages for smooth transitions

### 6. **ResNet Backbone Architecture**
- **Pretrained weights:** ImageNet initialization (optional)
- **Freeze modes:** none, all, until_layer3, last_block
- **Input adapters:** repeat3 (RGB duplication) or 1ch_conv (1×1 convolution)
- **Separate learning rates:** backbone_lr and head_lr for fine-tuning control

### 7. **Numerical Stability & Safety**
- NaN/Inf detection and replacement in encoder inputs
- Gradient clipping (max_norm=0.5)
- Value clipping for PPO stability
- Adaptive gradient scaling with momentum

---

## Common Workflows

### Starting Fresh Training
```bash
python train.py
```

### Resume Training from Checkpoint
```yaml
# In config.yaml
trainer:
  resume_training:
    enabled: true
    checkpoint_path: "./training_results/run0001/checkpoint_episode_500.pt"
    reset_optimizer: false      # Keep optimizer state
    reset_episode_count: false  # Continue episode numbering
```

### Ablation Study: Disable Simplicial Embedding
```yaml
# In config.yaml
actor_critic:
  simplicial_embedding:
    enabled: false  # Ablation: Remove SEM constraint
```

### Ablation Study: Freeze ResNet Backbone
```yaml
# In config.yaml
actor_critic:
  backbone:
    freeze_mode: all           # none | all | until_layer3 | last_block
    pretrained_weights: imagenet  # or 'random' for random init
```

### Experimental Mode: Centroid Distance Only
```yaml
# In config.yaml
environment:
  centroid_distance_only_mode: true  # Dense distance-based reward
  include_termination_rewards: true  # Include terminal rewards (recommended)
  
  distance_mode:
    use_delta_distance: true         # Dense ∆-distance shaping
    distance_reward_scale: 5.0       # Scale factor
    terminal_reward_scale: 0.02      # Downscale terminals to avoid dominance
    scheduler_enabled: false         # Adaptive terminal reduction
```

### Two-Stage Training Curriculum
```yaml
# In config.yaml
algorithm:
  two_stage_curriculum:
    stage: 1  # Stage 1: Learn delete_ratio only (spawn params fixed)
    
    # Fixed spawn parameters for Stage 1
    stage_1_fixed_spawn_params:
      gamma: 0.5
      alpha: 2.0
      noise: 0.1
      theta: 0.0

# After Stage 1 converges, switch to Stage 2:
# stage: 2  # Stage 2: Learn all 5 continuous actions
```

### Change Learning Rate via CLI
```bash
python train_cli.py --learning-rate 0.0003 --total-episodes 2000
```

---

## Summary

This codebase implements a sophisticated **graph-based RL system** for durotaxis simulation with:

✅ **Multi-component reward system** (6 components with adaptive weighting)  
✅ **Delete-ratio continuous action architecture** (5D continuous: delete_ratio + spawn params)  
✅ **Graph Transformer encoder** (PyTorch Geometric TransformerConv with edge features)  
✅ **Enriched feature representations** (19 graph-level, 11 node-level, 3 edge-level)  
✅ **Simplicial Embedding (SEM)** (optional geometric constraint for ablation studies)  
✅ **Experimental reward modes** (Centroid Distance Only with adaptive scaling)  
✅ **Two-stage training curriculum** (Stage 1: delete_ratio only, Stage 2: full 5D)  
✅ **ResNet18 backbone** (ImageNet pretrained, configurable freezing modes)  
✅ **Curriculum learning** (3-stage progressive difficulty with overlap)  
✅ **PPO with value clipping** (stable policy updates with NaN safeguards)  
✅ **Numerical stability** (gradient clipping, NaN/Inf detection, adaptive scaling)  
✅ **Comprehensive logging** (metrics, checkpoints, detailed node tracking in JSON)  
✅ **Flexible configuration** (YAML-based with CLI overrides)  
✅ **Ablation study support** (SEM, pretrained weights, freeze modes, reward modes)

### Recent Architectural Updates (2025)

**Encoder Architecture:**
- Migrated from custom attention to **PyTorch Geometric TransformerConv**
- Added **edge_attr parameter** for edge-aware message passing
- Enriched features: graph (14→19), node (9→11), edge (1→3)
- Optional **Simplicial Embedding** layer for geometric inductive bias

**Action Space:**
- Switched from hybrid (discrete+continuous) to **pure continuous (delete_ratio)**
- Delete leftmost nodes by fraction instead of per-node discrete decisions
- Two-stage curriculum for progressive learning complexity

**Reward System:**
- **Centroid Distance Only Mode:** Dense delta-distance reward with adaptive terminal scaling
- Distance mode scheduler automatically reduces terminal rewards as agent improves
- Enhanced boundary penalties (edge/danger/critical zones) for safety

**Training Stability:**
- Value clipping for PPO (0.2 epsilon)
- Adaptive gradient scaling with momentum
- Residual connections in graph transformer (when dims match)
- NaN/Inf detection throughout forward pass

The workflow is designed for **research reproducibility** and **production stability**, with extensive safeguards against training failures, comprehensive ablation study support, and detailed monitoring of learning progress.
