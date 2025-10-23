# RL-Durotaxis Codebase Workflow

## Overview

This codebase implements a **Reinforcement Learning (RL) system** for simulating and optimizing **durotaxis** - the directed migration of cells in response to substrate stiffness gradients. The agent learns to control a dynamic graph topology representing a cellular network, making decisions about cell spawning, deletion, and migration to maximize rightward movement along substrate gradients.

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
  │    └─▶ position_rewards (boundary awareness)             │
  │                                                           │
  │    Edge Rewards:                                         │
  │    ├─▶ rightward_bonus (edges pointing right)            │
  │    └─▶ leftward_penalty (edges pointing left)            │
  │                                                           │
  │    Spawn/Delete Rewards:                                 │
  │    ├─▶ spawn_success_reward                              │
  │    ├─▶ spawn_failure_penalty                             │
  │    ├─▶ proper_deletion_reward                            │
  │    └─▶ spawn_near_boundary_penalty                       │
  │                                                           │
  │    Milestone Rewards:                                    │
  │    ├─▶ distance_25_percent (reached 25% of width)        │
  │    ├─▶ distance_50_percent (reached 50%)                 │
  │    ├─▶ distance_75_percent (reached 75%)                 │
  │    └─▶ distance_90_percent (almost at goal)              │
  │                                                           │
  │    Termination Rewards:                                  │
  │    ├─▶ success_reward (reached rightmost boundary)       │
  │    ├─▶ out_of_bounds_penalty (boundary violation)        │
  │    ├─▶ no_nodes_penalty (lost all nodes)                 │
  │    └─▶ leftward_drift_penalty (persistent left movement) │
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
  ├─ node_features: [N, 3] (x, y, intensity)
  ├─ edge_features: [E, 1] (edge weights)
  ├─ graph_features: [G] (global features)
  └─ edge_index: [2, E] (connectivity)

        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  GRAPH INPUT ENCODER (encoder.py)                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  1. Linear Projection                                      │ │
│  │     - Node: [3] → [64]                                     │ │
│  │     - Edge: [1] → [64]                                     │ │
│  │     - Graph: [G] → [64]                                    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  2. Multi-Head Self-Attention (4 layers)                   │ │
│  │     - Node-to-node attention                               │ │
│  │     - Graph-to-node attention                              │ │
│  │     - Residual connections                                 │ │
│  │     - Layer normalization                                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  OUTPUT: [graph_token, node_tokens]                             │
│    - graph_token: [1, 64] (global representation)               │
│    - node_tokens: [N, 64] (per-node embeddings)                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────┬─────────────────────────────┐
        ▼                      ▼                             ▼
┌─────────────────┐  ┌─────────────────┐    ┌─────────────────────┐
│     ACTOR       │  │     CRITIC      │    │   SIMPLICIAL        │
│                 │  │                 │    │   EMBEDDING         │
│  node_tokens    │  │  graph_token    │    │   (Optional)        │
│  graph_token    │  │                 │    │                     │
│       │         │  │       │         │    │  - Group nodes      │
│       ▼         │  │       ▼         │    │  - Cluster features │
│  ┌──────────┐  │  │  ┌──────────┐  │    │  - Reduce dims      │
│  │ ResNet18 │  │  │  │ ResNet18 │  │    └─────────────────────┘
│  │ Backbone │  │  │  │ Backbone │  │
│  │(ImageNet)│  │  │  │(ImageNet)│  │    ┌─────────────────────┐
│  └────┬─────┘  │  │  └────┬─────┘  │    │   WSA (Optional)    │
│       │         │  │       │         │    │   Weight Sharing    │
│       ▼         │  │       ▼         │    │   Attention         │
│  ┌──────────┐  │  │  ┌──────────┐  │    │                     │
│  │   MLP    │  │  │  │   MLP    │  │    │  - Multi-PTM        │
│  └────┬─────┘  │  │  └────┬─────┘  │    │  - Dynamic weights  │
│       │         │  │       │         │    │  - Feature fusion   │
│       ▼         │  │       ▼         │    └─────────────────────┘
│  ┌──────────┐  │  │  ┌──────────┐  │
│  │ Outputs: │  │  │  │ Outputs: │  │
│  │          │  │  │  │          │  │
│  │ discrete │  │  │  │  values  │  │
│  │  logits  │  │  │  │  (6 comp)│  │
│  │  [N, 2]  │  │  │  │          │  │
│  │          │  │  │  │ - total  │  │
│  │continuous│  │  │  │ - graph  │  │
│  │   mu     │  │  │  │ - spawn  │  │
│  │  [N, 4]  │  │  │  │ - delete │  │
│  │          │  │  │  │ - edge   │  │
│  │continuous│  │  │  │ - node   │  │
│  │  logstd  │  │  │  │          │  │
│  │  [N, 4]  │  │  │  └──────────┘  │
│  └──────────┘  │  │                 │
└─────────────────┘  └─────────────────┘

ACTOR OUTPUT PROCESSING:
  ├─▶ discrete_logits → _safe_masked_logits() → Categorical(logits)
  │   └─ Sample: spawn/delete per node
  │
  └─▶ continuous_mu, continuous_std → Normal() → Sample: [γ, α, noise, θ]

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
    'graph_features': torch.Tensor,     # [global_features]
    'node_features': torch.Tensor,      # [N, 3] (x, y, intensity)
    'edge_features': torch.Tensor,      # [E, 1] (edge weights)
    'edge_index': torch.Tensor,         # [2, E] (connectivity)
    'num_nodes': int                    # Number of nodes
}

# Action Dictionary
actions = {
    'discrete_actions': torch.Tensor,   # [N] (0=spawn, 1=delete)
    'continuous_actions': torch.Tensor, # [N, 4] (γ, α, noise, θ)
    'discrete_log_probs': torch.Tensor, # [N]
    'continuous_log_probs': torch.Tensor, # [N]
    'total_log_probs': torch.Tensor     # [N]
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
| **encoder.py** | Graph neural network encoder | `GraphInputEncoder` |
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

### 2. **Hybrid Action Space**
- Discrete: spawn/delete (Categorical distribution)
- Continuous: spawn parameters (Normal distribution)
- Combined policy loss with gradient balancing

### 3. **Numerical Stability**
- `_safe_masked_logits()`: Conditional normalization
- `Categorical(logits=...)`: Use logits instead of probs
- Gradient clipping and NaN/Inf detection

### 4. **Curriculum Learning**
- 3 progressive stages with adaptive transitions
- Stage-specific reward multipliers
- Success criteria for stage advancement

### 5. **Graph Neural Networks**
- Multi-head self-attention for node embeddings
- Graph-level pooling for global features
- Edge-aware message passing

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
```

### Ablation Study (Disable WSA)
```yaml
# In config.yaml
actor_critic:
  wsa:
    enabled: false  # Use standard ResNet backbone
```

### Change Learning Rate via CLI
```bash
python train_cli.py --learning-rate 0.0003
```

---

## Summary

This codebase implements a sophisticated **graph-based RL system** for durotaxis simulation with:

✅ **Multi-component reward system** (6 components)  
✅ **Hybrid action space** (discrete + continuous)  
✅ **Graph neural networks** (attention-based encoder)  
✅ **Curriculum learning** (3-stage progressive difficulty)  
✅ **Numerical stability** (NaN-safe operations)  
✅ **PPO with value clipping** (stable policy updates)  
✅ **Comprehensive logging** (metrics, checkpoints, detailed node tracking)  
✅ **Flexible configuration** (YAML + CLI overrides)  
✅ **Ablation study support** (WSA, SEM, pretrained weights)

The workflow is designed for **research reproducibility** and **production stability**, with extensive safeguards against training failures and comprehensive monitoring of learning progress.
