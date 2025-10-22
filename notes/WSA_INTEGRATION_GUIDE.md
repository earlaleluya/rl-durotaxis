# Weight Sharing Attention (WSA) Integration Guide

## Overview

This guide explains how to integrate the **Weight Sharing Attention (WSA)** architecture into your durotaxis RL system to enhance feature representation using multiple pre-trained models.

Based on: *"Combining Pre-Trained Models for Enhanced Feature Representation in Reinforcement Learning"*

---

## What is WSA?

### Key Concept
Instead of using a single pre-trained model (like ResNet18), WSA combines **multiple diverse pre-trained models** and dynamically weights their contributions based on the current state context.

### Architecture Components

```
Graph State (nodes, edges, features)
           ↓
    ┌──────┴──────┐
    │   Encoder   │ (GNN + Simplicial Embedding)
    └──────┬──────┘
           ↓
    [Node Tokens, Graph Token]
           ↓
┌──────────┴──────────┐
│  Convert to Image   │ (24x24 "visual" representation)
└──────────┬──────────┘
           ↓
    ┌─────┴─────┬─────────┬─────────┐
    ↓           ↓         ↓         ↓
┌───────┐  ┌────────┐ ┌────────┐ ...
│ PTM 1 │  │ PTM 2  │ │ PTM 3  │
│ResNet │  │GraphCNN│ │ResNet  │
│ImageNet│ │        │ │Random  │
└───┬───┘  └────┬───┘ └────┬───┘
    │           │          │
    └─────┬─────┴──────────┘
          ↓
    [E₁, E₂, E₃, ...] (Embeddings)
          ↓
    ┌─────────────┐
    │State Encoder│ (Context from graph token)
    └──────┬──────┘
           ↓
      [Context C]
           ↓
    ┌──────────────────┐
    │Weight Sharing    │
    │Attention (WSA)   │
    │                  │
    │ Shared MLP(C,Eᵢ) │
    │    ↓             │
    │ Softmax(w₁...wₙ) │
    │    ↓             │
    │ R = Σ wᵢ * Eᵢ   │
    └──────┬───────────┘
           ↓
    [Rich Representation]
           ↓
    ┌──────┴──────┐
    │ Action Heads│
    └─────────────┘
```

---

## Benefits for Durotaxis

### 1. **Multiple Perspectives**
Different PTMs provide different views of the same graph state:
- **ResNet-ImageNet**: General visual patterns, spatial relationships
- **Graph-CNN**: Graph-specific structure patterns
- **ResNet-Random**: Diverse, uncorrelated features

### 2. **Adaptive Attention**
WSA dynamically emphasizes the most relevant model for each situation:
- **Early episode**: May focus on ResNet-ImageNet for general structure
- **Migration phase**: May focus on Graph-CNN for connectivity patterns  
- **Dense graphs**: May balance all models for comprehensive understanding

### 3. **Reduced Learning Burden**
By providing richer representations, the RL agent can focus on:
- **Policy optimization** (which actions to take)
- Rather than **feature learning** (what the state means)

### 4. **Robustness**
Multiple models provide redundancy - if one model's features are less useful in a situation, others can compensate.

---

## Integration Options

### Option 1: Replace Current Actor (Full Integration)

**Modify `actor_critic.py`:**

```python
from pretrained_fusion import WSAEnhancedActor

# In HybridActorCritic.__init__:
self.actor = WSAEnhancedActor(
    encoder_out_dim=self.encoder.out_dim,
    hidden_dim=self.hidden_dim,
    num_discrete_actions=self.num_discrete_actions,
    continuous_dim=self.continuous_dim,
    dropout_rate=self.dropout_rate,
    embedding_dim=256,  # WSA embedding dimension
    use_wsa=True        # Enable WSA
)
```

**Pros:**
- Most significant improvement
- Leverages full WSA benefits
- Directly applicable to current architecture

**Cons:**
- Increases model parameters (~3x due to multiple PTMs)
- Requires more GPU memory
- Training may be slower initially

### Option 2: Hybrid Approach (Gradual Integration)

Keep current Actor, but add WSA as an **optional enhancement**:

```python
# In config.yaml
actor_critic:
  use_wsa_enhancement: false  # Start with false, enable later
  wsa_embedding_dim: 256
  wsa_num_ptms: 3
```

**In code:**
```python
if config.get('use_wsa_enhancement', False):
    self.actor = WSAEnhancedActor(...)
else:
    self.actor = Actor(...)  # Current implementation
```

**Pros:**
- Can A/B test performance
- Gradual migration
- Fallback to simpler model if needed

**Cons:**
- More code complexity
- Maintains two parallel implementations

### Option 3: Feature Extractor Only

Use WSA just for feature extraction, keep current action heads:

```python
from pretrained_fusion import MultiPTMFeatureExtractor

class Actor(nn.Module):
    def __init__(self, ...):
        # Replace self.resnet_body with:
        self.feature_extractor = MultiPTMFeatureExtractor(
            embedding_dim=256,
            context_dim=128
        )
        # Keep existing action_mlp, discrete_head, etc.
```

**Pros:**
- Minimal changes to existing code
- Can compare feature quality directly
- Easier debugging

**Cons:**
- Doesn't leverage full WSA benefits
- Still increases parameters

---

## Configuration

### Add to `config.yaml`:

```yaml
actor_critic:
  # ... existing config ...
  
  # WSA Configuration
  use_wsa: false              # Enable WSA enhancement
  wsa_config:
    embedding_dim: 256        # Common embedding dimension
    context_dim: 128          # State encoder context dimension
    
    # Pre-trained models to use
    ptms:
      - model_type: 'resnet18_imagenet'
        freeze: true          # Freeze backbone (faster, less memory)
      
      - model_type: 'graph_cnn'
        freeze: false         # Train from scratch
      
      - model_type: 'resnet18_random'
        freeze: false         # Random init, trainable
    
    # Optional: Temperature for attention softmax
    attention_temperature: 1.0
```

---

## Training Strategy

### Phase 1: Baseline (Current System)
```bash
# Train with current Actor for comparison
python train.py --config config.yaml
```

**Metrics to track:**
- Episode rewards
- Success rate
- Training time per episode
- GPU memory usage

### Phase 2: WSA Integration
```bash
# Enable WSA in config.yaml
use_wsa: true

# Train with same hyperparameters
python train.py --config config.yaml
```

**Expected outcomes:**
- **Initial slowdown**: More parameters = slower initial steps
- **Faster convergence**: Better features = faster learning
- **Higher peak performance**: Richer representations = better policy

### Phase 3: Fine-Tuning

If WSA performs well, consider:

1. **Unfreezing ImageNet ResNet** after initial training:
   ```python
   # After episode 500, unfreeze for fine-tuning
   if episode > 500:
       for param in self.actor.feature_extractor.ptms[0].backbone.parameters():
           param.requires_grad = True
   ```

2. **Curriculum for PTMs**:
   - Stage 1: Only Graph-CNN active
   - Stage 2: Add ResNet-Random
   - Stage 3: Add ResNet-ImageNet

3. **Attention analysis**:
   - Log attention weights to see which models are used when
   - Identify patterns (e.g., "Graph-CNN dominates in dense regions")

---

## Monitoring WSA Performance

### Attention Weight Visualization

Add to your logging:

```python
# In training loop
if episode % 10 == 0 and hasattr(actor, 'feature_extractor'):
    with torch.no_grad():
        _, attention_weights = actor(
            node_tokens, graph_token, return_attention=True
        )
        
        # Log average attention per model
        avg_attention = attention_weights.mean(dim=0)
        for i, weight in enumerate(avg_attention):
            print(f"  PTM {i}: {weight:.3f}")
```

### Metrics to Track

1. **Attention Distribution**:
   - Are all models being used?
   - Does attention shift during curriculum stages?
   
2. **Feature Quality**:
   - Value prediction accuracy
   - Policy gradient magnitudes
   
3. **Computational Cost**:
   - Forward pass time
   - Memory usage
   - Training throughput (episodes/hour)

---

## Expected Performance Gains

Based on the WSA paper and your durotaxis domain:

### Conservative Estimate
- **10-15% faster convergence** to baseline performance
- **5-10% higher peak performance** after full training
- **More stable learning** (lower variance in rewards)

### Optimistic Estimate  
- **20-30% faster convergence**
- **15-20% higher peak performance**
- **Significant robustness** to initial conditions and hyperparameters

### Computational Cost
- **1.5-2x slower** per forward pass (3 models instead of 1)
- **~3x more parameters** (3 PTM backbones + WSA)
- **~2x more GPU memory** during training

---

## Debugging Tips

### If WSA is slower than expected:
1. **Reduce batch size**: WSA uses more memory
2. **Use gradient checkpointing**: Already implemented for large graphs
3. **Freeze more PTMs**: Set all `freeze: true` initially

### If attention weights don't change:
1. **Check temperature**: Lower temperature = sharper attention
2. **Increase context_dim**: Richer context = more discriminative attention
3. **Add regularization**: Encourage diverse attention (entropy bonus)

### If performance is worse:
1. **Start simpler**: Use only 2 PTMs initially
2. **Pretrain WSA**: Train WSA to match current ResNet first
3. **Gradual integration**: Add PTMs one at a time

---

## Code Example: Full Integration

```python
# actor_critic.py modifications

from pretrained_fusion import WSAEnhancedActor, MultiPTMFeatureExtractor

class HybridActorCritic(nn.Module):
    def __init__(self, encoder, config_path="config.yaml", **overrides):
        super().__init__()
        
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_actor_critic_config()
        
        # ... existing config loading ...
        
        # Check if WSA is enabled
        use_wsa = config.get('use_wsa', False)
        
        if use_wsa:
            print("🔄 Initializing WSA-Enhanced Actor...")
            wsa_config = config.get('wsa_config', {})
            
            self.actor = WSAEnhancedActor(
                encoder_out_dim=self.encoder.out_dim,
                hidden_dim=self.hidden_dim,
                num_discrete_actions=self.num_discrete_actions,
                continuous_dim=self.continuous_dim,
                dropout_rate=self.dropout_rate,
                embedding_dim=wsa_config.get('embedding_dim', 256),
                use_wsa=True
            )
            
            print(f"  ✅ WSA enabled with {len(wsa_config.get('ptms', 3))} PTMs")
        else:
            # Use current Actor implementation
            self.actor = Actor(
                encoder_out_dim=self.encoder.out_dim,
                hidden_dim=self.hidden_dim,
                num_discrete_actions=self.num_discrete_actions,
                continuous_dim=self.continuous_dim,
                dropout_rate=self.dropout_rate
            )
        
        # Critic remains unchanged
        self.critic = Critic(...)
```

---

## Research Opportunities

### Experiments to Try:

1. **Domain-Specific PTMs**:
   - Pre-train a model specifically on graph connectivity tasks
   - Use biological network datasets for pre-training

2. **Attention Mechanisms**:
   - Try multi-head attention instead of single weight per model
   - Explore learned temperature scheduling

3. **Curriculum for PTMs**:
   - Align PTM usage with curriculum learning stages
   - Stage 1: Simpler models, Stage 3: All models

4. **Transfer Learning**:
   - Train WSA on one substrate type
   - Transfer to different substrate (linear → exponential)

5. **Attention Analysis**:
   - Visualize which models activate for different graph structures
   - Correlate attention patterns with reward components

---

## Next Steps

1. **Baseline Testing** (1-2 days):
   - Run current system, establish performance metrics
   
2. **WSA Integration** (1 day):
   - Add configuration options
   - Integrate WSAEnhancedActor

3. **Comparative Training** (3-5 days):
   - Train both versions on same curriculum
   - Track all metrics

4. **Analysis** (1-2 days):
   - Compare learning curves
   - Analyze attention patterns
   - Identify best use cases

5. **Optimization** (ongoing):
   - Fine-tune WSA hyperparameters
   - Experiment with different PTM combinations

---

## References

- **Paper**: "Combining Pre-Trained Models for Enhanced Feature Representation in Reinforcement Learning"
- **Implementation**: `pretrained_fusion.py`
- **Key Innovation**: Weight Sharing Attention (WSA) for dynamic model combination
- **Your Domain**: Durotaxis navigation with graph-based representation

---

**Ready to enhance your RL agent with WSA? Start with Option 2 (Hybrid Approach) for a safe, gradual integration!** 🚀
