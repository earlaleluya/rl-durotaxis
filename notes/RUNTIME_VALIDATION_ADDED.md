# Runtime Validation for Reward Components

**Date**: 2025-10-31  
**Purpose**: Added defensive runtime validation to catch config/architecture mismatches

---

## What Was Added

### 1. Value Head Validation (`actor_critic.py`)

#### Method: `_validate_value_heads()`
Located in `HybridActorCritic` class, called during `__init__`.

**Purpose**: Ensures critic value heads exactly match configured `value_components`.

**Behavior**:
```python
def _validate_value_heads(self):
    expected_components = set(self.value_components)
    actual_heads = set(self.critic.value_heads.keys())
    
    if expected_components != actual_heads:
        # Raises ValueError with detailed error message
        raise ValueError(error_msg)
    
    # Prints success message with component list
```

**When It Runs**: Automatically during network initialization

**What It Catches**:
- Missing value heads (component in config but no critic head)
- Extra value heads (critic head exists but not in config)
- Any mismatch between config and architecture

**Example Output**:
```
✅ Value head validation passed: 5 components
   - delete_reward
   - distance_signal
   - graph_reward
   - spawn_reward
   - total_reward
```

---

### 2. Component Weights Validation (`actor_critic.py`)

#### Method: `validate_component_weights(trainer_component_weights)`
Located in `HybridActorCritic` class, called by trainer after initialization.

**Purpose**: Validates trainer component weights are compatible with value heads.

**Behavior**:
```python
def validate_component_weights(self, trainer_component_weights: Dict[str, float]):
    vc_set = set(self.value_components)
    tcw_set = set(trainer_component_weights.keys())
    
    # CRITICAL: Check for components in trainer but not in critic
    missing_in_critic = tcw_set - vc_set
    if missing_in_critic:
        raise ValueError(...)  # This is an error
    
    # WARNING: Check for components in critic but not in trainer
    missing_in_trainer = vc_set - tcw_set
    if missing_in_trainer:
        print("WARNING: ...")  # This is just a warning
```

**When It Runs**: Called in `train.py` after network initialization

**What It Catches**:
- **ERROR**: Trainer wants to weight a component that doesn't have a value head
- **WARNING**: Value head exists but trainer doesn't weight it (uses default 0)

**Example Output**:
```
✅ Component weights validation passed: All 5 components aligned
```

---

### 3. Trainer Integration (`train.py`)

#### Location: After network initialization (line ~698)

**Code Added**:
```python
self.network = HybridActorCritic(
    encoder=self.encoder,
    hidden_dim=actor_critic_config.get('hidden_dim', 128),
    value_components=actor_critic_config.get('value_components', self.component_names),
    dropout_rate=actor_critic_config.get('dropout_rate', 0.1)
).to(self.device)

# Validate component weights match network value heads
self.network.validate_component_weights(self.component_weights)

# Enhanced Learnable Component Weighting System
self._initialize_learnable_weights()
```

**Purpose**: Ensures trainer and network are aligned before training starts.

---

## Error Examples

### Example 1: Missing Value Head

**Scenario**: Config has component in `component_weights` but not in `value_components`

```yaml
actor_critic:
  value_components:
    - 'total_reward'
    - 'delete_reward'
    # Missing: spawn_reward

trainer:
  component_weights:
    total_reward: 1.0
    delete_reward: 0.4
    spawn_reward: 0.3  # ERROR: No value head for this!
```

**Error Message**:
```
❌ Trainer component_weights has keys not present in critic value_components:
  Missing in critic: {'spawn_reward'}
  Critic components: ['delete_reward', 'total_reward']
  Trainer weights: ['delete_reward', 'spawn_reward', 'total_reward']
```

---

### Example 2: Value Head Mismatch

**Scenario**: Internal bug where critic heads don't match config

```python
# Bug in code: Hardcoded value heads
self.value_heads = nn.ModuleDict({
    'total_reward': nn.Linear(128, 1),
    'graph_reward': nn.Linear(128, 1),
    # Missing other components!
})
```

**Error Message**:
```
❌ Value head configuration mismatch!
  Missing heads: {'delete_reward', 'spawn_reward', 'distance_signal'}
  Expected: ['delete_reward', 'distance_signal', 'graph_reward', 'spawn_reward', 'total_reward']
  Actual: ['graph_reward', 'total_reward']
```

---

### Example 3: Harmless Warning

**Scenario**: Value head exists but trainer doesn't use it (harmless)

```yaml
actor_critic:
  value_components:
    - 'total_reward'
    - 'delete_reward'
    - 'spawn_reward'  # Has value head

trainer:
  component_weights:
    total_reward: 1.0
    delete_reward: 0.4
    # Missing: spawn_reward (will default to weight=0)
```

**Warning Message**:
```
⚠️  WARNING: Value components not in trainer.component_weights (will use weight=0):
   - spawn_reward
```

---

## Benefits

### 1. **Early Error Detection**
- Catches mismatches during initialization (not during training)
- Fails fast with clear error messages
- Prevents silent bugs where components are ignored

### 2. **Clear Diagnostics**
- Detailed error messages show exactly what's wrong
- Lists expected vs actual components
- Makes debugging trivial

### 3. **Zero Runtime Overhead**
- Validation only runs during initialization
- No impact on training performance
- One-time check per training run

### 4. **Prevents Silent Failures**
- Old system: Component mismatch → training continues with wrong values
- New system: Component mismatch → immediate error with explanation

---

## Testing

### Test 1: Normal Configuration
```python
# All components aligned
network = HybridActorCritic(encoder, config_path='config.yaml')
network.validate_component_weights(trainer_weights)
# Output: ✅ Component weights validation passed: All 5 components aligned
```

### Test 2: Intentional Mismatch
```python
# Add invalid component to weights
bad_weights = {'total_reward': 1.0, 'invalid_component': 0.5}
network.validate_component_weights(bad_weights)
# Output: ValueError: Trainer component_weights has keys not present...
```

### Test 3: Automated Validation
```bash
python tools/validate_reward_components.py
# Output: 🎉 ALL VALIDATIONS PASSED - System is consistent!
```

---

## Migration Guide

### For Existing Code

**No changes needed!** The validation is automatic and transparent.

If your config is already consistent (which it should be after the review), you'll just see success messages:
```
✅ Value head validation passed: 5 components
✅ Component weights validation passed: All 5 components aligned
```

### For New Components

When adding a new reward component:

1. Add to `config.yaml` → `actor_critic.value_components`
2. Add to `config.yaml` → `trainer.component_weights`
3. Make sure environment returns it in reward breakdown
4. **Validation will automatically verify everything is aligned!**

---

## Summary

**What Changed**:
- ✅ Added `_validate_value_heads()` in `HybridActorCritic.__init__`
- ✅ Added `validate_component_weights()` method in `HybridActorCritic`
- ✅ Added validation call in `train.py` after network initialization
- ✅ All existing tests still pass

**What You Get**:
- ✅ Early detection of config/architecture mismatches
- ✅ Clear error messages for debugging
- ✅ Zero runtime overhead during training
- ✅ Prevention of silent bugs

**Impact**:
- No changes to existing functionality
- No performance impact
- Improved reliability and debuggability
- **Production ready** ✅
