# Integrated Curriculum Learning & Reward System

## Overview

The curriculum learning system has been **completely redesigned** to work synergistically with the new reward structure, creating a comprehensive training pipeline that progressively guides the agent from basic survival to successful rightmost substrate migration.

## 🎓 Three-Stage Progressive Curriculum

### Stage 1: Survival & Basic Movement (Episodes 0-200)
**Goal**: Learn to stay alive and move rightward without catastrophic failures

**Environment Settings:**
- Substrate: 600×400 (full size from start)
- Initial nodes: 3 (stable start)
- Max nodes: 30 (moderate complexity)
- Max steps: 1000 (plenty of time)

**Reward Multipliers:**
```yaml
survival_reward: 3.0x              # Strong survival incentive
movement_reward: 2.0x              # Learn rightward direction
centroid_movement_reward: 2.0x     # Collective movement
boundary_penalty: 5.0x             # Avoid catastrophic failures
milestone_rewards: 2.0x            # Celebrate early progress
```

**Success Criteria:**
- Survive ≥50 steps
- ≤2 boundary violations (learning forgiveness)
- Reach ≥50 units rightward (~8% of substrate)
- Centroid position ≥100 units

**Expected Behavior:**
- Episodes increasing from ~50 to 100+ steps
- Consistent rightward movement (→)
- Reduced boundary violations
- First survival milestones triggered

---

### Stage 2: Distance Milestones (Episodes 201-500)
**Goal**: Reach progressive distance milestones (25%, 50%, 75%)

**Environment Settings:**
- Max nodes: 50 (standard complexity)
- Full action space unlocked
- Milestone tracking active

**Reward Multipliers:**
```yaml
survival_reward: 2.0x              # Still important
movement_reward: 1.5x              # Well-learned, reduced
centroid_movement_reward: 2.5x     # Group progress crucial!
milestone_rewards: 3.0x            # PRIMARY FOCUS
spawn_success_reward: 1.5x         # Efficient expansion
intensity_bonus: 2.0x              # Seek better substrate
```

**Success Criteria:**
- Survive ≥100 steps
- ≤1 boundary violation
- Reach ≥150 units (25% of substrate)
- Trigger 25% milestone at least once
- Aim for 50% milestone

**Expected Behavior:**
- Episodes lasting 100-300 steps
- 25% milestone (150 units) consistently reached
- 50% milestone (300 units) occasionally reached
- 75% milestone (450 units) rarely reached
- Reduced exploration accidents

---

### Stage 3: Goal Achievement (Episodes 501-1000)
**Goal**: Consistently reach the rightmost substrate (goal completion)

**Environment Settings:**
- Max nodes: 75 (maximum capacity)
- All constraints enabled
- Full optimization mode

**Reward Multipliers:**
```yaml
survival_reward: 1.5x              # Reduced (well-learned)
movement_reward: 1.0x              # Standard
centroid_movement_reward: 2.0x     # Group coordination
milestone_rewards: 1.5x            # Easier now
success_reward: 2.0x               # MAIN GOAL!
efficiency_bonus: 3.0x             # Optimize path
```

**Success Criteria:**
- Survive ≥200 steps
- 0 boundary violations (mastery)
- Reach ≥450 units (75%+ of substrate)
- 15% completion rate
- Trigger 90% milestone
- ≥70% efficiency

**Expected Behavior:**
- Episodes lasting 200-1000 steps
- Consistent 75%+ progress
- 90% milestone frequently triggered
- **Goal completion (rightmost reach) achieved!**
- Optimized migration paths

---

## 🎯 Milestone Reward System

### Stage 1 Milestones (Basic Achievements)
```yaml
first_survival_50_steps: +30       # First time surviving 50 steps
first_survival_100_steps: +50      # First time surviving 100 steps
first_rightward_50_units: +40      # First time reaching 50 units
boundary_violation_free: +40       # First episode without violations
```

### Stage 2 Milestones (Distance Progress)
```yaml
first_rightward_150_units: +60     # First time reaching 25% (150 units)
first_milestone_25_percent: +80    # First 25% milestone trigger
first_rightward_300_units: +100    # First time reaching 50% (300 units)
first_milestone_50_percent: +120   # First 50% milestone trigger
```

### Stage 3 Milestones (Goal Approach)
```yaml
first_rightward_450_units: +150    # First time reaching 75% (450 units)
first_milestone_75_percent: +180   # First 75% milestone trigger
first_milestone_90_percent: +250   # First 90% milestone trigger
task_completion: +300              # FIRST SUCCESSFUL MIGRATION!
```

**Total Possible Milestone Rewards: 1,200+**

---

## 🔄 Curriculum Progression System

### Automatic Stage Advancement
```yaml
auto_advance: true                 # Enable automatic progression
advancement_criteria: "mixed"      # Both success rate AND episodes
min_success_rate: 0.10            # Only 10% success needed (achievable)
evaluation_window: 100            # Evaluate over 100 episodes
allow_early_advance: true         # Can advance before episode_end
force_advance_at_end: true        # Must advance at episode_end
stage_overlap: 50                 # 50-episode gradual transition
```

### Advancement Logic:
1. **Episode-Based**: Automatically advance at episode thresholds (200, 500, 1000)
2. **Performance-Based**: Can advance early if 10% success rate achieved
3. **Forced Progression**: Won't get stuck in a stage forever

---

## 📊 Integration with Base Reward System

### Base Rewards (Active at All Times)
These rewards are ALWAYS active, then multiplied by curriculum stage multipliers:

```yaml
# Movement incentives
movement_reward: 2.0               # Strong base for rightward movement
leftward_penalty: 1.0              # Discourage wrong direction
centroid_movement_reward: 2.0      # Collective migration reward

# Survival incentives
survival_reward: 0.1               # Base per-step survival
survival_bonus_after_100: +0.05    # Additional after 100 steps
progressive_scaling: up to 2x      # Increases over time

# Milestone rewards (environment-level)
distance_25_percent: +25.0         # Each time 25% reached
distance_50_percent: +50.0         # Each time 50% reached
distance_75_percent: +100.0        # Each time 75% reached
distance_90_percent: +200.0        # Each time 90% reached

# Curriculum milestones (one-time bonuses)
first_milestone_X: varies          # See milestone table above

# Goal achievement
success_reward: 500.0              # Reaching rightmost substrate
```

### Total Reward Calculation
```
Total Reward = (Base Rewards × Curriculum Multipliers) + Milestone Bonuses
```

**Example (Stage 2, Episode 250):**
- Base movement reward: 2.0
- Stage 2 multiplier: 1.5×
- Effective movement reward: 3.0 per unit rightward
- Plus 50% milestone (+50) if triggered
- Plus first_milestone_50_percent (+120) if first time
- Plus survival rewards, centroid rewards, etc.

---

## 🎮 Training Expectations

### Stage 1 (Episodes 0-200): Foundation
- **Week 1**: Agent learns to survive 50+ steps
- **Week 2**: Consistent rightward movement
- **Week 3**: Reaching 50-100 units regularly
- **Advancement**: When reaching 100 units consistently (10% success)

### Stage 2 (Episodes 201-500): Exploration
- **Week 4**: 25% milestone (150 units) achieved
- **Week 5**: 50% milestone (300 units) occasional
- **Week 6-7**: 75% milestone (450 units) rare successes
- **Advancement**: When 25% milestone consistent (10% success)

### Stage 3 (Episodes 501-1000): Mastery
- **Week 8-10**: 75%+ progress common
- **Week 11-12**: 90% milestone frequent
- **Week 13+**: **GOAL ACHIEVED!** Rightmost substrate reached
- **Success**: 15%+ completion rate = mastery

---

## 🔧 Monitoring & Debugging

### Key Metrics to Watch

**Stage 1 Success Indicators:**
```
✅ Episode length: 50 → 100+ steps
✅ Centroid movement: mostly → (rightward)
✅ Boundary violations: decreasing
✅ Max X-position: reaching 50-100 units
```

**Stage 2 Success Indicators:**
```
✅ Episode length: 100 → 300+ steps
✅ Milestone messages: "25%" and "50%" appearing
✅ Max X-position: reaching 150-300 units
✅ Spawn efficiency: better substrate targeting
```

**Stage 3 Success Indicators:**
```
✅ Episode length: 200 → 1000 steps
✅ Milestone messages: "75%" and "90%" frequent
✅ Success messages: "Node X reached rightmost location - SUCCESS!"
✅ Completion rate: 15%+ episodes successful
```

### Terminal Output Examples

**Stage 1 Progress:**
```
🎓 Stage 1: Survival & Basic Movement (Episode 50/200)
🏆 Milestone Achieved: first_survival_50_steps (+30 reward)
📊 Ep 50: N=5 | Steps=52 | Centroid=65.3→ | Max_X=89.2
```

**Stage 2 Progress:**
```
🎓 Stage 2: Distance Milestones (Episode 300/500)
🎯 MILESTONE REACHED! 25% of substrate width! Reward: +25.0
🏆 Milestone Achieved: first_milestone_25_percent (+80 reward)
📊 Ep 300: N=12 | Steps=145 | Centroid=168.4→ | Max_X=187.6
```

**Stage 3 Success:**
```
🎓 Stage 3: Goal Achievement (Episode 750/1000)
🎯 MILESTONE REACHED! 90% of substrate width! Reward: +200.0
🎯 Episode terminated: Node 8 reached rightmost location (x=599.8 >= 599) - SUCCESS!
🏆 Milestone Achieved: task_completion (+300 reward)
📊 Ep 750: R=+842.5 | Steps=487 | SUCCESS=True
```

---

## 🚀 Quick Start

1. **Verify Configuration:**
   ```bash
   # Check curriculum is enabled in config.yaml
   grep "enable_curriculum: true" config.yaml
   ```

2. **Start Training:**
   ```bash
   conda activate durotaxis && python train.py
   ```

3. **Monitor Progress:**
   - Watch for stage announcements: `🎓 Stage X: ...`
   - Track milestone achievements: `🏆 Milestone Achieved: ...`
   - Look for success messages: `🎯 Episode terminated: ... - SUCCESS!`

4. **Expected Timeline:**
   - **Episodes 0-200**: Learn survival and basic movement
   - **Episodes 201-500**: Reach 25% and 50% milestones
   - **Episodes 501-1000**: Achieve goal completion (rightmost substrate)

---

## 📝 Summary

The integrated curriculum + reward system creates a **comprehensive learning pipeline**:

1. **Strong Foundation (Stage 1)**: 3x survival rewards + forgiving criteria = learn to stay alive
2. **Progressive Goals (Stage 2)**: 3x milestone rewards + 25%/50% targets = learn to go far
3. **Goal Mastery (Stage 3)**: 2x success rewards + efficiency focus = learn to complete

**This system transforms a difficult long-horizon task (migrate 600 units) into manageable progressive goals (survive → 50 units → 150 units → 300 units → 450 units → 600 units), with massive rewards at each milestone to maintain motivation!**

The agent should successfully reach the rightmost substrate by **Episode 500-800** with this integrated system! 🎯
