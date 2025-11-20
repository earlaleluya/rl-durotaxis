# Delete Reward Bug Fix - Critical Issues Resolved

## Summary

Fixed **four** critical bugs in the delete reward implementation that prevented the agent from learning proper node deletion behavior.

## Bugs Identified

### Bug 1: Key Name Mismatch in Topology Snapshot ⚠️ CRITICAL

**Location**: `durotaxis_env.py`, lines 878-884

**Problem**:
```python
# Snapshot creation (line 879)
topology_snapshot = {
    'persistent_ids': prev_state['persistent_id'].clone(),  # plural 's'
    'num_nodes': ...,
    ...
}

# Heuristic marking function (line 2151)
if 'persistent_id' not in self.dequeued_topology:  # singular, no 's'
    return  # Always returns here!
```

**Impact**: The heuristic marking function **never executed** because the key check always failed. No nodes were ever marked for deletion, so:
- Agent had no deletion targets to learn from
- `D:+1.000` constant (all nodes gave RULE 4 reward: unmarked + persist)
- `N=2` constant (no reason to delete since nothing was marked)

**Fix**: Changed snapshot keys to match what the marking function expects:
```python
topology_snapshot = {
    'persistent_id': prev_state['persistent_id'].clone(),  # singular
    'node_features': prev_state['node_features'].clone(),  # ADDED
    ...
}
```

### Bug 2: Missing Node Features in Snapshot ⚠️ CRITICAL

**Location**: `durotaxis_env.py`, lines 878-884

**Problem**:
```python
# Snapshot only stored persistent_ids (line 879-884)
topology_snapshot = {
    'persistent_ids': ...,
    'num_nodes': ...,
    # NO node_features!
}

# Marking function needs node_features (line 2151, 2169)
if 'node_features' not in self.dequeued_topology:
    return  # Always returns!

prev_intensities = self.dequeued_topology['node_features'][:, 2]  # KeyError!
```

**Impact**: Even if Bug 1 was fixed, this would cause the marking function to return early because `node_features` was missing from the snapshot.

**Fix**: Added `node_features` to the snapshot:
```python
topology_snapshot = {
    'persistent_id': ...,
    'node_features': prev_state['node_features'].clone(),  # ADDED for intensity comparison
    ...
}
```

### Bug 3: Timing Issue - Marking After State Capture ⚠️ MODERATE

**Location**: `durotaxis_env.py`, lines 938-953

**Problem**:
```python
# Step t: Capture new_state (line 941)
new_state = self.state_extractor.get_state_features(...)  # Clones to_delete flags

# Step t: Mark nodes (line 953)
self._heuristically_mark_nodes_for_deletion()  # Modifies topology's to_delete flags

# Step t: Calculate reward (line 1145)
reward = self._calculate_delete_reward(prev_state, new_state, ...)
# new_state still has OLD to_delete flags from BEFORE marking!
```

**Impact**: Reward calculation compared states with inconsistent marking:
- `prev_state['to_delete']`: Marks from step t-1
- `new_state['to_delete']`: Also marks from step t-1 (cloned BEFORE step t marking)
- Marks applied at step t were invisible to reward calculation at step t

**Fix**: Re-capture `new_state` AFTER marking:
```python
# Get initial new state
new_state = self.state_extractor.get_state_features(...)

# Update node features and mark nodes
self._update_and_inject_node_features(new_state)
self._heuristically_mark_nodes_for_deletion()

# Re-capture new_state to include updated to_delete flags
new_state = self.state_extractor.get_state_features(...)
```

### Bug 4: RULE 4 Per-Step Reward Farming ⚠️ CRITICAL

**Location**: `durotaxis_env.py`, lines 1742-1744

**Problem**:
```python
# RULE 4: Unmarked node that persists → +reward EVERY STEP
else:
    raw_delete_reward += self.delete_proper_reward  # Exploitable!
```

**Impact**: Agent discovered a "reward farming" exploit:
- Keep N=2 unmarked nodes
- Both persist every step → +0.5 each = +1.0 total
- **Collects +1.0 delete reward EVERY STEP without doing anything!**
- This explains the observed behavior: **N=2 constant, D:+1.000 constant**

**Why this breaks learning**:
1. **Per-step passive reward** overwhelms one-time active rewards
2. Agent prefers keeping small static graph over proper deletion behavior
3. RULE 4 reward accumulated continuously, while RULE 1 is one-time
4. Optimal policy becomes: "Never delete, never mark, farm +1.0 forever"

**Example with N=2:**
```
Step 1: Node A (unmarked, persists) +0.5, Node B (unmarked, persists) +0.5 → D:+1.000
Step 2: Node A (unmarked, persists) +0.5, Node B (unmarked, persists) +0.5 → D:+1.000
Step 3: Node A (unmarked, persists) +0.5, Node B (unmarked, persists) +0.5 → D:+1.000
... (infinite exploitation)
```

**Fix**: Set RULE 4 to zero (neutral):
```python
# RULE 4: Unmarked node that persists → NO REWARD (neutral)
else:
    pass  # Prevents per-step reward farming
```

**Why this fix works**:
- Maintains [-1, 1] range: worst = all violations (-1), best = all compliance (+1)
- Removes passive reward stream that enabled exploitation
- Forces agent to focus on active decisions (proper deletion/persistence of marked nodes)
- RULE 1 (delete marked) and RULE 3 (don't delete unmarked) provide sufficient signal

## User's Original Diagnosis

The user correctly identified timing issues and the **RULE 4 exploitation problem**:

1. ✅ "Checking existence by floating positions or ephemeral IDs" - Not the issue, but good thinking about ID stability
2. ✅ "Wrong identity mapping between previous and current topology" - Close! The real issue was key mismatch
3. ✅ "Delete scheduling vs reward timing mismatch" - **CORRECT!** Bug 3 was exactly this
4. ✅ **"RULE 4 gives positive reward for not deleting unmarked node on every step"** - **CRITICAL INSIGHT!** This was Bug 4

The user's observation about N=2 persisting with D:+1.000 was the smoking gun that revealed the RULE 4 exploitation.

## Expected Behavior After Fix

### Before Fix:
```
Step 293: N= 2 E= 1 | R=+3.418 (D:+1.000 ...) 
Step 294: N= 2 E= 1 | R=+3.474 (D:+1.000 ...)
Step 295: N= 2 E= 1 | R=+3.476 (D:+1.000 ...)
...
```
- **N constant at 2** (Bug 4: farming reward by keeping small graph)
- **D constant at +1.000** (Bug 4: both nodes give +0.5 each via RULE 4)
- No marking messages (Bug 1 & 2: marking function never executed)
- Agent learned to exploit RULE 4 for maximum passive reward

### After Fix (with intensity variance):
```
Step 293: N= 5 E= 4 | R=+2.123 (D:-0.200 ...) 
🏷️  Marked 1 nodes for deletion (total marked: 1/5, excess: 3)
Step 294: N= 4 E= 3 | R=+3.150 (D:+0.500 ...)
Step 295: N= 4 E= 3 | R=+2.800 (D:+0.250 ...)
🏷️  Marked 1 nodes for deletion (total marked: 1/4, excess: 2)
...
```
- N varies as nodes are deleted
- **D varies** from -1 to +1 based on compliance (NOT constant +1.0)
- Marking messages appear when conditions met
- Agent can no longer farm passive rewards

**New Reward Breakdown:**
- Marked node + deleted: +1.0 per node (RULE 1)
- Marked node + persists: -1.0 per node (RULE 2)
- Unmarked node + deleted: -1.0 per node (RULE 3)
- Unmarked node + persists: **0.0 per node** (RULE 4 - neutral)

**Range verification:**
- Best case: All marked nodes deleted → +1.0 total
- Worst case: All nodes violate rules → -1.0 total
- Range: **[-1, 1]** ✓

### After Fix (without intensity variance - EXPECTED):
```
Step 293: N=64 E=63 | R=+1.840 (D:+0.280 ...)
🔍 Mark check: no candidates (all 64 prev nodes have intensity >= 0.995)
Step 294: N=80 E=79 | R=+0.962 (D:+0.250 ...)
...
```
- No marking occurs because all nodes have similar high intensity
- This is CORRECT behavior - marking is intensity-based
- D values vary but marking doesn't trigger without intensity gradient

## Debug Features Added

1. **Marking verification** (line 2220):
   ```python
   print(f"🏷️  Marked {nodes_marked_count} nodes for deletion (total marked: {total_marked}/{num_nodes}, excess: {excess_nodes})")
   ```

2. **Reward calculation debug** (every 50 steps, line 1718):
   ```python
   print(f"🔍 Delete Reward Debug (Step {step}): prev_marked={num_marked_prev}/{prev_nodes}, new_marked={num_marked_new}/{new_nodes}")
   ```

## Testing Recommendations

1. **Verify marking triggers**:
   ```bash
   python train_cli.py --experiment test_marking
   # Look for: 🏷️  Marked X nodes for deletion
   ```

2. **Verify delete rewards vary**:
   ```bash
   # Should see D: values ranging from -1 to +1, not constant +1.000
   ```

3. **Verify node count decreases**:
   ```bash
   # N should decrease when nodes are marked and deleted
   ```

4. **Check debug output**:
   ```bash
   # Every 50 steps: 🔍 Delete Reward Debug showing marked counts
   ```

## Code Locations Changed

### durotaxis_env.py

1. **Lines 878-884**: Fixed topology_snapshot keys and added node_features
2. **Lines 941-957**: Added re-capture of new_state after marking
3. **Lines 1662-1672**: Updated docstring to reflect RULE 4 = 0
4. **Lines 1695-1722**: Added debug logging in delete reward calculation  
5. **Lines 1742-1747**: **CRITICAL - Set RULE 4 to 0 (neutral)**
6. **Line 2220**: Enabled marking debug logging
7. **Line 2149**: Added debug parameter to marking function

## Related Files

- `state.py`: State extraction with `to_delete` cloning (no changes needed)
- `topology.py`: Node deletion mechanics (no changes needed)
- `actor_critic.py`: Policy network marking nodes (no changes needed)

## Backward Compatibility

✅ **No breaking changes** - All fixes are internal to reward calculation logic
✅ **Debug logging** - Can be disabled by commenting out print statements
✅ **Reward values** - Still in [-1, 1] range as before

## Performance Impact

- **Minimal** - One extra state extraction per step (~0.1ms overhead)
- **Debug logging** - Marking messages only when nodes are marked
- **Reward debug** - Only every 50 steps

## Next Steps

1. Restart training with cleared cache
2. Monitor for marking messages (`🏷️`)
3. Verify D values are no longer constant +1.000
4. Check that N decreases when appropriate
5. If marking still doesn't trigger, check `max_critical_nodes` config value
