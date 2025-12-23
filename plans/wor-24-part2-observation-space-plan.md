# WOR-24 Part 2: Observation Space Optimization Plan

## Summary of Changes

1. **Normalize player state** from int32 to float32 [0, 1]
2. **Remove turn and score** from observation (not actionable for agent)
3. **Merge flags into player state** - add showActivated and scheduledTasksDisabled
4. **Add missing scheduledTasksDisabled** to observation (Calm program effect)
5. **Add program inventory** as explicit 26-dimensional binary vector
6. **Optimize grid encoding** - one-hot encode enemy/block types, normalize all values to [0, 1]
7. **Player state size**: 10 → 10 values (remove turn + score, add 2 flags)
8. **Grid features**: Same 20 features but one-hot encoded and normalized

## Current vs Proposed Observation Space

### Current (Before)
```python
{
    "player": Box(shape=(10,), dtype=int32)
        [row, col, hp, credits, energy, stage, turn, siphons, attack, score]
        ranges: [0-5, 0-5, 0-3, 0-999, 0-999, 1-8, 0-9999, 0-99, 1-2, 0-9999]

    "grid": Box(shape=(6, 6, 20), dtype=int32)

    "flags": Box(shape=(1,), dtype=int32)
        [showActivated]
        # scheduledTasksDisabled missing!
}
```

### Proposed (After)
```python
{
    "player": Box(shape=(10,), dtype=float32, low=0.0, high=1.0)
        [row, col, hp, credits, energy, stage, siphons, attack,
         showActivated, scheduledTasksDisabled]
        # Turn removed - agent can infer from temporal patterns
        # Score removed - not actionable, reward signal sufficient
        # Flags merged in - cleaner structure
        # scheduledTasksDisabled added - Calm program feedback

    "programs": Box(shape=(26,), dtype=int32, low=0, high=1)
        # Binary vector of owned programs

    "grid": Box(shape=(6, 6, 20), dtype=float32, low=0.0, high=1.0)
        # One-hot encoded enemy/block types
        # All values normalized to [0, 1]
        # transmission_spawn renamed to transmission_spawncount

    # "flags" removed - merged into player state
}
```

## Detailed Changes

### 1. Player State Normalization

**Old parsing:**
```python
player = np.array([
    obs_dict["playerRow"],
    obs_dict["playerCol"],
    obs_dict["playerHP"],
    obs_dict["credits"],
    obs_dict["energy"],
    obs_dict["stage"],
    obs_dict["turn"],
    obs_dict["dataSiphons"],
    obs_dict["baseAttack"],
    obs_dict["score"]
], dtype=np.int32)
```

**New parsing:**
```python
player = np.array([
    obs_dict["playerRow"] / 5.0,                        # 0-5 → 0-1
    obs_dict["playerCol"] / 5.0,                        # 0-5 → 0-1
    obs_dict["playerHP"] / 3.0,                         # 0-3 → 0-1
    min(obs_dict["credits"] / 100.0, 1.0),              # 0-100+ → 0-1 (capped)
    min(obs_dict["energy"] / 100.0, 1.0),               # 0-100+ → 0-1 (capped)
    (obs_dict["stage"] - 1) / 7.0,                      # 1-8 → 0-1
    obs_dict["dataSiphons"] / 10.0,                     # 0-10 → 0-1
    (obs_dict["baseAttack"] - 1) / 1.0,                 # 1-2 → 0-1
    1.0 if obs_dict["showActivated"] else 0.0,          # Binary flag
    1.0 if obs_dict["scheduledTasksDisabled"] else 0.0  # Binary flag
], dtype=np.float32)
```

**Rationale for each normalization:**
- **row/col**: Divide by max (5) - precise position encoding
- **hp**: Divide by max (3) - full range
- **credits/energy**: Cap at 100 - rarely exceed this, better resolution
- **stage**: Subtract 1 (range 1-8 → 0-7), divide by 7 - precise stage encoding
- **siphons**: Divide by 10 - reasonable max
- **attack**: Subtract 1 (range 1-2 → 0-1) - binary effectively
- **showActivated**: Binary 0/1 - Show program effect
- **scheduledTasksDisabled**: Binary 0/1 - Calm program effect (was missing!)

**Removed fields:**
- **turn**: Agent can infer temporal patterns from observation sequences
- **score**: Not actionable - reward signal already captures score deltas

### 2. Program Inventory

**Add to observation:**
```python
programs = np.zeros(26, dtype=np.int32)
if "ownedPrograms" in obs_dict:
    for action_idx in obs_dict["ownedPrograms"]:
        # Action indices 5-30 are programs
        # Map to array indices 0-25
        programs[action_idx - 5] = 1
```

**Benefits:**
- Agent directly sees available programs
- No need to infer from action mask
- Clearer strategic decision-making

**Example:**
If player owns programs at action indices [5, 7, 12, 18]:
```python
programs = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, ...]
#          ^     ^              ^              ^
#          5     7              12             18
```

### 3. Grid Encoding Optimization

**Old encoding (per cell, 20 features, int32):**
```python
[
    enemy_type,            # 0-4 (ordinal encoding)
    enemy_hp,              # 0-3
    enemy_stunned,         # 0/1
    block_type,            # 0-3 (ordinal encoding)
    block_points,          # 0-9
    block_siphoned,        # 0/1
    program_action_index,  # 0-30
    transmission_spawn,    # 0-9
    transmission_turns,    # 0-9
    credits,               # 0-9
    energy,                # 0-9
    is_data_siphon,        # 0/1
    is_exit,               # 0/1
    0, 0, 0, 0, 0, 0, 0   # 7 padding zeros
]
```

**New encoding (per cell, 20 features, float32 [0, 1]):**
```python
[
    # Enemy (6 features) - one-hot type encoding
    is_virus,              # 1.0 if virus present, else 0.0
    is_daemon,             # 1.0 if daemon present, else 0.0
    is_glitch,             # 1.0 if glitch present, else 0.0
    is_cryptog,            # 1.0 if cryptog present, else 0.0
    enemy_hp / 3.0,        # 0-3 → 0-1
    enemy_stunned,         # 0.0 or 1.0

    # Block (5 features) - one-hot type encoding
    is_data_block,         # 1.0 if data block, else 0.0
    is_program_block,      # 1.0 if program block, else 0.0
    is_question_block,     # 1.0 if question block, else 0.0
    block_points / 9.0,    # 0-9 → 0-1
    block_siphoned,        # 0.0 or 1.0

    # Program/Transmission (3 features)
    program_action_index / 30.0,    # 0-30 → 0-1
    transmission_spawncount / 9.0,  # 0-9 → 0-1 (renamed from transmission_spawn)
    transmission_turns / 9.0,       # 0-9 → 0-1

    # Resources (2 features)
    credits / 9.0,         # 0-9 → 0-1
    energy / 9.0,          # 0-9 → 0-1

    # Special cells (2 features)
    is_data_siphon,        # 0.0 or 1.0
    is_exit,               # 0.0 or 1.0

    # Future expansion (2 features)
    0.0, 0.0
]
```

**Key changes:**
- **One-hot encoding**: Enemy and block types now use binary indicators instead of ordinal numbers
- **Normalization**: All values in [0, 1] range for consistent neural network input
- **Reduced padding**: 7 → 2 unused features
- **Clearer naming**: `transmission_spawn` → `transmission_spawncount`
- **Type change**: int32 → float32

**Benefits:**
- Neural network doesn't assume false ordinal relationships (e.g., "daemon=1 is between virus=0 and glitch=2")
- Consistent scaling across all features
- Each enemy/block type gets its own "neuron"
- Better gradient flow during training

## Files to Modify

### 1. `python/hack_env.py`

**Line ~50-76: Update observation space definition**
```python
self.observation_space = spaces.Dict({
    "player": spaces.Box(
        low=0.0,
        high=1.0,
        shape=(11,),  # Changed from 10: removed turn, added 2 flags
        dtype=np.float32  # Changed from int32
    ),

    "programs": spaces.Box(  # NEW
        low=0,
        high=1,
        shape=(26,),
        dtype=np.int32
    ),

    "grid": spaces.Box(
        low=0, high=999,
        shape=(6, 6, 20),
        dtype=np.int32
    )

    # "flags" removed - merged into player state
})
```

**Line ~137-220: Update _observation_to_array()**
- Remove turn from player array
- Add showActivated and scheduledTasksDisabled to player array
- Apply normalization formulas to all values
- Add programs parsing
- Remove flags from return dict (merged into player)
- Add "programs" to return dict

### 2. `HackMatrix/HeadlessGame.swift` or `HackMatrix/HeadlessGameCLI.swift`

Need to find where observation JSON is built and add:
```swift
"ownedPrograms": gameState.ownedPrograms.map { $0.actionIndex },
"scheduledTasksDisabled": gameState.scheduledTasksDisabled
```

**Changes needed:**
- Add `ownedPrograms` array (action indices 5-30)
- Add `scheduledTasksDisabled` boolean (Calm program effect)
- Verify `showActivated` is already sent (it should be)

**Search strategy:**
- Grep for "buildObservation" or "observation" in Swift files
- Look for where JSON is constructed with playerRow, playerCol, etc.
- Add ownedPrograms array to that JSON structure

## Testing Plan

### 1. Verify observation structure
```python
import hack_env
import numpy as np

env = hack_env.HackEnv()
obs, info = env.reset()

# Check shapes
assert obs['player'].shape == (9,), f"Expected (9,), got {obs['player'].shape}"
assert obs['player'].dtype == np.float32, f"Expected float32, got {obs['player'].dtype}"
assert obs['programs'].shape == (26,), f"Expected (26,), got {obs['programs'].shape}"
assert obs['programs'].dtype == np.int32, f"Expected int32, got {obs['programs'].dtype}"

# Check ranges
assert np.all(obs['player'] >= 0.0) and np.all(obs['player'] <= 1.0), "Player values out of [0,1] range"
assert np.all((obs['programs'] == 0) | (obs['programs'] == 1)), "Programs should be binary"

print('✓ Observation space validation passed')
env.close()
```

### 2. Run environment test
```bash
cd python
source venv/bin/activate
python test_env.py
```

### 3. Quick training test
```bash
# Verify MaskablePPO accepts new observation space
python train_maskable_ppo.py --total-timesteps 1000
```

### 4. Inspect sample observations
```python
env = hack_env.HackEnv()
obs, _ = env.reset()

print("Player state:", obs['player'])
print("Programs owned:", np.where(obs['programs'] == 1)[0] + 5)  # Convert back to action indices
```

## Breaking Changes

⚠️ **This invalidates all existing model checkpoints**

**Reason:** Observation space shape and dtype changed
- Old: `player` int32 with variable ranges, 10 values
- New: `player` float32 [0, 1], 9 values, plus new `programs` component

**Impact:**
- Cannot load old model checkpoints
- Must retrain from scratch
- Previous training runs/logs are still valid for comparison

**Migration:** No migration path - start fresh training

## Success Criteria

- [ ] Player state normalized to [0, 1] with 9 values
- [ ] Turn removed from observation
- [ ] Programs added as 26-dim binary vector
- [ ] Swift sends ownedPrograms in JSON
- [ ] Python correctly parses all fields
- [ ] All values in expected ranges
- [ ] Environment test passes
- [ ] Training script runs without errors
- [ ] No runtime exceptions during episode rollout

## Rollback Plan

If issues arise:
1. `git checkout main` to revert changes
2. Old observation space still works with existing code
3. No data loss - only code changes
4. Can iterate on branch without affecting main
