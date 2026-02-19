# Reward System Improvements: Resource Rewards, HP Penalties, and Program Usage Optimization

## User Requirements

### 1. Resource Acquisition & Holding Rewards
- **Reward resource gains** (credits/energy acquired)
- **Reward resource holding** (small continuous bonus for maintaining reserves)
- **Discourage suboptimal siphon usage** by comparing actual siphon yield against BEST possible position on entire board using **strict dominance**

### 2. HP Damage Punishment with Recovery Offset
- **Penalize taking damage** with negative reward
- **Reward HP recovery** (RESET program or stage completion) with full offset of previous penalty
- Net effect: Taking damage then healing = neutral, taking damage without healing = penalty remains

### 3. Program Usage Optimization
- **Penalize wasteful RESET usage**: -0.3 penalty when RESET used at 2 HP (none at 1 HP)
- Potentially extend to other suboptimal program patterns in future

### 4. Siphon-Caused Death Penalty
- **Extra penalty for dying to siphon-spawned enemies**: If player dies to an enemy that was spawned from siphoning a block, apply additional death penalty
- Rationale: Siphoning is player's choice - if that choice leads to death, it's almost always a bad decision
- Need to track which enemies were spawned from siphoning vs naturally spawned

---

## Current Reward System Analysis

### Existing Components (from RewardCalculator.swift)

| Component | Value | Purpose |
|-----------|-------|---------|
| Stage completion | `[1, 2, 4, 8, 16, 32, 64, 100]` | Exponential rewards for advancing stages |
| Score gain | `(score_delta) × 0.5` | Rewards collecting data blocks |
| Kills | `(kill_count) × 0.3` | Rewards defeating enemies |
| Data siphon collected | `1.0` | Flat bonus for collecting siphon resource |
| Distance shaping | `(distance_delta) × 0.05` | Guides toward exit |
| Victory | `500.0 + (score × 100.0)` | Massive win bonus |
| Death penalty | `-0.5 × cumulative_stage_rewards` | Scales with progress |

### Key Architectural Facts

1. **Rewards are calculated in Swift** (`RewardCalculator.swift`)
2. **Python receives single reward value** + detailed breakdown in `info["reward_breakdown"]`
3. **All reward constants are hard-coded** in RewardCalculator.swift
4. **Siphon mechanics**: Auto-siphons 5-cell cross pattern (no choice of which cells)
5. **Player position is chosen** before siphoning, so optimality = choosing best position to siphon from
6. **HP tracking**: Player HP is in observations, damage/recovery events are tracked
7. **Enemy spawning**: Siphoning blocks spawns enemies (Transmissions)
8. **Data blocks have values**: Each data block has a specific point value (number on it)

---

## Implementation Design

### Phase 1: Add New Reward Components to RewardCalculator

**File**: `HackMatrix/RewardCalculator.swift`

#### New Reward Components

```swift
// Add to RewardBreakdown struct (line 9):
struct RewardBreakdown {
    // ... existing fields ...

    // NEW: Resource rewards
    let resourceGain: Double        // Reward for acquiring credits/energy
    let resourceHolding: Double     // Small bonus for maintaining reserves

    // NEW: HP penalties and recovery
    let damagePenalty: Double       // Negative reward for taking damage
    let hpRecovery: Double          // Positive reward for healing (offsets penalty)

    // NEW: Siphon optimization
    let siphonQuality: Double       // Penalty if siphon was suboptimal

    // NEW: Program usage optimization
    let programWaste: Double        // Penalty for wasteful program usage (e.g., RESET at 2 HP)

    // NEW: Siphon-caused death
    let siphonDeathPenalty: Double  // Extra penalty for dying to siphon-spawned enemy

    var total: Double {
        return stage + scoreGain + kills + dataSiphonCollected +
               distanceShaping + victory + death +
               resourceGain + resourceHolding +  // NEW
               damagePenalty + hpRecovery +      // NEW
               siphonQuality + programWaste +    // NEW
               siphonDeathPenalty                // NEW
    }
}
```

#### New Reward Constants

```swift
// Add to top of RewardCalculator (around line 30):

// MARK: Resource Reward Constants
private static let creditGainMultiplier: Double = 0.05      // Reward per credit gained
private static let energyGainMultiplier: Double = 0.05      // Reward per energy gained
private static let creditHoldingMultiplier: Double = 0.001  // Small bonus per credit held
private static let energyHoldingMultiplier: Double = 0.001  // Small bonus per energy held

// MARK: HP Penalty Constants
private static let damagePenaltyPerHP: Double = -1.0        // Penalty per HP lost
private static let hpRecoveryRewardPerHP: Double = 1.0      // Reward per HP recovered (full offset)

// MARK: Siphon Optimization Constants
private static let siphonSuboptimalPenalty: Double = -0.5   // Penalty if strictly better option existed

// MARK: Program Waste Constants
private static let resetAt2HPPenalty: Double = -0.3         // Penalty for RESET at 2 HP

// MARK: Siphon Death Constants
private static let siphonCausedDeathPenalty: Double = -10.0 // Extra penalty for dying to siphon-spawned enemy
```

---

### Phase 2: Resource Acquisition & Holding

#### Calculate Resource Gain Reward

```swift
// In calculate() method, add after line 50:

// MARK: Resource Gain Reward
let creditsGained = currentCredits - oldCredits
let energyGained = currentEnergy - oldEnergy
let resourceGain = Double(creditsGained) * creditGainMultiplier +
                   Double(energyGained) * energyGainMultiplier
```

#### Calculate Resource Holding Reward

```swift
// MARK: Resource Holding Reward
let resourceHolding = Double(currentCredits) * creditHoldingMultiplier +
                      Double(currentEnergy) * energyHoldingMultiplier
```

#### Required Changes to GameState.swift

**Current situation**: `GameState.tryExecuteAction()` doesn't pass old credits/energy to reward calculator

**Changes needed** (around line 1993):

```swift
// BEFORE:
let rewardBreakdown = RewardCalculator.calculate(
    oldScore: oldScore,
    currentScore: currentScore,
    // ... other params
)

// AFTER:
let rewardBreakdown = RewardCalculator.calculate(
    oldScore: oldScore,
    currentScore: currentScore,
    oldCredits: oldCredits,        // NEW
    currentCredits: player.credits, // NEW
    oldEnergy: oldEnergy,          // NEW
    currentEnergy: player.energy,   // NEW
    // ... other params
)
```

**Add tracking** (around line 1954):

```swift
// Capture old values before action
let oldScore = player.score
let oldCredits = player.credits  // NEW
let oldEnergy = player.energy    // NEW
```

---

### Phase 3: HP Damage Penalty & Recovery

#### Track HP Changes

**In GameState.swift** (around line 1954):

```swift
let oldHP = player.health.rawValue  // NEW: Track HP before action
```

#### Calculate Damage Penalty

**In RewardCalculator.swift**:

```swift
// MARK: Damage Penalty
let hpLost = oldHP - currentHP
let damagePenalty = hpLost > 0 ? Double(hpLost) * damagePenaltyPerHP : 0.0
```

#### Calculate Recovery Reward

```swift
// MARK: HP Recovery Reward
let hpGained = currentHP - oldHP
let hpRecovery = hpGained > 0 ? Double(hpGained) * hpRecoveryRewardPerHP : 0.0
```

#### Required Parameter Changes

**Update RewardCalculator.calculate() signature** (line ~35):

```swift
static func calculate(
    // ... existing params ...
    oldHP: Int,             // NEW
    currentHP: Int,         // NEW
    // ... rest of params
) -> RewardBreakdown
```

**Update call site in GameState.swift** (line ~1993):

```swift
let rewardBreakdown = RewardCalculator.calculate(
    // ... existing args ...
    oldHP: oldHP,                       // NEW
    currentHP: player.health.rawValue,  // NEW
    // ... rest of args
)
```

---

### Phase 4: Siphon Quality - Strict Dominance Check

**IMPORTANT**: Position A is only "strictly better" than position B if ALL of these are true:
- `creditsA >= creditsB`
- `energyA >= energyB`
- Same exact set of data block values (e.g., [1, 3, 5] must match exactly)
- Same exact set of programs (e.g., {PUSH, RESET} must match exactly)

**Having MORE or DIFFERENT blocks is NOT better** because:
- Different data blocks have different point values
- Different programs have different utility
- More blocks spawn more enemies (more risk)

We only penalize if there was a position with equal or more resources AND the exact same blocks.

#### Implementation Approach

**Add helper method to GameState.swift** (around line 600, before `performSiphon()`):

```swift
/// Calculate total resource value that would be obtained by siphoning from given position
/// - Returns: (credits, energy, dataBlockValues, programs) tuple
func calculateSiphonYieldAt(row: Int, col: Int) -> (credits: Int, energy: Int, dataBlockValues: [Int], programs: Set<ProgramType>) {
    // OPTIMIZATION: Can't siphon if standing on a block (invalid position)
    if grid[row, col].block != nil {
        return (0, 0, [], [])
    }

    let siphonCells = grid.getSiphonCells(centerRow: row, centerCol: col)

    var totalCredits = 0
    var totalEnergy = 0
    var dataBlockValues: [Int] = []  // Track specific point values
    var programs: Set<ProgramType> = []

    for cell in siphonCells {
        // Skip already-siphoned cells
        guard !cell.isSiphoned else { continue }

        // Count blocks with specific values
        if let block = cell.block {
            switch block.type {
            case .data:
                dataBlockValues.append(block.value)  // Track specific point value
            case .program:
                // Only count if we don't already own this program
                if !ownedPrograms.contains(block.programType!) {
                    programs.insert(block.programType!)
                }
            case .question:
                // Unknown value - treat as incomparable
                dataBlockValues.append(-1)  // Sentinel for unknown block
            }
        } else {
            // No block - count resources
            totalCredits += cell.credits
            totalEnergy += cell.energy
        }
    }

    // Sort for consistent comparison
    dataBlockValues.sort()

    return (totalCredits, totalEnergy, dataBlockValues, programs)
}
```

#### Find Strictly Dominating Siphon Position (with Optimizations)

**Add method to check for strictly better positions**:

```swift
/// Check if there exists a strictly better siphon position than current position
/// A position is strictly better if it has >= credits, >= energy, and EXACT same blocks
/// - Returns: (exists, missedCredits, missedEnergy) tuple
func checkForBetterSiphonPosition() -> (exists: Bool, missedCredits: Int, missedEnergy: Int) {
    let currentYield = calculateSiphonYieldAt(row: player.row, col: player.col)

    // OPTIMIZATION: Determine which positions to check
    var positionsToCheck: [(Int, Int)] = []

    if !currentYield.dataBlockValues.isEmpty || !currentYield.programs.isEmpty {
        // Current yield has blocks - only check positions that would siphon SAME blocks
        let currentSiphonCells = grid.getSiphonCells(centerRow: player.row, centerCol: player.col)
        var blockPositions: [(Int, Int)] = []

        for cell in currentSiphonCells {
            if cell.block != nil && !cell.isSiphoned {
                blockPositions.append((cell.row, cell.col))
            }
        }

        if !blockPositions.isEmpty {
            // Find positions that could siphon ALL these blocks
            var candidateSets: [Set<(Int, Int)>] = []

            for (blockRow, blockCol) in blockPositions {
                var candidates: Set<(Int, Int)> = []

                // Positions that could have this block in their siphon pattern (cross pattern)
                let possibleCenters = [
                    (blockRow-1, blockCol),  // Block is below center
                    (blockRow+1, blockCol),  // Block is above center
                    (blockRow, blockCol-1),  // Block is to right of center
                    (blockRow, blockCol+1),  // Block is to left of center
                    (blockRow, blockCol)     // Block is at center
                ]

                for (centerRow, centerCol) in possibleCenters {
                    if centerRow >= 0 && centerRow < 6 && centerCol >= 0 && centerCol < 6 {
                        if grid[centerRow, centerCol].block == nil {  // Can't stand on block
                            candidates.insert((centerRow, centerCol))
                        }
                    }
                }

                candidateSets.append(candidates)
            }

            // Intersection: positions that can siphon ALL the blocks
            if !candidateSets.isEmpty {
                var intersection = candidateSets[0]
                for candidateSet in candidateSets.dropFirst() {
                    intersection = intersection.intersection(candidateSet)
                }

                positionsToCheck = Array(intersection)
            }
        }
    } else {
        // No blocks in current yield - check all valid positions
        for row in 0..<6 {
            for col in 0..<6 {
                if grid[row, col].block == nil {  // Can't siphon from block position
                    positionsToCheck.append((row, col))
                }
            }
        }
    }

    var bestCredits = currentYield.credits
    var bestEnergy = currentYield.energy
    var foundBetter = false

    // Check each candidate position for strict dominance
    for (row, col) in positionsToCheck {
        let yield = calculateSiphonYieldAt(row: row, col: col)

        // Check for strict dominance:
        // 1. EXACT same data block values (sorted arrays)
        // 2. EXACT same program set
        // 3. >= credits AND >= energy
        // 4. At least one resource is strictly greater
        if yield.dataBlockValues == currentYield.dataBlockValues &&  // Exact same data blocks
           yield.programs == currentYield.programs {                  // Exact same programs

            let betterCredits = yield.credits >= currentYield.credits
            let betterEnergy = yield.energy >= currentYield.energy
            let strictlyBetter = (yield.credits > currentYield.credits) ||
                               (yield.energy > currentYield.energy)

            if betterCredits && betterEnergy && strictlyBetter {
                foundBetter = true
                bestCredits = max(bestCredits, yield.credits)
                bestEnergy = max(bestEnergy, yield.energy)
            }
        }
    }

    let missedCredits = bestCredits - currentYield.credits
    let missedEnergy = bestEnergy - currentYield.energy

    return (foundBetter, missedCredits, missedEnergy)
}
```

#### Track Siphon Quality in ActionResult

**Modify GameState.performSiphon()** (around line 609):

```swift
func performSiphon() -> (
    success: Bool,
    blocksSiphoned: Int,
    programsAcquired: Int,
    creditsGained: Int,
    energyGained: Int,
    // NEW: Add siphon quality info
    wasOptimal: Bool,
    missedCredits: Int,
    missedEnergy: Int
) {
    // ... existing siphon logic ...

    // NEW: Check if there was a strictly better position
    let (betterExists, missedCredits, missedEnergy) = checkForBetterSiphonPosition()
    let wasOptimal = !betterExists

    return (
        success: true,
        blocksSiphoned: totalBlocksSiphoned,
        programsAcquired: newProgramsCount,
        creditsGained: totalCredits,
        energyGained: totalEnergy,
        wasOptimal: wasOptimal,        // NEW
        missedCredits: missedCredits,  // NEW
        missedEnergy: missedEnergy     // NEW
    )
}
```

#### Calculate Siphon Quality Reward

**In RewardCalculator.swift**:

```swift
// MARK: Siphon Quality
let siphonQuality: Double
if siphonWasUsed {
    if siphonWasOptimal {
        siphonQuality = 0.0  // No penalty for optimal choice
    } else {
        // Penalty proportional to missed resources
        let missedValue = Double(siphonMissedCredits) * creditGainMultiplier +
                         Double(siphonMissedEnergy) * energyGainMultiplier
        siphonQuality = -siphonSuboptimalPenalty * missedValue
    }
} else {
    siphonQuality = 0.0  // No siphon used, no penalty
}
```

#### Required Parameter Changes

**Update RewardCalculator.calculate() signature**:

```swift
static func calculate(
    // ... existing params ...
    siphonWasUsed: Bool,          // NEW
    siphonWasOptimal: Bool,       // NEW
    siphonMissedCredits: Int,     // NEW
    siphonMissedEnergy: Int,      // NEW
    // ... rest of params
) -> RewardBreakdown
```

**Update call site in GameState.swift**:

```swift
// Track siphon quality before action (default to no siphon)
var siphonWasUsed = false
var siphonWasOptimal = true
var siphonMissedCredits = 0
var siphonMissedEnergy = 0

// If action is siphon, capture quality metrics
if case .siphon = action {
    let siphonResult = performSiphon()
    siphonWasUsed = true
    siphonWasOptimal = siphonResult.wasOptimal
    siphonMissedCredits = siphonResult.missedCredits
    siphonMissedEnergy = siphonResult.missedEnergy
}

let rewardBreakdown = RewardCalculator.calculate(
    // ... existing args ...
    siphonWasUsed: siphonWasUsed,             // NEW
    siphonWasOptimal: siphonWasOptimal,       // NEW
    siphonMissedCredits: siphonMissedCredits, // NEW
    siphonMissedEnergy: siphonMissedEnergy,   // NEW
    // ... rest of args
)
```

---

### Phase 5: Program Waste Detection (RESET at 2 HP)

#### Detect Wasteful RESET Usage

**In GameState.swift**, around where programs are executed (line ~1169):

```swift
// Track if RESET was used wastefully
var resetWasWasteful = false

if case .program(.reset) = action {
    if oldHP == 2 {  // Using RESET at 2 HP is wasteful
        resetWasWasteful = true
    }
}
```

#### Calculate Program Waste Penalty

**In RewardCalculator.swift**:

```swift
// MARK: Program Waste Penalty
let programWaste = resetWasWasteful ? resetAt2HPPenalty : 0.0
```

#### Required Parameter Changes

**Update RewardCalculator.calculate() signature**:

```swift
static func calculate(
    // ... existing params ...
    resetWasWasteful: Bool,  // NEW
    // ... rest of params
) -> RewardBreakdown
```

**Update call site in GameState.swift**:

```swift
let rewardBreakdown = RewardCalculator.calculate(
    // ... existing args ...
    resetWasWasteful: resetWasWasteful,  // NEW
    // ... rest of args
)
```

---

### Phase 6: Siphon-Caused Death Penalty

This requires tracking which enemies were spawned from siphoning blocks vs naturally spawned.

#### Track Enemy Spawn Source

**Add property to Enemy struct** (in Enemy.swift or wherever Enemy is defined):

```swift
struct Enemy {
    // ... existing properties ...
    let spawnedFromSiphon: Bool  // NEW: Track if spawned from siphoning a block

    // Update initializer
    init(/* existing params */, spawnedFromSiphon: Bool = false) {
        // ... existing init ...
        self.spawnedFromSiphon = spawnedFromSiphon
    }
}
```

#### Mark Siphon-Spawned Enemies

**In GameState.performSiphon()** (around line 609):

When siphoning a block spawns a Transmission, mark it:

```swift
// When spawning transmission from siphoned block
let transmission = Transmission(
    row: cell.row,
    col: cell.col,
    turnsUntilConversion: conversionTime,
    targetEnemyType: enemyType,
    spawnedFromSiphon: true  // NEW: Mark as siphon-spawned
)
```

**In Transmission → Enemy conversion** (wherever transmissions convert to enemies):

```swift
// When transmission converts to enemy
let enemy = Enemy(
    // ... existing params ...
    spawnedFromSiphon: transmission.spawnedFromSiphon  // NEW: Preserve flag
)
```

#### Detect Siphon-Caused Death

**In GameState.tryExecuteAction()** (around line 1993):

```swift
// Track if death was caused by siphon-spawned enemy
var diedToSiphonEnemy = false

if playerDied {
    // Check if the killing enemy was spawned from siphoning
    // This requires tracking which enemy killed the player
    if let killingEnemy = /* find enemy that killed player */ {
        diedToSiphonEnemy = killingEnemy.spawnedFromSiphon
    }
}
```

#### Calculate Siphon Death Penalty

**In RewardCalculator.swift**:

```swift
// MARK: Siphon-Caused Death Penalty
let siphonDeathPenalty: Double
if diedToSiphonEnemy {
    siphonDeathPenalty = siphonCausedDeathPenalty  // Large negative penalty
} else {
    siphonDeathPenalty = 0.0
}
```

#### Required Parameter Changes

**Update RewardCalculator.calculate() signature**:

```swift
static func calculate(
    // ... existing params ...
    diedToSiphonEnemy: Bool,  // NEW
    // ... rest of params
) -> RewardBreakdown
```

**Update call site in GameState.swift**:

```swift
let rewardBreakdown = RewardCalculator.calculate(
    // ... existing args ...
    diedToSiphonEnemy: diedToSiphonEnemy,  // NEW
    // ... rest of args
)
```

---

## Implementation Order

### Step 1: Update RewardBreakdown Struct
- Add 7 new reward component fields (including siphonDeathPenalty)
- Update `total` computed property
- Add new reward constants

### Step 2: Implement Simple Rewards First (Low Risk)
1. **Resource gain/holding** - straightforward calculation
2. **HP damage/recovery** - simple HP delta tracking
3. **RESET waste detection** - single condition check

### Step 3: Implement Siphon Quality with Strict Dominance (Medium Risk)
1. Add `calculateSiphonYieldAt()` helper (track specific data block values + program sets)
2. Add `checkForBetterSiphonPosition()` method (strict dominance check with optimizations)
3. Update `performSiphon()` return signature
4. Calculate siphon quality in reward calculator

### Step 4: Implement Siphon-Caused Death Tracking (High Risk)
1. Add `spawnedFromSiphon` property to Enemy/Transmission
2. Mark siphon-spawned transmissions/enemies
3. Track which enemy killed the player
4. Calculate siphon death penalty

### Step 5: Update Python Logging
- `HackMatrix/HeadlessGame.swift`: Pass new breakdown fields to Python
- `python/hackmatrix/gym_env.py`: Log new reward components in `info["reward_breakdown"]`
- `python/hackmatrix/training_db.py`: Add new columns to database schema

### Step 6: Update W&B Config
- Add new reward component weights to W&B config in `train.py`

---

## Testing Strategy

### Unit Tests (GameLogicTests.swift)

```swift
// Test resource rewards
func testResourceGainReward() {
    let state = GameState()
    let oldCredits = state.player.credits
    // ... acquire credits ...
    // Assert reward includes creditGainMultiplier * delta
}

// Test HP penalty and recovery
func testHPPenaltyAndRecovery() {
    let state = GameState()
    // ... take damage ...
    // Assert damage penalty applied
    // ... heal with RESET ...
    // Assert recovery reward offsets penalty
}

// Test siphon optimality - strict dominance with specific block values
func testSiphonStrictDominance() {
    let state = GameState()
    // Set up board with two positions:
    // Position A: 5 credits, 3 energy, data blocks [1, 5], programs {PUSH}
    // Position B: 6 credits, 4 energy, data blocks [1, 5], programs {PUSH} (strictly better - same blocks, more resources)
    // Position C: 7 credits, 5 energy, data blocks [2, 5], programs {PUSH} (NOT comparable - different data blocks)
    // Position D: 8 credits, 6 energy, data blocks [1, 5], programs {RESET} (NOT comparable - different programs)

    // Move to Position A and siphon
    // Assert penalty applied (Position B was strictly better)

    // Move to Position B and siphon
    // Assert no penalty (Position B is optimal)
}

// Test RESET waste detection
func testResetWastePenalty() {
    let state = GameState()
    state.player.health = .damaged  // 2 HP
    // Execute RESET program
    // Assert waste penalty applied
}

// Test siphon-caused death penalty
func testSiphonCausedDeathPenalty() {
    let state = GameState()
    // Siphon a block to spawn enemy
    // Let that enemy kill the player
    // Assert extra siphon death penalty applied

    // Compare to natural enemy killing player
    // Assert no extra penalty for natural enemy death
}
```

### Integration Tests

1. **Resource accumulation test**: Play 10 turns collecting credits/energy, verify continuous holding bonus
2. **HP cycle test**: Take damage, heal with RESET, verify net-zero reward from HP changes
3. **Siphon comparison test**: Create board with strictly dominating position, siphon from suboptimal position, verify penalty
4. **RESET timing test**: Use RESET at 1 HP (no penalty), then at 2 HP (penalty), verify difference
5. **Siphon death test**: Siphon block to spawn enemy, die to that enemy, verify extra penalty

### Manual Validation

```bash
# Build and run with test scenario
xcodebuild -scheme HackMatrix -configuration Debug
cd python && source venv/bin/activate

# Test resource rewards
python -c "
from hackmatrix import HackEnv
env = HackEnv(debug=True)
obs, info = env.reset()
# Move to resource-rich cell
obs, reward, done, truncated, info = env.step(0)
print('Reward breakdown:', info['reward_breakdown'])
print('Resource gain:', info['reward_breakdown'].get('resourceGain', 'NOT_FOUND'))
"

# Test siphon optimality
python scripts/inspect_policy.py models/hackmatrix-dec29-25-4/interrupted_model --steps 20
# Watch for siphon quality penalties in reward breakdown
```

---

## Risk Assessment

### Low Risk
- ✅ Resource gain/holding calculation (simple math)
- ✅ HP damage/recovery tracking (HP already tracked)
- ✅ RESET waste detection (single condition)

### Medium Risk
- ⚠️ Parameter passing complexity (many new params to RewardCalculator)
- ⚠️ Database schema changes (need migration or rebuild)
- ⚠️ Siphon quality check (strict dominance logic with optimizations)

### High Risk
- 🔴 **Siphon-caused death tracking** - requires propagating spawn source through Transmission → Enemy lifecycle
  - **Mitigation**: Add property to existing structs, preserve through conversions
  - **Testing**: Need careful test cases to verify flag propagates correctly
  - **Edge cases**: Multiple enemies, player damage from multiple sources

---

## Performance Considerations

### Siphon Optimization Computational Cost

**Current**: `performSiphon()` evaluates 5 cells (cross pattern)
**New**: `checkForBetterSiphonPosition()` with optimizations:
- **No blocks case**: Evaluates all 36 positions × 5 cells = 180 cell evaluations
- **With blocks case**: Only evaluates positions that could siphon the same blocks (typically 1-4 positions)

**Estimated impact**:
- Only runs when siphon action is taken (~10-20% of actions)
- Each cell evaluation is cheap (check block type, count resources)
- Optimization: positions with blocks are skipped (can't stand there)
- Optimization: when current yield has blocks, only check ~1-4 positions instead of 36
- Training throughput: 570 steps/sec → expect ~540-550 steps/sec (3-5% slowdown)

**Optimization benefits**:
1. **Skip invalid positions**: Can't siphon from positions with blocks
2. **Early exit**: When blocks present, only check positions that siphon same blocks
3. **Array equality**: Sorted data block arrays allow fast comparison
4. **Set equality**: Program sets use efficient Set comparison

---

## Database Schema Updates

**File**: `python/hackmatrix/training_db.py`

Add columns to `episode_stats` table:

```python
reward_resource_gain REAL,
reward_resource_holding REAL,
reward_damage_penalty REAL,
reward_hp_recovery REAL,
reward_siphon_quality REAL,
reward_program_waste REAL,
reward_siphon_death REAL
```

**Migration**: Drop and recreate table (training history will be lost, but acceptable for experiment)

---

## Files to Modify

### Swift (Core Logic)
1. **HackMatrix/RewardCalculator.swift** (~250 lines modified)
   - Add 7 new reward components to struct
   - Add reward calculation logic
   - Update calculate() signature

2. **HackMatrix/GameState.swift** (~200 lines modified)
   - Add siphon optimization methods (strict dominance with optimizations)
   - Track old HP, credits, energy before actions
   - Detect RESET waste
   - Track which enemy killed player
   - Update RewardCalculator.calculate() calls

3. **HackMatrix/Enemy.swift** (~20 lines modified)
   - Add `spawnedFromSiphon` property
   - Update initializer

4. **HackMatrix/Transmission.swift** (~20 lines modified)
   - Add `spawnedFromSiphon` property
   - Preserve flag when converting to Enemy

5. **HackMatrix/GameLogicTests.swift** (~200 lines added)
   - Add test cases for new reward components

### Python (Training Infrastructure)
6. **python/hackmatrix/training_db.py** (~40 lines modified)
   - Add new reward component columns
   - Update insert/query logic

7. **python/scripts/train.py** (~15 lines modified)
   - Add new reward component weights to W&B config

---

## Success Criteria

1. ✅ **Resource rewards working**: Agent receives +0.05/credit gained, +0.001/credit held
2. ✅ **HP penalties working**: Taking 1 damage = -1.0, recovering 1 HP = +1.0
3. ✅ **Siphon quality working**: Suboptimal siphon position (strict dominance with exact blocks) triggers penalty
4. ✅ **RESET waste working**: RESET at 2 HP triggers -0.3 penalty
5. ✅ **Siphon death penalty working**: Dying to siphon-spawned enemy triggers -10.0 extra penalty
6. ✅ **Training still fast**: >500 steps/sec maintained
7. ✅ **W&B tracking**: New reward components visible in dashboard
8. ✅ **Database logging**: New columns populated correctly

---

## Expected Training Impact

### Agent Behavior Changes

**Before:**
- No incentive to collect/save resources → may ignore credits/energy
- No damage avoidance beyond death penalty → takes unnecessary hits
- No siphon positioning strategy → siphons from arbitrary positions
- Uses RESET inefficiently → may heal at high HP
- Siphons blocks carelessly without considering enemy spawn risk

**After:**
- **Resource hoarding**: Agent learns to collect and maintain reserves
- **HP preservation**: Agent actively avoids damage, heals efficiently
- **Strategic positioning**: Agent moves to optimal siphon locations (strict dominance with exact blocks)
- **Efficient healing**: Agent saves RESET for 1 HP (maximum value)
- **Risk-aware siphoning**: Agent avoids siphoning blocks when it would spawn enemies that could kill them

### Expected Reward Magnitudes

| Component | Typical Range | Impact |
|-----------|---------------|--------|
| Resource gain | 0.0 - 0.5 | Moderate (steady income) |
| Resource holding | 0.0 - 0.2 | Small (continuous bonus) |
| Damage penalty | -3.0 - 0.0 | High (avoid at all costs) |
| HP recovery | 0.0 - 3.0 | High (offsets damage) |
| Siphon quality | -0.3 - 0.0 | Small-Moderate (efficiency nudge) |
| Program waste | -0.3 - 0.0 | Small (efficiency nudge) |
| Siphon death penalty | -10.0 or 0.0 | Very High (catastrophic failure) |

**Total new reward range**: -14.0 to +4.0 per episode

---

## Future Extensions

### Potential Additions
1. **Program usage optimization**: Detect other wasteful program patterns (e.g., POLY when no enemies, CRASH when no blocks)
2. **Resource spending efficiency**: Reward using resources on high-value programs at optimal times
3. **Turn efficiency**: Penalize wasting turns (e.g., moving back and forth with no progress)
4. **Enemy positioning awareness**: Reward maintaining safe distance from fast enemies (Virus)
5. **Transmission spawn prediction**: Reward avoiding siphoning when transmissions will convert to enemies before player can escape

### Code Structure for Extensibility
- Keep reward components modular in RewardCalculator
- Each component has its own calculation method
- Easy to add/remove/tune individual components
- Clear separation between Swift logic and Python logging
