# Game Execution Refactoring Plan

## Task Checklist

### Phase 1: Add Unified Turn Execution to GameState
- ☐ Add TurnResult struct to GameState.swift
- ☐ Add EnemyMovement and EnemyAttack structs
- ☐ Implement executeTurn() method
- ☐ Test: Verify method produces correct turn progression

### Phase 2: Fix HeadlessGame Turn Processing
- ☐ Modify HeadlessGame.step() to call executeTurn() after actions
- ☐ Test: Verify turn counter increments in headless mode
- ☐ Test: Verify enemies move in headless mode
- ☐ Test: Verify transmissions spawn in headless mode
- ☐ Test: Python wrapper (test_env.py) still works

### Phase 3: Remove GameScene Duplication
- ☐ Add executeTurnWithAnimation() helper to GameScene
- ☐ Replace turn initiation after siphon (line 651-663)
- ☐ Replace turn initiation after wait program (line 676-687)
- ☐ Replace turn initiation after other programs (line 689-700)
- ☐ Replace turn initiation in mouseDown wait (line 749-762)
- ☐ Replace turn initiation in handlePlayerMoveComplete (line 862-874)
- ☐ Test: Verify GUI animations work correctly

---

## Critical Problem

**HeadlessGame does not process game turns.** After executing an action:
- Turn counter stays at 0
- Enemies never move
- Transmissions never spawn enemies
- Scheduled tasks never trigger

This means **ML training happens in a fundamentally different game** than human play. The agent learns strategies that only work in a frozen environment.

**Root cause:** HeadlessGame.step() calls action methods (tryMove, performSiphon, executeProgram) but never calls turn advancement (advanceTurn, beginAnimatedTurn, etc.).

## Solution Overview

1. **Create unified turn execution** in GameState that both modes use
2. **Fix HeadlessGame** to process full turns after actions
3. **Eliminate duplication** in GameScene (5 copies of same turn initiation code)

---

## Phase 1: Add Unified Turn Execution to GameState

**File:** 868-hack/GameState.swift

**Location:** After line 510 (after finalizeAnimatedTurn)

### Add Result Structures

```swift
struct EnemyMovement {
    let enemyId: UUID
    let fromRow: Int
    let fromCol: Int
    let toRow: Int
    let toCol: Int
}

struct EnemyAttack {
    let enemyId: UUID
    let fromRow: Int
    let fromCol: Int
    let targetRow: Int
    let targetCol: Int
    let damage: Int
}

struct TurnResult {
    let enemySteps: [(step: Int, movements: [EnemyMovement], attacks: [EnemyAttack])]
    let transmissionsSpawned: Int
    let scheduledTaskExecuted: Bool
    let pendingSiphonSpawned: Int
}
```

### Add executeTurn() Method

```swift
func executeTurn() -> TurnResult {
    turnCount += 1

    var result = TurnResult(
        enemySteps: [],
        transmissionsSpawned: 0,
        scheduledTaskExecuted: false,
        pendingSiphonSpawned: 0
    )

    // Process transmissions (spawn enemies from countdown)
    let transmissionCountBefore = enemies.count
    if !stepActive {
        processTransmissions()
        result.transmissionsSpawned = enemies.count - transmissionCountBefore
    }

    // Process enemy turn synchronously (for headless mode)
    if !stepActive {
        processEnemyTurn()
    }

    stepActive = false

    // Process scheduled task (spawn new transmission)
    if turnCount > 0 && turnCount % Constants.scheduledTaskIntervals[currentStage - 1] == 0 {
        processScheduledTask()
        result.scheduledTaskExecuted = true
    }

    // Spawn pending siphon transmissions
    if pendingSiphonTransmissions > 0 {
        result.pendingSiphonSpawned = pendingSiphonTransmissions
        spawnRandomTransmissions(count: pendingSiphonTransmissions)
        pendingSiphonTransmissions = 0
    }

    // Reset enemy status
    for enemy in enemies {
        enemy.decrementDisable()
        enemy.isStunned = false
    }

    // Save snapshot for undo
    saveSnapshot()

    return result
}
```

**Why this works:**
- Consolidates logic from advanceTurn() (line 460-475)
- Includes finalization from finalizeAnimatedTurn() (line 496-510)
- Returns data that GameScene could use for animations (future enhancement)
- Pure game logic, no animation dependencies

**Testing:**
- Create simple test: reset game, execute action, call executeTurn()
- Verify: turn counter increments
- Verify: if transmission exists with timer=1, enemy spawns
- Verify: enemies move closer to player

---

## Phase 2: Fix HeadlessGame Turn Processing

**File:** 868-hack/HeadlessGame.swift

**Current code (lines 19-76):** Executes actions but never processes turns

**Changes needed:**

### Modify step() Method

Replace entire step() method:

```swift
func step(action: GameAction) -> (GameObservation, Double, Bool, [String: Any]) {
    let oldScore = gameState.player.score

    var isDone = false
    var info: [String: Any] = [:]

    // Execute the action
    switch action {
    case .move(let direction):
        let result = gameState.tryMove(direction: direction)
        if result.exitReached {
            // Reached exit - advance to next stage
            let continues = gameState.completeStage()
            isDone = !continues  // Game ends on victory
            info["stage_complete"] = true
        } else {
            // CRITICAL FIX: Process turn after movement
            _ = gameState.executeTurn()
        }

    case .siphon:
        if gameState.performSiphon() {
            // CRITICAL FIX: Process turn after siphon
            _ = gameState.executeTurn()
        }

    case .program(let programType):
        if gameState.canExecuteProgram(programType).canExecute {
            _ = gameState.executeProgram(programType)
            // CRITICAL FIX: Most programs advance turn (except undo)
            if programType != .undo {
                _ = gameState.executeTurn()
            }
        } else {
            // Invalid action - penalize slightly
            info["invalid_action"] = true
        }
    }

    // Check if player died (could happen during enemy turn)
    if gameState.player.health == .dead {
        isDone = true
        info["death"] = true
    }

    // Calculate reward
    let scoreDelta = Double(gameState.player.score - oldScore)
    var reward = scoreDelta * 0.01  // Small reward for gaining points

    // Episode end rewards
    if isDone {
        if gameState.player.health == .dead {
            // Death: No reward
            reward = 0.0
        } else {
            // Victory (completed stage 8): BIG reward based on final score
            reward = Double(gameState.player.score) * 10.0
        }
    }

    let observation = getObservation()
    return (observation, reward, isDone, info)
}
```

**Key changes:**
- After movement (non-exit): call executeTurn()
- After siphon: call executeTurn()
- After programs (except undo): call executeTurn()
- Check for death after turn (enemies may have killed player)

**Testing:**
- Run python/test_env.py - should still work
- Add test: 5 random steps, verify turn counter = 5
- Add test: place enemy adjacent to player, step, verify player took damage
- Add test: place transmission with timer=1, step, verify enemy spawned

**Backward compatibility:** Python wrapper unchanged, observations unchanged, only internal game state now progresses correctly.

---

## Phase 3: Remove GameScene Duplication

**File:** 868-hack/GameScene.swift

**Problem:** Turn initiation sequence duplicated 5 times

### Add Helper Method

Insert after line 920 (after animateEnemySteps):

```swift
private func executeTurnWithAnimation() {
    isAnimating = true
    let shouldEnemiesMove = gameState.beginAnimatedTurn()
    updateDisplay()

    if shouldEnemiesMove {
        enemiesWhoAttacked = Set<UUID>()
        animateEnemySteps(currentStep: 0)
    } else {
        gameState.finalizeAnimatedTurn()
        isAnimating = false
    }
}
```

### Replace Duplicate #1: Siphon Handler (lines 651-663)

**Current:**
```swift
if gameState.performSiphon() {
    // Begin animated turn
    isAnimating = true
    let shouldEnemiesMove = gameState.beginAnimatedTurn()
    updateDisplay()

    if shouldEnemiesMove {
        enemiesWhoAttacked = Set<UUID>()
        animateEnemySteps(currentStep: 0)
    } else {
        gameState.finalizeAnimatedTurn()
        isAnimating = false
    }
}
```

**Replace with:**
```swift
if gameState.performSiphon() {
    executeTurnWithAnimation()
}
```

### Replace Duplicate #2: Wait Program (lines 676-687)

**Current:**
```swift
if programType == .wait {
    isAnimating = true
    let shouldEnemiesMove = gameState.beginAnimatedTurn()
    updateDisplay()

    if shouldEnemiesMove {
        enemiesWhoAttacked = Set<UUID>()
        animateEnemySteps(currentStep: 0)
    } else {
        gameState.finalizeAnimatedTurn()
        isAnimating = false
    }
}
```

**Replace with:**
```swift
if programType == .wait {
    executeTurnWithAnimation()
}
```

### Replace Duplicate #3: Other Programs (lines 689-700)

**Current:**
```swift
} else if programType != .undo {
    // All other programs except undo advance the turn
    isAnimating = true
    let shouldEnemiesMove = gameState.beginAnimatedTurn()
    updateDisplay()

    if shouldEnemiesMove {
        enemiesWhoAttacked = Set<UUID>()
        animateEnemySteps(currentStep: 0)
    } else {
        gameState.finalizeAnimatedTurn()
        isAnimating = false
    }
}
```

**Replace with:**
```swift
} else if programType != .undo {
    // All other programs except undo advance the turn
    executeTurnWithAnimation()
}
```

### Replace Duplicate #4: Mouse Click Wait (lines 749-762)

**Current (in mouseDown after wait program execution):**
```swift
if programType == .wait {
    isAnimating = true
    let shouldEnemiesMove = gameState.beginAnimatedTurn()
    updateDisplay()

    if shouldEnemiesMove {
        enemiesWhoAttacked = Set<UUID>()
        animateEnemySteps(currentStep: 0)
    } else {
        gameState.finalizeAnimatedTurn()
        isAnimating = false
    }
}
```

**Replace with:**
```swift
if programType == .wait {
    executeTurnWithAnimation()
}
```

### Replace Duplicate #5: Player Movement Complete (lines 862-874)

**Current (in handlePlayerMoveComplete):**
```swift
isAnimating = true
let shouldEnemiesMove = gameState.beginAnimatedTurn()
updateDisplay()

if shouldEnemiesMove {
    enemiesWhoAttacked = Set<UUID>()
    animateEnemySteps(currentStep: 0)
} else {
    gameState.finalizeAnimatedTurn()
    isAnimating = false
}
```

**Replace with:**
```swift
executeTurnWithAnimation()
```

**Result:** 5 duplicate blocks (13 lines each = 65 lines) replaced with 5 single-line calls (5 lines total). 60 lines eliminated.

**Testing:**
- Play game in GUI
- Test siphon action - verify animations work
- Test wait program - verify animations work
- Test other programs - verify animations work
- Test movement - verify animations work
- Verify no visual regressions

---

## Validation Tests

### After Phase 1
No behavioral changes yet. GameState has new method but nothing calls it.

### After Phase 2 (CRITICAL TESTS)

**Test 1: Turn Counter**
```python
env = HackEnv()
env.reset()
initial_turn = env.game_state.turnCount  # Should be 0
env.step(0)  # Move up
assert env.game_state.turnCount == initial_turn + 1
```

**Test 2: Enemy Movement**
```python
# Set up: Place enemy at (3, 3), player at (1, 1)
# Enemy should path toward player
env.step(0)  # Move
# Verify enemy moved closer to player
```

**Test 3: Transmission Spawning**
```python
# Set up: Place transmission with timer=1
env.step(0)  # Move
# Verify new enemy spawned
```

**Test 4: Python Wrapper**
```bash
cd python && python test_env.py
# Should pass without modifications
```

### After Phase 3

**Test 5: GUI Animations**
- Launch GUI with "Run GUI" task
- Press S (siphon) - verify animation completes
- Click program button - verify animation completes
- Move with arrow keys - verify animation completes
- No hangs, no visual glitches

**Test 6: State Consistency**
Run same sequence in both modes:
```python
# HeadlessGame
game1 = HeadlessGame()
game1.step(.move(.up))
game1.step(.siphon)
state1 = game1.getObservation()

# GameScene (simulate)
game2 = GameState()
game2.tryMove(.up)
game2.executeTurn()
game2.performSiphon()
game2.executeTurn()
state2 = extractObservation(game2)

assert state1 == state2  # Same final state
```

---

## Files Modified

### Primary Files
1. **868-hack/GameState.swift**
   - Add: TurnResult, EnemyMovement, EnemyAttack structs (after line 510)
   - Add: executeTurn() method (after line 510)
   - Lines added: ~60

2. **868-hack/HeadlessGame.swift**
   - Modify: step() method (lines 19-76)
   - Add: executeTurn() calls after actions
   - Lines changed: ~60

3. **868-hack/GameScene.swift**
   - Add: executeTurnWithAnimation() helper (after line 920)
   - Replace: 5 duplicate sequences with helper calls
   - Lines: +10 added, -60 removed

### Test Files
4. **python/test_env.py** (validation only, no changes)
   - Run to verify backward compatibility

---

## Why This Approach

**Simplicity:** Minimal new abstractions, uses existing methods where possible

**Composability:** executeTurn() is pure game logic, both modes call it

**Separation of concerns:**
- GameState: game logic only
- HeadlessGame: game logic execution
- GameScene: game logic + animation wrapping

**Backward compatible:** Python wrapper unchanged, observation format unchanged

**Incremental:** Each phase testable independently, can rollback if issues

**Fixes critical bug:** ML training now happens in real game environment

---

## Risk Mitigation

**Risk 1: Animation timing breaks**
- Mitigation: Phase 3 is pure refactoring, no logic change
- Test after each replacement

**Risk 2: Python wrapper breaks**
- Mitigation: Observation format unchanged
- Test with existing test_env.py

**Risk 3: Game state diverges**
- Mitigation: Use same executeTurn() in both modes
- Add comparison test

**Risk 4: Performance regression**
- Mitigation: No new allocations, same algorithm
- executeTurn() just consolidates existing calls
