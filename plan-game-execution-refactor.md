# Game Execution Refactoring

## Task Checklist

### Phase 1: Add Unified Turn Execution
- ☐ Add TurnResult struct to GameState.swift
- ☐ Add EnemyMovement and EnemyAttack structs
- ☐ Implement executeTurn() method

### Phase 2: Fix HeadlessGame Turn Processing
- ☐ Modify HeadlessGame.step() to call executeTurn() after actions
- ☐ Test: Verify enemies move in headless mode
- ☐ Test: Verify transmissions spawn

### Phase 3: Remove GameScene Duplication
- ☐ Add executeTurnWithAnimation() helper
- ☐ Replace 5 duplicate turn initiation sequences

---

## Critical Bug

HeadlessGame does not process game turns. After actions:
- Turn counter stays at 0
- Enemies never move
- Transmissions never spawn
- ML training happens in a frozen game environment

---

## Phase 1: Add Unified Turn Execution

**Affected files:**
- 868-hack/GameState.swift

**Changes:**
Add structs and executeTurn() method that consolidates turn logic for both headless and animated modes.

### Add after line 510 in GameState.swift

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

func executeTurn() -> TurnResult {
    turnCount += 1

    var result = TurnResult(
        enemySteps: [],
        transmissionsSpawned: 0,
        scheduledTaskExecuted: false,
        pendingSiphonSpawned: 0
    )

    let transmissionCountBefore = enemies.count
    if !stepActive {
        processTransmissions()
        result.transmissionsSpawned = enemies.count - transmissionCountBefore
    }

    if !stepActive {
        processEnemyTurn()
    }

    stepActive = false

    if turnCount > 0 && turnCount % Constants.scheduledTaskIntervals[currentStage - 1] == 0 {
        processScheduledTask()
        result.scheduledTaskExecuted = true
    }

    if pendingSiphonTransmissions > 0 {
        result.pendingSiphonSpawned = pendingSiphonTransmissions
        spawnRandomTransmissions(count: pendingSiphonTransmissions)
        pendingSiphonTransmissions = 0
    }

    for enemy in enemies {
        enemy.decrementDisable()
        enemy.isStunned = false
    }

    saveSnapshot()

    return result
}
```

**Why:** Consolidates advanceTurn() logic (line 460-475) plus finalizeAnimatedTurn() finalization (line 496-510). Both modes can call this single method.

---

## Phase 2: Fix HeadlessGame Turn Processing

**Affected files:**
- 868-hack/HeadlessGame.swift

**Changes:**
Call executeTurn() after each action to process enemy turns, transmissions, and scheduled tasks.

### Replace step() method (lines 19-76)

```swift
func step(action: GameAction) -> (GameObservation, Double, Bool, [String: Any]) {
    let oldScore = gameState.player.score

    var isDone = false
    var info: [String: Any] = [:]

    switch action {
    case .move(let direction):
        let result = gameState.tryMove(direction: direction)
        if result.exitReached {
            let continues = gameState.completeStage()
            isDone = !continues
            info["stage_complete"] = true
        } else {
            _ = gameState.executeTurn()
        }

    case .siphon:
        if gameState.performSiphon() {
            _ = gameState.executeTurn()
        }

    case .program(let programType):
        if gameState.canExecuteProgram(programType).canExecute {
            _ = gameState.executeProgram(programType)
            if programType != .undo {
                _ = gameState.executeTurn()
            }
        } else {
            info["invalid_action"] = true
        }
    }

    if gameState.player.health == .dead {
        isDone = true
        info["death"] = true
    }

    let scoreDelta = Double(gameState.player.score - oldScore)
    var reward = scoreDelta * 0.01

    if isDone {
        if gameState.player.health == .dead {
            reward = 0.0
        } else {
            reward = Double(gameState.player.score) * 10.0
        }
    }

    let observation = getObservation()
    return (observation, reward, isDone, info)
}
```

**Key changes:**
- After movement (non-exit): call executeTurn()
- After siphon success: call executeTurn()
- After programs (except undo): call executeTurn()
- Check for death after turn

---

## Phase 3: Remove GameScene Duplication

**Affected files:**
- 868-hack/GameScene.swift

**Changes:**
Extract duplicated turn initiation code into single helper method.

### Add helper after line 920

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

### Replace duplicate #1 (lines 651-663)

Replace siphon handler:
```swift
if gameState.performSiphon() {
    executeTurnWithAnimation()
}
```

### Replace duplicate #2 (lines 676-687)

Replace wait program handler:
```swift
if programType == .wait {
    executeTurnWithAnimation()
}
```

### Replace duplicate #3 (lines 689-700)

Replace other programs handler:
```swift
} else if programType != .undo {
    executeTurnWithAnimation()
}
```

### Replace duplicate #4 (lines 749-762)

Replace mouseDown wait handler:
```swift
if programType == .wait {
    executeTurnWithAnimation()
}
```

### Replace duplicate #5 (lines 862-874)

Replace handlePlayerMoveComplete:
```swift
executeTurnWithAnimation()
```

**Result:** 5 duplicates (65 lines total) → 5 single-line calls (5 lines).
