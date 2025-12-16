# Clean Architecture Refactor

## Goal

Single source of truth for game action processing. Clear separation:
- **Input Layer**: Translates input to GameAction (no game logic)
- **Game Logic Layer**: GameState processes everything
- **Presentation Layer**: GameScene only animates results

## Completed

- [x] Fix stepActive bug in beginAnimatedEnemyTurn (turnCount/scheduledTasks now respect stepActive)

---

## Phase 1: Add processAction to GameState

**File:** GameState.swift

### Add ActionResult struct

```swift
struct ActionResult {
    let success: Bool
    let exitReached: Bool
    let shouldAdvanceEnemyTurn: Bool
    let affectedPositions: [(Int, Int)]  // for explosion animations

    static let failed = ActionResult(
        success: false,
        exitReached: false,
        shouldAdvanceEnemyTurn: false,
        affectedPositions: []
    )
}
```

### Add processAction method

```swift
func processAction(_ action: GameAction) -> ActionResult {
    switch action {
    case .move(let direction):
        let result = tryMove(direction: direction)
        if result.blocked {
            return .failed
        }
        return ActionResult(
            success: true,
            exitReached: result.exitReached,
            shouldAdvanceEnemyTurn: !result.exitReached,
            affectedPositions: []
        )

    case .siphon:
        let success = performSiphon()
        return ActionResult(
            success: success,
            exitReached: false,
            shouldAdvanceEnemyTurn: success,
            affectedPositions: []
        )

    case .program(let programType):
        let execResult = executeProgram(programType)
        // Only wait program advances enemy turn (undo doesn't, other programs don't)
        let shouldAdvance = programType == .wait && execResult.success
        return ActionResult(
            success: execResult.success,
            exitReached: false,
            shouldAdvanceEnemyTurn: shouldAdvance,
            affectedPositions: execResult.affectedPositions
        )
    }
}
```

---

## Phase 2: Add Synchronous Enemy Turn

**File:** GameState.swift

For HeadlessGame - no animations, just execute the full enemy turn.

```swift
func executeSynchronousEnemyTurn() {
    guard !stepActive else {
        stepActive = false
        return
    }

    turnCount += 1
    processTransmissions()
    processEnemyTurn()
    maybeExecuteScheduledTask()

    // Spawn pending siphon transmissions
    if pendingSiphonTransmissions > 0 {
        spawnRandomTransmissions(count: pendingSiphonTransmissions)
        pendingSiphonTransmissions = 0
    }

    // Reset enemy status
    for enemy in enemies {
        enemy.decrementDisable()
        enemy.isStunned = false
    }

    saveSnapshot()
}
```

---

## Phase 3: Update HeadlessGame

**File:** HeadlessGame.swift

Replace step() to use processAction + executeSynchronousEnemyTurn:

```swift
func step(action: GameAction) -> (GameObservation, Double, Bool, [String: Any]) {
    let oldScore = gameState.player.score
    var isDone = false
    var info: [String: Any] = [:]

    let result = gameState.processAction(action)

    if !result.success {
        info["invalid_action"] = true
    }

    if result.exitReached {
        let continues = gameState.completeStage()
        isDone = !continues
        info["stage_complete"] = true
    }

    if result.shouldAdvanceEnemyTurn {
        gameState.executeSynchronousEnemyTurn()
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

**Tests after this phase:**
- Turn counter increments after move/siphon/wait
- Turn counter does NOT increment after other programs
- Enemies move in headless mode
- Transmissions spawn in headless mode
- Python wrapper (test_env.py) still works

---

## Phase 4: Refactor GameScene Input

**File:** GameScene.swift

### Add handleAction helper

```swift
private func handleAction(_ action: GameAction) {
    let result = gameState.processAction(action)

    if !result.success {
        return
    }

    if result.exitReached {
        handleStageComplete()
        return
    }

    updateDisplay()

    if !result.affectedPositions.isEmpty {
        isAnimating = true
        animateExplosions(at: result.affectedPositions) { [weak self] in
            guard let self = self else { return }
            if result.shouldAdvanceEnemyTurn {
                self.executeEnemyTurnWithAnimation()
            } else {
                self.isAnimating = false
            }
        }
        return
    }

    if result.shouldAdvanceEnemyTurn {
        executeEnemyTurnWithAnimation()
    }
}
```

### Simplify keyDown

```swift
override func keyDown(with event: NSEvent) {
    guard !isAnimating else { return }

    if isGameOver && event.keyCode == 15 {
        restartGame()
        return
    }

    guard gameState.player.health != .dead else { return }

    let action: GameAction?

    if event.keyCode == 1 {
        action = .siphon
    } else if let programType = getProgramForKeyCode(event.keyCode) {
        action = .program(programType)
    } else {
        switch event.keyCode {
        case 126: action = .move(.up)
        case 125: action = .move(.down)
        case 123: action = .move(.left)
        case 124: action = .move(.right)
        default: action = nil
        }
    }

    if let action = action {
        handleAction(action)
    }
}
```

### Simplify mouseDown

Similar - extract to just create GameAction, then call handleAction.

---

## Phase 5 (Future): Decouple Animation from Game Logic

Current problem: `executeAndAnimateEnemyStep()` calls `gameState.executeEnemyStep()` - view driving model.

Solution: Callback-based architecture where GameState drives execution and provides animation hooks. Defer this for now.

---

## Files Modified

1. **GameState.swift**
   - Add ActionResult struct
   - Add processAction() method
   - Add executeSynchronousEnemyTurn() method

2. **HeadlessGame.swift**
   - Replace step() with processAction-based implementation

3. **GameScene.swift**
   - Add handleAction() helper
   - Simplify keyDown
   - Simplify mouseDown

---

## Execution Order

1. ~~Fix stepActive bug~~ (DONE)
2. Phase 1: Add processAction to GameState
3. Phase 2: Add executeSynchronousEnemyTurn
4. Phase 3: Update HeadlessGame (fixes ML training bug)
5. Phase 4: Refactor GameScene input
6. Phase 5: Decouple animation (future)
