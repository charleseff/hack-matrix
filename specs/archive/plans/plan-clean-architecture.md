# Clean Architecture Refactor

## Goal

`tryExecuteAction()` is THE function that advances the game until next user input. Both HeadlessGame and GUI call it. Enemy turn logic runs inside it. Returns data for GUI to animate.

**Key insight:** Game state advances synchronously, animation is just replay of what happened.

---

## Current Status

### Completed
- [x] Fix stepActive bug in beginAnimatedEnemyTurn
- [x] Move GameAction enum to GameState.swift (renamed `.move` → `.direction`)
- [x] Add `EnemyStepResult` struct
- [x] Add `PlayerActionResult` struct (fromRow, fromCol, movedTo, attackedTarget)
- [x] Add `ActionResult` struct with all animation data
- [x] Add `runEnemyTurn()` helper
- [x] Add `executeEnemyStepWithCapture()` method
- [x] Rename `tryMove()` → `tryAttackOrMove()` (since direction can attack OR move)
- [x] Update `tryExecuteAction()` to include enemy turn AND stage completion
- [x] Update `HeadlessGame.step()` to use `tryExecuteAction()` (uses result.stageAdvanced/gameWon)
- [x] Refactor `moveEnemiesSimultaneously()` to filter enemies upfront

### Remaining
- [ ] Update GameScene to use `tryExecuteActionAndAnimate()` + `animateActionResult()`

---

## What's Done

### GameState.swift

**New structs:**
```swift
struct PlayerActionResult {
    let fromRow: Int
    let fromCol: Int
    let movedTo: (row: Int, col: Int)?        // if player moved
    let attackedTarget: (row: Int, col: Int)?  // if player attacked
}

struct EnemyStepResult {
    let step: Int  // 0, 1, etc (virus moves twice per turn)
    let movements: [(enemyId: UUID, fromRow: Int, fromCol: Int, toRow: Int, toCol: Int)]
    let attacks: [(enemyId: UUID, damage: Int)]
}

struct ActionResult {
    let success: Bool
    let exitReached: Bool
    let stageAdvanced: Bool  // stage was completed
    let gameWon: Bool        // player beat final stage
    let playerDied: Bool
    let playerAction: PlayerActionResult?  // nil if action failed
    let affectedPositions: [(row: Int, col: Int)]  // for explosion animations
    let enemySteps: [EnemyStepResult]  // for enemy movement animations

    static let failed = ActionResult(...)
}
```

**tryExecuteAction()** handles everything until next user input:
```swift
func tryExecuteAction(_ action: GameAction) -> ActionResult {
    // 1. Capture player position before action
    // 2. Handle player action (direction/siphon/program)
    // 3. If exit reached: call completeStage(), set stageAdvanced/gameWon
    // 4. Run enemy turn if needed via runEnemyTurn()
    // 5. Return ActionResult with all animation data
}
```

**runEnemyTurn()** executes full enemy turn and captures step data for animation.

**executeEnemyStepWithCapture()** executes one enemy step and captures movements/attacks.

**tryAttackOrMove()** (renamed from tryMove) returns movedTo and attackedTarget for animation.

### HeadlessGame.swift

**step()** is now trivially simple:
```swift
func step(action: GameAction) -> (GameObservation, Double, Bool, [String: Any]) {
    let oldScore = gameState.player.score
    let result = gameState.tryExecuteAction(action)

    if !result.success { info["invalid_action"] = true }
    if result.stageAdvanced {
        isDone = result.gameWon
        info["stage_complete"] = true
    }
    if result.playerDied {
        isDone = true
        info["death"] = true
    }

    // Calculate reward
    return (observation, reward, isDone, info)
}
```

**Critical ML bug fixed:** Enemies now move during headless training!

---

## What Remains: GameScene Refactor

### Plan

```swift
// GameScene simplified flow:
private func tryExecuteActionAndAnimate(_ action: GameAction) {
    let result = gameState.tryExecuteAction(action)
    if !result.success { return }

    // Animate player action (move OR attack)
    // Animate explosions (affectedPositions)
    // Animate enemy steps (attacks first, then movements per step)

    if result.exitReached {
        // Still animate player moving to exit
        // Then handle stage complete
    }
}

private func animateActionResult(_ result: ActionResult) {
    // 1. Animate player action (from result.playerAction)
    //    - If attackedTarget: show attack animation toward target
    //    - If movedTo: animate player movement
    // 2. Animate explosions (from result.affectedPositions)
    // 3. Animate enemy steps sequentially:
    //    - For each step: show attacks first, then movements
}
```

---

## Benefits Achieved

- ✅ Single code path for game logic (`tryExecuteAction()`)
- ✅ Stage completion handled inside `tryExecuteAction()`
- ✅ HeadlessGame is trivially simple
- ✅ Enemy turn bug fixed (enemies now move in ML training)
- ⏳ GUI animates from data (pending GameScene refactor)
