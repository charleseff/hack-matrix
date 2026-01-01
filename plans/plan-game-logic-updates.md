# Game Logic Updates: Starting Bonus & Scheduled Task Timing

## Task Checklist

### Phase 1: Random Starting Bonus
- [x] Add `StartingBonus` enum to Constants.swift
- [x] Apply random bonus in GameState.init()

### Phase 2: Scheduled Task Timing Redesign
- [x] Add `scheduledTaskInterval` and `nextScheduledTaskTurn` properties to GameState
- [x] Update `maybeExecuteScheduledTask()` to use threshold-based triggering
- [x] Update `performSiphon()` to decrease interval and add delay
- [x] Update stage transition to recalculate interval

### Phase 3: Test Suite
- [x] Add `--run-tests` flag to App.swift
- [x] Create GameLogicTests.swift with lightweight test runner
- [x] All 13 tests passing

---

## Phase 1: Random Starting Bonus

**Files:** `HackMatrix/Constants.swift`, `HackMatrix/GameState.swift`

**Constants.swift:** Add enum before `Constants`:
```swift
enum StartingBonus: CaseIterable {
    case credits10
    case energy11
    case dataSiphon1
}
```

**GameState.swift:** In `init()`, after creating player:
```swift
let bonus = StartingBonus.allCases.randomElement()!
switch bonus {
case .credits10: player.credits = 10
case .energy11: player.energy = 11
case .dataSiphon1: player.dataSiphons = 1
}
```

**Tests:**
- `testStartingBonusIsApplied`: Verify one of the three bonuses is present
- `testStartingBonusRandomness`: Run 60 trials, verify all three bonus types appear

---

## Phase 2: Scheduled Task Timing Redesign

**Files:** `HackMatrix/GameState.swift`

**Design:** Replace modulo-based scheduling (`turnCount % interval == 0`) with explicit tracking:
- `scheduledTaskInterval`: Current spawn interval (starts at 12, decreases)
- `nextScheduledTaskTurn`: Turn number when next spawn occurs

**GameState properties:**
```swift
var scheduledTaskInterval: Int = 12
var nextScheduledTaskTurn: Int = 12
```

**maybeExecuteScheduledTask():**
```swift
func maybeExecuteScheduledTask() {
    guard !scheduledTasksDisabled else { return }
    guard scheduledTaskInterval > 0 else { return }

    if turnCount >= nextScheduledTaskTurn {
        spawnRandomTransmissions(count: 1, isFromScheduledTask: true)
        nextScheduledTaskTurn = turnCount + max(1, scheduledTaskInterval)
    }
}
```

**performSiphon():** Add before return:
```swift
scheduledTaskInterval = max(1, scheduledTaskInterval - 1)  // Permanent decrease
nextScheduledTaskTurn += 5  // Temporary delay
```

**Stage transition:** After incrementing `currentStage`:
```swift
// Resets to base for new stage - siphon reductions don't carry over
scheduledTaskInterval = 13 - currentStage
nextScheduledTaskTurn = turnCount + scheduledTaskInterval
```

**Tests:**
- `testInitialScheduledTaskState`: interval=12, nextTurn=12
- `testSiphonAffectsScheduledTiming`: interval decreases by 1, nextTurn increases by 5
- `testScheduledTaskIntervalMinimum`: interval stays at 1 minimum
- `testScheduledTaskTriggersAtCorrectTurn`: spawns at turn 12, not before
- `testStageChangeResetsInterval`: interval resets to 11 for stage 2 (ignores siphon reductions)

---

## Phase 3: Test Suite

**Files:** `HackMatrix/App.swift`, `HackMatrix/GameLogicTests.swift`

**App.swift:** Add in `init()`:
```swift
if CommandLine.arguments.contains("--run-tests") {
    GameLogicTests.runAllTests()
    exit(0)
}
```

**GameLogicTests.swift:** Lightweight test runner with:
- `assert()`, `assertEqual()`, `assertGreaterThan()` helpers
- Test methods for each feature
- Summary output with pass/fail counts

**Run tests:** `HackMatrix.app/Contents/MacOS/HackMatrix --run-tests`
