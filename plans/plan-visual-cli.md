# Visual CLI Mode

Goal: Run GUI driven by stdin/stdout JSON (same protocol as headless), to visualize ML gameplay.

---

## Task Checklist

### Phase 1: Extract Shared Protocol
- [ ] Create `GameCommandProtocol.swift` with protocol + stdin/stdout handling
- [ ] Refactor HeadlessGameCLI to use protocol
- [ ] Verify headless mode still works

### Phase 2: Visual CLI Implementation
- [ ] Add `--visual-cli` flag to App.swift
- [ ] Implement protocol in GameScene
- [ ] Wire animation completion to send observations

### Phase 3: Python Integration
- [ ] Add `visual=True` to HackEnv
- [ ] Create `test_visual.py`

---

## Phase 1: Extract Shared Protocol

**Files:**
- `868-hack/GameCommandProtocol.swift` (new)
- `868-hack/HeadlessGameCLI.swift`

### GameCommandProtocol.swift

Create protocol and shared stdin/stdout coordinator:

```swift
import Foundation

protocol GameCommandExecutor: AnyObject {
    func executeReset() -> GameObservation
    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any])
    func executeGetValidActions() -> [Int]
}

struct Command: Codable {
    let action: String
    let actionIndex: Int?
}

class StdinCommandReader {
    weak var executor: GameCommandExecutor?
    private var outputFile: FileHandle?

    func start() {
        let originalStdout = dup(STDOUT_FILENO)
        outputFile = FileHandle(fileDescriptor: originalStdout, closeOnDealloc: false)
        dup2(STDERR_FILENO, STDOUT_FILENO)

        while let line = readLine() {
            guard let data = line.data(using: .utf8),
                  let command = try? JSONDecoder().decode(Command.self, from: data) else {
                sendError("Invalid JSON")
                continue
            }
            handleCommand(command)
        }
    }

    private func handleCommand(_ command: Command) {
        guard let executor = executor else { return }

        switch command.action {
        case "reset":
            let obs = executor.executeReset()
            sendResponse(["observation": encodeObservation(obs)])

        case "step":
            guard let actionIndex = command.actionIndex else {
                sendError("Missing actionIndex")
                return
            }
            let (obs, reward, done, info) = executor.executeStep(actionIndex: actionIndex)
            sendResponse([
                "observation": encodeObservation(obs),
                "reward": reward,
                "done": done,
                "info": info
            ])

        case "getValidActions":
            let actions = executor.executeGetValidActions()
            sendResponse(["validActions": actions])

        case "getActionSpace":
            sendResponse(["actionSpaceSize": 31])

        case "getObservationSpace":
            sendResponse([
                "gridSize": 6,
                "playerFeatures": 9,
                "cellFeatures": 20
            ])

        default:
            sendError("Unknown command: \(command.action)")
        }
    }

    private func encodeObservation(_ obs: GameObservation) -> [String: Any] {
        // Move from HeadlessGameCLI - same implementation
    }

    private func sendResponse(_ data: [String: Any]) {
        // Move from HeadlessGameCLI - same implementation
    }

    private func sendError(_ message: String) {
        // Move from HeadlessGameCLI - same implementation
    }
}
```

### HeadlessGameCLI.swift refactor

Simplify to just implement protocol:

```swift
class HeadlessGameCLI: GameCommandExecutor {
    private var game: HeadlessGame?
    private let reader = StdinCommandReader()

    func run() {
        guard CommandLine.arguments.contains("--headless-cli") else { return }
        reader.executor = self
        reader.start()
    }

    func executeReset() -> GameObservation {
        game = HeadlessGame()
        return game!.getObservation()
    }

    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let game = game,
              let action = GameAction.fromIndex(actionIndex) else {
            fatalError("Invalid step")
        }
        return game.step(action: action)
    }

    func executeGetValidActions() -> [Int] {
        guard let game = game else { return [] }
        return game.getValidActions().map { $0.toIndex() }
    }
}
```

Remove old `handleCommand`, `sendResponse`, `sendError`, `encodeObservation` - now in shared reader.

### Test

Run `python/test_env.py` to verify headless mode still works.

---

## Phase 2: Visual CLI Implementation

**Files:**
- `868-hack/App.swift`
- `868-hack/GameScene.swift`

### App.swift

Add after headless-cli check:

```swift
if CommandLine.arguments.contains("--visual-cli") {
    UserDefaults.standard.set(true, forKey: "visualCliMode")
}
```

### GameScene.swift

Implement protocol with async response:

```swift
class GameScene: SKScene, GameCommandExecutor {
    private var visualCliMode = false
    private let commandReader = StdinCommandReader()
    private var pendingStepCallback: ((GameObservation, Double, Bool, [String: Any]) -> Void)?

    override func didMove(to view: SKView) {
        // ... existing setup ...

        if UserDefaults.standard.bool(forKey: "visualCliMode") {
            visualCliMode = true
            commandReader.executor = self
            Thread {
                self.commandReader.start()
            }.start()
        }
    }

    // MARK: - GameCommandExecutor

    func executeReset() -> GameObservation {
        gameState = GameState()
        updateDisplay()
        return buildObservation()
    }

    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let action = GameAction.fromIndex(actionIndex) else {
            return (buildObservation(), 0, true, [:])
        }

        // This is async - need to wait for animation
        var result: (GameObservation, Double, Bool, [String: Any])?
        let semaphore = DispatchSemaphore(value: 0)

        DispatchQueue.main.async { [weak self] in
            guard let self = self, !self.isAnimating else {
                result = (self?.buildObservation() ?? GameObservation(), 0, true, [:])
                semaphore.signal()
                return
            }

            self.pendingStepCallback = { obs, reward, done, info in
                result = (obs, reward, done, info)
                semaphore.signal()
            }

            self.tryExecuteActionAndAnimate(action)
        }

        semaphore.wait()
        return result!
    }

    func executeGetValidActions() -> [Int] {
        return gameState.getValidActions().map { $0.toIndex() }
    }

    // MARK: - Animation completion

    // At end of animateActionResult completion chain:
    private func onAnimationComplete() {
        if let callback = pendingStepCallback {
            pendingStepCallback = nil
            let obs = buildObservation()
            let done = gameState.isGameOver || gameState.isVictory
            callback(obs, 0.0, done, [:])
        }
    }

    private func buildObservation() -> GameObservation {
        // Build GameObservation from current gameState
        // Same structure as HeadlessGame.getObservation()
    }
}
```

---

## Phase 3: Python Integration

**Files:**
- `python/hack_env.py`
- `python/test_visual.py` (new)

### hack_env.py

```python
def __init__(self, app_path: str = "...", visual: bool = False):
    self.visual = visual
    # ...

def _start_process(self):
    flag = "--visual-cli" if self.visual else "--headless-cli"
    self.process = subprocess.Popen([self.app_path, flag], ...)
```

### test_visual.py

```python
from hack_env import HackEnv
import numpy as np

env = HackEnv(visual=True)
obs, info = env.reset()

for i in range(30):
    valid = env.get_valid_actions()
    if not valid:
        break
    action = np.random.choice(valid)
    obs, reward, done, _, info = env.step(action)
    if done:
        break

env.close()
```
