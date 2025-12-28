# HackMatrix Game Architecture

## Project Conventions

- **Plan files** go in `plans/` directory

### Python Scripts

- **Always activate venv** when running Python scripts
- Command pattern: `cd python && source venv/bin/activate && python <script>`
- Example: `cd python && source venv/bin/activate && python scripts/train.py`

### Building

- **Always use Xcode build**, NOT `swift build`
- Build command: `xcodebuild -scheme HackMatrix -configuration Debug`
- Output location: `DerivedData/HackMatrix/Build/Products/Debug/HackMatrix.app/Contents/MacOS/HackMatrix`
- Python expects the executable at this location (Xcode default)

### Git Workflow

- **Always create a new branch** for any non-trivial work
- Branch naming: descriptive kebab-case (e.g., `reward-system-refactor`, `fix-movement-bug`)
- Workflow:
  1. Create branch: `git checkout -b feature-name`
  2. Make changes and test thoroughly
  3. Review code before committing
  4. Commit with descriptive messages
  5. Merge to main when complete and tested
  6. Delete feature branch after merge

---

## Turn Structure

### Player's Turn
Player takes one of these actions:
- **Move** → Turn ends, enemy turn begins
- **Attack** → Turn ends, enemy turn begins
- **Siphon** → Turn ends, enemy turn begins
- **Execute Program** → Turn does NOT end (can chain multiple programs)
  - **Exception: Wait program** → Turn ends without requiring move/attack/siphon

### Turn Transition (when player's turn ends)
1. Turn counter increments
2. Enemy turn begins

### Enemy's Turn
1. Transmissions spawn (convert to enemies based on timer)
2. Enemies move/attack
3. Scheduled tasks execute
4. Enemy status resets (disable counters, stun flags)

---

## Architecture Overview

### Entry Points

The app has multiple entry points for different use cases:

| Flag | Purpose | GUI | Execution |
|------|---------|-----|-----------|
| (none) | Human plays game | Yes | Interactive |
| `--headless-cli` | ML training | No | Instant |
| `--visual-cli` | Watch ML play | Yes | Animated |
| `--debug-scenario` | Test specific scenario | Yes | Interactive |

### Call Hierarchies

**GUI Mode (Human Player):**
```
App → ContentView → GameScene
  User Input → GameScene.keyDown/mouseDown
    → tryExecuteActionAndAnimate()
      → GameState.tryExecuteAction() [game logic]
      → animateActionResult() [visuals]
```

**Headless CLI Mode (ML Training):**
```
App → HeadlessGameCLI → StdinCommandReader
  Python stdin → executeStep()
    → HeadlessGame.step()
      → GameState.tryExecuteAction() [same logic]
      → ObservationBuilder.build() [state → observation]
```

**Visual CLI Mode (Watch ML):**
```
App → ContentView → GameScene + VisualGameController
  Python stdin → executeStep()
    → GameScene.tryExecuteActionAndAnimate()
      → GameState.tryExecuteAction() [same logic]
      → animateActionResult() [visuals]
      → Wait for animation → return observation
```

### Key Components

| Component | Responsibility |
|-----------|----------------|
| **GameState** | All game logic (movement, combat, programs, stage gen) - single source of truth |
| **GameScene** | Visual rendering, animation, user input handling |
| **ObservationBuilder** | Convert GameState → GameObservation for ML |
| **HeadlessGameCLI** | stdin/stdout protocol for headless mode |
| **VisualGameController** | stdin/stdout protocol for visual mode (syncs with animations) |
| **StdinCommandReader** | Parse JSON commands, encode responses |

---

## Python-Swift Bridge (ML Training)

### Architecture

```
python/test_env.py or train_maskable_ppo.py
    └── python/hack_env.py (Gymnasium environment)
            └── subprocess: HackMatrix --headless-cli
                    └── HeadlessGameCLI.swift (JSON stdin/stdout)
                            └── HeadlessGame.swift (game wrapper)
                                    └── GameState.tryExecuteAction()
```

Both GUI and headless modes use the same `GameState.tryExecuteAction()` core logic.

### JSON Protocol

Python sends one JSON command per line to stdin:
```json
{"action": "reset"}
{"action": "step", "actionIndex": 0}
{"action": "getValidActions"}
```

Swift responds with JSON on stdout:
```json
{"observation": {...}, "reward": 0.0, "done": false, "info": {}}
{"validActions": [0, 2, 4]}
```

### Action Space (31 actions)

| Index | Action |
|-------|--------|
| 0 | Move up |
| 1 | Move down |
| 2 | Move left |
| 3 | Move right |
| 4 | Siphon |
| 5-30 | Programs (26 total) |

### Observation Space

**Player state** (9 values): `[row, col, hp, credits, energy, stage, turn, dataSiphons, baseAttack]`

**Grid** (6×6×20): Each cell has 20 features encoding enemies, blocks, transmissions, resources, special cells.

**Flags**: `[showActivated]` - whether the "show" program has been used.

### Running Tests

```bash
cd python
source venv/bin/activate
python test_env.py
```

### Key Files

| File | Purpose |
|------|---------|
| `python/hack_env.py` | Gymnasium environment wrapper |
| `python/test_env.py` | Basic functionality test |
| `python/train_maskable_ppo.py` | MaskablePPO training script |
| `HackMatrix/HeadlessGameCLI.swift` | JSON protocol handler |
| `HackMatrix/HeadlessGame.swift` | Game state + observation encoding |

### Build Path

The Python environment expects the app at:
```
DerivedData/Build/Products/Debug/HackMatrix.app/Contents/MacOS/HackMatrix
```
This matches VSCode's build output location (see `.vscode/launch.json`).
