# HackMatrix Game Architecture

## Project Conventions

- **Plan files** go in `plans/` directory

### Python Scripts

- **Always activate venv** when running Python scripts
- Command pattern: `cd python && source venv/bin/activate && python <script>`
- Example: `cd python && source venv/bin/activate && python scripts/train.py`

### Building

**Hybrid build approach:**
- **Training (headless)**: `swift build` → `.build/debug/HackMatrix`
- **GUI app**: `xcodebuild -scheme HackMatrix -configuration Debug build` → `DerivedData/.../HackMatrix.app`

Python auto-selects the right binary based on mode.

**Validation commands:**
- Tests: `swift test`
- Build (headless): `swift build`
- Build (GUI): `xcodebuild -scheme HackMatrix -configuration Debug build`

**Notes:**
- SPM excludes `App.swift` and GUI code via conditional compilation
- Game logic is shared between SPM and Xcode builds
- Source code in `HackMatrix/` and `Sources/`, tests in `Tests/HackMatrixTests/`

### Dev Container (Linux Training)

Use VS Code Dev Containers for a full Linux dev environment with Swift, Python, and Claude Code.

**Setup:**
1. Install "Dev Containers" VS Code extension
2. Command Palette → "Dev Containers: Reopen in Container"
3. Wait for build (first run builds Swift and installs Python deps)

**Inside the container:**
```bash
# Run training
python3 python/scripts/train.py

# Run Claude Code
claude

# Rebuild Swift after code changes
swift build -c release
```

The `HACKMATRIX_BINARY` env var is automatically set to the built executable.

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

### Action Space (28 actions)

| Index | Action |
|-------|--------|
| 0 | Move up |
| 1 | Move down |
| 2 | Move left |
| 3 | Move right |
| 4 | Siphon |
| 5-27 | Programs (23 total) |

### Observation Space

**Player state** (10 values): `[row, col, hp, credits, energy, stage, dataSiphons, baseAttack, showActivated, scheduledTasksDisabled]`

**Programs** (23 values): Binary int32 vector indicating owned programs.

**Grid** (6x6x40): Each cell has 40 features encoding enemies, blocks, transmissions, resources, special cells.

### Running Python Tests

```bash
# Run all tests with pytest
cd python && source venv/bin/activate && pytest tests/ -v

# Run specific test file
cd python && source venv/bin/activate && pytest tests/test_movement.py -v
```

### Key Files

| File | Purpose |
|------|---------|
| `python/hack_env.py` | Gymnasium environment wrapper |
| `python/test_env.py` | Basic functionality test |
| `python/train_maskable_ppo.py` | MaskablePPO training script |
| `HackMatrix/HeadlessGameCLI.swift` | JSON protocol handler |
| `HackMatrix/HeadlessGame.swift` | Game state + observation encoding |

### Build Paths

Python uses different binaries based on mode:
- **Training** (headless): `.build/debug/HackMatrix` (SPM build)
- **Visual mode** (GUI): `DerivedData/HackMatrix/Build/Products/Debug/HackMatrix.app` (Xcode build)
