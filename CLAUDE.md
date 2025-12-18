# 868-HACK Game Architecture

## Project Conventions

- **Plan files** go in `plans/` directory

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

## Python-Swift Bridge (ML Training)

### Architecture

```
python/test_env.py or train_maskable_ppo.py
    └── python/hack_env.py (Gymnasium environment)
            └── subprocess: 868-hack --headless-cli
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
| `868-hack/HeadlessGameCLI.swift` | JSON protocol handler |
| `868-hack/HeadlessGame.swift` | Game state + observation encoding |

### Build Path

The Python environment expects the app at:
```
DerivedData/Build/Products/Debug/868-hack.app/Contents/MacOS/868-hack
```
This matches VSCode's build output location (see `.vscode/launch.json`).
