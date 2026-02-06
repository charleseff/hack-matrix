# Module Boundaries & Information Hiding Spec

**Status:** Draft
**Depends on:** None (foundational, can be implemented incrementally)

## Goal

Restructure both the Swift and Python codebases into **deep modules** with narrow, well-documented interfaces -- so that an agent (or new developer) can read a module's interface to understand what it does and how to use it, without reading any implementation code.

Inspired by Ousterhout's *A Philosophy of Software Design*: the best modules provide powerful functionality behind simple interfaces. The current codebase has several "god objects" with wide, leaky surfaces and no enforced boundaries. This spec defines the target module structure, the interface contracts, and the incremental path to get there.

## Principles (from Ousterhout)

These principles guide every decision in this spec:

1. **Deep modules over shallow modules.** A module's value = functionality provided minus interface complexity. Favor fewer modules with simple interfaces and rich implementations over many small modules with thin implementations.

2. **Information hiding.** Each module encapsulates design decisions. Implementation details (data structures, algorithms, internal state shapes) must not leak through the interface. If a change to module internals forces changes elsewhere, the boundary is wrong.

3. **Pull complexity downward.** A module's interface should be simpler than its implementation. The module author absorbs complexity so every caller doesn't have to. Configuration parameters and options pushed to callers are complexity exported upward.

4. **No classitis.** Don't split code into many small units for the sake of "small files." Split only where a meaningful abstraction boundary exists -- where the resulting interface genuinely hides something. Three similar lines of code in one place beat a premature abstraction.

5. **Different layer, different abstraction.** Adjacent layers should not mirror each other's interfaces. Pass-through methods that merely delegate with the same signature are a red flag.

6. **Better together or better apart?** Combine code that shares information, is always used together, or is hard to understand in isolation. Separate code that is genuinely independent.

## Current State

### Swift

| Problem | Location | Ousterhout Red Flag |
|---------|----------|-------------------|
| **God class** (2100 lines): game logic, 23 programs, enemy AI, stage gen, undo, rewards all in one type | `GameState.swift` | Shallow module (wide interface relative to abstraction) |
| **Mixed concerns** (960 lines): rendering + animation + input + HUD + game-over screens | `GameScene.swift` | Information leakage (animation details leak into game logic via ActionResult) |
| **ActionResult conflates 3 concerns**: gameplay outcomes, animation data, reward breakdown | `GameState.ActionResult` | Information leakage across layers |
| **No access control on internals**: GameScene reads `gameState.grid.cells[r][c]`, `gameState.enemies`, etc. directly | Throughout | No information hiding boundary |
| **Bidirectional coupling**: GameScene holds VisualGameController, which holds weak ref back to GameScene | `GameScene` / `VisualGameController` | Conjoined modules |
| **Flat namespace**: all types in one module, no enforced visibility boundaries | All `.swift` files | No mechanism to enforce hiding |

**Bright spots (keep these):**
- `GameCommandExecutor` protocol -- clean abstraction over headless vs. visual
- `ObservationBuilder` -- pure, single-responsibility, read-only
- `RewardCalculator` -- pure function, no dependencies
- `StdinCommandReader` -- clean protocol dispatch

### Python

| Problem | Location | Ousterhout Red Flag |
|---------|----------|-------------------|
| **Constants dumping ground**: 50+ constants + dataclasses + helpers all in one file, imported by every module | `jax_env/state.py` | Information leakage (every module depends on state internals) |
| **God orchestrator**: imports 8 sibling modules, orchestrates complex game loop | `jax_env/env.py` | Shallow orchestration (little hiding, mostly delegation) |
| **Monolithic dispatch**: 500+ lines, 23 program implementations in one function | `jax_env/programs.py` | Could be deeper (table-driven dispatch would narrow the interface) |
| **Mixed concerns**: subprocess management + JSON protocol + observation conversion + episode tracking | `gym_env.py` | Information leakage (subprocess details visible to callers) |
| **No documented public API**: `jax_env/__init__.py` re-exports 30+ symbols without documenting what they are or why you need them | `jax_env/__init__.py` | Interface wider than it needs to be |

**Bright spots (keep these):**
- `jax_env/` is pure-functional (no I/O, deterministic, vmappable)
- `purejaxrl/` is cleanly separated from game logic
- `pathfinding.py` -- genuinely deep (1 function, complex wavefront implementation)
- `siphon_quality.py` -- deep (1 public function, matrix operations hidden)
- No circular dependencies anywhere

## Design

### Guiding Heuristic

For each proposed module: **can an agent read only the interface and correctly use the module?** If not, the interface is too thin or leaks too much. If the interface is longer than a page, the module may be too shallow or doing too many unrelated things.

### Swift Target Structure

#### Module: `GameEngine` (the deep core)

**What it hides:** Turn sequencing, action validation, program execution details, enemy AI algorithms, stage generation, undo snapshots, scheduled tasks.

**Interface (protocol):**

```swift
protocol GameEngine {
    // State queries (read-only view)
    var observation: GameObservation { get }
    var validActions: [GameAction] { get }
    var isGameOver: Bool { get }
    var isGameWon: Bool { get }

    // Mutations
    func executeAction(_ action: GameAction) -> StepResult
    func reset()
}

struct StepResult {
    let outcome: TurnOutcome       // what happened (for any consumer)
    let animationData: AnimationData?  // nil in headless mode
    let rewardBreakdown: RewardBreakdown
}

enum TurnOutcome {
    case continued
    case stageAdvanced(newStage: Int)
    case playerDied
    case gameWon
}
```

**What this changes:** Today `GameScene` reads `gameState.grid.cells[r][c]` and `gameState.enemies` directly. Under this design, it reads `engine.observation` -- the same `GameObservation` the ML path uses. The grid internals, enemy arrays, and transmission timers are hidden.

**Why one protocol, not five:** Splitting into `ActionExecutor`, `StateQuerier`, `TurnManager`, etc. would be classitis -- each would be shallow and they would share most of their internal state. A single deep `GameEngine` protocol with ~6 members hides the most.

**ActionResult split rationale:** Today's `ActionResult` carries gameplay flags, animation keyframes, and reward components in one struct. These serve three unrelated consumers (game flow, renderer, RL). Splitting into `TurnOutcome` + `AnimationData` + `RewardBreakdown` means each consumer sees only what it needs. The renderer never sees reward math; the RL loop never sees sprite animation data.

#### Module: `ProgramExecutor` (internal to GameEngine)

**What it hides:** The 23 individual program implementations, cost validation, resource deduction.

**Interface (internal, not public):**

```swift
// Used only by GameEngine implementation, not by external callers
protocol ProgramExecutor {
    func execute(_ program: ProgramType, state: inout GameStateData, rng: inout RNG)
        -> ProgramResult
    func isValid(_ program: ProgramType, state: GameStateData) -> Bool
}
```

This is an internal seam, not a public API. It exists so that program logic can be tested independently and understood without reading the full engine. External callers never see it -- they just call `engine.executeAction(.program(.push))`.

#### Module: `EnemyController` (internal to GameEngine)

**What it hides:** Enemy pathfinding, attack patterns, transmission spawning, scheduled tasks, status effect reset.

**Interface (internal):**

```swift
protocol EnemyController {
    func processTurn(state: inout GameStateData) -> [EnemyAction]
    func tickTransmissions(state: inout GameStateData)
}
```

#### Module: `Renderer` (consumes GameEngine)

**What it hides:** SpriteKit nodes, sprite textures, cell coloring, HUD layout, animation timing.

**Interface:**

```swift
protocol GameRenderer {
    func render(observation: GameObservation)
    func animate(data: AnimationData, completion: @escaping () -> Void)
}
```

Today `GameScene` does rendering, animation, input, and game flow. This protocol carves out the rendering/animation responsibility. `GameScene` becomes a thin coordinator: receives input, calls `engine.executeAction()`, passes results to `renderer.animate()`.

**What this does NOT do:** It does not split `GameScene` into 5 files. That would be classitis. The `GameRenderer` protocol exists so that:
- Headless mode can provide a no-op renderer
- The animation contract is documented in one place
- GameScene's dependency on GameState internals is replaced by the `GameObservation` type

### Python Target Structure

#### Module: `jax_env` (the deep core)

**What it hides:** Grid representation, enemy/transmission array layout, program dispatch, reward component calculations, stage generation, pathfinding algorithms, siphon quality math.

**Public interface (`__init__.py`):**

```python
"""
HackMatrix JAX environment -- pure-functional, JIT-compilable game.

State lifecycle:
    state, obs = reset(key)
    state, obs, reward, done, breakdown = step(state, action, key)
    mask = get_valid_actions(state)  # boolean[28]

Batched (vmapped) variants:
    batched_reset, batched_step, batched_get_valid_actions

Types:
    EnvState   -- opaque game state (pass to step/get_valid_actions)
    Observation -- (player_state[10], programs[23], grid[6,6,42])

Constants:
    NUM_ACTIONS = 28
    GRID_SIZE = 6
    OBS_SHAPE = (1545,)  # flattened observation size
"""

# Lifecycle
from .env import reset, step, get_valid_actions
from .env import batched_reset, batched_step, batched_get_valid_actions

# Types (opaque to callers -- fields are implementation details)
from .state import EnvState
from .observation import Observation, get_observation

# Dimensional constants (needed by network architecture, wrappers)
NUM_ACTIONS: int
GRID_SIZE: int
OBS_SHAPE: tuple
```

**What changes:** Today `__init__.py` re-exports ~30 symbols including internal constants like `BLOCK_WALL`, `ENEMY_TROJAN`, `PROGRAM_PUSH_COST_CREDITS`. These are implementation details of how the grid is encoded. External callers (purejaxrl, tests) should not depend on them. Only `env_wrapper.py` and parity tests legitimately need grid encoding details -- they can import from `jax_env.state` directly as an "internal" import, documented as such.

**Narrowing the export surface does not mean moving code.** The files stay the same. Only `__init__.py` changes to export fewer symbols, with docstrings explaining the public API.

#### Module: `jax_env.state` (internal, not re-exported)

**What it hides (from external callers):** Grid feature layout, enemy type encoding, block type encoding, program cost tables, array sizing constants.

**Who may import it directly:** `env_wrapper.py` (needs `GRID_FEATURES` for obs flattening), parity tests (need encoding details to compare with Swift). These are documented exceptions.

**What changes:** Add a module docstring explaining that this is internal. Group constants semantically:

```python
# Instead of 50 flat constants:
#   ENEMY_TROJAN = 1, ENEMY_WORM = 2, ENEMY_CRYPTOG = 3, ...
#   BLOCK_WALL = 1, BLOCK_QUESTION = 2, ...
#
# Group into namespaces (frozen dataclasses or SimpleNamespace):
class EnemyType:
    TROJAN = 1
    WORM = 2
    CRYPTOG = 3
    # ...

class BlockType:
    WALL = 1
    QUESTION = 2
    # ...
```

This is a mechanical change that improves discoverability without altering any logic.

#### Module: `purejaxrl` (training framework)

**What it hides:** PPO math, GAE computation, action masking implementation, network architecture, checkpoint format, wandb integration.

**Interface (`__init__.py`):**

```python
"""
PureJaxRL training for HackMatrix.

Usage:
    config = TrainConfig(total_timesteps=10_000_000)
    config = config.auto_tune_for_device()
    train = make_chunked_train(config)
    train()

Resume from checkpoint:
    config = TrainConfig(checkpoint_dir="runs/run-42")
    # ... same as above, checkpoint is loaded automatically
"""

from .config import TrainConfig
from .training_loop import make_chunked_train
```

**What changes:** Today `__init__.py` exports `ActorCritic`, `Transition`, `HackMatrixGymnax`, and other internals. These are implementation details of the training loop. External callers (scripts) only need `TrainConfig` and `make_chunked_train`.

#### Module: `gym_env` (Swift bridge)

**What it hides:** Subprocess lifecycle, JSON protocol encoding/decoding, platform-specific binary paths, episode statistics accumulation.

No structural changes needed -- this module is already reasonably deep. The main improvement is documenting its interface clearly so that callers (training scripts, test_env.py) know they get a standard `gym.Env` and don't need to understand the Swift subprocess underneath.

### Interface Documentation Strategy

For an AI agent to benefit from module boundaries, the interfaces must be **readable without the implementation**. The approach differs by language:

**Swift:** Protocols in dedicated files (e.g., `GameEngineProtocol.swift`). An agent reads the protocol file to understand the module. The protocol includes doc comments explaining semantics, preconditions, and invariants.

**Python:** `__init__.py` with module-level docstrings and explicit `__all__`. An agent reads `__init__.py` to understand what the module provides. Type hints on the public functions provide the contract.

**Both:** Each module's interface should fit on one screen (~40 lines). If it doesn't, the module is either doing too many unrelated things (split it) or exposing too much (hide more).

## Incremental Migration Path

This is not a rewrite. Each step is independently valuable and mergeable:

### Phase 1: Document & Narrow (no logic changes)

1. Write `__init__.py` docstrings for `jax_env` and `purejaxrl` defining the public API
2. Reduce `jax_env/__init__.py` exports to the minimal public surface
3. Group Python constants into semantic namespaces in `state.py`
4. Add `__all__` to Python modules
5. Write `GameEngineProtocol.swift` expressing the target Swift interface (protocol only, not yet adopted)

### Phase 2: Swift Structural Splits (logic changes, same behavior)

6. Split `ActionResult` into `TurnOutcome` + `AnimationData` + `RewardBreakdown` (RewardBreakdown already exists)
7. Extract `ProgramExecutor` from `GameState` (move program logic to own file, called by GameState)
8. Extract `EnemyController` from `GameState` (move enemy turn logic to own file)
9. Have `GameState` conform to `GameEngine` protocol

### Phase 3: Swift Access Control

10. Make `GameState` internals (`grid`, `enemies`, `transmissions`, `gameHistory`) `private` or `internal`
11. Route `GameScene` through `GameEngine` protocol / `GameObservation` instead of direct field access
12. Break `GameScene` <-> `VisualGameController` bidirectional dependency via callback/delegate

### Phase 4: Validation

13. Verify all existing tests pass
14. Verify training parity (JAX rewards unchanged)
15. Verify headless and visual CLI modes work identically

## Files Affected

| File | Phase | Change |
|------|-------|--------|
| `python/hackmatrix/jax_env/__init__.py` | 1 | Narrow exports, add docstring |
| `python/hackmatrix/jax_env/state.py` | 1 | Group constants into namespaces |
| `python/hackmatrix/purejaxrl/__init__.py` | 1 | Narrow exports, add docstring |
| `HackMatrix/GameEngineProtocol.swift` | 1 | New: protocol definition |
| `HackMatrix/GameState.swift` | 2-3 | Extract programs/enemies, conform to protocol, restrict access |
| `HackMatrix/ProgramExecutor.swift` | 2 | New: extracted from GameState |
| `HackMatrix/EnemyController.swift` | 2 | New: extracted from GameState |
| `HackMatrix/GameScene.swift` | 3 | Use GameEngine protocol instead of direct GameState access |
| `HackMatrix/VisualGameController.swift` | 3 | Remove weak GameScene ref, use delegate/callback |

## Testing

### Behavioral equivalence (all phases)

- All existing Swift tests (`GameLogicTests`, `RewardCalculatorTests`) must pass unchanged
- All existing Python tests (`test_reward_parity`, `test_purejaxrl`) must pass unchanged
- Headless CLI JSON protocol must produce identical output for identical inputs
- No changes to game mechanics, reward values, or observation encoding

### Interface compliance (phase 1+)

- Python: `mypy --strict` on `__init__.py` public surface (if type stubs are added)
- Swift: compile-time protocol conformance check (if it compiles, the interface is satisfied)

### Information hiding verification (phase 3)

- `GameScene` has zero direct references to `GameState.grid`, `GameState.enemies`, `GameState.transmissions`, `GameState.gameHistory`
- `purejaxrl/` has zero imports from `jax_env.state` (only from `jax_env` top-level), except `env_wrapper.py`

## Success Criteria

- [ ] An agent can read `jax_env/__init__.py` (~40 lines) and correctly use `reset`/`step`/`get_valid_actions` without reading any implementation files
- [ ] An agent can read `GameEngineProtocol.swift` (~30 lines) and understand how to drive the game without reading `GameState.swift`
- [ ] `purejaxrl/` imports only `NUM_ACTIONS`, `OBS_SHAPE`, and `EnvState` from `jax_env` (plus `env_wrapper`'s documented internal import)
- [ ] `GameState.swift` is under 800 lines (down from 2100) after extracting programs and enemies
- [ ] `GameScene` accesses game state exclusively through the `GameEngine` protocol
- [ ] All existing tests pass with zero changes to test assertions
- [ ] No training regression (reward curves match pre-refactor within noise)

## Open Questions (Draft)

- **Swift package structure:** Should `GameEngine` be a separate Swift package/target to enforce visibility at the compiler level? Or are protocols + access control sufficient?
- **Python `state.py` namespacing:** Frozen dataclasses vs. `enum.IntEnum` vs. `SimpleNamespace` for constant groups? IntEnum has nice debug printing but adds overhead in JAX tracing.
- **How far to restrict `EnvState` field access?** JAX dataclasses need public fields for `jax.tree_util`. Could wrap with accessor functions, but that may be fighting the JAX idiom.
- **Animation data ownership:** Should `AnimationData` be produced by `GameEngine` (which then knows about animation) or derived by the renderer from `TurnOutcome` + before/after observations?
