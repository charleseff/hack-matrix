# Environment Parity Test Suite Spec

## Goal

Create a comprehensive test suite that validates environment implementations through a common interface. The Swift environment serves as the reference implementation.

This spec covers:
1. Defining the `EnvInterface` contract
2. Implementing `SwiftEnvWrapper` (including new `set_state` capability)
3. Writing exhaustive tests validated against Swift

The JAX implementation will later use these same tests to verify parity (see `jax-implementation.md`).

## Principles

1. **Interface-only testing**: All tests interact with environments through a common interface—no implementation-specific code
2. **Swift as source of truth**: Tests are written/validated against Swift env first
3. **Deterministic scenarios**: Tests set specific game states before sending actions
4. **Flexible assertions**: Each test checks what matters for that scenario (not necessarily every observation bit)

## Interface Contract

Both Swift and JAX environments must implement this interface:

```python
class EnvInterface(Protocol):
    def reset(self) -> Observation:
        """Reset to initial game state, return observation."""
        ...

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """Execute action, return (observation, reward, done, info)."""
        ...

    def get_valid_actions(self) -> list[int]:
        """Return list of valid action indices."""
        ...

    def set_state(self, state: GameState) -> Observation:
        """Set complete game state for test setup, return observation."""
        ...
```

### Observation Structure

```python
@dataclass
class Observation:
    player: PlayerObs      # [row, col, hp, credits, energy, stage, dataSiphons, baseAttack, showActivated, scheduledTasksDisabled]
    programs: list[int]    # 23 binary values indicating owned programs
    grid: np.ndarray       # (6, 6, 40) cell features
```

### GameState Structure (for set_state)

```python
@dataclass
class GameState:
    player: PlayerState
    enemies: list[Enemy]
    transmissions: list[Transmission]
    blocks: list[Block]
    resources: list[Resource]  # credits/energy on cells
    owned_programs: list[int]  # program indices
    stage: int
    turn: int
    # ... other state fields
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

## Architecture

### Test Runner

- **Framework**: pytest
- **Fixtures**:
  - `swift_env` — provides `SwiftEnvWrapper` (for comprehensive tests)
  - `env` — parameterized for smoke tests (both Swift and JAX)

```python
@pytest.fixture
def swift_env():
    return SwiftEnvWrapper()

@pytest.fixture(params=["swift", "jax"])
def env(request):
    """Used for interface smoke tests only."""
    if request.param == "swift":
        return SwiftEnvWrapper()
    else:
        return JaxEnvWrapper()
```

Comprehensive tests use `swift_env`. Interface smoke tests use `env` to verify both wrappers.

### Swift Environment Wrapper

Wraps the existing Swift binary (communicates via JSON stdin/stdout protocol).

Must implement:
- Existing `reset()`, `step()`, `get_valid_actions()` — already supported
- New `set_state()` — needs to be added (extend `--debug-scenario` capability to the protocol)

### JAX Environment Wrapper (Skeleton)

A minimal `JaxEnvWrapper` that implements `EnvInterface` with stub/dummy logic. This verifies the interface contract is implementable by JAX.

- `reset()` → returns dummy observation
- `step(action)` → returns dummy (obs, reward, done, info)
- `get_valid_actions()` → returns dummy mask
- `set_state(state)` → accepts state, returns dummy observation

A smoke test verifies the skeleton implements the interface. Real tests will fail against it until `jax-implementation.md` is complete.

## Implementation Order

### Phase 0: Corrections & Game Mechanics Documentation

1. **Read `everything_wrong_with_impl_plan.txt`** — absorb all corrections
2. **Update `IMPLEMENTATION_PLAN.md`** — fix incorrect test specifications
3. **Create `specs/game-mechanics.md`** — document authoritative game mechanics based on Swift code analysis

### Phase 1: Interface & Both Wrappers

1. Define the `EnvInterface` protocol in Python
2. Implement `SwiftEnvWrapper` with existing functionality
3. Add `set_state` command to Swift JSON protocol
4. Implement `set_state` in `SwiftEnvWrapper`
5. Implement `JaxEnvWrapper` skeleton (stub/dummy returns)
6. Write interface smoke tests that run against both wrappers (verifies interface contract)

### Phase 2: Comprehensive Test Cases

Enumerate and implement all test cases (see Test Cases section below). Use Swift env to validate tests are correct.

> **Note**: Full JAX implementation (making real tests pass) is deferred to `jax-implementation.md`. The skeleton just verifies interface compliance.

## Test Cases

> **Planning Phase Requirement**: Before implementation, expand `IMPLEMENTATION_PLAN.md` with fully detailed test cases. Each test case listed below (e.g., "Move to empty cell") must be specified with:
>
> 1. **Preconditions**: Exact `set_state` setup (player position, enemies, blocks, resources, owned programs, etc.)
> 2. **Action**: Which action index to execute
> 3. **Expected Observation Changes**: Specific changes to player state (10 values), grid channels (40 per cell), programs array
> 4. **Expected Reward**: Exact reward value with breakdown of components (per `RewardCalculator.swift`)
> 5. **Expected Valid Actions**: How the action mask should change after the action
> 6. **Variants**: All scenario variations that need separate test functions
>
> The planner must explore the Swift codebase (`GameState.swift`, `RewardCalculator.swift`, `Program.swift`, `ObservationBuilder.swift`, etc.) to extract exact mechanics and enumerate all edge cases. The resulting `IMPLEMENTATION_PLAN.md` will be substantially larger than the current version.

> **CRITICAL: Corrections Required**
>
> The file `everything_wrong_with_impl_plan.txt` documents known errors and missing details in the current `IMPLEMENTATION_PLAN.md`. Before implementing tests, the planner **MUST**:
>
> 1. Read and absorb all corrections in `everything_wrong_with_impl_plan.txt`
> 2. Update `IMPLEMENTATION_PLAN.md` to fix incorrect test specifications
> 3. Add any missing test cases identified in the corrections file
> 4. Create or update a game mechanics reference spec based on learnings
>
> Key corrections include (but are not limited to):
> - **Credits/energy are NOT collected by moving** — only by siphoning blocks
> - **Data siphons ARE collected by moving** — walk into a cell with a data siphon to pick it up
> - **Siphon is always valid** when player has data siphons (not dependent on adjacent blocks)
> - **Line-of-sight attacks** — player attacks enemies in line-of-sight even when >1 cell away
> - **Program side effects** — many programs have stun effects, block destruction, etc. not documented
> - **Data blocks** always have matching score and transmission count
> - **Program applicability** — many conditions beyond ownership/resources (see `isProgramApplicable` in `GameState.swift`)

### Categories

#### Movement Actions (0-3)

- Move to empty cell
- Move into wall (blocked)
- Move off grid edge (blocked)
- ~~Move onto cell with credits~~ *(INCORRECT: credits/energy are NOT collected by moving)*
- ~~Move onto cell with energy~~ *(INCORRECT: credits/energy are NOT collected by moving)*
- ~~Move onto cell with both resources~~ *(INCORRECT: credits/energy are NOT collected by moving)*
- **Move onto cell with data siphon**: Player collects the data siphon (this IS collected by moving)
- Move into adjacent enemy (attack, kill if damage >= HP)
- Move into adjacent enemy (attack, enemy survives)
- Move into block (blocked)
- Move into transmission (attack, destroys transmission)
- **Line-of-sight attack**: Direction toward enemy >1 cell away triggers attack (not move)
- **Line-of-sight attack on block**: Enemy on block in line-of-sight is attacked

#### Siphon Action (4)

- Siphon adjacent block (gain score, spawn transmissions)
- ~~Siphon with no adjacent block (invalid action?)~~ *(INCORRECT: siphon is always valid when player has data siphons)*
- ~~Siphon block that's already been siphoned~~ *(INCORRECT: siphon validity doesn't depend on block state)*
- Siphon effects on different block types (data blocks vs program blocks)
- **Data block invariant**: score and spawnCount are always equal
- Siphon reveals resources underneath block (after block is siphoned)
- Siphon spawns transmissions based on block's spawnCount
- **Data siphon acquisition**: Player gains data siphons by **walking into a cell containing one** (or via SIPH+ program)

#### Programs (5-27)

Each of the 23 programs needs tests for their specific effects and edge cases.

> **TODO**: List all 23 programs and their expected behaviors

**Important program details** (see `everything_wrong_with_impl_plan.txt` for full list):
- Many programs have **stun effects** in addition to damage (DEBUG, ROW, COL, ANTI-V, D_BOM)
- **CRASH** destroys blocks (siphoned or not), exposing resources underneath
- **WARP** teleports to a random enemy's position, killing that enemy
- **POLY** edge case: Daemon with 1 HP → Cryptog with 0 HP → dead
- **UNDO** reverses enemy positions too, not just player
- **SHOW** makes transmissions display incoming enemy type
- **SIPH+** grants player a data siphon (alternative to walking into data siphon cells)
- **D_BOM** does splash damage/stun to enemies adjacent to the destroyed daemon
- **HACK** destroys siphoned blocks, exposing resources underneath
- **Program applicability**: Many conditions beyond ownership/resources — see `isProgramApplicable` in `GameState.swift`

Note: We don't test "program not owned" or "insufficient energy" as invalid actions—action masking prevents these. Instead, we verify the action mask is correct (see Action Masking section).

#### Turn Mechanics

- Player action ends turn (move/attack/siphon)
- Program execution does NOT end turn (can chain)
- Wait program ends turn
- Turn counter increments correctly
- Enemy turn executes after player turn ends

#### Enemy Behavior

- Enemy spawning from transmissions
- Enemy movement (pathfinding toward player)
- Enemy movement with multiple equally-good options (assert one of valid outcomes)
- Enemy attack when adjacent to player
- Enemy status effects (stunned, disabled)

#### Stage Transitions

- Stage completion triggers
- New stage generation (test non-random parts):
  - Number of enemy spawns
  - Positions of enemies carried over
  - Player state preserved/reset appropriately
  - **Data block invariant**: all data blocks have matching score and spawnCount

#### Edge Cases

- Player death (HP reaches 0)
- Win condition

#### Rewards

Reward verification is critical for RL training correctness. Every test should verify the reward returned by `step()` matches expected values.

Reward tests should cover:
- Killing an enemy → positive reward (verify exact value)
- Taking damage → negative reward (verify exact value)
- ~~Collecting credits → reward (if applicable)~~ *(resources collected via siphon, not movement)*
- ~~Collecting energy → reward (if applicable)~~ *(resources collected via siphon, not movement)*
- Completing a stage → reward
- Player death → terminal reward
- Neutral actions (moving to empty cell) → zero or baseline reward
- Program execution rewards (if any)
- Siphon rewards
- Compound scenarios (e.g., kill enemy but take damage in same turn)

**Resource verification** (critical for parity):
- Program costs: verify correct credits/energy deducted for each program
- Test setup must ensure player has sufficient resources before executing programs

> **TODO**: Extract exact reward values from Swift implementation during planning phase.

#### Action Masking

Action mask verification is a core part of every test. After each `step()`, tests should call `get_valid_actions()` and verify the mask is correct for the resulting state.

Action mask tests should cover:
- Movement blocked by walls/edges → those directions masked
- Movement blocked by blocks → masked
- ~~Siphon only valid when adjacent to unsiphoned block~~ *(INCORRECT: siphon valid when player has data siphons)*
- **Siphon valid when player has data siphons available** (regardless of adjacent blocks)
- Programs masked when not owned
- Programs masked when insufficient credits/energy
- Programs masked based on other conditions (see `isProgramApplicable` in `GameState.swift`)
- All 4 directions independently calculated
- Mask changes correctly as state changes (e.g., gain energy → program becomes valid)

### Test Template

```python
def test_move_up_to_empty_cell(env):
    """Moving up to an empty cell should move the player."""
    # Arrange: Set up specific game state
    state = GameState(
        player=PlayerState(row=3, col=3, hp=3, ...),
        enemies=[],
        ...
    )
    obs = env.set_state(state)

    # Act: Send action
    obs, reward, done, info = env.step(0)  # Move up

    # Assert: Check observation
    assert obs.player.row == 2
    assert obs.player.col == 3
    assert done == False

    # Assert: Verify action mask is correct for new state
    valid_actions = env.get_valid_actions()
    assert 0 in valid_actions      # can move up (row 2 -> 1)
    assert 1 in valid_actions      # can move down (row 2 -> 3)
    # ... etc based on expected state
```

## Handling Non-Determinism

### Enemy Movement Ties

When an enemy has multiple equally-good movement options, both implementations may choose differently. Tests should:

```python
def test_enemy_moves_toward_player(env):
    # ... setup with enemy that has 2 equally good moves ...
    obs, _, _, _ = env.step(action)

    # Assert enemy is in ONE of the valid positions
    valid_positions = [(2, 3), (3, 2)]  # both equally good
    actual_position = (obs.enemies[0].row, obs.enemies[0].col)
    assert actual_position in valid_positions
```

### Stage Generation

New stages have randomness. Tests should check deterministic properties:

```python
def test_stage_transition_enemy_count(env):
    # ... trigger stage completion ...
    obs, _, _, _ = env.step(action)

    # Check non-random properties
    assert obs.player.stage == 2
    assert count_enemies(obs) == expected_new_enemy_count
    # Don't assert specific positions of newly spawned enemies
```

## Files

| File | Purpose |
|------|---------|
| `everything_wrong_with_impl_plan.txt` | **CRITICAL**: Corrections to IMPLEMENTATION_PLAN.md — must be absorbed before implementation |
| `IMPLEMENTATION_PLAN.md` | Detailed test specifications (must be updated with corrections) |
| `specs/game-mechanics.md` | *(To be created)* Authoritative game mechanics reference based on Swift code analysis |
| `python/tests/conftest.py` | pytest fixtures |
| `python/tests/env_interface.py` | `EnvInterface` protocol definition |
| `python/tests/swift_env_wrapper.py` | Swift env wrapper implementation |
| `python/tests/jax_env_wrapper.py` | JAX env wrapper skeleton (stub implementation) |
| `python/tests/test_interface_smoke.py` | Smoke tests verifying both wrappers implement interface |
| `python/tests/test_movement.py` | Movement action tests |
| `python/tests/test_siphon.py` | Siphon action tests |
| `python/tests/test_programs.py` | Program execution tests |
| `python/tests/test_enemies.py` | Enemy behavior tests |
| `python/tests/test_turns.py` | Turn mechanics tests |
| `python/tests/test_stages.py` | Stage transition tests |
| `python/tests/test_action_mask.py` | Action mask verification tests |

## Open Questions

1. **State serialization format**: What format should `set_state` accept? JSON matching Swift's internal representation?
2. **Observation comparison helpers**: Should we build utilities for comparing observations with tolerance for non-deterministic parts?
3. **Test coverage tooling**: How do we ensure we've covered all game mechanics?

## Success Criteria

1. **Corrections absorbed**: `everything_wrong_with_impl_plan.txt` reviewed and `IMPLEMENTATION_PLAN.md` updated
2. **Game mechanics documented**: `specs/game-mechanics.md` created with authoritative mechanics reference
3. `EnvInterface` protocol defined
4. `SwiftEnvWrapper` fully implements `EnvInterface`
5. `JaxEnvWrapper` skeleton implements `EnvInterface` (stub returns)
6. Interface smoke tests pass for both wrappers
7. `set_state` functionality added to Swift JSON protocol
8. All comprehensive tests pass against Swift environment
9. Test coverage includes all programs, all action types, key edge cases
