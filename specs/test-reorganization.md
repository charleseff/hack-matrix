# Test Reorganization Spec

## Goal

Reorganize and expand the test suite before proceeding with the JAX implementation:

1. Rename and restructure tests as "parity tests"
2. Add scheduled tasks testing to parity tests
3. Create implementation-level tests for hidden/latent state
4. Make test running idiomatic (pytest config in pyproject.toml)

## Background

The current test suite validates environment behavior through the public interface (`reset`, `step`, `get_valid_actions`, `set_state`). These are **parity tests** - they ensure Swift and JAX implementations behave identically.

However, significant game state is **not exposed in observations**. This hidden state affects gameplay but can't be verified through parity tests alone. We need **implementation-level tests** to verify both Swift and JAX handle this hidden state correctly.

## Hidden State Analysis

Based on Swift codebase exploration, the following state is NOT in observations:

### High Priority (Game-Breaking if Wrong)

| Hidden State | Location | Impact |
|--------------|----------|--------|
| `scheduledTaskInterval` | GameState | Decreases with each siphon (min 4), persists through stages. Controls spawn pressure. |
| `nextScheduledTaskTurn` | GameState | Determines when transmissions spawn. Pure internal state. |
| Question block contents | Cell | Hidden until siphoned. Affects strategy and rewards. |
| `spawnedFromSiphon` flag | Enemy | **Significant negative reward** if player dies from a siphon-spawned enemy. Will be in obs space after prerequisite spec. |
| `isFromScheduledTask` flag | Enemy | Enemies with this flag give **no reward** when killed. NOT in obs space (internal only). |
| `lastKnownRow/Col` | Enemy (Cryptog) | Hidden position hints for invisible enemies. |

### Medium Priority (Economy Impact)

| Hidden State | Location | Impact |
|--------------|----------|--------|
| `disabledTurns` | Enemy | Counter that decrements each turn. Enemies spawn with `disabledTurns=1`. Enemies with `disabledTurns > 0` don't act. |
| `isStunned` | Enemy | Boolean flag set when enemy takes damage but survives. Auto-resets to `false` at start of enemy turn (one-turn effect). |
| `pendingSiphonTransmissions` | GameState | Defers siphon-spawned enemy creation. |
| Block placement validation | Stage gen | Can fail silently after 100 attempts. |

### Lower Priority (Edge Cases)

| Hidden State | Location | Impact |
|--------------|----------|--------|
| `gameHistory` | GameState | Undo program restore fidelity. |

**Note:** `siphonCenter`, `spawnedFromSiphon` (enemy obs), and `atkPlusUsesThisStage` are addressed in [observation-and-attack-fixes.md](./observation-and-attack-fixes.md).

## Implementation Plan

### Phase 1: Test Infrastructure

1. **Create `pyproject.toml`** with pytest configuration
2. **Delete `run_all_tests.py`** (redundant)
3. **Reorganize directory structure**:
   ```
   python/tests/
   ├── parity/                    # Interface-level parity tests
   │   ├── __init__.py
   │   ├── test_movement.py
   │   ├── test_siphon.py
   │   ├── test_programs.py
   │   ├── test_enemies.py
   │   ├── test_turns.py
   │   ├── test_stages.py
   │   ├── test_action_mask.py
   │   ├── test_edge_cases.py
   │   ├── test_rewards.py
   │   └── test_interface_smoke.py
   ├── implementation/            # Implementation-level tests
   │   ├── __init__.py
   │   ├── test_scheduled_tasks.py
   │   ├── test_hidden_state.py
   │   └── test_stage_generation.py
   ├── conftest.py               # Shared fixtures
   ├── env_interface.py          # Interface protocol
   ├── swift_env_wrapper.py      # Swift wrapper
   └── jax_env_wrapper.py        # JAX wrapper
   ```

4. **Update imports** in moved files

### Phase 2: Add Scheduled Tasks to Parity Tests

Add tests to `parity/test_scheduled_tasks.py` (new file) that verify observable effects:

```python
def test_scheduled_transmission_spawns_after_interval(swift_env):
    """Verify transmissions spawn at expected intervals."""
    # Set up state at specific stage
    # Execute WAIT actions to advance turns
    # Verify transmission appears in grid observation
    pass

def test_siphon_delays_scheduled_spawn(swift_env):
    """Siphoning adds +5 turn delay to next scheduled spawn."""
    # Compare transmission timing with/without siphon
    pass

def test_calm_program_disables_scheduled_tasks(swift_env):
    """CALM program sets scheduledTasksDisabled, stopping spawns."""
    # Execute CALM, verify no spawns for expected duration
    pass
```

These test **observable effects** of scheduled tasks, not internal state.

### Phase 3: Implementation-Level Tests

Create tests that verify hidden state behavior by extending the interface.

**Decision:** Add `get_internal_state()` to `EnvInterface`. Even though it's only used for tests, this preserves parity - JAX must implement the same internal mechanics.

```python
class EnvInterface(Protocol):
    # ... existing methods ...

    def get_internal_state(self) -> InternalState:
        """Return internal state for implementation-level testing."""
        ...

@dataclass
class InternalState:
    scheduled_task_interval: int
    next_scheduled_task_turn: int
    pending_siphon_transmissions: int
    enemies: list[EnemyInternalState]
    # ... etc

@dataclass
class EnemyInternalState:
    row: int
    col: int
    hp: int
    disabled_turns: int
    is_stunned: bool
    spawned_from_siphon: bool
    is_from_scheduled_task: bool
```

### Phase 4: Specific Implementation-Level Tests

#### Scheduled Task Mechanics (`test_scheduled_tasks.py`)

**Note:** Many scheduled task mechanics can be tested via observation space by moving multiple times on an empty board (no enemies to kill player). Use `get_internal_state()` for verifying exact internal values.

```python
def test_initial_interval_depends_on_stage(env):
    """scheduledTaskInterval = 13 - stage at stage start."""
    # Stage 1: interval = 12
    # Stage 5: interval = 8
    # Verify via get_internal_state()
    pass

def test_siphon_permanently_reduces_interval(env):
    """Each siphon reduces interval by 1 (min 4)."""
    # Siphon 5 times at stage 1
    # Interval should be 12 - 5 = 7
    pass

def test_interval_reduction_persists_through_stages(env):
    """Siphon reductions carry over to next stage."""
    # Siphon 3 times at stage 1 (interval: 12 -> 9)
    # Complete stage
    # Stage 2 base: 11, but with reductions: 11 - 3 = 8
    pass

def test_siphon_adds_temporary_delay(env):
    """Siphoning adds +5 turns to nextScheduledTaskTurn."""
    pass

def test_transmission_spawns_observable_via_moves(env):
    """Verify transmissions appear in observation after enough moves."""
    # Set up empty board (no enemies)
    # Move repeatedly until transmission appears in grid observation
    # This tests observable effects without get_internal_state()
    pass
```

#### Enemy Spawn Source (`test_hidden_state.py`)

```python
def test_siphon_spawned_enemy_flag_in_observation(env):
    """Enemies from siphoning have spawnedFromSiphon visible in obs."""
    # After prerequisite spec, spawnedFromSiphon is in enemy observation
    # Siphon a block, verify spawned enemy has flag set in observation
    pass

def test_scheduled_spawned_enemy_flag(env):
    """Enemies from scheduled tasks have isFromScheduledTask = true."""
    # Via get_internal_state()
    pass

def test_death_from_siphon_enemy_gives_large_negative_reward(env):
    """Dying to siphon-spawned enemy gives significant negative reward."""
    # Set up via set_state:
    # - Player with 1 HP
    # - spawnedFromSiphon enemy adjacent to player
    # - Another enemy adjacent for player to attack
    # Player attacks the other enemy, causing siphon enemy to attack and kill player
    # Verify large negative reward
    pass

def test_scheduled_enemy_kill_gives_no_reward(env):
    """Killing scheduled-task enemies gives 0 reward."""
    # Set up via set_state with isFromScheduledTask=true enemy
    # Kill it, verify 0 reward
    pass
```

#### Stage Generation (`test_stage_generation.py`)

```python
def test_block_count_matches_stage(env):
    """Correct number of blocks generated per stage."""
    # Derive from grid observation - count cells with block features
    pass

def test_exit_position_is_corner(env):
    """Exit is always in a corner (not player's starting corner)."""
    pass

def test_data_siphons_in_other_two_corners(env):
    """2 data siphons placed in the 2 corners without exit/player."""
    # Player starts in one corner, exit in another
    # Data siphons must be in the remaining 2 corners
    pass

def test_question_block_contents_consistent(env):
    """Question block reveals consistent contents when siphoned."""
    # Multiple siphons of same question block (via undo) give same result
    pass
```

## pyproject.toml Configuration

```toml
[project]
name = "hackmatrix-tests"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short"
markers = [
    "requires_set_state: mark test as requiring set_state() support",
    "implementation: mark test as implementation-level (not interface parity)",
]

# Run parity tests by default, use -m for filtering:
#   pytest                        # all tests
#   pytest tests/parity/          # parity tests only
#   pytest tests/implementation/  # implementation tests only
#   pytest -k swift               # swift env only
#   pytest -k jax                 # jax env only
```

## Migration Steps

1. Create `python/pyproject.toml`
2. Create `python/tests/parity/` directory
3. Move test files: `test_*.py` → `parity/test_*.py`
4. Create `python/tests/parity/__init__.py`
5. Create `python/tests/implementation/` directory
6. Create `python/tests/implementation/__init__.py`
7. Update imports in conftest.py if needed
8. Delete `run_all_tests.py`
9. Verify tests still pass: `cd python && pytest tests/`

## Success Criteria

1. [ ] `pyproject.toml` created with pytest config
2. [ ] `run_all_tests.py` deleted
3. [ ] Tests reorganized into `parity/` and `implementation/` subdirectories
4. [ ] All existing tests pass after reorganization
5. [ ] Scheduled task parity tests added (observable effects)
6. [ ] Decision made on interface extension for implementation-level tests
7. [ ] Implementation-level tests added for high-priority hidden state

## Resolved Questions

1. **Interface extension**: ✅ Yes, `get_internal_state()` will be added to `EnvInterface` to preserve parity in tests.
2. **Question block determinism**: Not needed - can test consistency via undo/redo without seeding.
3. **Lifetime stats**: Don't exist in codebase - removed from spec.

## Dependencies

- [observation-and-attack-fixes.md](./observation-and-attack-fixes.md) - Must be completed first (adds `siphonCenter` to obs, ATK+ twice per stage, `set_state` enemy flags)
- This spec should be completed before `jax-implementation.md`

## References

- [observation-and-attack-fixes.md](./observation-and-attack-fixes.md) - Prerequisite fixes
- [env-parity-tests.md](./env-parity-tests.md) - Original parity test spec
- [game-mechanics.md](./game-mechanics.md) - Authoritative game mechanics
