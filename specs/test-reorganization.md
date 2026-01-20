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
| `spawnedFromSiphon` flag | Enemy | Siphon-spawned enemies adjacent to player cause extra damage. |
| `isFromScheduledTask` flag | Enemy | Affects kill reward calculation. |
| `lastKnownRow/Col` | Enemy (Cryptog) | Hidden position hints for invisible enemies. |

### Medium Priority (Economy Impact)

| Hidden State | Location | Impact |
|--------------|----------|--------|
| `disabledTurns` | Enemy | Enemies disabled on spawn turn. |
| `isStunned` | Enemy | Persists exactly one turn, then resets. |
| `pendingSiphonTransmissions` | GameState | Defers siphon-spawned enemy creation. |
| `atkPlusUsedThisStage` | GameState | Prevents duplicate ATK+ usage per stage. |
| Block placement validation | Stage gen | Can fail silently after 100 attempts. |

### Lower Priority (Edge Cases)

| Hidden State | Location | Impact |
|--------------|----------|--------|
| `siphonCenter` | Cell | Vestigial? Purpose unclear. |
| `gameHistory` | GameState | Undo program restore fidelity. |
| Lifetime stats | GameState | totalSiphons, totalKills, etc. May affect nothing. |

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

Create tests that verify hidden state behavior. These require **extending the interface** or **direct implementation access**.

#### Option A: Extend Interface (Preferred for JAX parity)

Add `get_internal_state()` to `EnvInterface`:

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
```

Pros: JAX must implement same internal state, ensuring parity
Cons: Pollutes interface with test-only methods

#### Option B: Swift-Only Unit Tests

Test Swift internals directly without going through the wrapper:

```python
# tests/implementation/test_swift_internals.py
def test_siphon_reduces_scheduled_interval(swift_env):
    """Each siphon permanently reduces scheduledTaskInterval by 1."""
    # Would need Swift to expose this via debug protocol
    pass
```

Cons: Doesn't help JAX parity, requires Swift protocol changes anyway

#### Recommendation

Use **Option A** for critical hidden state that affects gameplay determinism. The interface extension is justified because JAX *must* implement the same internal mechanics.

For truly internal details (like `siphonCenter`), document them but don't test - they're implementation details that don't affect game outcomes.

### Phase 4: Specific Implementation-Level Tests

#### Scheduled Task Mechanics (`test_scheduled_tasks.py`)

```python
def test_initial_interval_depends_on_stage(env):
    """scheduledTaskInterval = 13 - stage at stage start."""
    # Stage 1: interval = 12
    # Stage 5: interval = 8
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
```

#### Enemy Spawn Source (`test_hidden_state.py`)

```python
def test_siphon_spawned_enemy_flag(env):
    """Enemies from siphoning have spawnedFromSiphon = true."""
    pass

def test_scheduled_spawned_enemy_flag(env):
    """Enemies from scheduled tasks have isFromScheduledTask = true."""
    pass

def test_siphon_spawned_adjacent_causes_extra_damage(env):
    """Adjacent siphon-spawned enemy deals +1 damage."""
    # This IS observable via HP change
    pass

def test_scheduled_enemy_kill_gives_no_reward(env):
    """Killing scheduled-task enemies gives 0 reward."""
    # This IS observable via reward
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

def test_data_siphons_placed_in_remaining_corners(env):
    """2 data siphons placed in corners without exit/player."""
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

## Open Questions

1. **Interface extension**: Should `get_internal_state()` be added to `EnvInterface`, or should implementation tests be Swift-only?
2. **Question block determinism**: Should we add seeding to make question block contents deterministic for testing?
3. **Lifetime stats**: Are these used anywhere? If not, can they be removed from Swift?

## Dependencies

- None (this spec should be completed before `jax-implementation.md`)

## References

- [env-parity-tests.md](./env-parity-tests.md) - Original parity test spec
- [game-mechanics.md](./game-mechanics.md) - Authoritative game mechanics
