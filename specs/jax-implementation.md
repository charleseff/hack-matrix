# JAX Environment Implementation Spec

## Goal

Port the HackMatrix game environment to JAX to enable TPU-accelerated training via Google TPU Research Cloud.

## Background

### Current Architecture

```
┌─────────────────┐      JSON/stdin       ┌─────────────────┐
│  Python (SB3)   │ ◄──────────────────► │   Swift Game    │
│  MaskablePPO    │                       │   (CPU)         │
│  PyTorch        │                       │                 │
└─────────────────┘                       └─────────────────┘
```

- **RL Library**: Stable Baselines 3 (sb3-contrib) with MaskablePPO
- **Framework**: PyTorch
- **Environment**: Swift binary communicating via subprocess stdin/stdout JSON protocol
- **Parallelization**: Multiple subprocess instances (~8-32 practical limit)

### Why JAX?

With larger neural networks, compute becomes the bottleneck. Google TPU Research Cloud provides TPU access, but requires JAX-native code for the full benefit.

**Key insight**: PureJaxRL offers the best TPU performance by JIT-compiling the entire training loop (including environment). This requires the environment itself to be written in JAX.

### Target Architecture

```
┌─────────────────────────────────────────────────────┐
│                    TPU / GPU                        │
│  ┌─────────────────┐    ┌────────────────────────┐  │
│  │  HackMatrix Env │───►│   PPO Policy Network   │  │
│  │     (JAX)       │◄───│        (JAX)           │  │
│  └─────────────────┘    └────────────────────────┘  │
│            (entire loop JIT-compiled)               │
└─────────────────────────────────────────────────────┘
```

**Benefits**:
- Run 1000+ environments in parallel on a single TPU
- Entire training loop JIT-compiled (no Python overhead)
- Native TPU support via JAX

## Success Criteria

### Primary: Parity Tests Pass

All tests in `python/tests/parity/` must pass with `pytest -k jax`:

```bash
cd python && source venv/bin/activate && pytest tests/parity/ -k jax -v
```

This includes:
- `test_movement.py` - Player movement and line-of-sight attacks
- `test_enemies.py` - Enemy types, movement, pathfinding
- `test_programs.py` - All 23 programs with correct effects
- `test_siphon.py` - Siphon mechanics and transmissions
- `test_stages.py` - Stage generation and transitions
- `test_rewards.py` - Reward calculation
- `test_action_mask.py` - Action validity masking
- `test_turns.py` - Turn structure
- `test_edge_cases.py` - Edge cases
- `test_scheduled_tasks.py` - Scheduled task mechanics
- `test_interface_smoke.py` - Basic interface compliance

### Secondary: Implementation Tests Pass

Tests in `python/tests/implementation/` that use `get_internal_state()`:

```bash
cd python && source venv/bin/activate && pytest tests/implementation/ -k jax -v
```

### Additional Parity Criteria

Beyond passing tests, verify:

1. **Exact observation values** - Same normalized values for identical game states
2. **Deterministic behavior** - Same RNG seed produces identical trajectories
3. **Action masks match** - Identical valid action sets for same states
4. **Reward values match** - Identical reward calculations

## Implementation Plan

### Incremental Milestones

The implementation builds from simple to complex, with each milestone verified by a subset of parity tests.

#### Milestone 1: Core State and Observation

**Goal**: JAX env produces valid observations matching Swift format.

**Components**:
- `EnvState` dataclass with all game state fields
- `Observation` dataclass matching Swift's 42-feature grid
- `reset()` that initializes a valid stage 1 state
- `get_observation()` that encodes state to observation arrays
- `set_state()` for test compatibility

**Verification**: `test_interface_smoke.py` passes

#### Milestone 2: Player Movement

**Goal**: Player can move on the grid.

**Components**:
- Grid bounds checking
- Block collision detection
- Movement execution
- Turn counter increment

**Verification**: Basic movement tests in `test_movement.py` pass

#### Milestone 3: Enemy System

**Goal**: Enemies exist, are visible, and block movement.

**Components**:
- Enemy representation (fixed-size array with mask)
- Enemy visibility rules (Cryptog line-of-sight)
- Enemy presence in observations

**Verification**: `test_enemies.py` visibility tests pass

#### Milestone 4: Combat

**Goal**: Player can attack enemies, enemies can attack player.

**Components**:
- Line-of-sight attack detection
- Damage calculation (baseAttack)
- Enemy death and removal
- Player HP reduction
- Game over on HP = 0

**Verification**: Combat tests in `test_movement.py`, `test_enemies.py` pass

#### Milestone 5: Enemy Movement and Pathfinding

**Goal**: Enemies move toward player each turn.

**Components**:
- BFS pathfinding in JAX
- Enemy movement speeds (Virus = 2)
- Glitch can path through blocks
- Stunned/disabled enemy handling

**Verification**: Enemy movement tests pass

#### Milestone 6: Siphon and Transmissions

**Goal**: Siphon action works, transmissions spawn and tick.

**Components**:
- Siphon action execution
- Block siphoning logic
- Transmission creation
- Transmission countdown
- Enemy spawning from transmissions

**Verification**: `test_siphon.py` passes

#### Milestone 7: Stage Transitions

**Goal**: Completing stages works correctly.

**Components**:
- Exit detection (dynamic position, random corner different from player)
- Stage completion triggers: movement to exit OR WARP to exit
- Player position preserved (stays at exit after transition)
- HP gains +1 (up to max 3)
- Player resources preserved (credits, energy, score)
- Full stage generation:
  - New exit at random corner
  - Data siphons at remaining corners
  - 5-11 blocks placed randomly
  - Resources on empty cells
  - Transmissions based on stage number

**Verification**: `test_stages.py` passes

#### Milestone 8: Programs (Basic)

**Goal**: Simple programs work (WAIT, SIPH+, EXCH, SHOW, RESET, CALM).

**Components**:
- Program ownership tracking
- Cost deduction (credits, energy)
- Program applicability checking
- Basic program effects

**Verification**: Corresponding tests in `test_programs.py` pass

#### Milestone 9: Programs (Combat)

**Goal**: Combat programs work (PUSH, PULL, ROW, COL, DEBUG, HACK, D_BOM, ANTI-V).

**Components**:
- Enemy manipulation (push/pull)
- Area damage
- Stun effects

**Verification**: Combat program tests pass

#### Milestone 10: Programs (Complex)

**Goal**: Complex programs work (WARP, POLY, CRASH, UNDO, STEP, SCORE, REDUC, DELAY, ATK+).

**Components**:
- Random target selection (WARP) - triggers stage completion if target at exit
- Type transformation (POLY)
- State history (UNDO)
- Stage-scoped effects (ATK+)

**Verification**: All `test_programs.py` tests pass

#### Milestone 11: Scheduled Tasks

**Goal**: Scheduled spawns work correctly.

**Components**:
- Scheduled task interval calculation
- Next scheduled task turn tracking
- Transmission spawning from scheduled tasks
- CALM program disabling
- Siphon delay effect (+5 turns)

**Verification**: `test_scheduled_tasks.py` passes

#### Milestone 12: Rewards and Action Masking

**Goal**: Rewards match Swift, action masks are correct.

**Components**:
- Full reward calculation
- Complete action masking logic

**Verification**: `test_rewards.py`, `test_action_mask.py` pass

#### Milestone 13: Final Parity

**Goal**: All parity tests pass, implementation tests pass.

**Verification**: Full test suite green for JAX

### Phase 2: PureJaxRL Integration

Once the JAX environment is complete:

1. Implement action-masked PPO (or find existing implementation)
2. Integrate with PureJaxRL's training loop
3. Benchmark on TPU
4. Tune hyperparameters for massively parallel training

## Technical Approach

### JAX Constraints

JAX requires functional programming patterns:

1. **Pure functions**: No mutation, no side effects
   ```python
   # Bad (mutation)
   state.player.hp -= damage

   # Good (return new state)
   new_player = player._replace(hp=player.hp - damage)
   ```

2. **Fixed-size arrays**: Variable-length lists become max-size arrays with masks
   ```python
   # Max 20 enemies, with active mask
   enemies = jnp.zeros((MAX_ENEMIES, ENEMY_FEATURES))
   enemy_mask = jnp.zeros(MAX_ENEMIES, dtype=bool)
   ```

3. **No Python control flow on traced values**: Use JAX primitives
   ```python
   # Bad
   if enemy.hp <= 0:
       remove_enemy()

   # Good
   enemy_alive = enemy.hp > 0
   enemy_mask = enemy_mask & enemy_alive
   ```

4. **Use `jax.lax` for control flow**:
   - `jax.lax.cond(pred, true_fn, false_fn)` for if/else
   - `jax.lax.switch(index, branches)` for switch/case
   - `jax.lax.fori_loop(start, stop, body_fn, init)` for loops
   - `jax.lax.scan` for sequential operations

### State Structure

```python
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class Player:
    row: jnp.int32
    col: jnp.int32
    hp: jnp.int32           # 0-3
    credits: jnp.int32
    energy: jnp.int32
    data_siphons: jnp.int32
    base_attack: jnp.int32  # 1-2
    score: jnp.int32

@struct.dataclass
class EnvState:
    # Player
    player: Player

    # Grid: 6x6 cells
    # Content type: 0=empty, 1=data, 2=program, 3=question
    grid_content: jnp.ndarray      # (6, 6) int32
    grid_block_points: jnp.ndarray # (6, 6) int32 - points for data blocks
    grid_block_program: jnp.ndarray # (6, 6) int32 - program index for program blocks
    grid_block_siphoned: jnp.ndarray # (6, 6) bool
    grid_block_spawn_count: jnp.ndarray # (6, 6) int32
    grid_resources_credits: jnp.ndarray # (6, 6) int32
    grid_resources_energy: jnp.ndarray  # (6, 6) int32
    grid_data_siphon: jnp.ndarray  # (6, 6) bool
    grid_exit: jnp.ndarray         # (6, 6) bool - only (5,5) is True
    grid_siphon_center: jnp.ndarray # (6, 6) bool

    # Enemies: fixed-size array with mask
    enemies: jnp.ndarray           # (MAX_ENEMIES, 6) - type, row, col, hp, stunned, from_siphon
    enemy_mask: jnp.ndarray        # (MAX_ENEMIES,) bool

    # Transmissions: fixed-size array with mask
    transmissions: jnp.ndarray     # (MAX_TRANSMISSIONS, 4) - row, col, turns, enemy_type
    trans_mask: jnp.ndarray        # (MAX_TRANSMISSIONS,) bool

    # Programs owned: binary vector
    owned_programs: jnp.ndarray    # (23,) bool

    # Game state
    stage: jnp.int32
    turn: jnp.int32
    show_activated: jnp.bool_
    scheduled_tasks_disabled: jnp.bool_
    atk_plus_used_this_stage: jnp.bool_
    next_scheduled_task_turn: jnp.int32
    scheduled_task_interval: jnp.int32

    # For UNDO program
    previous_state: "EnvState | None"  # Optional, may need special handling

    # RNG key
    rng_key: jnp.ndarray

# Constants
# These are JAX implementation choices (Swift uses dynamic arrays)
# Values should be validated empirically - check max counts during training
MAX_ENEMIES = 20        # Estimate: 6x6 grid, ~half could be enemies at most
MAX_TRANSMISSIONS = 20  # Estimate: spawns are bounded by blocks siphoned
GRID_SIZE = 6
```

### Observation Encoding

Grid features (42 total per cell):
- Enemy (7): 4 type one-hot + hp/3 + isStunned + spawnedFromSiphon
- Block (5): 3 type one-hot + points/9 + isSiphoned
- Program (23): one-hot for program type on block
- Transmission (2): spawnCount/9 + turnsUntilSpawn/4
- Resources (2): credits/3 + energy/3
- Special (3): isDataSiphon + isExit + siphonCenter

Player state (10 values, normalized):
```
[row/5, col/5, hp/3, credits/50, energy/50, (stage-1)/7, dataSiphons/10, (baseAttack-1)/2, showActivated, scheduledTasksDisabled]
```

### Core Functions

```python
def reset(rng_key: jnp.ndarray) -> tuple[EnvState, Observation]:
    """Initialize a new game."""
    ...

def step(state: EnvState, action: int, rng_key: jnp.ndarray) -> tuple[EnvState, Observation, float, bool]:
    """Execute one action, return (new_state, observation, reward, done)."""
    ...

def get_valid_actions(state: EnvState) -> jnp.ndarray:
    """Return boolean mask of valid actions (28,)."""
    ...

def get_observation(state: EnvState) -> Observation:
    """Convert state to observation arrays."""
    ...

def set_state(state_dict: GameState) -> EnvState:
    """Create EnvState from test GameState (for parity testing)."""
    ...

def get_internal_state(state: EnvState) -> InternalState:
    """Extract internal state for implementation testing."""
    ...
```

## Implementation Plan Requirements

For the planning phase that produces an `IMPLEMENTATION_PLAN.md` , make sure it includes:

1. **JAX Code Structure** (required, detailed):
   - Module/file organization (what goes where)
   - Key class and function signatures with type annotations
   - Data flow between components (how state flows through step/reset/observe)
   - Helper function breakdown (what pure functions are needed)
   - How JAX constraints (no mutation, fixed arrays, lax control flow) are handled in each component

2. **Implementation Order**: Which functions/modules to implement first, with dependencies

3. **Test Strategy**: How to incrementally verify each piece against parity tests

The plan should be detailed enough that someone could implement the JAX environment by following it without needing to make significant architectural decisions.

## Development Workflow

1. **Local development**: Dev container with JAX CPU backend
   ```bash
   cd python && source venv/bin/activate
   pytest tests/parity/test_movement.py -k jax -v
   ```

2. **Incremental testing**: Run specific test files as milestones complete
   ```bash
   pytest tests/parity/test_movement.py -k jax -v  # After milestone 2
   pytest tests/parity/test_enemies.py -k jax -v   # After milestone 3
   # etc.
   ```

3. **Full parity check**: All tests against both implementations
   ```bash
   pytest tests/parity/ -v  # Runs both swift and jax
   ```

4. **TPU deployment**: Google TPU Research Cloud
   ```bash
   pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
   ```

## Open Questions

1. **UNDO program state**: How to handle previous_state in JAX? Options:
   - Store flattened state as array
   - Use pytree with special handling
   - Limit UNDO history to 1 step

2. **Action masking in PureJaxRL**: May need custom implementation.

3. **Vectorization limits**: How many parallel envs fit on TPU v2-8 / v3-8?

## References

- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax struct.dataclass](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html)
- [Action Masking in PPO](https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html)
