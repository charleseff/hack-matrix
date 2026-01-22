# JAX Environment Implementation Plan

**Goal**: Port HackMatrix game environment to pure JAX for TPU-accelerated training.

**Success Criteria**: All parity tests pass with `pytest tests/parity/ -k jax -v`

**Current Status**: 148 tests passing, 0 tests failing (ALL TESTS PASS!)

## Current State Assessment

### What Exists
- **Dummy JAX environment** (`python/hackmatrix/jax_env.py`): Stub with zeroed observations and random termination
- **JAX wrapper** (`python/tests/jax_env_wrapper.py`): Adapter implementing EnvInterface, raises NotImplementedError for set_state/get_internal_state
- **Parity test suite** (`python/tests/parity/`): 11 test files covering all game mechanics
- **Swift reference implementation**: Complete game logic in `HackMatrix/*.swift`
- **Observation encoding**: 42-feature grid documented in `observation_utils.py`

### What's Missing
- [x] Full JAX `EnvState` dataclass with all game state fields
- [x] Game logic (movement, combat, siphon, programs, enemy AI)
- [x] Observation building from JAX state
- [x] `set_state()` for test compatibility
- [x] `get_internal_state()` for implementation tests

## JAX Code Structure

### File Organization

| File | Purpose | Status |
|------|---------|--------|
| `jax_env.py` | Main environment - reset, step, get_valid_actions | [x] Complete |
| `jax_state.py` | EnvState dataclass and constants | [x] Complete |
| `jax_observation.py` | Observation building from state | [x] Complete |
| `jax_actions.py` | Action execution (movement, siphon, programs) | [x] Complete |
| `jax_enemy.py` | Enemy logic (movement, pathfinding, attacks) | [x] Complete |
| `jax_stage.py` | Stage generation | [x] Complete |
| `jax_rewards.py` | Reward calculation | [x] Complete |
| `jax_utils.py` | Common JAX utilities | [x] Complete |

### Data Structures

#### EnvState (in `jax_state.py`)

```python
import jax.numpy as jnp
from flax import struct

# Constants
GRID_SIZE = 6
MAX_ENEMIES = 20
MAX_TRANSMISSIONS = 20
NUM_PROGRAMS = 23
NUM_ACTIONS = 28

# Enemy type encoding
ENEMY_VIRUS = 0
ENEMY_DAEMON = 1
ENEMY_GLITCH = 2
ENEMY_CRYPTOG = 3

@struct.dataclass
class Player:
    row: jnp.int32
    col: jnp.int32
    hp: jnp.int32              # 0-3
    credits: jnp.int32
    energy: jnp.int32
    data_siphons: jnp.int32
    attack_damage: jnp.int32   # 1-3
    score: jnp.int32

@struct.dataclass
class EnvState:
    # Player
    player: Player

    # Grid: 6x6 - multiple arrays for different cell properties
    # Block type: 0=empty, 1=data, 2=program, 3=question
    grid_block_type: jnp.ndarray          # (6, 6) int32
    grid_block_points: jnp.ndarray        # (6, 6) int32, for data blocks
    grid_block_program: jnp.ndarray       # (6, 6) int32, program index (0-22) for program blocks
    grid_block_spawn_count: jnp.ndarray   # (6, 6) int32, transmissions to spawn
    grid_block_siphoned: jnp.ndarray      # (6, 6) bool
    grid_siphon_center: jnp.ndarray       # (6, 6) bool, where player siphoned from
    grid_resources_credits: jnp.ndarray   # (6, 6) int32
    grid_resources_energy: jnp.ndarray    # (6, 6) int32
    grid_data_siphon: jnp.ndarray         # (6, 6) bool, data siphon pickups
    grid_exit: jnp.ndarray                # (6, 6) bool, only (5,5) is True

    # Enemies: fixed-size array with mask
    # Each enemy: [type, row, col, hp, disabled_turns, is_stunned, spawned_from_siphon, is_from_scheduled_task]
    enemies: jnp.ndarray                  # (MAX_ENEMIES, 8) int32
    enemy_mask: jnp.ndarray               # (MAX_ENEMIES,) bool

    # Transmissions: fixed-size array with mask
    # Each: [row, col, turns_remaining, enemy_type, spawned_from_siphon, is_from_scheduled_task]
    transmissions: jnp.ndarray            # (MAX_TRANSMISSIONS, 6) int32
    trans_mask: jnp.ndarray               # (MAX_TRANSMISSIONS,) bool

    # Programs owned: binary vector
    owned_programs: jnp.ndarray           # (23,) bool

    # Game state
    stage: jnp.int32                      # 1-8, 9 = victory
    turn: jnp.int32
    show_activated: jnp.bool_
    scheduled_tasks_disabled: jnp.bool_
    step_active: jnp.bool_                # STEP program effect
    atk_plus_uses_this_stage: jnp.int32   # 0, 1, or 2

    # Scheduled task timing
    next_scheduled_task_turn: jnp.int32
    scheduled_task_interval: jnp.int32
    pending_siphon_transmissions: jnp.int32

    # For UNDO program - store previous state as flattened array
    # We'll store key fields needed to restore state
    previous_state_valid: jnp.bool_
    previous_player: Player
    previous_enemies: jnp.ndarray         # (MAX_ENEMIES, 8)
    previous_enemy_mask: jnp.ndarray      # (MAX_ENEMIES,)
    previous_transmissions: jnp.ndarray   # (MAX_TRANSMISSIONS, 6)
    previous_trans_mask: jnp.ndarray      # (MAX_TRANSMISSIONS,)
    previous_turn: jnp.int32

    # RNG key
    rng_key: jnp.ndarray

    # Reward tracking (for computing delta rewards)
    prev_score: jnp.int32
    prev_hp: jnp.int32
    prev_credits: jnp.int32
    prev_energy: jnp.int32
    cumulative_reward: jnp.float32
```

### Core Functions

#### Main Environment (`jax_env.py`)

```python
def reset(key: jax.Array) -> tuple[EnvState, Observation]:
    """Initialize new game, generate stage 1."""
    key, subkey = jax.random.split(key)
    state = _generate_initial_state(subkey)
    obs = get_observation(state)
    return state, obs

def step(state: EnvState, action: int, key: jax.Array) -> tuple[EnvState, Observation, float, bool]:
    """Execute action, return (new_state, obs, reward, done)."""
    # 1. Save state for UNDO (if turn-ending action)
    # 2. Execute action based on type (0-3: move, 4: siphon, 5-27: program)
    # 3. If turn ends (move/siphon/WAIT):
    #    a. Tick transmissions, spawn enemies
    #    b. Move enemies (unless STEP active)
    #    c. Enemy attacks
    #    d. Check scheduled tasks
    #    e. Reset stun flags
    # 4. Check for stage completion (player at exit)
    # 5. Check for game over (player HP = 0 or stage > 8)
    # 6. Calculate reward
    # 7. Build observation
    return new_state, obs, reward, done

def get_valid_actions(state: EnvState) -> jax.Array:
    """Return boolean mask of shape (28,)."""
    # Movement (0-3): in bounds, no block (or enemy in LOS for attack)
    # Siphon (4): player.data_siphons > 0
    # Programs (5-27): owned, have resources, applicability condition met
    return mask
```

#### Observation Building (`jax_observation.py`)

```python
def get_observation(state: EnvState) -> Observation:
    """Convert EnvState to observation arrays."""
    player_obs = _encode_player(state.player, state.stage,
                                 state.show_activated, state.scheduled_tasks_disabled)
    program_obs = state.owned_programs.astype(jnp.int32)
    grid_obs = _encode_grid(state)
    return Observation(player_state=player_obs, programs=program_obs, grid=grid_obs)

def _encode_player(player: Player, stage: int, show: bool, calm: bool) -> jax.Array:
    """Encode player state to (10,) normalized array."""
    return jnp.array([
        player.row / 5.0,
        player.col / 5.0,
        player.hp / 3.0,
        jnp.minimum(player.credits / 50.0, 1.0),
        jnp.minimum(player.energy / 50.0, 1.0),
        (stage - 1) / 7.0,
        player.data_siphons / 10.0,
        (player.attack_damage - 1) / 2.0,
        jnp.float32(show),
        jnp.float32(calm),
    ], dtype=jnp.float32)

def _encode_grid(state: EnvState) -> jax.Array:
    """Encode grid to (6, 6, 42) array."""
    # For each cell, encode:
    # [0-3]: Enemy type one-hot (virus, daemon, glitch, cryptog)
    # [4]: Enemy HP / 3
    # [5]: Enemy stunned
    # [6]: Enemy spawned_from_siphon
    # [7-9]: Block type one-hot (data, program, question)
    # [10]: Block points / 9
    # [11]: Block siphoned
    # [12-34]: Program type one-hot (23 programs)
    # [35]: Transmission spawn count / 9
    # [36]: Transmission turns remaining / 4
    # [37]: Cell credits / 3
    # [38]: Cell energy / 3
    # [39]: Is data siphon pickup
    # [40]: Is exit
    # [41]: Is siphon center
    grid = jnp.zeros((6, 6, 42), dtype=jnp.float32)
    # ... populate using vmap over cells
    return grid
```

#### Action Execution (`jax_actions.py`)

```python
def execute_action(state: EnvState, action: int, key: jax.Array) -> tuple[EnvState, bool]:
    """Execute action, return (new_state, turn_ends)."""
    return jax.lax.switch(
        jnp.minimum(action, 5),  # Clamp programs to single branch
        [
            lambda: _execute_move(state, 0, key),  # up
            lambda: _execute_move(state, 1, key),  # down
            lambda: _execute_move(state, 2, key),  # left
            lambda: _execute_move(state, 3, key),  # right
            lambda: _execute_siphon(state, key),   # siphon
            lambda: _execute_program(state, action, key),  # programs 5-27
        ]
    )

def _execute_move(state: EnvState, direction: int, key: jax.Array) -> tuple[EnvState, bool]:
    """Execute movement or line-of-sight attack."""
    # Check for enemy/transmission in line of sight
    # If found: attack (damage enemy/destroy transmission)
    # If not: move player to adjacent cell
    # Collect data siphon if present
    # Turn ends
    return new_state, True

def _execute_siphon(state: EnvState, key: jax.Array) -> tuple[EnvState, bool]:
    """Execute siphon action on best adjacent block."""
    # Find best adjacent block (program > high-point data > low-point data)
    # Mark block as siphoned, collect rewards (program/points/credits/energy)
    # Spawn transmissions based on block spawn count
    # Decrement data_siphons
    # Delay next scheduled task by 5 turns
    # Turn ends
    return new_state, True

def _execute_program(state: EnvState, action: int, key: jax.Array) -> tuple[EnvState, bool]:
    """Execute a program (action indices 5-27)."""
    program_idx = action - 5
    # Deduct costs, execute effect based on program type
    # Only WAIT ends the turn
    turn_ends = (program_idx == 5)  # WAIT is index 5 (action 10)
    return new_state, turn_ends
```

#### Enemy Logic (`jax_enemy.py`)

```python
def tick_transmissions(state: EnvState) -> EnvState:
    """Decrement transmission timers, spawn enemies at 0."""
    # For each transmission with turns_remaining == 0:
    #   Create enemy at that position
    #   Remove transmission from array
    return new_state

def move_enemies(state: EnvState) -> EnvState:
    """Move all non-stunned, non-disabled enemies toward player."""
    # Skip if step_active
    # For each enemy:
    #   If stunned or disabled: skip
    #   Use BFS to find path to player
    #   Move 1 step (2 for virus)
    #   Glitch can path through blocks
    return new_state

def enemy_attacks(state: EnvState) -> EnvState:
    """Adjacent enemies attack player."""
    # Count adjacent enemies (non-stunned)
    # Reduce player HP by count
    return new_state

def bfs_pathfind(state: EnvState, start_row: int, start_col: int,
                  can_walk_blocks: bool) -> jax.Array:
    """BFS to find next step toward player. Returns (row_delta, col_delta)."""
    # Implemented using jax.lax.while_loop or fori_loop
    # Returns direction to move
    pass
```

#### Stage Generation (`jax_stage.py`)

```python
def generate_stage(state: EnvState, key: jax.Array) -> EnvState:
    """Generate a new stage."""
    # Reset player position to (0, 0)
    # Clear grid, enemies, transmissions
    # Place blocks based on stage number
    # Place data siphon pickups
    # Place resources (credits/energy)
    # Spawn initial enemies
    # Reset stage-scoped flags (atk_plus_uses)
    return new_state

def _place_blocks(key: jax.Array, stage: int) -> tuple[jax.Array, ...]:
    """Randomly place blocks on grid."""
    # Returns grid arrays for blocks
    pass
```

#### Reward Calculation (`jax_rewards.py`)

```python
def calculate_reward(prev_state: EnvState, new_state: EnvState,
                      stage_completed: bool, game_won: bool,
                      player_died: bool) -> float:
    """Calculate reward for this transition."""
    reward = 0.0

    # Stage completion: [1, 2, 4, 8, 16, 32, 64, 100]
    stage_rewards = jnp.array([1, 2, 4, 8, 16, 32, 64, 100])
    reward += jax.lax.cond(stage_completed,
                           lambda: stage_rewards[prev_state.stage - 1],
                           lambda: 0.0)

    # Score gain: delta * 0.5
    score_delta = new_state.player.score - prev_state.player.score
    reward += score_delta * 0.5

    # Kills: count * 0.3 (track via enemy_mask changes)
    # Data siphon collected: 1.0
    # Distance shaping: delta * 0.05
    # Victory: 500 + score * 100
    # Death: -cumulative * 0.5
    # HP loss: -1.0 per HP
    # HP gain: +1.0 per HP
    # Resource gain: (credits + energy) * 0.05

    return reward
```

### JAX Constraint Handling

#### Fixed-Size Arrays

Enemies and transmissions use fixed-size arrays with masks:

```python
# Adding an enemy
def add_enemy(enemies: jax.Array, mask: jax.Array,
              enemy_data: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Find first False in mask
    idx = jnp.argmin(mask)
    # Only add if there's space
    has_space = ~mask.all()
    new_enemies = jax.lax.cond(
        has_space,
        lambda: enemies.at[idx].set(enemy_data),
        lambda: enemies
    )
    new_mask = jax.lax.cond(
        has_space,
        lambda: mask.at[idx].set(True),
        lambda: mask
    )
    return new_enemies, new_mask

# Removing an enemy (by setting mask to False)
def remove_enemy(enemies: jax.Array, mask: jax.Array, idx: int) -> jax.Array:
    return mask.at[idx].set(False)
```

#### Control Flow

Use JAX control flow primitives:

```python
# Conditional execution
new_hp = jax.lax.cond(
    enemy.hp > damage,
    lambda: enemy.hp - damage,
    lambda: 0
)

# Switch on action type
result = jax.lax.switch(action_type, [fn0, fn1, fn2, fn3])

# Loop over enemies
def body_fn(i, state):
    enemy_active = state.enemy_mask[i]
    # ... process enemy i if active
    return updated_state

final_state = jax.lax.fori_loop(0, MAX_ENEMIES, body_fn, state)
```

#### Pure Functions

All state updates return new state:

```python
# Bad: mutation
state.player.hp -= damage

# Good: return new state
new_player = state.player.replace(hp=state.player.hp - damage)
new_state = state.replace(player=new_player)
```

## Implementation Order

### Phase 1: Foundation (Milestone 1) - COMPLETE
- [x] `jax_state.py`: Define EnvState dataclass with all fields
- [x] `jax_observation.py`: Implement observation encoding
- [x] `jax_env.py`: Basic reset() that creates valid initial state
- [x] Update `jax_env_wrapper.py`: Implement set_state() for tests
- [x] **Test**: `test_interface_smoke.py` passes

### Phase 2: Movement (Milestone 2) - COMPLETE
- [x] `jax_actions.py`: Implement _execute_move() for basic movement
- [x] Grid bounds checking and block collision
- [x] Turn counter increment
- [x] **Test**: Basic movement tests in `test_movement.py` pass

### Phase 3: Enemies (Milestones 3-5) - COMPLETE
- [x] Enemy representation and visibility (Cryptog line-of-sight)
- [x] Line-of-sight attack detection
- [x] Combat (damage calculation, enemy death)
- [x] Enemy movement and BFS pathfinding
- [x] Enemy attacks on player
- [x] **Test**: `test_enemies.py`, combat tests in `test_movement.py` pass

### Phase 4: Siphon (Milestone 6) - COMPLETE
- [x] Siphon action execution
- [x] Block siphoning logic (collecting resources, marking blocks)
- [x] Transmission creation and countdown
- [x] Enemy spawning from transmissions
- [x] **Test**: `test_siphon.py` passes

### Phase 5: Stage Transitions (Milestone 7) - COMPLETE
- [x] Exit detection at (5,5) - only movement actions (0-3) trigger stage completion
- [x] Stage generation
- [x] Player stat preservation (HP not restored)
- [x] **Test**: `test_stages.py` passes

**Key Fix**: Only movement actions (0-3) trigger stage completion when player reaches exit at (5,5). Programs like WARP can teleport to exit position but don't complete the stage, matching Swift behavior.

### Phase 6: Basic Programs (Milestone 8) - COMPLETE
- [x] Program ownership tracking
- [x] Cost deduction (credits, energy)
- [x] Applicability checking
- [x] Implement: WAIT
- [x] Implement: SIPH+
- [x] Implement: EXCH
- [x] Implement: SHOW
- [x] Implement: RESET
- [x] Implement: CALM
- [x] **Test**: Corresponding tests in `test_programs.py` pass

### Phase 7: Combat Programs (Milestone 9) - COMPLETE
- [x] Implement: PUSH
- [x] Implement: PULL
- [x] Implement: ROW
- [x] Implement: COL
- [x] Implement: DEBUG
- [x] Implement: HACK
- [x] Implement: D_BOM
- [x] Implement: ANTI-V
- [x] **Test**: Combat program tests pass

### Phase 8: Complex Programs (Milestone 10) - COMPLETE
- [x] Implement: WARP
- [x] Implement: POLY
- [x] Implement: CRASH
- [x] Implement: UNDO
- [x] Implement: STEP
- [x] Implement: SCORE
- [x] Implement: REDUC
- [x] Implement: DELAY
- [x] Implement: ATK+
- [x] **Test**: All `test_programs.py` tests pass

### Phase 9: Scheduled Tasks (Milestone 11) - COMPLETE
- [x] Scheduled task interval calculation
- [x] Transmission spawning from scheduled tasks
- [x] CALM program effect
- [x] Siphon delay effect
- [x] **Test**: `test_scheduled_tasks.py` passes

### Phase 10: Rewards and Action Masking (Milestone 12) - COMPLETE
- [x] Full reward calculation (kill rewards, data siphon rewards)
- [x] Complete action masking logic
- [x] **Test**: `test_rewards.py`, `test_action_mask.py` pass

### Phase 11: Final Parity (Milestone 13) - COMPLETE
- [x] Edge case fixes
- [x] Determinism verification
- [x] Full test suite
- [x] **Test**: All 148 tests pass

## Test Strategy

### Running Tests

```bash
# Run all parity tests for JAX only
cd python && source venv/bin/activate && pytest tests/parity/ -k jax -v

# Run specific milestone tests
pytest tests/parity/test_interface_smoke.py -k jax -v  # Phase 1
pytest tests/parity/test_movement.py -k jax -v         # Phase 2
pytest tests/parity/test_enemies.py -k jax -v          # Phase 3
```

### Test Fixture Updates

Update `jax_env_wrapper.py` to implement:

- [x] **set_state()**: Convert GameState dataclass to JAX EnvState
- [x] **get_internal_state()**: Extract hidden state for implementation tests

```python
def set_state(self, state: GameState) -> Observation:
    """Convert test GameState to JAX EnvState."""
    # Convert player
    player = Player(
        row=jnp.int32(state.player.row),
        col=jnp.int32(state.player.col),
        hp=jnp.int32(state.player.hp),
        # ...
    )

    # Convert enemies to fixed-size array
    enemies = jnp.zeros((MAX_ENEMIES, 8), dtype=jnp.int32)
    enemy_mask = jnp.zeros(MAX_ENEMIES, dtype=jnp.bool_)
    for i, e in enumerate(state.enemies[:MAX_ENEMIES]):
        enemies = enemies.at[i].set([
            ENEMY_TYPE_MAP[e.type], e.row, e.col, e.hp, 0,
            int(e.stunned), int(e.spawnedFromSiphon), int(e.isFromScheduledTask)
        ])
        enemy_mask = enemy_mask.at[i].set(True)

    # ... convert rest of state

    self.state = EnvState(player=player, enemies=enemies, ...)
    return self._convert_observation(get_observation(self.state))
```

## Open Questions Resolved

- [x] **UNDO state storage**: Store previous state fields directly in EnvState (not nested pytree)
- [x] **BFS pathfinding**: Implement using `jax.lax.while_loop` with fixed iteration limit
- [x] **Random selection (WARP/POLY)**: Use `jax.random.choice` with masking

## Dependencies

- [x] `jax`: Core JAX library
- [x] `flax`: For `struct.dataclass`
- [x] `numpy`: Existing, for wrapper conversions

## Verification Checklist

- [x] `test_interface_smoke.py` - JAX passes
- [x] `test_movement.py` - JAX passes
- [x] `test_enemies.py` - JAX passes
- [x] `test_programs.py` - JAX passes
- [x] `test_siphon.py` - JAX passes
- [x] `test_stages.py` - JAX passes
- [x] `test_rewards.py` - JAX passes
- [x] `test_action_mask.py` - JAX passes
- [x] `test_turns.py` - JAX passes
- [x] `test_edge_cases.py` - JAX passes
- [x] `test_scheduled_tasks.py` - JAX passes
- [x] Implementation tests pass
