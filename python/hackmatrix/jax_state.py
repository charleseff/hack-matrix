"""
JAX State Definitions for HackMatrix.

This module defines all state dataclasses and constants for the JAX environment.
Uses flax.struct.dataclass for JIT-compatible, immutable state.

Why this design:
- Fixed-size arrays with masks for enemies/transmissions (JAX requires static shapes)
- Separate grid arrays for different cell properties (easier indexing)
- Previous state stored flat for UNDO program support
- All fields are JAX-compatible dtypes
"""

import jax
import jax.numpy as jnp
from flax import struct

# =============================================================================
# Constants
# =============================================================================

# Grid dimensions
GRID_SIZE = 6

# Fixed array sizes (JAX requires static shapes)
MAX_ENEMIES = 20
MAX_TRANSMISSIONS = 20

# Action space
NUM_ACTIONS = 28
NUM_PROGRAMS = 23

# Action indices
ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_SIPHON = 4
ACTION_PROGRAM_START = 5  # Programs are 5-27

# Direction offsets: (row_delta, col_delta) for each movement direction
# Up increases row (toward exit), Down decreases row
DIRECTION_OFFSETS = jnp.array([
    [1, 0],   # Up (row +1)
    [-1, 0],  # Down (row -1)
    [0, -1],  # Left (col -1)
    [0, 1],   # Right (col +1)
], dtype=jnp.int32)

# Enemy type encoding (for enemies array column 0)
ENEMY_VIRUS = 0
ENEMY_DAEMON = 1
ENEMY_GLITCH = 2
ENEMY_CRYPTOG = 3

# Enemy HP by type
ENEMY_MAX_HP = jnp.array([2, 3, 2, 2], dtype=jnp.int32)  # virus, daemon, glitch, cryptog

# Enemy speed by type (cells per turn)
ENEMY_SPEED = jnp.array([2, 1, 1, 1], dtype=jnp.int32)  # virus moves 2, others 1

# Block type encoding (for grid_block_type)
BLOCK_EMPTY = 0
BLOCK_DATA = 1
BLOCK_PROGRAM = 2
BLOCK_QUESTION = 3

# Program indices (0-22, maps to action indices 5-27)
PROGRAM_PUSH = 0       # Action 5
PROGRAM_PULL = 1       # Action 6
PROGRAM_CRASH = 2      # Action 7
PROGRAM_WARP = 3       # Action 8
PROGRAM_POLY = 4       # Action 9
PROGRAM_WAIT = 5       # Action 10
PROGRAM_DEBUG = 6      # Action 11
PROGRAM_ROW = 7        # Action 12
PROGRAM_COL = 8        # Action 13
PROGRAM_UNDO = 9       # Action 14
PROGRAM_STEP = 10      # Action 15
PROGRAM_SIPH_PLUS = 11 # Action 16
PROGRAM_EXCH = 12      # Action 17
PROGRAM_SHOW = 13      # Action 18
PROGRAM_RESET = 14     # Action 19
PROGRAM_CALM = 15      # Action 20
PROGRAM_D_BOM = 16     # Action 21
PROGRAM_DELAY = 17     # Action 22
PROGRAM_ANTI_V = 18    # Action 23
PROGRAM_SCORE = 19     # Action 24
PROGRAM_REDUC = 20     # Action 25
PROGRAM_ATK_PLUS = 21  # Action 26
PROGRAM_HACK = 22      # Action 27

# Program costs: [credits, energy] for each program (0-22)
# Based on Swift reference in HackMatrix/Program.swift
PROGRAM_COSTS = jnp.array([
    [0, 2],   # PUSH - 2 energy
    [0, 2],   # PULL - 2 energy
    [3, 2],   # CRASH - 3 credits, 2 energy
    [2, 2],   # WARP - 2 credits, 2 energy
    [1, 1],   # POLY - 1 credit, 1 energy
    [0, 1],   # WAIT - 1 energy
    [3, 0],   # DEBUG - 3 credits
    [3, 1],   # ROW - 3 credits, 1 energy
    [3, 1],   # COL - 3 credits, 1 energy
    [1, 0],   # UNDO - 1 credit
    [0, 3],   # STEP - 3 energy
    [5, 0],   # SIPH+ - 5 credits
    [4, 0],   # EXCH - 4 credits
    [2, 0],   # SHOW - 2 credits
    [0, 4],   # RESET - 4 energy
    [2, 4],   # CALM - 2 credits, 4 energy
    [3, 0],   # D_BOM - 3 credits
    [1, 2],   # DELAY - 1 credit, 2 energy
    [3, 0],   # ANTI-V - 3 credits
    [0, 5],   # SCORE - 5 energy
    [2, 1],   # REDUC - 2 credits, 1 energy
    [4, 4],   # ATK+ - 4 credits, 4 energy
    [2, 2],   # HACK - 2 credits, 2 energy
], dtype=jnp.int32)

# Exit position
EXIT_ROW = 5
EXIT_COL = 5

# Player stats
PLAYER_MAX_HP = 3
PLAYER_INITIAL_ATTACK = 1
PLAYER_MAX_ATTACK = 3

# Scheduled task defaults
DEFAULT_SCHEDULED_TASK_INTERVAL = 12
SIPHON_DELAY_TURNS = 5

# Observation features
GRID_FEATURES = 42  # 7 enemy + 5 block + 23 program + 2 transmission + 2 resource + 3 special
PLAYER_STATE_SIZE = 10

# Stage rewards for completion
STAGE_COMPLETION_REWARDS = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0], dtype=jnp.float32)

# =============================================================================
# State Dataclasses
# =============================================================================


@struct.dataclass
class Player:
    """Player state.

    All fields are scalars for simple indexing and updates.
    """
    row: jnp.int32
    col: jnp.int32
    hp: jnp.int32              # 0-3 (0 = dead)
    credits: jnp.int32
    energy: jnp.int32
    data_siphons: jnp.int32
    attack_damage: jnp.int32   # 1-3
    score: jnp.int32


@struct.dataclass
class EnvState:
    """Complete environment state.

    Design decisions:
    - Grid uses separate arrays per property for efficient JAX indexing
    - Enemies/transmissions use fixed-size arrays with boolean masks
    - Previous state stored for UNDO program functionality
    - RNG key stored for reproducibility

    Enemy array columns: [type, row, col, hp, disabled_turns, is_stunned, spawned_from_siphon, is_from_scheduled_task]
    Transmission array columns: [row, col, turns_remaining, enemy_type, spawned_from_siphon, is_from_scheduled_task]
    """
    # Player state
    player: Player

    # Grid arrays (6x6 each)
    # Block information
    grid_block_type: jax.Array          # int32: 0=empty, 1=data, 2=program, 3=question
    grid_block_points: jax.Array        # int32: points for data blocks (1-9)
    grid_block_program: jax.Array       # int32: program index (0-22) for program blocks
    grid_block_spawn_count: jax.Array   # int32: transmissions to spawn when siphoned
    grid_block_siphoned: jax.Array      # bool: whether block has been siphoned
    grid_siphon_center: jax.Array       # bool: where player stood when siphoning

    # Resources on grid
    grid_resources_credits: jax.Array   # int32: credits available at cell
    grid_resources_energy: jax.Array    # int32: energy available at cell
    grid_data_siphon: jax.Array         # bool: data siphon pickup present
    grid_exit: jax.Array                # bool: exit cell (only (5,5) is True)

    # Enemies: fixed-size array with mask
    # Shape: (MAX_ENEMIES, 8) - columns: [type, row, col, hp, disabled_turns, is_stunned, spawned_from_siphon, is_from_scheduled_task]
    enemies: jax.Array                  # int32
    enemy_mask: jax.Array               # bool: which enemy slots are active

    # Transmissions: fixed-size array with mask
    # Shape: (MAX_TRANSMISSIONS, 6) - columns: [row, col, turns_remaining, enemy_type, spawned_from_siphon, is_from_scheduled_task]
    transmissions: jax.Array            # int32
    trans_mask: jax.Array               # bool: which transmission slots are active

    # Program ownership
    owned_programs: jax.Array           # bool (23,): which programs are owned

    # Game state flags
    stage: jnp.int32                    # 1-8, 9 = victory
    turn: jnp.int32
    show_activated: jnp.bool_           # SHOW program effect active
    scheduled_tasks_disabled: jnp.bool_ # CALM program effect active
    step_active: jnp.bool_              # STEP program effect active
    atk_plus_uses_this_stage: jnp.int32 # 0, 1, or 2 (max 2 uses per stage)

    # Exit position (random corner, different from player)
    exit_row: jnp.int32
    exit_col: jnp.int32

    # Scheduled task timing
    next_scheduled_task_turn: jnp.int32
    scheduled_task_interval: jnp.int32
    pending_siphon_transmissions: jnp.int32  # Queued transmissions from siphoning

    # UNDO support - store previous state
    previous_state_valid: jnp.bool_
    previous_player: Player
    previous_enemies: jax.Array         # (MAX_ENEMIES, 8)
    previous_enemy_mask: jax.Array      # (MAX_ENEMIES,)
    previous_transmissions: jax.Array   # (MAX_TRANSMISSIONS, 6)
    previous_trans_mask: jax.Array      # (MAX_TRANSMISSIONS,)
    previous_turn: jnp.int32
    previous_grid_block_siphoned: jax.Array  # (6, 6) bool
    previous_grid_siphon_center: jax.Array   # (6, 6) bool

    # RNG key for reproducibility
    rng_key: jax.Array

    # Reward tracking (for computing delta rewards)
    prev_score: jnp.int32
    prev_hp: jnp.int32
    prev_credits: jnp.int32
    prev_energy: jnp.int32
    cumulative_reward: jnp.float32


# =============================================================================
# State Creation Helpers
# =============================================================================


def create_empty_player() -> Player:
    """Create a player with default initial state."""
    return Player(
        row=jnp.int32(0),
        col=jnp.int32(0),
        hp=jnp.int32(PLAYER_MAX_HP),
        credits=jnp.int32(0),
        energy=jnp.int32(0),
        data_siphons=jnp.int32(0),
        attack_damage=jnp.int32(PLAYER_INITIAL_ATTACK),
        score=jnp.int32(0),
    )


def create_empty_state(rng_key: jax.Array) -> EnvState:
    """Create an empty environment state with zeroed arrays.

    This is useful for initialization before setting specific state.
    """
    empty_player = create_empty_player()

    # Grid arrays
    grid_shape = (GRID_SIZE, GRID_SIZE)
    grid_block_type = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_points = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_program = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_spawn_count = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_siphoned = jnp.zeros(grid_shape, dtype=jnp.bool_)
    grid_siphon_center = jnp.zeros(grid_shape, dtype=jnp.bool_)
    grid_resources_credits = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_resources_energy = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_data_siphon = jnp.zeros(grid_shape, dtype=jnp.bool_)

    # Exit at (5, 5)
    grid_exit = jnp.zeros(grid_shape, dtype=jnp.bool_)
    grid_exit = grid_exit.at[EXIT_ROW, EXIT_COL].set(True)

    # Enemy and transmission arrays
    enemies = jnp.zeros((MAX_ENEMIES, 8), dtype=jnp.int32)
    enemy_mask = jnp.zeros(MAX_ENEMIES, dtype=jnp.bool_)
    transmissions = jnp.zeros((MAX_TRANSMISSIONS, 6), dtype=jnp.int32)
    trans_mask = jnp.zeros(MAX_TRANSMISSIONS, dtype=jnp.bool_)

    # Programs
    owned_programs = jnp.zeros(NUM_PROGRAMS, dtype=jnp.bool_)

    return EnvState(
        player=empty_player,
        grid_block_type=grid_block_type,
        grid_block_points=grid_block_points,
        grid_block_program=grid_block_program,
        grid_block_spawn_count=grid_block_spawn_count,
        grid_block_siphoned=grid_block_siphoned,
        grid_siphon_center=grid_siphon_center,
        grid_resources_credits=grid_resources_credits,
        grid_resources_energy=grid_resources_energy,
        grid_data_siphon=grid_data_siphon,
        grid_exit=grid_exit,
        enemies=enemies,
        enemy_mask=enemy_mask,
        transmissions=transmissions,
        trans_mask=trans_mask,
        owned_programs=owned_programs,
        stage=jnp.int32(1),
        turn=jnp.int32(0),
        show_activated=jnp.bool_(False),
        scheduled_tasks_disabled=jnp.bool_(False),
        step_active=jnp.bool_(False),
        atk_plus_uses_this_stage=jnp.int32(0),
        exit_row=jnp.int32(EXIT_ROW),
        exit_col=jnp.int32(EXIT_COL),
        next_scheduled_task_turn=jnp.int32(DEFAULT_SCHEDULED_TASK_INTERVAL),
        scheduled_task_interval=jnp.int32(DEFAULT_SCHEDULED_TASK_INTERVAL),
        pending_siphon_transmissions=jnp.int32(0),
        previous_state_valid=jnp.bool_(False),
        previous_player=empty_player,
        previous_enemies=enemies,
        previous_enemy_mask=enemy_mask,
        previous_transmissions=transmissions,
        previous_trans_mask=trans_mask,
        previous_turn=jnp.int32(0),
        previous_grid_block_siphoned=grid_block_siphoned,
        previous_grid_siphon_center=grid_siphon_center,
        rng_key=rng_key,
        prev_score=jnp.int32(0),
        prev_hp=jnp.int32(PLAYER_MAX_HP),
        prev_credits=jnp.int32(0),
        prev_energy=jnp.int32(0),
        cumulative_reward=jnp.float32(0.0),
    )


# =============================================================================
# Enemy and Transmission Array Helpers
# =============================================================================


def add_enemy(
    enemies: jax.Array,
    enemy_mask: jax.Array,
    enemy_type: int,
    row: int,
    col: int,
    hp: int,
    spawned_from_siphon: bool = False,
    is_from_scheduled_task: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Add an enemy to the fixed-size array.

    Returns updated (enemies, enemy_mask) arrays.
    If array is full, returns unchanged arrays.
    """
    # Find first empty slot
    idx = jnp.argmin(enemy_mask)
    has_space = ~enemy_mask.all()

    # Create enemy data
    enemy_data = jnp.array([
        enemy_type,
        row,
        col,
        hp,
        0,  # disabled_turns
        0,  # is_stunned (False)
        int(spawned_from_siphon),
        int(is_from_scheduled_task),
    ], dtype=jnp.int32)

    # Update arrays conditionally
    new_enemies = jax.lax.cond(
        has_space,
        lambda: enemies.at[idx].set(enemy_data),
        lambda: enemies,
    )
    new_mask = jax.lax.cond(
        has_space,
        lambda: enemy_mask.at[idx].set(True),
        lambda: enemy_mask,
    )

    return new_enemies, new_mask


def remove_enemy(enemy_mask: jax.Array, idx: int) -> jax.Array:
    """Remove an enemy by setting its mask to False."""
    return enemy_mask.at[idx].set(False)


def add_transmission(
    transmissions: jax.Array,
    trans_mask: jax.Array,
    row: int,
    col: int,
    turns_remaining: int,
    enemy_type: int,
    spawned_from_siphon: bool = False,
    is_from_scheduled_task: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Add a transmission to the fixed-size array.

    Returns updated (transmissions, trans_mask) arrays.
    """
    idx = jnp.argmin(trans_mask)
    has_space = ~trans_mask.all()

    trans_data = jnp.array([
        row,
        col,
        turns_remaining,
        enemy_type,
        int(spawned_from_siphon),
        int(is_from_scheduled_task),
    ], dtype=jnp.int32)

    new_transmissions = jax.lax.cond(
        has_space,
        lambda: transmissions.at[idx].set(trans_data),
        lambda: transmissions,
    )
    new_mask = jax.lax.cond(
        has_space,
        lambda: trans_mask.at[idx].set(True),
        lambda: trans_mask,
    )

    return new_transmissions, new_mask


def remove_transmission(trans_mask: jax.Array, idx: int) -> jax.Array:
    """Remove a transmission by setting its mask to False."""
    return trans_mask.at[idx].set(False)


# =============================================================================
# Enemy Type String Mapping (for test compatibility)
# =============================================================================

ENEMY_TYPE_TO_INT = {
    "virus": ENEMY_VIRUS,
    "daemon": ENEMY_DAEMON,
    "glitch": ENEMY_GLITCH,
    "cryptog": ENEMY_CRYPTOG,
}

ENEMY_INT_TO_TYPE = {
    ENEMY_VIRUS: "virus",
    ENEMY_DAEMON: "daemon",
    ENEMY_GLITCH: "glitch",
    ENEMY_CRYPTOG: "cryptog",
}
