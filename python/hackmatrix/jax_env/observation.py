"""
JAX Observation Building for HackMatrix.

This module converts EnvState to observations for the agent.
Observation encoding must match Swift exactly for parity testing.

Grid features (42 per cell):
- [0-3]: Enemy type one-hot (virus, daemon, glitch, cryptog)
- [4]: Enemy HP / 3
- [5]: Enemy stunned
- [6]: Enemy spawned_from_siphon
- [7-9]: Block type one-hot (data, program, question)
- [10]: Block points / 9
- [11]: Block siphoned
- [12-34]: Program type one-hot (23 programs)
- [35]: Transmission spawn count / 9
- [36]: Transmission turns remaining / 4
- [37]: Cell credits / 3
- [38]: Cell energy / 3
- [39]: Is data siphon pickup
- [40]: Is exit
- [41]: Is siphon center
"""

import jax
import jax.numpy as jnp
from flax import struct

from .state import (
    EnvState,
    GRID_SIZE,
    GRID_FEATURES,
    PLAYER_STATE_SIZE,
    NUM_PROGRAMS,
    MAX_ENEMIES,
    MAX_TRANSMISSIONS,
    ENEMY_VIRUS,
    ENEMY_DAEMON,
    ENEMY_GLITCH,
    ENEMY_CRYPTOG,
    BLOCK_EMPTY,
    BLOCK_DATA,
    BLOCK_PROGRAM,
    BLOCK_QUESTION,
)


@struct.dataclass
class Observation:
    """Observation returned to agent.

    Matches the structure from Swift for parity testing.
    """
    player_state: jax.Array  # (10,) float32
    programs: jax.Array      # (23,) int32
    grid: jax.Array          # (6, 6, 42) float32


def get_observation(state: EnvState) -> Observation:
    """Convert EnvState to observation arrays.

    This must match Swift's ObservationBuilder exactly.
    """
    player_obs = _encode_player(state)
    program_obs = state.owned_programs.astype(jnp.int32)
    grid_obs = _encode_grid(state)

    return Observation(
        player_state=player_obs,
        programs=program_obs,
        grid=grid_obs,
    )


def _encode_player(state: EnvState) -> jax.Array:
    """Encode player state to (10,) normalized array.

    Encoding:
    0: row / 5.0
    1: col / 5.0
    2: hp / 3.0
    3: min(credits / 50.0, 1.0)
    4: min(energy / 50.0, 1.0)
    5: (stage - 1) / 7.0
    6: dataSiphons / 10.0
    7: (attackDamage - 1) / 2.0
    8: showActivated (binary)
    9: scheduledTasksDisabled (binary)
    """
    player = state.player

    return jnp.array([
        player.row / 5.0,
        player.col / 5.0,
        player.hp / 3.0,
        jnp.minimum(player.credits / 50.0, 1.0),
        jnp.minimum(player.energy / 50.0, 1.0),
        (state.stage - 1) / 7.0,
        player.data_siphons / 10.0,
        (player.attack_damage - 1) / 2.0,
        jnp.float32(state.show_activated),
        jnp.float32(state.scheduled_tasks_disabled),
    ], dtype=jnp.float32)


def _encode_grid(state: EnvState) -> jax.Array:
    """Encode grid to (6, 6, 42) array.

    Uses vectorized operations for efficiency.
    """
    grid = jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_FEATURES), dtype=jnp.float32)

    # Encode enemies into grid
    grid = _encode_enemies(grid, state)

    # Encode blocks
    grid = _encode_blocks(grid, state)

    # Encode transmissions
    grid = _encode_transmissions(grid, state)

    # Encode resources
    grid = _encode_resources(grid, state)

    # Encode special cells
    grid = _encode_special_cells(grid, state)

    return grid


def _encode_enemies(grid: jax.Array, state: EnvState) -> jax.Array:
    """Encode enemy information into grid.

    For Cryptog enemies: only visible if in same row/col as player OR show_activated.

    Features per cell:
    - [0-3]: Enemy type one-hot
    - [4]: HP / 3
    - [5]: Is stunned
    - [6]: Spawned from siphon
    """
    def encode_single_enemy(carry, idx):
        grid, state = carry
        is_active = state.enemy_mask[idx]

        enemy = state.enemies[idx]
        enemy_type = enemy[0]
        row = enemy[1]
        col = enemy[2]
        hp = enemy[3]
        is_stunned = enemy[5]
        spawned_from_siphon = enemy[6]

        # Check visibility for Cryptog
        is_cryptog = enemy_type == ENEMY_CRYPTOG
        in_same_row = row == state.player.row
        in_same_col = col == state.player.col
        is_visible = ~is_cryptog | in_same_row | in_same_col | state.show_activated

        # Only encode if active and visible
        should_encode = is_active & is_visible

        # Enemy type one-hot
        grid = jax.lax.cond(
            should_encode & (enemy_type == ENEMY_VIRUS),
            lambda g: g.at[row, col, 0].set(1.0),
            lambda g: g,
            grid,
        )
        grid = jax.lax.cond(
            should_encode & (enemy_type == ENEMY_DAEMON),
            lambda g: g.at[row, col, 1].set(1.0),
            lambda g: g,
            grid,
        )
        grid = jax.lax.cond(
            should_encode & (enemy_type == ENEMY_GLITCH),
            lambda g: g.at[row, col, 2].set(1.0),
            lambda g: g,
            grid,
        )
        grid = jax.lax.cond(
            should_encode & (enemy_type == ENEMY_CRYPTOG),
            lambda g: g.at[row, col, 3].set(1.0),
            lambda g: g,
            grid,
        )

        # HP, stunned, spawned_from_siphon
        grid = jax.lax.cond(
            should_encode,
            lambda g: g.at[row, col, 4].set(hp / 3.0),
            lambda g: g,
            grid,
        )
        grid = jax.lax.cond(
            should_encode,
            lambda g: g.at[row, col, 5].set(jnp.float32(is_stunned)),
            lambda g: g,
            grid,
        )
        grid = jax.lax.cond(
            should_encode,
            lambda g: g.at[row, col, 6].set(jnp.float32(spawned_from_siphon)),
            lambda g: g,
            grid,
        )

        return (grid, state), None

    (grid, _), _ = jax.lax.scan(
        encode_single_enemy,
        (grid, state),
        jnp.arange(MAX_ENEMIES),
    )

    return grid


def _encode_blocks(grid: jax.Array, state: EnvState) -> jax.Array:
    """Encode block information into grid.

    Features:
    - [7-9]: Block type one-hot (data=7, program=8, question=9)
    - [10]: Block points / 9
    - [11]: Block siphoned
    - [12-34]: Program type one-hot (23 programs)
    """
    # Block type one-hot
    is_data = state.grid_block_type == BLOCK_DATA
    is_program = state.grid_block_type == BLOCK_PROGRAM
    is_question = state.grid_block_type == BLOCK_QUESTION

    grid = grid.at[:, :, 7].set(is_data.astype(jnp.float32))
    grid = grid.at[:, :, 8].set(is_program.astype(jnp.float32))
    grid = grid.at[:, :, 9].set(is_question.astype(jnp.float32))

    # Block points (only for data blocks, but store for all)
    grid = grid.at[:, :, 10].set(state.grid_block_points / 9.0)

    # Block siphoned
    grid = grid.at[:, :, 11].set(state.grid_block_siphoned.astype(jnp.float32))

    # Program type one-hot (channels 12-34)
    # Only for program blocks
    has_program = (state.grid_block_type == BLOCK_PROGRAM) | (state.grid_block_type == BLOCK_QUESTION)

    # Create program one-hot encoding for each cell
    def encode_program_for_cell(row_col):
        row, col = row_col[0], row_col[1]
        program_idx = state.grid_block_program[row, col]
        is_program_block = has_program[row, col]

        # Create one-hot vector for this program
        one_hot = jnp.zeros(NUM_PROGRAMS, dtype=jnp.float32)
        one_hot = jax.lax.cond(
            is_program_block & (program_idx >= 0) & (program_idx < NUM_PROGRAMS),
            lambda: one_hot.at[program_idx].set(1.0),
            lambda: one_hot,
        )
        return one_hot

    # Generate all row/col combinations
    rows, cols = jnp.meshgrid(jnp.arange(GRID_SIZE), jnp.arange(GRID_SIZE), indexing='ij')
    row_col_pairs = jnp.stack([rows.ravel(), cols.ravel()], axis=1)

    # Vectorize program encoding
    program_features = jax.vmap(encode_program_for_cell)(row_col_pairs)
    program_features = program_features.reshape(GRID_SIZE, GRID_SIZE, NUM_PROGRAMS)

    grid = grid.at[:, :, 12:35].set(program_features)

    return grid


def _encode_transmissions(grid: jax.Array, state: EnvState) -> jax.Array:
    """Encode transmission information into grid.

    Features:
    - [35]: Transmission spawn count / 9 (from block)
    - [36]: Transmission turns remaining / 4
    """
    # Spawn count from blocks (channels 35)
    grid = grid.at[:, :, 35].set(state.grid_block_spawn_count / 9.0)

    # Transmission turns remaining (channel 36)
    def encode_single_transmission(carry, idx):
        grid, state = carry
        is_active = state.trans_mask[idx]

        trans = state.transmissions[idx]
        row = trans[0]
        col = trans[1]
        turns_remaining = trans[2]

        # Set turns remaining if active
        grid = jax.lax.cond(
            is_active,
            lambda g: g.at[row, col, 36].set(jnp.minimum(turns_remaining / 4.0, 1.0)),
            lambda g: g,
            grid,
        )

        return (grid, state), None

    (grid, _), _ = jax.lax.scan(
        encode_single_transmission,
        (grid, state),
        jnp.arange(MAX_TRANSMISSIONS),
    )

    return grid


def _encode_resources(grid: jax.Array, state: EnvState) -> jax.Array:
    """Encode resource information into grid.

    Features:
    - [37]: Credits / 3
    - [38]: Energy / 3
    """
    grid = grid.at[:, :, 37].set(state.grid_resources_credits / 3.0)
    grid = grid.at[:, :, 38].set(state.grid_resources_energy / 3.0)

    return grid


def _encode_special_cells(grid: jax.Array, state: EnvState) -> jax.Array:
    """Encode special cell information into grid.

    Features:
    - [39]: Is data siphon pickup
    - [40]: Is exit
    - [41]: Is siphon center
    """
    grid = grid.at[:, :, 39].set(state.grid_data_siphon.astype(jnp.float32))
    grid = grid.at[:, :, 40].set(state.grid_exit.astype(jnp.float32))
    grid = grid.at[:, :, 41].set(state.grid_siphon_center.astype(jnp.float32))

    return grid
