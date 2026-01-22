"""
Stage generation and state management for HackMatrix JAX environment.
"""

import jax
import jax.numpy as jnp

from .state import (
    EnvState,
    GRID_SIZE,
    PLAYER_MAX_HP,
    BLOCK_DATA,
    BLOCK_PROGRAM,
    NUM_PROGRAMS,
    MAX_TRANSMISSIONS,
)


def save_previous_state(state: EnvState) -> EnvState:
    """Save current state for UNDO functionality."""
    return state.replace(
        previous_state_valid=jnp.bool_(True),
        previous_player=state.player,
        previous_enemies=state.enemies,
        previous_enemy_mask=state.enemy_mask,
        previous_transmissions=state.transmissions,
        previous_trans_mask=state.trans_mask,
        previous_turn=state.turn,
        previous_grid_block_siphoned=state.grid_block_siphoned,
        previous_grid_siphon_center=state.grid_siphon_center,
        prev_score=state.player.score,
        prev_hp=state.player.hp,
        prev_credits=state.player.credits,
        prev_energy=state.player.energy,
    )


def advance_stage(state: EnvState) -> EnvState:
    """Advance to next stage."""
    new_stage = state.stage + 1

    # Player position is preserved (they stay at exit position)
    # HP gains 1 (up to max), not reset to max
    new_hp = jnp.minimum(state.player.hp + 1, PLAYER_MAX_HP)
    player = state.player.replace(
        hp=new_hp,
    )

    # Pick a random exit corner different from player's position
    # Corners: (0,0), (0,5), (5,0), (5,5) - indices 0-3
    corners = jnp.array([[0, 0], [0, 5], [5, 0], [5, 5]], dtype=jnp.int32)

    # Create mask for corners that are NOT the player's position
    player_row, player_col = player.row, player.col
    corner_mask = ~((corners[:, 0] == player_row) & (corners[:, 1] == player_col))

    # Use RNG to pick from valid corners with categorical sampling
    key, subkey = jax.random.split(state.rng_key)
    probs = corner_mask.astype(jnp.float32)
    probs = probs / jnp.sum(probs)
    corner_idx = jax.random.choice(subkey, jnp.arange(4), p=probs)

    new_exit_row = corners[corner_idx, 0]
    new_exit_col = corners[corner_idx, 1]

    # Update grid_exit array
    grid_shape = (GRID_SIZE, GRID_SIZE)
    grid_exit = jnp.zeros(grid_shape, dtype=jnp.bool_)
    grid_exit = grid_exit.at[new_exit_row, new_exit_col].set(True)

    # Reset stage-scoped flags
    state = state.replace(
        player=player,
        stage=new_stage,
        atk_plus_uses_this_stage=jnp.int32(0),
        step_active=jnp.bool_(False),
        show_activated=jnp.bool_(False),
        exit_row=new_exit_row,
        exit_col=new_exit_col,
        grid_exit=grid_exit,
        rng_key=key,
    )

    # Clear grid (except exit which was just set)
    grid_block_type = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_points = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_program = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_spawn_count = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_block_siphoned = jnp.zeros(grid_shape, dtype=jnp.bool_)
    grid_siphon_center = jnp.zeros(grid_shape, dtype=jnp.bool_)
    grid_resources_credits = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_resources_energy = jnp.zeros(grid_shape, dtype=jnp.int32)
    grid_data_siphon = jnp.zeros(grid_shape, dtype=jnp.bool_)

    # Place data siphons at remaining corners (not player's, not exit's)
    for i in range(4):
        corner_row, corner_col = corners[i, 0], corners[i, 1]
        is_player = (corner_row == player_row) & (corner_col == player_col)
        is_exit = (corner_row == new_exit_row) & (corner_col == new_exit_col)
        should_place_siphon = ~is_player & ~is_exit
        grid_data_siphon = jax.lax.cond(
            should_place_siphon,
            lambda g: g.at[corner_row, corner_col].set(True),
            lambda g: g,
            grid_data_siphon,
        )

    state = state.replace(
        grid_block_type=grid_block_type,
        grid_block_points=grid_block_points,
        grid_block_program=grid_block_program,
        grid_block_spawn_count=grid_block_spawn_count,
        grid_block_siphoned=grid_block_siphoned,
        grid_siphon_center=grid_siphon_center,
        grid_resources_credits=grid_resources_credits,
        grid_resources_energy=grid_resources_energy,
        grid_data_siphon=grid_data_siphon,
    )

    # Clear enemies and transmissions
    state = state.replace(
        enemy_mask=jnp.zeros(state.enemy_mask.shape, dtype=jnp.bool_),
        trans_mask=jnp.zeros(state.trans_mask.shape, dtype=jnp.bool_),
    )

    # Generate new stage content
    state = generate_stage_content(state, new_stage)

    return state


def generate_stage_content(state: EnvState, stage: jnp.int32) -> EnvState:
    """Generate blocks, resources, and transmissions for a new stage.

    This mimics Swift's stage initialization:
    - 5-11 blocks (50% data, 50% program)
    - Resources on empty cells
    - Transmissions based on stage number
    """
    key = state.rng_key

    # === Place Blocks ===
    # Decide block count (5-11)
    key, subkey = jax.random.split(key)
    block_count = jax.random.randint(subkey, (), 5, 12)

    # Create mask for valid block positions (not corners)
    corner_mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)
    corner_mask = corner_mask.at[0, 0].set(True)
    corner_mask = corner_mask.at[0, GRID_SIZE - 1].set(True)
    corner_mask = corner_mask.at[GRID_SIZE - 1, 0].set(True)
    corner_mask = corner_mask.at[GRID_SIZE - 1, GRID_SIZE - 1].set(True)

    # Non-corner positions are valid for blocks
    valid_for_blocks = ~corner_mask

    # Get all valid positions as flat indices
    valid_flat = valid_for_blocks.flatten()

    # Randomly select positions for blocks
    key, subkey = jax.random.split(key)
    # Create random priorities for each cell
    priorities = jax.random.uniform(subkey, (GRID_SIZE * GRID_SIZE,))
    # Set invalid cells to -1 so they're never selected
    priorities = jnp.where(valid_flat, priorities, -1.0)

    # Get top block_count positions by sorting
    sorted_indices = jnp.argsort(-priorities)  # Descending
    selected_indices = sorted_indices[:11]  # Max 11 blocks

    # Create block placement mask
    block_placement = jnp.zeros(GRID_SIZE * GRID_SIZE, dtype=jnp.bool_)
    for i in range(11):
        should_place = i < block_count
        block_placement = jax.lax.cond(
            should_place,
            lambda bp: bp.at[selected_indices[i]].set(True),
            lambda bp: bp,
            block_placement,
        )
    block_placement = block_placement.reshape((GRID_SIZE, GRID_SIZE))

    # Decide block types (50% data, 50% program)
    key, subkey = jax.random.split(key)
    is_data = jax.random.uniform(subkey, (GRID_SIZE, GRID_SIZE)) < 0.5

    # Generate points for data blocks (1-9)
    key, subkey = jax.random.split(key)
    points = jax.random.randint(subkey, (GRID_SIZE, GRID_SIZE), 1, 10)

    # Generate program indices for program blocks (0-22)
    key, subkey = jax.random.split(key)
    program_indices = jax.random.randint(subkey, (GRID_SIZE, GRID_SIZE), 0, NUM_PROGRAMS)

    # Set block types
    grid_block_type = jnp.where(
        block_placement & is_data,
        BLOCK_DATA,
        jnp.where(block_placement, BLOCK_PROGRAM, 0)
    )

    # Set points (for data blocks, spawnCount == points)
    grid_block_points = jnp.where(block_placement & is_data, points, 0)
    grid_block_spawn_count = jnp.where(
        block_placement & is_data,
        points,  # Data: spawnCount == points
        jnp.where(block_placement, 2, 0)  # Program: spawnCount = 2
    )

    # Set program indices
    grid_block_program = jnp.where(block_placement & ~is_data, program_indices, 0)

    state = state.replace(
        grid_block_type=grid_block_type,
        grid_block_points=grid_block_points,
        grid_block_program=grid_block_program,
        grid_block_spawn_count=grid_block_spawn_count,
    )

    # === Place Resources ===
    # Resources go on empty cells (not blocks, not corners)
    empty_cells = ~block_placement & ~corner_mask

    # Resource amounts: 45% 1, 45% 2, 10% 3
    key, subkey = jax.random.split(key)
    amount_roll = jax.random.uniform(subkey, (GRID_SIZE, GRID_SIZE))
    resource_amount = jnp.where(
        amount_roll < 0.45, 1,
        jnp.where(amount_roll < 0.9, 2, 3)
    )

    # 50% credits, 50% energy
    key, subkey = jax.random.split(key)
    is_credits = jax.random.uniform(subkey, (GRID_SIZE, GRID_SIZE)) < 0.5

    grid_resources_credits = jnp.where(empty_cells & is_credits, resource_amount, state.grid_resources_credits)
    grid_resources_energy = jnp.where(empty_cells & ~is_credits, resource_amount, state.grid_resources_energy)

    state = state.replace(
        grid_resources_credits=grid_resources_credits,
        grid_resources_energy=grid_resources_energy,
    )

    # === Spawn Transmissions ===
    # Stage 1-8 spawns 1-8 transmissions respectively
    starting_enemies = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
    transmission_count = starting_enemies[jnp.minimum(stage - 1, 7)]

    # Find empty cells for transmissions (not player, not blocks, not corners)
    player_row, player_col = state.player.row, state.player.col
    player_mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)
    player_mask = player_mask.at[player_row, player_col].set(True)

    valid_for_trans = ~block_placement & ~corner_mask & ~player_mask

    # Randomly place transmissions
    key, subkey = jax.random.split(key)
    trans_priorities = jax.random.uniform(subkey, (GRID_SIZE * GRID_SIZE,))
    trans_priorities = jnp.where(valid_for_trans.flatten(), trans_priorities, -1.0)
    trans_sorted = jnp.argsort(-trans_priorities)

    # Create transmission array
    transmissions = state.transmissions
    trans_mask = state.trans_mask

    # Enemy types: 0=virus, 1=daemon, 2=glitch, 3=cryptog (25% each)
    key, subkey = jax.random.split(key)
    enemy_types = jax.random.randint(subkey, (MAX_TRANSMISSIONS,), 0, 4)

    for i in range(8):  # Max 8 transmissions
        should_place = i < transmission_count
        flat_idx = trans_sorted[i]
        row = flat_idx // GRID_SIZE
        col = flat_idx % GRID_SIZE

        # Transmission: [row, col, turns_remaining, enemy_type, spawned_from_siphon, is_from_scheduled_task]
        trans_data = jnp.array([row, col, 1, enemy_types[i], 0, 0], dtype=jnp.int32)

        transmissions = jax.lax.cond(
            should_place,
            lambda t: t.at[i].set(trans_data),
            lambda t: t,
            transmissions,
        )
        trans_mask = jax.lax.cond(
            should_place,
            lambda m: m.at[i].set(True),
            lambda m: m,
            trans_mask,
        )

    state = state.replace(
        transmissions=transmissions,
        trans_mask=trans_mask,
        rng_key=key,
    )

    return state
