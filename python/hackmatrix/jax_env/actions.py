"""
Action execution helpers for HackMatrix JAX environment.
Movement, siphon, and combat actions.
"""

import jax
import jax.numpy as jnp

from .state import (
    EnvState,
    GRID_SIZE,
    BLOCK_EMPTY,
    BLOCK_DATA,
    BLOCK_PROGRAM,
    DIRECTION_OFFSETS,
    SIPHON_DELAY_TURNS,
    ENEMY_VIRUS,
)
from .programs import execute_program, is_program_valid


def execute_action(
    state: EnvState, action: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute an action, return (new_state, turn_ends)."""
    from .state import ACTION_PROGRAM_START

    # Branch on action type
    is_move = action < 4
    is_siphon = action == 4
    is_program = action >= 5

    # Execute move
    state_after_move, turn_ends_move = jax.lax.cond(
        is_move,
        lambda: execute_move(state, action, key),
        lambda: (state, jnp.bool_(False)),
    )

    # Execute siphon
    state_after_siphon, turn_ends_siphon = jax.lax.cond(
        is_siphon,
        lambda: execute_siphon(state_after_move, key),
        lambda: (state_after_move, jnp.bool_(False)),
    )

    # Execute program
    state_after_program, turn_ends_program = jax.lax.cond(
        is_program,
        lambda: execute_program(state_after_siphon, action - ACTION_PROGRAM_START, key),
        lambda: (state_after_siphon, jnp.bool_(False)),
    )

    # Determine if turn ends
    turn_ends = jax.lax.cond(
        is_move,
        lambda: turn_ends_move,
        lambda: jax.lax.cond(
            is_siphon,
            lambda: turn_ends_siphon,
            lambda: turn_ends_program,
        ),
    )

    return state_after_program, turn_ends


def execute_move(
    state: EnvState, direction: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute a movement action.

    Movement can result in:
    1. Attack if enemy or transmission in line of sight (player stays in place)
    2. Move to target cell if no LOS target and cell is empty
    """
    offset = DIRECTION_OFFSETS[direction]
    target_row = state.player.row + offset[0]
    target_col = state.player.col + offset[1]

    # Check for enemy or transmission in line of sight
    has_target, target_idx, is_transmission = find_enemy_in_los(state, direction)

    # If enemy found, attack it (player stays in place)
    state = jax.lax.cond(
        has_target & ~is_transmission,
        lambda s: attack_enemy(s, target_idx),
        lambda s: s,
        state,
    )

    # If transmission found, destroy it (player stays in place)
    state = jax.lax.cond(
        has_target & is_transmission,
        lambda s: destroy_transmission(s, target_idx),
        lambda s: s,
        state,
    )

    # If no target in LOS, move player
    state = jax.lax.cond(
        ~has_target,
        lambda s: move_player(s, target_row, target_col),
        lambda s: s,
        state,
    )

    # Increment turn counter
    state = state.replace(turn=state.turn + 1)

    return state, jnp.bool_(True)


def destroy_transmission(state: EnvState, trans_idx: jnp.int32) -> EnvState:
    """Destroy a transmission."""
    new_mask = state.trans_mask.at[trans_idx].set(False)
    return state.replace(trans_mask=new_mask)


def execute_siphon(state: EnvState, key: jax.Array) -> tuple[EnvState, jnp.bool_]:
    """Execute siphon action on best adjacent block.

    Siphon priority: program > high-point data > low-point data
    Effects:
    - Mark block as siphoned
    - Mark siphon center (player position)
    - Add score for data blocks
    - Add program for program blocks
    - Spawn transmissions based on block's spawn count
    - Delay next scheduled task by 5 turns
    """
    # Find best adjacent block
    player_row = state.player.row
    player_col = state.player.col

    # Check each adjacent cell
    def check_cell(row, col):
        in_bounds = (row >= 0) & (row < GRID_SIZE) & (col >= 0) & (col < GRID_SIZE)
        block_type = jax.lax.cond(
            in_bounds,
            lambda: state.grid_block_type[row, col],
            lambda: jnp.int32(0),
        )
        is_siphoned = jax.lax.cond(
            in_bounds,
            lambda: state.grid_block_siphoned[row, col],
            lambda: jnp.bool_(True),
        )
        has_block = (block_type != BLOCK_EMPTY) & ~is_siphoned
        points = jax.lax.cond(
            in_bounds & has_block,
            lambda: state.grid_block_points[row, col],
            lambda: jnp.int32(0),
        )
        is_program = block_type == BLOCK_PROGRAM
        # Score: programs get 100, data gets points
        score = jax.lax.cond(is_program, lambda: jnp.int32(100), lambda: points)
        return has_block, score, row, col

    # Check all 4 directions
    up_valid, up_score, up_row, up_col = check_cell(player_row + 1, player_col)
    down_valid, down_score, down_row, down_col = check_cell(player_row - 1, player_col)
    left_valid, left_score, left_row, left_col = check_cell(player_row, player_col - 1)
    right_valid, right_score, right_row, right_col = check_cell(player_row, player_col + 1)

    # Find best (highest score)
    best_score = jnp.int32(0)
    best_row = jnp.int32(-1)
    best_col = jnp.int32(-1)
    has_target = jnp.bool_(False)

    # Check up
    update_up = up_valid & (up_score > best_score)
    best_score = jax.lax.cond(update_up, lambda: up_score, lambda: best_score)
    best_row = jax.lax.cond(update_up, lambda: up_row, lambda: best_row)
    best_col = jax.lax.cond(update_up, lambda: up_col, lambda: best_col)
    has_target = has_target | up_valid

    # Check down
    update_down = down_valid & (down_score > best_score)
    best_score = jax.lax.cond(update_down, lambda: down_score, lambda: best_score)
    best_row = jax.lax.cond(update_down, lambda: down_row, lambda: best_row)
    best_col = jax.lax.cond(update_down, lambda: down_col, lambda: best_col)
    has_target = has_target | down_valid

    # Check left
    update_left = left_valid & (left_score > best_score)
    best_score = jax.lax.cond(update_left, lambda: left_score, lambda: best_score)
    best_row = jax.lax.cond(update_left, lambda: left_row, lambda: best_row)
    best_col = jax.lax.cond(update_left, lambda: left_col, lambda: best_col)
    has_target = has_target | left_valid

    # Check right
    update_right = right_valid & (right_score > best_score)
    best_score = jax.lax.cond(update_right, lambda: right_score, lambda: best_score)
    best_row = jax.lax.cond(update_right, lambda: right_row, lambda: best_row)
    best_col = jax.lax.cond(update_right, lambda: right_col, lambda: best_col)
    has_target = has_target | right_valid

    # Apply siphon effects
    def apply_siphon(state):
        block_type = state.grid_block_type[best_row, best_col]
        block_points = state.grid_block_points[best_row, best_col]
        block_program = state.grid_block_program[best_row, best_col]
        spawn_count = state.grid_block_spawn_count[best_row, best_col]

        # Mark block as siphoned
        new_siphoned = state.grid_block_siphoned.at[best_row, best_col].set(True)

        # Mark siphon center (player position)
        new_siphon_center = state.grid_siphon_center.at[player_row, player_col].set(True)

        # Add score for data blocks
        new_score = jax.lax.cond(
            block_type == BLOCK_DATA,
            lambda: state.player.score + block_points,
            lambda: state.player.score,
        )

        # Add program for program blocks
        new_owned = jax.lax.cond(
            block_type == BLOCK_PROGRAM,
            lambda: state.owned_programs.at[block_program].set(True),
            lambda: state.owned_programs,
        )

        # Decrement data siphons
        new_siphons = state.player.data_siphons - 1

        player = state.player.replace(
            score=new_score,
            data_siphons=new_siphons,
        )

        # Delay scheduled tasks
        new_next_task = state.next_scheduled_task_turn + SIPHON_DELAY_TURNS

        state = state.replace(
            player=player,
            grid_block_siphoned=new_siphoned,
            grid_siphon_center=new_siphon_center,
            owned_programs=new_owned,
            next_scheduled_task_turn=new_next_task,
        )

        # Spawn transmissions
        key_for_trans = state.rng_key

        def spawn_transmission(carry, i):
            s, key = carry
            should_spawn = i < spawn_count

            # Find empty cell for transmission (use player position + offset)
            # Simplified: spawn at block position
            trans_row = best_row
            trans_col = best_col

            # Find empty slot
            slot = jnp.argmin(s.trans_mask)
            has_space = ~s.trans_mask.all()

            # Random enemy type
            key, subkey = jax.random.split(key)
            enemy_type = jax.random.randint(subkey, (), 0, 4)

            trans_data = jnp.array([
                trans_row, trans_col, 3, enemy_type, 1, 0  # 3 turns, spawned_from_siphon=True
            ], dtype=jnp.int32)

            new_trans = jax.lax.cond(
                should_spawn & has_space,
                lambda: s.transmissions.at[slot].set(trans_data),
                lambda: s.transmissions,
            )
            new_mask = jax.lax.cond(
                should_spawn & has_space,
                lambda: s.trans_mask.at[slot].set(True),
                lambda: s.trans_mask,
            )

            return (s.replace(transmissions=new_trans, trans_mask=new_mask, rng_key=key), key), None

        (state, _), _ = jax.lax.scan(
            spawn_transmission,
            (state, key_for_trans),
            jnp.arange(9),  # Max spawn count is 9
        )

        return state

    # Only apply if we found a target, otherwise just decrement siphons
    state = jax.lax.cond(
        has_target,
        apply_siphon,
        lambda s: s.replace(
            player=s.player.replace(data_siphons=s.player.data_siphons - 1)
        ),
        state,
    )

    # Increment turn
    state = state.replace(turn=state.turn + 1)

    return state, jnp.bool_(True)


def is_move_valid(state: EnvState, direction: jnp.int32) -> jnp.bool_:
    """Check if movement in direction is valid.

    Valid if:
    - Target cell is in bounds AND
    - (Target cell has no unsiphoned block OR there's an enemy/transmission in LOS)
    """
    offset = DIRECTION_OFFSETS[direction]
    target_row = state.player.row + offset[0]
    target_col = state.player.col + offset[1]

    # Check bounds
    in_bounds = (
        (target_row >= 0) & (target_row < GRID_SIZE) &
        (target_col >= 0) & (target_col < GRID_SIZE)
    )

    # Check for blocking block at target
    has_blocking_block = jax.lax.cond(
        in_bounds,
        lambda: (state.grid_block_type[target_row, target_col] != BLOCK_EMPTY) &
                (~state.grid_block_siphoned[target_row, target_col]),
        lambda: jnp.bool_(False),
    )

    # Check for enemy/transmission in line of sight (allows "attack move")
    has_los_target, _, _ = find_enemy_in_los(state, direction)

    return in_bounds & (~has_blocking_block | has_los_target)


def find_enemy_in_los(
    state: EnvState, direction: jnp.int32
) -> tuple[jnp.bool_, jnp.int32, jnp.bool_]:
    """Find first enemy or transmission in line of sight.

    Returns (found, idx, is_transmission).
    - If found and is_transmission=False: idx is enemy index
    - If found and is_transmission=True: idx is transmission index
    - If not found: idx is invalid

    Note: Line-of-sight attack can reach through blocks (blocks don't block LOS attacks).
    """
    offset = DIRECTION_OFFSETS[direction]

    # Search along direction for enemy or transmission
    def scan_cell(carry, step):
        found, idx, is_trans, state, offset = carry
        row = state.player.row + offset[0] * (step + 1)
        col = state.player.col + offset[1] * (step + 1)

        # Check bounds
        in_bounds = (row >= 0) & (row < GRID_SIZE) & (col >= 0) & (col < GRID_SIZE)

        # Check for enemy at this cell
        def check_enemies_at_cell(carry, i):
            enemy_found, enemy_idx, row, col, state = carry
            is_at_cell = (
                state.enemy_mask[i] &
                (state.enemies[i, 1] == row) &
                (state.enemies[i, 2] == col)
            )
            new_found = enemy_found | is_at_cell
            new_idx = jax.lax.cond(
                is_at_cell & ~enemy_found,
                lambda: jnp.int32(i),
                lambda: enemy_idx,
            )
            return (new_found, new_idx, row, col, state), None

        enemy_result = jax.lax.cond(
            in_bounds & ~found,
            lambda: jax.lax.scan(
                check_enemies_at_cell,
                (jnp.bool_(False), jnp.int32(-1), row, col, state),
                jnp.arange(state.enemy_mask.shape[0]),
            )[0][:2],
            lambda: (jnp.bool_(False), jnp.int32(-1)),
        )

        # Check for transmission at this cell
        def check_trans_at_cell(carry, i):
            trans_found, trans_idx, row, col, state = carry
            is_at_cell = (
                state.trans_mask[i] &
                (state.transmissions[i, 0] == row) &
                (state.transmissions[i, 1] == col)
            )
            new_found = trans_found | is_at_cell
            new_idx = jax.lax.cond(
                is_at_cell & ~trans_found,
                lambda: jnp.int32(i),
                lambda: trans_idx,
            )
            return (new_found, new_idx, row, col, state), None

        trans_result = jax.lax.cond(
            in_bounds & ~found & ~enemy_result[0],
            lambda: jax.lax.scan(
                check_trans_at_cell,
                (jnp.bool_(False), jnp.int32(-1), row, col, state),
                jnp.arange(state.trans_mask.shape[0]),
            )[0][:2],
            lambda: (jnp.bool_(False), jnp.int32(-1)),
        )

        # Update found status - prioritize enemy over transmission
        found_enemy = enemy_result[0]
        found_trans = trans_result[0] & ~found_enemy

        new_found = found | found_enemy | found_trans
        new_idx = jax.lax.cond(
            found_enemy & ~found,
            lambda: enemy_result[1],
            lambda: jax.lax.cond(
                found_trans & ~found,
                lambda: trans_result[1],
                lambda: idx,
            ),
        )
        new_is_trans = jax.lax.cond(
            found_trans & ~found & ~found_enemy,
            lambda: jnp.bool_(True),
            lambda: is_trans,
        )

        return (new_found, new_idx, new_is_trans, state, offset), None

    # Scan up to GRID_SIZE cells (blocks don't stop LOS)
    (found, idx, is_trans, _, _), _ = jax.lax.scan(
        scan_cell,
        (jnp.bool_(False), jnp.int32(-1), jnp.bool_(False), state, offset),
        jnp.arange(GRID_SIZE),
    )

    return found, idx, is_trans


def move_player(state: EnvState, target_row: jnp.int32, target_col: jnp.int32) -> EnvState:
    """Move player to target cell and collect pickups."""
    # Check bounds
    in_bounds = (
        (target_row >= 0) & (target_row < GRID_SIZE) &
        (target_col >= 0) & (target_col < GRID_SIZE)
    )

    new_row = jax.lax.cond(in_bounds, lambda: target_row, lambda: state.player.row)
    new_col = jax.lax.cond(in_bounds, lambda: target_col, lambda: state.player.col)

    # Collect data siphon if present
    collected_siphon = jax.lax.cond(
        in_bounds,
        lambda: state.grid_data_siphon[new_row, new_col],
        lambda: jnp.bool_(False),
    )

    new_siphons = jax.lax.cond(
        collected_siphon,
        lambda: state.player.data_siphons + 1,
        lambda: state.player.data_siphons,
    )

    # Clear the data siphon from grid
    new_grid_data_siphon = jax.lax.cond(
        collected_siphon,
        lambda: state.grid_data_siphon.at[new_row, new_col].set(False),
        lambda: state.grid_data_siphon,
    )

    # Collect resources (credits)
    collected_credits = jax.lax.cond(
        in_bounds,
        lambda: state.grid_resources_credits[new_row, new_col],
        lambda: jnp.int32(0),
    )
    new_credits = state.player.credits + collected_credits
    new_grid_credits = jax.lax.cond(
        in_bounds & (collected_credits > 0),
        lambda: state.grid_resources_credits.at[new_row, new_col].set(0),
        lambda: state.grid_resources_credits,
    )

    # Collect resources (energy)
    collected_energy = jax.lax.cond(
        in_bounds,
        lambda: state.grid_resources_energy[new_row, new_col],
        lambda: jnp.int32(0),
    )
    new_energy = state.player.energy + collected_energy
    new_grid_energy = jax.lax.cond(
        in_bounds & (collected_energy > 0),
        lambda: state.grid_resources_energy.at[new_row, new_col].set(0),
        lambda: state.grid_resources_energy,
    )

    player = state.player.replace(
        row=new_row,
        col=new_col,
        data_siphons=new_siphons,
        credits=new_credits,
        energy=new_energy,
    )

    return state.replace(
        player=player,
        grid_data_siphon=new_grid_data_siphon,
        grid_resources_credits=new_grid_credits,
        grid_resources_energy=new_grid_energy,
    )


def attack_enemy(state: EnvState, enemy_idx: jnp.int32) -> EnvState:
    """Attack an enemy, dealing player's attack damage."""
    enemy = state.enemies[enemy_idx]
    new_hp = enemy[3] - state.player.attack_damage

    # Update enemy HP
    new_enemies = state.enemies.at[enemy_idx, 3].set(jnp.maximum(new_hp, 0))

    # Remove enemy if dead
    enemy_killed = new_hp <= 0
    new_mask = jax.lax.cond(
        enemy_killed,
        lambda: state.enemy_mask.at[enemy_idx].set(False),
        lambda: state.enemy_mask,
    )

    return state.replace(enemies=new_enemies, enemy_mask=new_mask)
