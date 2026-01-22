"""
Pure functional JAX environment for HackMatrix.
Designed for JIT compilation and TPU training with PureJaxRL.

This is the main environment file that provides reset(), step(), and get_valid_actions().
Game logic is implemented in separate modules for maintainability.
"""

import jax
import jax.numpy as jnp

from .jax_state import (
    EnvState,
    Player,
    create_empty_state,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_PROGRAMS,
    GRID_FEATURES,
    PLAYER_STATE_SIZE,
    PLAYER_MAX_HP,
    EXIT_ROW,
    EXIT_COL,
    DEFAULT_SCHEDULED_TASK_INTERVAL,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_SIPHON,
    ACTION_PROGRAM_START,
    DIRECTION_OFFSETS,
    BLOCK_EMPTY,
    PROGRAM_WAIT,
    PROGRAM_COSTS,
)
from .jax_observation import Observation, get_observation


# =============================================================================
# Core Environment Functions
# =============================================================================


def reset(key: jax.Array) -> tuple[EnvState, Observation]:
    """
    Initialize environment state.

    Args:
        key: JAX PRNG key for random stage generation

    Returns:
        (initial_state, initial_observation)
    """
    state = create_empty_state(key)

    # Set initial player position at (0, 0)
    player = Player(
        row=jnp.int32(0),
        col=jnp.int32(0),
        hp=jnp.int32(PLAYER_MAX_HP),
        credits=jnp.int32(0),
        energy=jnp.int32(0),
        data_siphons=jnp.int32(0),
        attack_damage=jnp.int32(1),
        score=jnp.int32(0),
    )
    state = state.replace(
        player=player,
        prev_hp=jnp.int32(PLAYER_MAX_HP),
    )

    # TODO: Generate stage 1 with blocks, enemies, resources
    # For now, just return basic state with exit set

    obs = get_observation(state)
    return state, obs


def step(
    state: EnvState, action: jnp.int32, key: jax.Array
) -> tuple[EnvState, Observation, jnp.float32, jnp.bool_]:
    """
    Take one environment step.

    Args:
        state: Current environment state
        action: Action index (0-27)
        key: JAX PRNG key

    Returns:
        (new_state, observation, reward, done)

    Action indices:
        0-3: Movement (up, down, left, right)
        4: Siphon
        5-27: Programs (23 total)
    """
    # Update RNG key
    key, subkey = jax.random.split(key)
    state = state.replace(rng_key=key)

    # Save previous state for reward calculation and UNDO
    state = _save_previous_state(state)

    # Execute action
    state, turn_ends = _execute_action(state, action, subkey)

    # If turn ends, process enemy turn
    state = jax.lax.cond(
        turn_ends,
        lambda s: _process_enemy_turn(s),
        lambda s: s,
        state,
    )

    # Check game end conditions
    player_died = state.player.hp <= 0
    reached_exit = (state.player.row == EXIT_ROW) & (state.player.col == EXIT_COL)
    stage_complete = reached_exit & (state.stage <= 8)

    # Advance stage if complete
    state = jax.lax.cond(
        stage_complete,
        lambda s: _advance_stage(s),
        lambda s: s,
        state,
    )

    # Game is done if player died or won (stage 9)
    game_won = state.stage > 8
    done = player_died | game_won

    # Calculate reward
    reward = _calculate_reward(state, stage_complete, game_won, player_died)
    state = state.replace(cumulative_reward=state.cumulative_reward + reward)

    obs = get_observation(state)
    return state, obs, reward, done


def get_valid_actions(state: EnvState) -> jax.Array:
    """
    Return mask of valid actions.

    Args:
        state: Current environment state

    Returns:
        Boolean array of shape (NUM_ACTIONS,) where True = valid
    """
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)

    # Movement actions (0-3): check bounds and obstacles
    mask = mask.at[ACTION_MOVE_UP].set(_is_move_valid(state, ACTION_MOVE_UP))
    mask = mask.at[ACTION_MOVE_DOWN].set(_is_move_valid(state, ACTION_MOVE_DOWN))
    mask = mask.at[ACTION_MOVE_LEFT].set(_is_move_valid(state, ACTION_MOVE_LEFT))
    mask = mask.at[ACTION_MOVE_RIGHT].set(_is_move_valid(state, ACTION_MOVE_RIGHT))

    # Siphon (4): requires data_siphons > 0
    mask = mask.at[ACTION_SIPHON].set(state.player.data_siphons > 0)

    # Programs (5-27): requires ownership, resources, and applicability
    for prog_idx in range(NUM_PROGRAMS):
        action_idx = ACTION_PROGRAM_START + prog_idx
        mask = mask.at[action_idx].set(_is_program_valid(state, prog_idx))

    return mask


# =============================================================================
# Action Execution
# =============================================================================


def _execute_action(
    state: EnvState, action: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute an action, return (new_state, turn_ends)."""

    # Branch on action type
    is_move = action < 4
    is_siphon = action == 4
    is_program = action >= 5

    # Execute move
    state_after_move, turn_ends_move = jax.lax.cond(
        is_move,
        lambda: _execute_move(state, action, key),
        lambda: (state, jnp.bool_(False)),
    )

    # Execute siphon
    state_after_siphon, turn_ends_siphon = jax.lax.cond(
        is_siphon,
        lambda: _execute_siphon(state_after_move, key),
        lambda: (state_after_move, jnp.bool_(False)),
    )

    # Execute program
    state_after_program, turn_ends_program = jax.lax.cond(
        is_program,
        lambda: _execute_program(state_after_siphon, action - ACTION_PROGRAM_START, key),
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


def _execute_move(
    state: EnvState, direction: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute a movement action.

    Movement can result in:
    1. Attack if enemy in line of sight
    2. Move to target cell if empty
    """
    offset = DIRECTION_OFFSETS[direction]
    target_row = state.player.row + offset[0]
    target_col = state.player.col + offset[1]

    # Check for enemy in line of sight
    has_enemy, enemy_idx = _find_enemy_in_los(state, direction)

    # If enemy found, attack it
    state = jax.lax.cond(
        has_enemy,
        lambda s: _attack_enemy(s, enemy_idx),
        lambda s: _move_player(s, target_row, target_col),
        state,
    )

    # Increment turn counter
    state = state.replace(turn=state.turn + 1)

    return state, jnp.bool_(True)


def _execute_siphon(state: EnvState, key: jax.Array) -> tuple[EnvState, jnp.bool_]:
    """Execute siphon action on best adjacent block."""
    # TODO: Implement full siphon logic
    # For now, just decrement siphons and end turn
    player = state.player.replace(data_siphons=state.player.data_siphons - 1)
    state = state.replace(player=player, turn=state.turn + 1)
    return state, jnp.bool_(True)


def _execute_program(
    state: EnvState, prog_idx: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute a program."""
    # Deduct costs
    cost = PROGRAM_COSTS[prog_idx]
    player = state.player.replace(
        credits=state.player.credits - cost[0],
        energy=state.player.energy - cost[1],
    )
    state = state.replace(player=player)

    # WAIT is the only program that ends the turn
    is_wait = prog_idx == PROGRAM_WAIT
    state = jax.lax.cond(
        is_wait,
        lambda s: s.replace(turn=s.turn + 1),
        lambda s: s,
        state,
    )

    # TODO: Implement program effects

    return state, is_wait


# =============================================================================
# Movement Helpers
# =============================================================================


def _is_move_valid(state: EnvState, direction: jnp.int32) -> jnp.bool_:
    """Check if movement in direction is valid.

    Valid if:
    - Target cell is in bounds AND
    - (Target cell has no unsiphoned block OR there's an enemy in LOS)
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

    # Check for enemy in line of sight (allows "attack move")
    has_los_enemy, _ = _find_enemy_in_los(state, direction)

    return in_bounds & (~has_blocking_block | has_los_enemy)


def _find_enemy_in_los(
    state: EnvState, direction: jnp.int32
) -> tuple[jnp.bool_, jnp.int32]:
    """Find first enemy or transmission in line of sight.

    Returns (found, enemy_idx). If found is False, enemy_idx is invalid.
    """
    offset = DIRECTION_OFFSETS[direction]

    # Search along direction for enemy
    def scan_cell(carry, step):
        found, idx, state, offset = carry
        row = state.player.row + offset[0] * (step + 1)
        col = state.player.col + offset[1] * (step + 1)

        # Check bounds
        in_bounds = (row >= 0) & (row < GRID_SIZE) & (col >= 0) & (col < GRID_SIZE)

        # Check for blocking block
        has_block = jax.lax.cond(
            in_bounds,
            lambda: (state.grid_block_type[row, col] != BLOCK_EMPTY) &
                    (~state.grid_block_siphoned[row, col]),
            lambda: jnp.bool_(False),
        )

        # Check for enemy at this cell
        def check_enemy(enemy_found_idx):
            enemy_found, enemy_idx = enemy_found_idx
            for i in range(state.enemy_mask.shape[0]):
                is_at_cell = (
                    state.enemy_mask[i] &
                    (state.enemies[i, 1] == row) &
                    (state.enemies[i, 2] == col)
                )
                enemy_found = enemy_found | is_at_cell
                enemy_idx = jax.lax.cond(
                    is_at_cell & (enemy_idx < 0),
                    lambda: jnp.int32(i),
                    lambda: enemy_idx,
                )
            return enemy_found, enemy_idx

        # Use fori_loop for enemy check
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
            in_bounds & ~found & ~has_block,
            lambda: jax.lax.scan(
                check_enemies_at_cell,
                (jnp.bool_(False), jnp.int32(-1), row, col, state),
                jnp.arange(state.enemy_mask.shape[0]),
            )[0][:2],
            lambda: (jnp.bool_(False), jnp.int32(-1)),
        )

        # Update found status - stop if blocked or found
        new_found = found | enemy_result[0]
        new_idx = jax.lax.cond(
            enemy_result[0] & ~found,
            lambda: enemy_result[1],
            lambda: idx,
        )

        # Stop searching if we hit a block
        stop = ~in_bounds | has_block | new_found

        return (new_found, new_idx, state, offset), stop

    # Scan up to GRID_SIZE cells
    (found, idx, _, _), _ = jax.lax.scan(
        scan_cell,
        (jnp.bool_(False), jnp.int32(-1), state, offset),
        jnp.arange(GRID_SIZE),
    )

    return found, idx


def _move_player(state: EnvState, target_row: jnp.int32, target_col: jnp.int32) -> EnvState:
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

    player = state.player.replace(
        row=new_row,
        col=new_col,
        data_siphons=new_siphons,
    )

    return state.replace(player=player, grid_data_siphon=new_grid_data_siphon)


def _attack_enemy(state: EnvState, enemy_idx: jnp.int32) -> EnvState:
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

    # TODO: Add score for kills

    return state.replace(enemies=new_enemies, enemy_mask=new_mask)


# =============================================================================
# Program Validation
# =============================================================================


def _is_program_valid(state: EnvState, prog_idx: jnp.int32) -> jnp.bool_:
    """Check if a program can be used."""
    # Must own the program
    owns_program = state.owned_programs[prog_idx]

    # Must have enough resources
    cost = PROGRAM_COSTS[prog_idx]
    has_credits = state.player.credits >= cost[0]
    has_energy = state.player.energy >= cost[1]

    # TODO: Add program-specific applicability checks

    return owns_program & has_credits & has_energy


# =============================================================================
# Enemy Turn Processing
# =============================================================================


def _process_enemy_turn(state: EnvState) -> EnvState:
    """Process enemy turn after player action ends their turn."""
    # 1. Tick transmissions and spawn enemies
    state = _tick_transmissions(state)

    # 2. Move enemies (unless STEP is active)
    state = jax.lax.cond(
        ~state.step_active,
        lambda s: _move_enemies(s),
        lambda s: s,
        state,
    )

    # 3. Enemy attacks
    state = _enemy_attacks(state)

    # 4. Check scheduled tasks
    state = _check_scheduled_tasks(state)

    # 5. Reset status effects
    state = _reset_status_effects(state)

    # Clear STEP effect after enemy turn
    state = state.replace(step_active=jnp.bool_(False))

    return state


def _tick_transmissions(state: EnvState) -> EnvState:
    """Decrement transmission timers and spawn enemies."""
    # TODO: Implement transmission spawning
    return state


def _move_enemies(state: EnvState) -> EnvState:
    """Move all non-stunned enemies toward player."""
    # TODO: Implement enemy movement with pathfinding
    return state


def _enemy_attacks(state: EnvState) -> EnvState:
    """Adjacent enemies attack player."""
    # Count adjacent non-stunned enemies
    def count_adjacent_attackers(carry, idx):
        count, state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        col = enemy[2]
        is_stunned = enemy[5] > 0

        # Check if adjacent (Manhattan distance = 1)
        row_dist = jnp.abs(row - state.player.row)
        col_dist = jnp.abs(col - state.player.col)
        is_adjacent = (row_dist + col_dist) == 1

        should_attack = is_active & is_adjacent & ~is_stunned
        new_count = count + jax.lax.cond(should_attack, lambda: 1, lambda: 0)

        return (new_count, state), None

    (attack_count, _), _ = jax.lax.scan(
        count_adjacent_attackers,
        (0, state),
        jnp.arange(state.enemy_mask.shape[0]),
    )

    # Reduce player HP
    new_hp = jnp.maximum(state.player.hp - attack_count, 0)
    player = state.player.replace(hp=new_hp)

    return state.replace(player=player)


def _check_scheduled_tasks(state: EnvState) -> EnvState:
    """Check and execute scheduled tasks."""
    # TODO: Implement scheduled task spawning
    return state


def _reset_status_effects(state: EnvState) -> EnvState:
    """Reset enemy stun flags and decrement disable counters."""
    # Clear stun flags
    new_enemies = state.enemies.at[:, 5].set(0)

    # Decrement disable counters (column 4)
    disable_counters = state.enemies[:, 4]
    new_disable = jnp.maximum(disable_counters - 1, 0)
    new_enemies = new_enemies.at[:, 4].set(new_disable)

    return state.replace(enemies=new_enemies)


# =============================================================================
# State Management
# =============================================================================


def _save_previous_state(state: EnvState) -> EnvState:
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


def _advance_stage(state: EnvState) -> EnvState:
    """Advance to next stage."""
    new_stage = state.stage + 1

    # Reset player position
    player = state.player.replace(row=jnp.int32(0), col=jnp.int32(0))

    # Reset stage-scoped flags
    state = state.replace(
        player=player,
        stage=new_stage,
        atk_plus_uses_this_stage=jnp.int32(0),
        step_active=jnp.bool_(False),
        show_activated=jnp.bool_(False),
    )

    # Clear grid (except exit)
    grid_shape = (GRID_SIZE, GRID_SIZE)
    state = state.replace(
        grid_block_type=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_points=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_program=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_spawn_count=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_block_siphoned=jnp.zeros(grid_shape, dtype=jnp.bool_),
        grid_siphon_center=jnp.zeros(grid_shape, dtype=jnp.bool_),
        grid_resources_credits=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_resources_energy=jnp.zeros(grid_shape, dtype=jnp.int32),
        grid_data_siphon=jnp.zeros(grid_shape, dtype=jnp.bool_),
    )

    # Clear enemies and transmissions
    state = state.replace(
        enemy_mask=jnp.zeros(state.enemy_mask.shape, dtype=jnp.bool_),
        trans_mask=jnp.zeros(state.trans_mask.shape, dtype=jnp.bool_),
    )

    # TODO: Generate new stage content

    return state


# =============================================================================
# Reward Calculation
# =============================================================================


def _calculate_reward(
    state: EnvState,
    stage_completed: jnp.bool_,
    game_won: jnp.bool_,
    player_died: jnp.bool_,
) -> jnp.float32:
    """Calculate reward for this transition."""
    reward = jnp.float32(0.0)

    # Stage completion reward
    from .jax_state import STAGE_COMPLETION_REWARDS
    stage_reward = jax.lax.cond(
        stage_completed & (state.stage <= 8),
        lambda: STAGE_COMPLETION_REWARDS[state.stage - 2],  # -2 because stage was incremented
        lambda: jnp.float32(0.0),
    )
    reward = reward + stage_reward

    # Score gain reward (0.5 per point)
    score_delta = state.player.score - state.prev_score
    reward = reward + score_delta * 0.5

    # HP change
    hp_delta = state.player.hp - state.prev_hp
    reward = reward + hp_delta * 1.0  # +1 per HP gained, -1 per HP lost

    # Victory bonus
    victory_bonus = jax.lax.cond(
        game_won,
        lambda: 500.0 + state.player.score * 100.0,
        lambda: jnp.float32(0.0),
    )
    reward = reward + victory_bonus

    # Death penalty
    death_penalty = jax.lax.cond(
        player_died,
        lambda: -state.cumulative_reward * 0.5,
        lambda: jnp.float32(0.0),
    )
    reward = reward + death_penalty

    return reward


# =============================================================================
# Vectorized Versions
# =============================================================================


# Note: These will be re-enabled once the basic functions work
# batched_reset = jax.vmap(reset)
# batched_step = jax.vmap(step)
# batched_get_valid_actions = jax.vmap(get_valid_actions)
