"""
Enemy turn processing for HackMatrix JAX environment.
Handles enemy movement, attacks, transmission spawning, and status effects.
"""

import jax
import jax.numpy as jnp

from .state import (
    EnvState,
    GRID_SIZE,
    BLOCK_EMPTY,
    ENEMY_MAX_HP,
    ENEMY_VIRUS,
    ENEMY_GLITCH,
    ENEMY_SPEED,
)


def process_enemy_turn(state: EnvState) -> EnvState:
    """Process enemy turn after player action ends their turn."""
    # 1. Tick transmissions and spawn enemies
    state = tick_transmissions(state)

    # 2. Move enemies (unless STEP is active)
    state = jax.lax.cond(
        ~state.step_active,
        lambda s: move_enemies(s),
        lambda s: s,
        state,
    )

    # 3. Enemy attacks
    state = enemy_attacks(state)

    # 4. Check scheduled tasks
    state = check_scheduled_tasks(state)

    # 5. Reset status effects
    state = reset_status_effects(state)

    # Clear STEP effect after enemy turn
    state = state.replace(step_active=jnp.bool_(False))

    return state


def tick_transmissions(state: EnvState) -> EnvState:
    """Decrement transmission timers and spawn enemies."""

    def process_transmission(carry, idx):
        state = carry
        is_active = state.trans_mask[idx]
        trans = state.transmissions[idx]
        row = trans[0]
        col = trans[1]
        turns_remaining = trans[2]
        enemy_type = trans[3]
        spawned_from_siphon = trans[4]
        is_from_scheduled_task = trans[5]

        # Decrement turns
        new_turns = turns_remaining - 1

        # If timer hits 0, spawn enemy
        should_spawn = is_active & (new_turns <= 0)

        # Update transmission turns
        new_transmissions = jax.lax.cond(
            is_active & (new_turns > 0),
            lambda: state.transmissions.at[idx, 2].set(new_turns),
            lambda: state.transmissions,
        )

        # Remove transmission if spawned
        new_trans_mask = jax.lax.cond(
            should_spawn,
            lambda: state.trans_mask.at[idx].set(False),
            lambda: state.trans_mask,
        )

        # Find slot for new enemy
        enemy_slot = jnp.argmin(state.enemy_mask)
        has_enemy_space = ~state.enemy_mask.all()

        # Create enemy data
        enemy_hp = ENEMY_MAX_HP[enemy_type]
        enemy_data = jnp.array([
            enemy_type, row, col, enemy_hp, 0,
            0, spawned_from_siphon, is_from_scheduled_task
        ], dtype=jnp.int32)

        # Add enemy if spawning and space available
        new_enemies = jax.lax.cond(
            should_spawn & has_enemy_space,
            lambda: state.enemies.at[enemy_slot].set(enemy_data),
            lambda: state.enemies,
        )
        new_enemy_mask = jax.lax.cond(
            should_spawn & has_enemy_space,
            lambda: state.enemy_mask.at[enemy_slot].set(True),
            lambda: state.enemy_mask,
        )

        new_state = state.replace(
            transmissions=new_transmissions,
            trans_mask=new_trans_mask,
            enemies=new_enemies,
            enemy_mask=new_enemy_mask,
        )

        return new_state, None

    state, _ = jax.lax.scan(
        process_transmission,
        state,
        jnp.arange(state.trans_mask.shape[0]),
    )

    return state


def move_enemies(state: EnvState) -> EnvState:
    """Move all non-stunned enemies toward player.

    Simple pathfinding: move directly toward player.
    Virus moves 2 cells, others move 1 cell.
    Glitch can move through blocks.
    """

    def move_single_enemy(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        enemy_type = enemy[0]
        row = enemy[1]
        col = enemy[2]
        disabled_turns = enemy[4]
        is_stunned = enemy[5] > 0

        # Check if enemy can move
        can_move = is_active & ~is_stunned & (disabled_turns <= 0)

        # Calculate direction toward player
        row_diff = state.player.row - row
        col_diff = state.player.col - col

        # Get movement speed for this enemy type
        speed = ENEMY_SPEED[enemy_type]

        # Move function - moves one step toward player
        def do_move(state_row_col):
            s, r, c = state_row_col
            # Prefer row movement if row diff is larger, else col
            row_delta = jnp.sign(s.player.row - r)
            col_delta = jnp.sign(s.player.col - c)

            # Try to move in primary direction
            # For simplicity, alternate between row and col movement
            abs_row_diff = jnp.abs(s.player.row - r)
            abs_col_diff = jnp.abs(s.player.col - c)

            # Move in direction of larger distance
            new_row = jax.lax.cond(
                abs_row_diff >= abs_col_diff,
                lambda: r + row_delta,
                lambda: r,
            )
            new_col = jax.lax.cond(
                abs_row_diff < abs_col_diff,
                lambda: c + col_delta,
                lambda: c,
            )

            # Bounds check
            new_row = jnp.clip(new_row, 0, GRID_SIZE - 1)
            new_col = jnp.clip(new_col, 0, GRID_SIZE - 1)

            # Don't move onto player position - stay adjacent instead
            is_player_pos = (new_row == s.player.row) & (new_col == s.player.col)
            new_row = jax.lax.cond(is_player_pos, lambda: r, lambda: new_row)
            new_col = jax.lax.cond(is_player_pos, lambda: c, lambda: new_col)

            # Check for blocking block (unless glitch)
            is_glitch = enemy_type == ENEMY_GLITCH
            has_block = (s.grid_block_type[new_row, new_col] != BLOCK_EMPTY) & \
                       (~s.grid_block_siphoned[new_row, new_col])
            blocked = has_block & ~is_glitch

            # If blocked, try alternate direction
            alt_row = jax.lax.cond(
                blocked & (abs_row_diff < abs_col_diff),
                lambda: r + row_delta,
                lambda: new_row,
            )
            alt_col = jax.lax.cond(
                blocked & (abs_row_diff >= abs_col_diff),
                lambda: c + col_delta,
                lambda: new_col,
            )

            # Use alternate if not blocked
            final_row = jax.lax.cond(
                blocked,
                lambda: jnp.clip(alt_row, 0, GRID_SIZE - 1),
                lambda: new_row,
            )
            final_col = jax.lax.cond(
                blocked,
                lambda: jnp.clip(alt_col, 0, GRID_SIZE - 1),
                lambda: new_col,
            )

            return final_row, final_col

        # Move up to 'speed' times
        new_row, new_col = row, col
        new_row, new_col = jax.lax.cond(
            can_move,
            lambda: do_move((state, new_row, new_col)),
            lambda: (new_row, new_col),
        )

        # Virus moves twice
        new_row, new_col = jax.lax.cond(
            can_move & (enemy_type == ENEMY_VIRUS),
            lambda: do_move((state, new_row, new_col)),
            lambda: (new_row, new_col),
        )

        # Update enemy position
        new_enemies = jax.lax.cond(
            can_move,
            lambda: state.enemies.at[idx, 1].set(new_row).at[idx, 2].set(new_col),
            lambda: state.enemies,
        )

        return state.replace(enemies=new_enemies), None

    state, _ = jax.lax.scan(
        move_single_enemy,
        state,
        jnp.arange(state.enemy_mask.shape[0]),
    )

    return state


def enemy_attacks(state: EnvState) -> EnvState:
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


def check_scheduled_tasks(state: EnvState) -> EnvState:
    """Check and execute scheduled tasks."""
    # TODO: Implement scheduled task spawning
    return state


def reset_status_effects(state: EnvState) -> EnvState:
    """Reset enemy stun flags and decrement disable counters."""
    # Clear stun flags
    new_enemies = state.enemies.at[:, 5].set(0)

    # Decrement disable counters (column 4)
    disable_counters = state.enemies[:, 4]
    new_disable = jnp.maximum(disable_counters - 1, 0)
    new_enemies = new_enemies.at[:, 4].set(new_disable)

    return state.replace(enemies=new_enemies)
