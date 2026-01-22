"""
Program execution and validity checking for HackMatrix JAX environment.
Contains all 23 program implementations.
"""

import jax
import jax.numpy as jnp

from .state import (
    EnvState,
    GRID_SIZE,
    BLOCK_EMPTY,
    PLAYER_MAX_HP,
    PROGRAM_COSTS,
    PROGRAM_PUSH,
    PROGRAM_PULL,
    PROGRAM_CRASH,
    PROGRAM_WARP,
    PROGRAM_POLY,
    PROGRAM_WAIT,
    PROGRAM_DEBUG,
    PROGRAM_ROW,
    PROGRAM_COL,
    PROGRAM_UNDO,
    PROGRAM_STEP,
    PROGRAM_SIPH_PLUS,
    PROGRAM_EXCH,
    PROGRAM_SHOW,
    PROGRAM_RESET,
    PROGRAM_CALM,
    PROGRAM_D_BOM,
    PROGRAM_DELAY,
    PROGRAM_ANTI_V,
    PROGRAM_SCORE,
    PROGRAM_REDUC,
    PROGRAM_ATK_PLUS,
    PROGRAM_HACK,
    ENEMY_VIRUS,
    ENEMY_DAEMON,
    ENEMY_MAX_HP,
)


def execute_program(
    state: EnvState, prog_idx: jnp.int32, key: jax.Array
) -> tuple[EnvState, jnp.bool_]:
    """Execute a program.

    Programs (0-22):
    0: PUSH - push enemies away
    1: PULL - pull enemies closer
    2: CRASH - damage all enemies
    3: WARP - teleport to random cell
    4: POLY - transform random enemy
    5: WAIT - end turn (free)
    6: DEBUG - stun adjacent enemies
    7: ROW - damage enemies in row
    8: COL - damage enemies in column
    9: UNDO - restore previous state
    10: STEP - enemies skip movement next turn
    11: SIPH+ - add 1 data siphon
    12: EXCH - swap credits/energy
    13: SHOW - reveal cryptogs
    14: RESET - restore HP to 3
    15: CALM - disable scheduled tasks
    16: D_BOM - damage based on data siphons
    17: DELAY - delay scheduled task
    18: ANTI-V - kill all viruses
    19: SCORE - convert credits to score
    20: REDUC - reduce enemy HP
    21: ATK+ - increase attack damage
    22: HACK - damage enemies on siphoned cells
    """
    # Deduct costs
    cost = PROGRAM_COSTS[prog_idx]
    player = state.player.replace(
        credits=state.player.credits - cost[0],
        energy=state.player.energy - cost[1],
    )
    state = state.replace(player=player)

    # WAIT ends the turn
    is_wait = prog_idx == PROGRAM_WAIT
    state = jax.lax.cond(
        is_wait,
        lambda s: s.replace(turn=s.turn + 1),
        lambda s: s,
        state,
    )

    # SIPH+ - add 1 data siphon
    state = jax.lax.cond(
        prog_idx == PROGRAM_SIPH_PLUS,
        lambda s: s.replace(
            player=s.player.replace(data_siphons=s.player.data_siphons + 1)
        ),
        lambda s: s,
        state,
    )

    # RESET - restore HP to max
    state = jax.lax.cond(
        prog_idx == PROGRAM_RESET,
        lambda s: s.replace(
            player=s.player.replace(hp=jnp.int32(PLAYER_MAX_HP))
        ),
        lambda s: s,
        state,
    )

    # STEP - set step_active flag (enemies skip movement)
    state = jax.lax.cond(
        prog_idx == PROGRAM_STEP,
        lambda s: s.replace(step_active=jnp.bool_(True)),
        lambda s: s,
        state,
    )

    # SHOW - reveal cryptogs
    state = jax.lax.cond(
        prog_idx == PROGRAM_SHOW,
        lambda s: s.replace(show_activated=jnp.bool_(True)),
        lambda s: s,
        state,
    )

    # CALM - disable scheduled tasks
    state = jax.lax.cond(
        prog_idx == PROGRAM_CALM,
        lambda s: s.replace(scheduled_tasks_disabled=jnp.bool_(True)),
        lambda s: s,
        state,
    )

    # ATK+ - increase attack damage (max 2 uses per stage)
    can_use_atkplus = (prog_idx == PROGRAM_ATK_PLUS) & (state.atk_plus_uses_this_stage < 2)
    state = jax.lax.cond(
        can_use_atkplus,
        lambda s: s.replace(
            player=s.player.replace(
                attack_damage=jnp.minimum(s.player.attack_damage + 1, 3)
            ),
            atk_plus_uses_this_stage=s.atk_plus_uses_this_stage + 1,
        ),
        lambda s: s,
        state,
    )

    # EXCH - costs 4 credits (already deducted), grants 4 energy
    state = jax.lax.cond(
        prog_idx == PROGRAM_EXCH,
        lambda s: s.replace(
            player=s.player.replace(
                energy=s.player.energy + 4,
            )
        ),
        lambda s: s,
        state,
    )

    # DEBUG - damage and stun enemies ON BLOCKS
    state = jax.lax.cond(
        prog_idx == PROGRAM_DEBUG,
        lambda s: _apply_debug(s),
        lambda s: s,
        state,
    )

    # ROW - damage enemies in player's row
    state = jax.lax.cond(
        prog_idx == PROGRAM_ROW,
        lambda s: _apply_row(s),
        lambda s: s,
        state,
    )

    # COL - damage enemies in player's column
    state = jax.lax.cond(
        prog_idx == PROGRAM_COL,
        lambda s: _apply_col(s),
        lambda s: s,
        state,
    )

    # CRASH - damage ALL enemies (1 damage)
    state = jax.lax.cond(
        prog_idx == PROGRAM_CRASH,
        lambda s: _apply_crash(s),
        lambda s: s,
        state,
    )

    # ANTI-V - damage and stun all viruses
    state = jax.lax.cond(
        prog_idx == PROGRAM_ANTI_V,
        lambda s: _apply_antiv(s),
        lambda s: s,
        state,
    )

    # HACK - damage enemies on siphoned cells
    state = jax.lax.cond(
        prog_idx == PROGRAM_HACK,
        lambda s: _apply_hack(s),
        lambda s: s,
        state,
    )

    # PUSH - push enemies away from player
    state = jax.lax.cond(
        prog_idx == PROGRAM_PUSH,
        lambda s: _apply_push(s),
        lambda s: s,
        state,
    )

    # PULL - pull enemies toward player
    state = jax.lax.cond(
        prog_idx == PROGRAM_PULL,
        lambda s: _apply_pull(s),
        lambda s: s,
        state,
    )

    # WARP - teleport to random enemy/transmission and kill it
    state = jax.lax.cond(
        prog_idx == PROGRAM_WARP,
        lambda s: _apply_warp(s),
        lambda s: s,
        state,
    )

    # POLY - change all enemies to a different type
    state = jax.lax.cond(
        prog_idx == PROGRAM_POLY,
        lambda s: _apply_poly(s),
        lambda s: s,
        state,
    )

    # D_BOM - destroy nearest daemon, splash damage + stun around it
    state = jax.lax.cond(
        prog_idx == PROGRAM_D_BOM,
        lambda s: _apply_dbom(s),
        lambda s: s,
        state,
    )

    # SCORE - gain points = (8 - stage)
    state = jax.lax.cond(
        prog_idx == PROGRAM_SCORE,
        lambda s: _apply_score(s),
        lambda s: s,
        state,
    )

    # REDUC - reduce spawn count of all unsiphoned blocks by 1
    state = jax.lax.cond(
        prog_idx == PROGRAM_REDUC,
        lambda s: _apply_reduc(s),
        lambda s: s,
        state,
    )

    # UNDO - restore previous state
    state = jax.lax.cond(
        prog_idx == PROGRAM_UNDO,
        lambda s: _apply_undo(s),
        lambda s: s,
        state,
    )

    # DELAY - add 3 turns to all active transmissions
    state = jax.lax.cond(
        prog_idx == PROGRAM_DELAY,
        lambda s: _apply_delay(s),
        lambda s: s,
        state,
    )

    return state, is_wait


# =============================================================================
# Program Effect Implementations
# =============================================================================


def _apply_debug(s: EnvState) -> EnvState:
    """DEBUG - damage and stun enemies ON BLOCKS."""
    def damage_on_block(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        col = enemy[2]
        on_block = state.grid_block_type[row, col] != BLOCK_EMPTY
        should_damage = is_active & on_block
        new_hp = jnp.maximum(enemy[3] - 1, 0)
        new_enemies = jax.lax.cond(
            should_damage,
            lambda: state.enemies.at[idx, 3].set(new_hp).at[idx, 5].set(jnp.int32(new_hp > 0)),
            lambda: state.enemies,
        )
        new_mask = jax.lax.cond(
            should_damage & (new_hp <= 0),
            lambda: state.enemy_mask.at[idx].set(False),
            lambda: state.enemy_mask,
        )
        return state.replace(enemies=new_enemies, enemy_mask=new_mask), None
    state_out, _ = jax.lax.scan(damage_on_block, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_row(s: EnvState) -> EnvState:
    """ROW - damage enemies in player's row."""
    def damage_in_row(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        in_row = row == state.player.row
        should_damage = is_active & in_row
        new_hp = jnp.maximum(enemy[3] - 1, 0)
        new_enemies = jax.lax.cond(
            should_damage,
            lambda: state.enemies.at[idx, 3].set(new_hp),
            lambda: state.enemies,
        )
        new_mask = jax.lax.cond(
            should_damage & (new_hp <= 0),
            lambda: state.enemy_mask.at[idx].set(False),
            lambda: state.enemy_mask,
        )
        return state.replace(enemies=new_enemies, enemy_mask=new_mask), None
    state_out, _ = jax.lax.scan(damage_in_row, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_col(s: EnvState) -> EnvState:
    """COL - damage enemies in player's column."""
    def damage_in_col(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        col = enemy[2]
        in_col = col == state.player.col
        should_damage = is_active & in_col
        new_hp = jnp.maximum(enemy[3] - 1, 0)
        new_enemies = jax.lax.cond(
            should_damage,
            lambda: state.enemies.at[idx, 3].set(new_hp),
            lambda: state.enemies,
        )
        new_mask = jax.lax.cond(
            should_damage & (new_hp <= 0),
            lambda: state.enemy_mask.at[idx].set(False),
            lambda: state.enemy_mask,
        )
        return state.replace(enemies=new_enemies, enemy_mask=new_mask), None
    state_out, _ = jax.lax.scan(damage_in_col, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_crash(s: EnvState) -> EnvState:
    """CRASH - damage ALL enemies (1 damage)."""
    def damage_enemy(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        new_hp = jnp.maximum(enemy[3] - 1, 0)
        new_enemies = jax.lax.cond(
            is_active,
            lambda: state.enemies.at[idx, 3].set(new_hp),
            lambda: state.enemies,
        )
        new_mask = jax.lax.cond(
            is_active & (new_hp <= 0),
            lambda: state.enemy_mask.at[idx].set(False),
            lambda: state.enemy_mask,
        )
        return state.replace(enemies=new_enemies, enemy_mask=new_mask), None
    state_out, _ = jax.lax.scan(damage_enemy, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_antiv(s: EnvState) -> EnvState:
    """ANTI-V - damage and stun all viruses."""
    def damage_virus(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        is_virus = enemy[0] == ENEMY_VIRUS
        should_damage = is_active & is_virus
        new_hp = jnp.maximum(enemy[3] - 1, 0)
        new_enemies = jax.lax.cond(
            should_damage,
            lambda: state.enemies.at[idx, 3].set(new_hp).at[idx, 5].set(jnp.int32(new_hp > 0)),
            lambda: state.enemies,
        )
        new_mask = jax.lax.cond(
            should_damage & (new_hp <= 0),
            lambda: state.enemy_mask.at[idx].set(False),
            lambda: state.enemy_mask,
        )
        return state.replace(enemies=new_enemies, enemy_mask=new_mask), None
    state_out, _ = jax.lax.scan(damage_virus, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_hack(s: EnvState) -> EnvState:
    """HACK - damage enemies on siphoned cells."""
    def damage_on_siphoned(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        col = enemy[2]
        on_siphoned = state.grid_block_siphoned[row, col]
        should_damage = is_active & on_siphoned
        new_hp = jnp.maximum(enemy[3] - 2, 0)  # HACK does 2 damage
        new_enemies = jax.lax.cond(
            should_damage,
            lambda: state.enemies.at[idx, 3].set(new_hp),
            lambda: state.enemies,
        )
        new_mask = jax.lax.cond(
            should_damage & (new_hp <= 0),
            lambda: state.enemy_mask.at[idx].set(False),
            lambda: state.enemy_mask,
        )
        return state.replace(enemies=new_enemies, enemy_mask=new_mask), None
    state_out, _ = jax.lax.scan(damage_on_siphoned, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_push(s: EnvState) -> EnvState:
    """PUSH - push enemies away from player."""
    def push_enemy(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        col = enemy[2]
        row_delta = jnp.sign(row - state.player.row)
        col_delta = jnp.sign(col - state.player.col)
        new_row = jnp.clip(row + row_delta, 0, GRID_SIZE - 1)
        new_col = jnp.clip(col + col_delta, 0, GRID_SIZE - 1)
        new_enemies = jax.lax.cond(
            is_active,
            lambda: state.enemies.at[idx, 1].set(new_row).at[idx, 2].set(new_col),
            lambda: state.enemies,
        )
        return state.replace(enemies=new_enemies), None
    state_out, _ = jax.lax.scan(push_enemy, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_pull(s: EnvState) -> EnvState:
    """PULL - pull enemies toward player."""
    def pull_enemy(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        col = enemy[2]
        row_delta = jnp.sign(state.player.row - row)
        col_delta = jnp.sign(state.player.col - col)
        new_row = jnp.clip(row + row_delta, 0, GRID_SIZE - 1)
        new_col = jnp.clip(col + col_delta, 0, GRID_SIZE - 1)
        new_enemies = jax.lax.cond(
            is_active,
            lambda: state.enemies.at[idx, 1].set(new_row).at[idx, 2].set(new_col),
            lambda: state.enemies,
        )
        return state.replace(enemies=new_enemies), None
    state_out, _ = jax.lax.scan(pull_enemy, s, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_warp(s: EnvState) -> EnvState:
    """WARP - teleport to random enemy/transmission and kill it."""
    key = s.rng_key
    enemy_count = jnp.sum(s.enemy_mask.astype(jnp.int32))
    trans_count = jnp.sum(s.trans_mask.astype(jnp.int32))
    total_targets = enemy_count + trans_count

    key, subkey = jax.random.split(key)
    target_choice = jax.random.randint(subkey, (), 0, jnp.maximum(total_targets, 1))

    def find_enemy_target(carry, idx):
        count, found_idx, target = carry
        is_active = s.enemy_mask[idx]
        is_target = is_active & (count == target)
        new_count = count + jax.lax.cond(is_active, lambda: 1, lambda: 0)
        new_found = jax.lax.cond(is_target, lambda: jnp.int32(idx), lambda: found_idx)
        return (new_count, new_found, target), None

    (_, enemy_idx, _), _ = jax.lax.scan(
        find_enemy_target,
        (jnp.int32(0), jnp.int32(-1), target_choice),
        jnp.arange(s.enemy_mask.shape[0])
    )

    is_enemy_target = target_choice < enemy_count
    enemy_row = jax.lax.cond(
        is_enemy_target & (enemy_idx >= 0),
        lambda: s.enemies[enemy_idx, 1],
        lambda: jnp.int32(0)
    )
    enemy_col = jax.lax.cond(
        is_enemy_target & (enemy_idx >= 0),
        lambda: s.enemies[enemy_idx, 2],
        lambda: jnp.int32(0)
    )

    trans_target = target_choice - enemy_count
    def find_trans_target(carry, idx):
        count, found_idx, target = carry
        is_active = s.trans_mask[idx]
        is_target = is_active & (count == target)
        new_count = count + jax.lax.cond(is_active, lambda: 1, lambda: 0)
        new_found = jax.lax.cond(is_target, lambda: jnp.int32(idx), lambda: found_idx)
        return (new_count, new_found, target), None

    (_, trans_idx, _), _ = jax.lax.scan(
        find_trans_target,
        (jnp.int32(0), jnp.int32(-1), trans_target),
        jnp.arange(s.trans_mask.shape[0])
    )

    trans_row = jax.lax.cond(
        ~is_enemy_target & (trans_idx >= 0),
        lambda: s.transmissions[trans_idx, 0],
        lambda: jnp.int32(0)
    )
    trans_col = jax.lax.cond(
        ~is_enemy_target & (trans_idx >= 0),
        lambda: s.transmissions[trans_idx, 1],
        lambda: jnp.int32(0)
    )

    new_row = jax.lax.cond(is_enemy_target, lambda: enemy_row, lambda: trans_row)
    new_col = jax.lax.cond(is_enemy_target, lambda: enemy_col, lambda: trans_col)
    new_player = s.player.replace(row=new_row, col=new_col)

    new_enemy_mask = jax.lax.cond(
        is_enemy_target & (enemy_idx >= 0),
        lambda: s.enemy_mask.at[enemy_idx].set(False),
        lambda: s.enemy_mask
    )
    new_trans_mask = jax.lax.cond(
        ~is_enemy_target & (trans_idx >= 0),
        lambda: s.trans_mask.at[trans_idx].set(False),
        lambda: s.trans_mask
    )

    return s.replace(
        player=new_player,
        enemy_mask=new_enemy_mask,
        trans_mask=new_trans_mask,
        rng_key=key
    )


def _apply_poly(s: EnvState) -> EnvState:
    """POLY - change all enemies to a different type."""
    key = s.rng_key

    def change_enemy_type(carry, idx):
        state, key = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        current_type = enemy[0]
        key, subkey = jax.random.split(key)
        type_offset = jax.random.randint(subkey, (), 1, 4)
        new_type = (current_type + type_offset) % 4
        new_hp = ENEMY_MAX_HP[new_type]
        new_enemies = jax.lax.cond(
            is_active,
            lambda: state.enemies.at[idx, 0].set(new_type).at[idx, 3].set(new_hp),
            lambda: state.enemies
        )
        return (state.replace(enemies=new_enemies), key), None

    (new_state, new_key), _ = jax.lax.scan(
        change_enemy_type,
        (s, key),
        jnp.arange(s.enemy_mask.shape[0])
    )
    return new_state.replace(rng_key=new_key)


def _apply_dbom(s: EnvState) -> EnvState:
    """D_BOM - destroy nearest daemon, splash damage + stun around it."""
    def find_nearest_daemon(carry, idx):
        min_dist, nearest_idx, state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        is_daemon = enemy[0] == ENEMY_DAEMON
        row = enemy[1]
        col = enemy[2]
        dist = jnp.abs(row - state.player.row) + jnp.abs(col - state.player.col)
        is_closer = is_active & is_daemon & (dist < min_dist)
        new_min_dist = jax.lax.cond(is_closer, lambda: dist, lambda: min_dist)
        new_nearest = jax.lax.cond(is_closer, lambda: jnp.int32(idx), lambda: nearest_idx)
        return (new_min_dist, new_nearest, state), None

    (_, daemon_idx, _), _ = jax.lax.scan(
        find_nearest_daemon,
        (jnp.int32(100), jnp.int32(-1), s),
        jnp.arange(s.enemy_mask.shape[0])
    )

    daemon_row = jax.lax.cond(daemon_idx >= 0, lambda: s.enemies[daemon_idx, 1], lambda: jnp.int32(-10))
    daemon_col = jax.lax.cond(daemon_idx >= 0, lambda: s.enemies[daemon_idx, 2], lambda: jnp.int32(-10))
    new_mask = jax.lax.cond(daemon_idx >= 0, lambda: s.enemy_mask.at[daemon_idx].set(False), lambda: s.enemy_mask)

    def splash_damage(carry, idx):
        state = carry
        is_active = state.enemy_mask[idx]
        enemy = state.enemies[idx]
        row = enemy[1]
        col = enemy[2]
        row_dist = jnp.abs(row - daemon_row)
        col_dist = jnp.abs(col - daemon_col)
        is_adjacent = (row_dist <= 1) & (col_dist <= 1) & ((row_dist > 0) | (col_dist > 0))
        is_not_daemon = idx != daemon_idx
        should_damage = is_active & is_adjacent & is_not_daemon
        new_hp = jnp.maximum(enemy[3] - 1, 0)
        new_enemies = jax.lax.cond(
            should_damage,
            lambda: state.enemies.at[idx, 3].set(new_hp).at[idx, 5].set(jnp.int32(new_hp > 0)),
            lambda: state.enemies
        )
        new_enemy_mask = jax.lax.cond(
            should_damage & (new_hp <= 0),
            lambda: state.enemy_mask.at[idx].set(False),
            lambda: state.enemy_mask
        )
        return state.replace(enemies=new_enemies, enemy_mask=new_enemy_mask), None

    state_with_killed = s.replace(enemy_mask=new_mask)
    state_out, _ = jax.lax.scan(splash_damage, state_with_killed, jnp.arange(s.enemy_mask.shape[0]))
    return state_out


def _apply_score(s: EnvState) -> EnvState:
    """SCORE - gain points = (8 - stage)."""
    points_gained = 8 - s.stage
    new_score = s.player.score + points_gained
    return s.replace(player=s.player.replace(score=new_score))


def _apply_reduc(s: EnvState) -> EnvState:
    """REDUC - reduce spawn count of all unsiphoned blocks by 1."""
    has_block = s.grid_block_type != BLOCK_EMPTY
    not_siphoned = ~s.grid_block_siphoned
    should_reduce = has_block & not_siphoned
    new_spawn_count = jnp.where(
        should_reduce,
        jnp.maximum(s.grid_block_spawn_count - 1, 0),
        s.grid_block_spawn_count
    )
    return s.replace(grid_block_spawn_count=new_spawn_count)


def _apply_undo(s: EnvState) -> EnvState:
    """UNDO - restore previous state."""
    return s.replace(
        player=s.previous_player,
        enemies=s.previous_enemies,
        enemy_mask=s.previous_enemy_mask,
        transmissions=s.previous_transmissions,
        trans_mask=s.previous_trans_mask,
        turn=s.previous_turn,
        grid_block_siphoned=s.previous_grid_block_siphoned,
        grid_siphon_center=s.previous_grid_siphon_center,
        previous_state_valid=jnp.bool_(False),
    )


def _apply_delay(s: EnvState) -> EnvState:
    """DELAY - add 3 turns to all active transmissions."""
    def delay_transmission(carry, idx):
        state = carry
        is_active = state.trans_mask[idx]
        new_trans = jax.lax.cond(
            is_active,
            lambda: state.transmissions.at[idx, 2].set(state.transmissions[idx, 2] + 3),
            lambda: state.transmissions
        )
        return state.replace(transmissions=new_trans), None
    state_out, _ = jax.lax.scan(delay_transmission, s, jnp.arange(s.trans_mask.shape[0]))
    return state_out


# =============================================================================
# Program Validity Checking
# =============================================================================


def is_program_valid(state: EnvState, prog_idx: jnp.int32) -> jnp.bool_:
    """Check if a program can be used."""
    # Must own the program
    owns_program = state.owned_programs[prog_idx]

    # Must have enough resources
    cost = PROGRAM_COSTS[prog_idx]
    has_credits = state.player.credits >= cost[0]
    has_energy = state.player.energy >= cost[1]
    base_valid = owns_program & has_credits & has_energy

    # Count enemies
    enemy_count = jnp.sum(state.enemy_mask.astype(jnp.int32))
    has_enemies = enemy_count > 0

    # Check for adjacent enemies
    def has_adjacent_enemy(state):
        def check_adjacent(carry, idx):
            found, state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            row_dist = jnp.abs(enemy[1] - state.player.row)
            col_dist = jnp.abs(enemy[2] - state.player.col)
            is_adjacent = (row_dist + col_dist) == 1
            return (found | (is_active & is_adjacent), state), None
        (found, _), _ = jax.lax.scan(check_adjacent, (jnp.bool_(False), state), jnp.arange(state.enemy_mask.shape[0]))
        return found

    # Check for enemy in row/column
    def has_enemy_in_row(state):
        def check_row(carry, idx):
            found, state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            in_row = enemy[1] == state.player.row
            return (found | (is_active & in_row), state), None
        (found, _), _ = jax.lax.scan(check_row, (jnp.bool_(False), state), jnp.arange(state.enemy_mask.shape[0]))
        return found

    def has_enemy_in_col(state):
        def check_col(carry, idx):
            found, state = carry
            is_active = state.enemy_mask[idx]
            enemy = state.enemies[idx]
            in_col = enemy[2] == state.player.col
            return (found | (is_active & in_col), state), None
        (found, _), _ = jax.lax.scan(check_col, (jnp.bool_(False), state), jnp.arange(state.enemy_mask.shape[0]))
        return found

    enemy_in_row = has_enemy_in_row(state)
    enemy_in_col = has_enemy_in_col(state)

    # Check for virus/daemon
    def has_virus(state):
        def check_virus(carry, idx):
            found, state = carry
            is_active = state.enemy_mask[idx]
            is_virus = state.enemies[idx, 0] == ENEMY_VIRUS
            return (found | (is_active & is_virus), state), None
        (found, _), _ = jax.lax.scan(check_virus, (jnp.bool_(False), state), jnp.arange(state.enemy_mask.shape[0]))
        return found

    def has_daemon(state):
        def check_daemon(carry, idx):
            found, state = carry
            is_active = state.enemy_mask[idx]
            is_daemon = state.enemies[idx, 0] == ENEMY_DAEMON
            return (found | (is_active & is_daemon), state), None
        (found, _), _ = jax.lax.scan(check_daemon, (jnp.bool_(False), state), jnp.arange(state.enemy_mask.shape[0]))
        return found

    has_any_virus = has_virus(state)
    has_any_daemon = has_daemon(state)

    # Check for enemy on block
    def has_enemy_on_block(state):
        def check_on_block(carry, idx):
            found, state = carry
            is_active = state.enemy_mask[idx]
            row = state.enemies[idx, 1]
            col = state.enemies[idx, 2]
            on_block = state.grid_block_type[row, col] != BLOCK_EMPTY
            return (found | (is_active & on_block), state), None
        (found, _), _ = jax.lax.scan(check_on_block, (jnp.bool_(False), state), jnp.arange(state.enemy_mask.shape[0]))
        return found

    enemy_on_block = has_enemy_on_block(state)

    # Check for blocks
    has_siphoned = jnp.any(state.grid_block_siphoned)
    has_unsiphoned_with_spawn = jnp.any(
        (state.grid_block_type != BLOCK_EMPTY) &
        ~state.grid_block_siphoned &
        (state.grid_block_spawn_count > 0)
    )

    # Check for transmissions
    has_transmissions = jnp.any(state.trans_mask)

    # Check for surrounding targets (for CRASH)
    def has_surrounding_targets(state):
        player_row = state.player.row
        player_col = state.player.col

        def check_cell(carry, offset_idx):
            found, state = carry
            row_off = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1])[offset_idx]
            col_off = jnp.array([-1, 0, 1, -1, 1, -1, 0, 1])[offset_idx]
            cell_row = player_row + row_off
            cell_col = player_col + col_off
            in_bounds = (cell_row >= 0) & (cell_row < GRID_SIZE) & (cell_col >= 0) & (cell_col < GRID_SIZE)

            has_block_at = jax.lax.cond(
                in_bounds,
                lambda: state.grid_block_type[cell_row, cell_col] != BLOCK_EMPTY,
                lambda: jnp.bool_(False)
            )

            def check_enemy_at(state, r, c):
                def check_one(carry, idx):
                    found, state, r, c = carry
                    is_at = state.enemy_mask[idx] & (state.enemies[idx, 1] == r) & (state.enemies[idx, 2] == c)
                    return (found | is_at, state, r, c), None
                (found, _, _, _), _ = jax.lax.scan(check_one, (jnp.bool_(False), state, r, c), jnp.arange(state.enemy_mask.shape[0]))
                return found

            has_enemy_at = jax.lax.cond(in_bounds, lambda: check_enemy_at(state, cell_row, cell_col), lambda: jnp.bool_(False))

            def check_trans_at(state, r, c):
                def check_one(carry, idx):
                    found, state, r, c = carry
                    is_at = state.trans_mask[idx] & (state.transmissions[idx, 0] == r) & (state.transmissions[idx, 1] == c)
                    return (found | is_at, state, r, c), None
                (found, _, _, _), _ = jax.lax.scan(check_one, (jnp.bool_(False), state, r, c), jnp.arange(state.trans_mask.shape[0]))
                return found

            has_trans_at = jax.lax.cond(in_bounds, lambda: check_trans_at(state, cell_row, cell_col), lambda: jnp.bool_(False))

            return (found | has_block_at | has_enemy_at | has_trans_at, state), None

        (found, _), _ = jax.lax.scan(check_cell, (jnp.bool_(False), state), jnp.arange(8))
        return found

    has_surrounding = has_surrounding_targets(state)

    # Apply applicability based on program type
    needs_enemies = (prog_idx == PROGRAM_PUSH) | (prog_idx == PROGRAM_PULL) | (prog_idx == PROGRAM_POLY)
    enemy_check = jax.lax.cond(needs_enemies, lambda: has_enemies, lambda: jnp.bool_(True))

    warp_check = jax.lax.cond(prog_idx == PROGRAM_WARP, lambda: has_enemies | has_transmissions, lambda: jnp.bool_(True))
    crash_check = jax.lax.cond(prog_idx == PROGRAM_CRASH, lambda: has_surrounding, lambda: jnp.bool_(True))
    debug_check = jax.lax.cond(prog_idx == PROGRAM_DEBUG, lambda: enemy_on_block, lambda: jnp.bool_(True))
    row_check = jax.lax.cond(prog_idx == PROGRAM_ROW, lambda: enemy_in_row, lambda: jnp.bool_(True))
    col_check = jax.lax.cond(prog_idx == PROGRAM_COL, lambda: enemy_in_col, lambda: jnp.bool_(True))
    undo_check = jax.lax.cond(prog_idx == PROGRAM_UNDO, lambda: state.previous_state_valid, lambda: jnp.bool_(True))
    exch_check = jax.lax.cond(prog_idx == PROGRAM_EXCH, lambda: state.player.credits > 0, lambda: jnp.bool_(True))
    show_check = jax.lax.cond(prog_idx == PROGRAM_SHOW, lambda: ~state.show_activated, lambda: jnp.bool_(True))
    reset_check = jax.lax.cond(prog_idx == PROGRAM_RESET, lambda: state.player.hp < PLAYER_MAX_HP, lambda: jnp.bool_(True))
    calm_check = jax.lax.cond(prog_idx == PROGRAM_CALM, lambda: ~state.scheduled_tasks_disabled, lambda: jnp.bool_(True))
    dbom_check = jax.lax.cond(prog_idx == PROGRAM_D_BOM, lambda: has_any_daemon, lambda: jnp.bool_(True))
    score_check = jax.lax.cond(prog_idx == PROGRAM_SCORE, lambda: state.stage < 8, lambda: jnp.bool_(True))
    delay_check = jax.lax.cond(prog_idx == PROGRAM_DELAY, lambda: has_transmissions, lambda: jnp.bool_(True))
    antiv_check = jax.lax.cond(prog_idx == PROGRAM_ANTI_V, lambda: has_any_virus, lambda: jnp.bool_(True))
    reduc_check = jax.lax.cond(prog_idx == PROGRAM_REDUC, lambda: has_unsiphoned_with_spawn, lambda: jnp.bool_(True))
    atkplus_check = jax.lax.cond(
        prog_idx == PROGRAM_ATK_PLUS,
        lambda: (state.atk_plus_uses_this_stage < 2) & (state.player.attack_damage < 3),
        lambda: jnp.bool_(True)
    )
    hack_check = jax.lax.cond(prog_idx == PROGRAM_HACK, lambda: has_siphoned, lambda: jnp.bool_(True))

    return (base_valid & enemy_check & warp_check & crash_check & debug_check &
            row_check & col_check & undo_check & exch_check & show_check &
            reset_check & calm_check & dbom_check & delay_check & antiv_check &
            score_check & reduc_check & atkplus_check & hack_check)
