"""
Pure functional JAX environment for HackMatrix.
Designed for JIT compilation and TPU training with PureJaxRL.

This is the main environment file that provides reset(), step(), and get_valid_actions().
Game logic is implemented in separate modules for maintainability.
"""

import jax
import jax.numpy as jnp

from .state import (
    EnvState,
    Player,
    create_empty_state,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_PROGRAMS,
    PLAYER_MAX_HP,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_SIPHON,
    ACTION_PROGRAM_START,
    PROGRAM_WARP,
)
from .observation import Observation, get_observation
from .actions import execute_action, is_move_valid
from .programs import is_program_valid
from .enemy import process_enemy_turn
from .stage import save_previous_state, advance_stage
from .rewards import calculate_reward


# =============================================================================
# Core Environment Functions
# =============================================================================


@jax.jit
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

    obs = get_observation(state)
    return state, obs


@jax.jit
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
    state = save_previous_state(state)

    # Track action type for exit checking
    is_move_action = action < 4
    is_warp_action = action == (ACTION_PROGRAM_START + PROGRAM_WARP)

    # Execute action
    state, turn_ends = execute_action(state, action, subkey)

    # If turn ends, process enemy turn
    state = jax.lax.cond(
        turn_ends,
        lambda s: process_enemy_turn(s),
        lambda s: s,
        state,
    )

    # Check game end conditions
    # Movement and WARP can complete the stage by reaching exit
    player_died = state.player.hp <= 0
    reached_exit = (state.player.row == state.exit_row) & (state.player.col == state.exit_col)
    stage_complete = reached_exit & (state.stage <= 8) & (is_move_action | is_warp_action)

    # Advance stage if complete
    state = jax.lax.cond(
        stage_complete,
        lambda s: advance_stage(s),
        lambda s: s,
        state,
    )

    # Game is done if player died or won (stage 9)
    game_won = state.stage > 8
    done = player_died | game_won

    # Calculate reward
    reward = calculate_reward(state, stage_complete, game_won, player_died)
    state = state.replace(cumulative_reward=state.cumulative_reward + reward)

    obs = get_observation(state)
    return state, obs, reward, done


@jax.jit
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
    mask = mask.at[ACTION_MOVE_UP].set(is_move_valid(state, ACTION_MOVE_UP))
    mask = mask.at[ACTION_MOVE_DOWN].set(is_move_valid(state, ACTION_MOVE_DOWN))
    mask = mask.at[ACTION_MOVE_LEFT].set(is_move_valid(state, ACTION_MOVE_LEFT))
    mask = mask.at[ACTION_MOVE_RIGHT].set(is_move_valid(state, ACTION_MOVE_RIGHT))

    # Siphon (4): requires data_siphons > 0
    mask = mask.at[ACTION_SIPHON].set(state.player.data_siphons > 0)

    # Programs (5-27): requires ownership, resources, and applicability
    # Use fori_loop instead of Python for loop to avoid repeated tracing
    def check_program(i, mask_state):
        mask, state = mask_state
        prog_idx = i
        action_idx = ACTION_PROGRAM_START + prog_idx
        is_valid = is_program_valid(state, prog_idx)
        mask = mask.at[action_idx].set(is_valid)
        return (mask, state)

    mask, _ = jax.lax.fori_loop(0, NUM_PROGRAMS, check_program, (mask, state))

    return mask


# =============================================================================
# Vectorized Versions for Parallel Training
# =============================================================================


batched_reset = jax.vmap(reset)
batched_step = jax.vmap(step)
batched_get_valid_actions = jax.vmap(get_valid_actions)
