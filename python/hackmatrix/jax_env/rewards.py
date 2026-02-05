"""
Reward calculation for HackMatrix JAX environment.
"""

import jax
import jax.numpy as jnp

from .state import STAGE_COMPLETION_REWARDS, EnvState

# Distance shaping coefficient (reward per cell closer to exit)
DISTANCE_SHAPING_COEF = 0.05

# Per-step penalty to create time pressure and discourage oscillation
STEP_PENALTY = -0.01


def calculate_reward(
    state: EnvState,
    stage_completed: jnp.bool_,
    game_won: jnp.bool_,
    player_died: jnp.bool_,
) -> jnp.float32:
    """Calculate reward for this transition.

    Rewards:
    - Step penalty: -0.01 per step (creates time pressure, discourages oscillation)
    - Stage completion: [1, 2, 4, 8, 16, 32, 64, 100]
    - Score gain: delta * 0.5
    - Kill: 0.3 per enemy killed
    - Data siphon collection: 1.0
    - Distance shaping: +0.05 per cell closer to exit (one-directional, no penalty for moving away)
    - HP gain: +1.0 per HP
    - HP loss: -1.0 per HP
    - Victory: 500 + score * 100
    - Death: -cumulative * 0.5
    """
    # Start with step penalty (creates urgency to reach exit)
    reward = jnp.float32(STEP_PENALTY)

    # Stage completion reward
    stage_reward = jax.lax.cond(
        stage_completed & (state.stage <= 8),
        lambda: STAGE_COMPLETION_REWARDS[state.stage - 2],  # -2 because stage was incremented
        lambda: jnp.float32(0.0),
    )
    reward = reward + stage_reward

    # Score gain reward (0.5 per point)
    score_delta = state.player.score - state.prev_score
    reward = reward + jnp.float32(score_delta) * 0.5

    # Kill reward (0.3 per kill)
    # Count enemies that were active before but not now
    prev_enemy_count = jnp.sum(state.previous_enemy_mask.astype(jnp.int32))
    curr_enemy_count = jnp.sum(state.enemy_mask.astype(jnp.int32))
    kills = jnp.maximum(prev_enemy_count - curr_enemy_count, 0)
    kill_reward = jnp.float32(kills) * 0.3
    reward = reward + kill_reward

    # Data siphon collection reward (1.0)
    prev_siphons = state.previous_player.data_siphons
    curr_siphons = state.player.data_siphons
    siphons_collected = jnp.maximum(curr_siphons - prev_siphons, 0)
    # Only count if not from SIPH+ (which deducts credits)
    # For simplicity, always reward siphon collection
    siphon_reward = jnp.float32(siphons_collected) * 1.0
    reward = reward + siphon_reward

    # Distance shaping (one-directional: only reward getting closer, no penalty for moving away)
    # This prevents the agent from "farming" reward by oscillating back and forth
    prev_distance = (
        jnp.abs(state.previous_player.row - state.exit_row)
        + jnp.abs(state.previous_player.col - state.exit_col)
    )
    curr_distance = (
        jnp.abs(state.player.row - state.exit_row)
        + jnp.abs(state.player.col - state.exit_col)
    )
    distance_delta = prev_distance - curr_distance  # Positive if moved closer
    # Only reward positive progress (moved closer), ignore moving away
    distance_reward = jnp.maximum(distance_delta, 0) * DISTANCE_SHAPING_COEF
    reward = reward + jnp.float32(distance_reward)

    # HP change
    hp_delta = state.player.hp - state.prev_hp
    reward = reward + jnp.float32(hp_delta) * 1.0  # +1 per HP gained, -1 per HP lost

    # Victory bonus
    victory_bonus = jax.lax.cond(
        game_won,
        lambda: 500.0 + jnp.float32(state.player.score) * 100.0,
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
