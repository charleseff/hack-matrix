"""
Reward calculation for HackMatrix JAX environment.

Implements all reward components matching Swift RewardCalculator.swift:
- Step penalty, stage completion, score gain, kills, data siphon collection
- Distance shaping (BFS pathfinding), HP change, victory bonus, death penalty
- Siphon-caused death penalty (extra -10.0 for dying to siphon-spawned enemy)
- Resource gain, resource holding, program waste penalty

Distance shaping uses BFS (not Manhattan) to match Swift Pathfinding.findDistance().
Death penalty uses stage-only cumulative calculation (sum of STAGE_COMPLETION_REWARDS
for completed stages), not the running cumulative_reward, matching Swift behavior.
"""

import jax
import jax.numpy as jnp

from .pathfinding import bfs_distance
from .state import ACTION_PROGRAM_START, PROGRAM_RESET, STAGE_COMPLETION_REWARDS, EnvState

# Distance shaping coefficient (reward per cell closer to exit)
DISTANCE_SHAPING_COEF = 0.05

# Per-step penalty to create time pressure and discourage oscillation
STEP_PENALTY = -0.01

# Resource reward multipliers (match Swift RewardCalculator constants)
CREDIT_GAIN_MULTIPLIER = 0.05
ENERGY_GAIN_MULTIPLIER = 0.05
CREDIT_HOLDING_MULTIPLIER = 0.01
ENERGY_HOLDING_MULTIPLIER = 0.01

# Program waste penalty
RESET_AT_2HP_PENALTY = -0.3

# Siphon-caused death: extra penalty when player dies to a siphon-spawned enemy
# Matches Swift RewardCalculator.siphonCausedDeathPenalty
SIPHON_CAUSED_DEATH_PENALTY = -10.0

# Action index for RESET program
ACTION_RESET = ACTION_PROGRAM_START + PROGRAM_RESET  # 5 + 14 = 19


def calculate_reward(
    state: EnvState,
    stage_completed: jnp.bool_,
    game_won: jnp.bool_,
    player_died: jnp.bool_,
    action: jnp.int32,
) -> jnp.float32:
    """Calculate reward for this transition.

    Matches Swift RewardCalculator.calculate() for full parity.

    Rewards:
    - Step penalty: -0.01 per step
    - Stage completion: [1, 2, 4, 8, 16, 32, 64, 100]
    - Score gain: delta * 0.5
    - Kill: 0.3 per enemy killed
    - Data siphon collection: 1.0
    - Distance shaping: +0.05 per cell closer to exit (one-directional)
    - HP change: +1.0/-1.0 per HP gained/lost
    - Victory: 500 + score * 100
    - Death penalty: -0.5 * sum(stage rewards for completed stages)
    - Siphon death: extra -10.0 when dying to a cardinally adjacent siphon-spawned enemy
    - Resource gain: credits_delta * 0.05 + energy_delta * 0.05
    - Resource holding: (credits * 0.01 + energy * 0.01) on stage completion
    - Program waste: -0.3 for RESET at 2 HP
    """
    # Start with step penalty (creates urgency to reach exit)
    reward = jnp.float32(STEP_PENALTY)

    # Stage completion reward
    # stage was already incremented by advance_stage, so -2 to get the completed stage index.
    # Guard: stage must be 2-9 after increment (completed stages 1-8).
    # Stage 9 means stage 8 was just completed (game won).
    stage_reward = jax.lax.cond(
        stage_completed & (state.stage >= 2) & (state.stage <= 9),
        lambda: STAGE_COMPLETION_REWARDS[state.stage - 2],
        lambda: jnp.float32(0.0),
    )
    reward = reward + stage_reward

    # Score gain reward (0.5 per point)
    score_delta = state.player.score - state.prev_score
    reward = reward + jnp.float32(score_delta) * 0.5

    # Kill reward (0.3 per kill)
    prev_enemy_count = jnp.sum(state.previous_enemy_mask.astype(jnp.int32))
    curr_enemy_count = jnp.sum(state.enemy_mask.astype(jnp.int32))
    kills = jnp.maximum(prev_enemy_count - curr_enemy_count, 0)
    reward = reward + jnp.float32(kills) * 0.3

    # Data siphon collection reward (1.0)
    prev_siphons = state.previous_player.data_siphons
    curr_siphons = state.player.data_siphons
    siphons_collected = jnp.maximum(curr_siphons - prev_siphons, 0)
    reward = reward + jnp.float32(siphons_collected) * 1.0

    # Distance shaping via BFS (one-directional: only reward getting closer)
    # prev_distance was pre-computed before the action in save_previous_state
    # to match Swift's oldDistanceToExit (uses grid before action modifies it).
    # Swift fallback: nil → 0 (no path means distance 0, so delta = 0).
    prev_distance = jnp.where(state.prev_distance_to_exit < 0, jnp.int32(0), state.prev_distance_to_exit)
    curr_bfs = bfs_distance(
        state.player.row, state.player.col,
        state.exit_row, state.exit_col,
        state.grid_block_type,
    )
    curr_distance = jnp.where(curr_bfs < 0, jnp.int32(0), curr_bfs)
    distance_delta = prev_distance - curr_distance  # Positive if moved closer
    distance_reward = jnp.maximum(distance_delta, 0) * DISTANCE_SHAPING_COEF
    reward = reward + jnp.float32(distance_reward)

    # HP change (damage penalty -1.0/HP, recovery reward +1.0/HP)
    hp_delta = state.player.hp - state.prev_hp
    reward = reward + jnp.float32(hp_delta) * 1.0

    # Victory bonus
    victory_bonus = jax.lax.cond(
        game_won,
        lambda: 500.0 + jnp.float32(state.player.score) * 100.0,
        lambda: jnp.float32(0.0),
    )
    reward = reward + victory_bonus

    # Death penalty: -0.5 * cumulative stage rewards for completed stages
    # Swift: for i in 0..<(currentStage - 1) { cumulativeReward += stageRewards[i] }
    # When player dies, stage has NOT been incremented, so state.stage is current stage.
    # Completed stages = stages before current = 0..<(stage-1)
    death_penalty = jax.lax.cond(
        player_died,
        lambda: _stage_death_penalty(state.stage),
        lambda: jnp.float32(0.0),
    )
    reward = reward + death_penalty

    # Siphon-caused death penalty: extra -10.0 when dying to siphon-spawned enemy
    # Swift: scan enemies for any adjacent to player with spawnedFromSiphon == true
    # Adjacent = cardinal only (Manhattan distance 1, not diagonal)
    siphon_death = jnp.where(
        player_died & _any_adjacent_siphon_enemy(state),
        jnp.float32(SIPHON_CAUSED_DEATH_PENALTY),
        jnp.float32(0.0),
    )
    reward = reward + siphon_death

    # Resource gain reward: credits_delta * 0.05 + energy_delta * 0.05
    credits_delta = state.player.credits - state.prev_credits
    energy_delta = state.player.energy - state.prev_energy
    resource_gain = (
        jnp.float32(credits_delta) * CREDIT_GAIN_MULTIPLIER
        + jnp.float32(energy_delta) * ENERGY_GAIN_MULTIPLIER
    )
    reward = reward + resource_gain

    # Resource holding bonus (only on stage completion)
    resource_holding = jax.lax.cond(
        stage_completed,
        lambda: (
            jnp.float32(state.player.credits) * CREDIT_HOLDING_MULTIPLIER
            + jnp.float32(state.player.energy) * ENERGY_HOLDING_MULTIPLIER
        ),
        lambda: jnp.float32(0.0),
    )
    reward = reward + resource_holding

    # Program waste penalty: RESET at 2 HP wastes 1 HP (max is 3, only gains 1 instead of 2)
    is_reset_action = action == ACTION_RESET
    was_at_2hp = state.prev_hp == 2
    program_waste = jnp.where(
        is_reset_action & was_at_2hp,
        jnp.float32(RESET_AT_2HP_PENALTY),
        jnp.float32(0.0),
    )
    reward = reward + program_waste

    return reward


def _any_adjacent_siphon_enemy(state: EnvState) -> jnp.bool_:
    """Check if any active enemy is cardinally adjacent and spawned from siphon.

    Matches Swift GameState.isAdjacentToPlayer: cardinal adjacency only
    (|row_diff| == 1 && col_diff == 0) || (row_diff == 0 && |col_diff| == 1).

    Enemy array columns: [type, row, col, hp, disabled_turns, is_stunned,
                          spawned_from_siphon, is_from_scheduled_task]
    """
    enemy_rows = state.enemies[:, 1]
    enemy_cols = state.enemies[:, 2]
    spawned_from_siphon = state.enemies[:, 6]

    row_diff = jnp.abs(enemy_rows - state.player.row)
    col_diff = jnp.abs(enemy_cols - state.player.col)

    is_cardinal = ((row_diff == 1) & (col_diff == 0)) | ((row_diff == 0) & (col_diff == 1))
    is_siphon = spawned_from_siphon == 1
    is_active = state.enemy_mask

    return jnp.any(is_cardinal & is_siphon & is_active)


def _stage_death_penalty(current_stage: jnp.int32) -> jnp.float32:
    """Calculate death penalty based on completed stage rewards.

    Matches Swift: sum of STAGE_COMPLETION_REWARDS[0..<(currentStage-1)] * -0.5
    Stage 1 death = 0 penalty (no stages completed).
    Stage 2 death = -0.5 (completed stage 1, reward 1.0).
    Stage 4 death = -3.5 (completed stages 1-3, rewards 1+2+4=7).
    """
    # Build mask: True for indices < (current_stage - 1)
    indices = jnp.arange(8)
    completed_mask = indices < (current_stage - 1)
    stage_cumulative = jnp.sum(STAGE_COMPLETION_REWARDS * completed_mask)
    return -stage_cumulative * 0.5
