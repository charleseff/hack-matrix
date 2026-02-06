"""
Reward calculation for HackMatrix JAX environment.

Implements all 15 reward components matching Swift RewardCalculator.swift:
- Step penalty, stage completion, score gain, kills, data siphon collection
- Distance shaping (BFS pathfinding), HP change, victory bonus, death penalty
- Siphon-caused death penalty (extra -10.0 for dying to siphon-spawned enemy)
- Resource gain, resource holding, program waste penalty
- Siphon quality penalty (suboptimal position penalized proportionally)

Distance shaping uses BFS (not Manhattan) to match Swift Pathfinding.findDistance().
Death penalty uses stage-only cumulative calculation (sum of STAGE_COMPLETION_REWARDS
for completed stages), not the running cumulative_reward, matching Swift behavior.
Siphon quality is pre-computed before the action to use correct pre-siphon state.
"""

import jax
import jax.numpy as jnp

from .pathfinding import bfs_distance
from .state import ACTION_PROGRAM_START, ACTION_SIPHON, PROGRAM_RESET, STAGE_COMPLETION_REWARDS, EnvState

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

# Siphon quality: penalty multiplier for suboptimal siphon position
# Matches Swift RewardCalculator.siphonSuboptimalPenalty
SIPHON_SUBOPTIMAL_PENALTY = -0.5

# Action index for RESET program
ACTION_RESET = ACTION_PROGRAM_START + PROGRAM_RESET  # 5 + 14 = 19


def calculate_reward(
    state: EnvState,
    stage_completed: jnp.bool_,
    game_won: jnp.bool_,
    player_died: jnp.bool_,
    action: jnp.int32,
) -> tuple[jnp.float32, dict[str, jnp.float32]]:
    """Calculate reward for this transition, returning total and per-component breakdown.

    Matches Swift RewardCalculator.calculate() for full parity.
    Returns (total_reward, breakdown_dict) where breakdown_dict has 15 keys
    that sum to total_reward.

    Breakdown keys:
    - step_penalty: -0.01 per step
    - stage_completion: [1, 2, 4, 8, 16, 32, 64, 100]
    - score_gain: delta * 0.5
    - kills: 0.3 per enemy killed
    - data_siphon: 1.0 per siphon collected
    - distance_shaping: +0.05 per cell closer to exit (one-directional)
    - damage_penalty: -1.0 per HP lost (negative part of hp_delta)
    - hp_recovery: +1.0 per HP gained (positive part of hp_delta)
    - victory: 500 + score * 100
    - death_penalty: -0.5 * sum(stage rewards for completed stages)
    - siphon_death_penalty: extra -10.0 for dying to siphon-spawned enemy
    - resource_gain: credits_delta * 0.05 + energy_delta * 0.05
    - resource_holding: (credits * 0.01 + energy * 0.01) on stage completion
    - program_waste: -0.3 for RESET at 2 HP
    - siphon_quality: -0.5 * (missed_credits * 0.05 + missed_energy * 0.05)
    """
    # Step penalty (creates urgency to reach exit)
    step_penalty = jnp.float32(STEP_PENALTY)

    # Stage completion reward
    # stage was already incremented by advance_stage, so -2 to get the completed stage index.
    # Guard: stage must be 2-9 after increment (completed stages 1-8).
    # Stage 9 means stage 8 was just completed (game won).
    stage_completion = jax.lax.cond(
        stage_completed & (state.stage >= 2) & (state.stage <= 9),
        lambda: STAGE_COMPLETION_REWARDS[state.stage - 2],
        lambda: jnp.float32(0.0),
    )

    # Score gain reward (0.5 per point)
    score_delta = state.player.score - state.prev_score
    score_gain = jnp.float32(score_delta) * 0.5

    # Kill reward (0.3 per kill)
    prev_enemy_count = jnp.sum(state.previous_enemy_mask.astype(jnp.int32))
    curr_enemy_count = jnp.sum(state.enemy_mask.astype(jnp.int32))
    kills_count = jnp.maximum(prev_enemy_count - curr_enemy_count, 0)
    kills = jnp.float32(kills_count) * 0.3

    # Data siphon collection reward (1.0)
    prev_siphons = state.previous_player.data_siphons
    curr_siphons = state.player.data_siphons
    siphons_collected = jnp.maximum(curr_siphons - prev_siphons, 0)
    data_siphon = jnp.float32(siphons_collected) * 1.0

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
    distance_shaping = jnp.float32(jnp.maximum(distance_delta, 0) * DISTANCE_SHAPING_COEF)

    # HP change — split into damage_penalty (negative) and hp_recovery (positive)
    # Total HP reward = hp_delta * 1.0 = damage_penalty + hp_recovery
    hp_delta = state.player.hp - state.prev_hp
    damage_penalty = jnp.float32(jnp.minimum(hp_delta, 0)) * 1.0
    hp_recovery = jnp.float32(jnp.maximum(hp_delta, 0)) * 1.0

    # Victory bonus
    victory = jax.lax.cond(
        game_won,
        lambda: 500.0 + jnp.float32(state.player.score) * 100.0,
        lambda: jnp.float32(0.0),
    )

    # Death penalty: -0.5 * cumulative stage rewards for completed stages
    # Swift: for i in 0..<(currentStage - 1) { cumulativeReward += stageRewards[i] }
    # When player dies, stage has NOT been incremented, so state.stage is current stage.
    # Completed stages = stages before current = 0..<(stage-1)
    death_penalty = jax.lax.cond(
        player_died,
        lambda: _stage_death_penalty(state.stage),
        lambda: jnp.float32(0.0),
    )

    # Siphon-caused death penalty: extra -10.0 when dying to siphon-spawned enemy
    # Swift: scan enemies for any adjacent to player with spawnedFromSiphon == true
    # Adjacent = cardinal only (Manhattan distance 1, not diagonal)
    siphon_death_penalty = jnp.where(
        player_died & _any_adjacent_siphon_enemy(state),
        jnp.float32(SIPHON_CAUSED_DEATH_PENALTY),
        jnp.float32(0.0),
    )

    # Resource gain reward: credits_delta * 0.05 + energy_delta * 0.05
    credits_delta = state.player.credits - state.prev_credits
    energy_delta = state.player.energy - state.prev_energy
    resource_gain = (
        jnp.float32(credits_delta) * CREDIT_GAIN_MULTIPLIER
        + jnp.float32(energy_delta) * ENERGY_GAIN_MULTIPLIER
    )

    # Resource holding bonus (only on stage completion)
    resource_holding = jax.lax.cond(
        stage_completed,
        lambda: (
            jnp.float32(state.player.credits) * CREDIT_HOLDING_MULTIPLIER
            + jnp.float32(state.player.energy) * ENERGY_HOLDING_MULTIPLIER
        ),
        lambda: jnp.float32(0.0),
    )

    # Program waste penalty: RESET at 2 HP wastes 1 HP (max is 3, only gains 1 instead of 2)
    is_reset_action = action == ACTION_RESET
    was_at_2hp = state.prev_hp == 2
    program_waste = jnp.where(
        is_reset_action & was_at_2hp,
        jnp.float32(RESET_AT_2HP_PENALTY),
        jnp.float32(0.0),
    )

    # Siphon quality penalty: penalize suboptimal siphon positioning.
    # Pre-computed in env.py before the action modifies grid state.
    # Only applies when action is siphon AND a strictly better position exists.
    is_siphon_action = action == ACTION_SIPHON
    missed_value = (
        jnp.float32(state.siphon_missed_credits) * CREDIT_GAIN_MULTIPLIER
        + jnp.float32(state.siphon_missed_energy) * ENERGY_GAIN_MULTIPLIER
    )
    siphon_quality = jnp.where(
        is_siphon_action & state.siphon_found_better,
        jnp.float32(SIPHON_SUBOPTIMAL_PENALTY) * missed_value,
        jnp.float32(0.0),
    )

    # Sum all components
    total = (
        step_penalty + stage_completion + score_gain + kills + data_siphon
        + distance_shaping + damage_penalty + hp_recovery + victory
        + death_penalty + siphon_death_penalty + resource_gain
        + resource_holding + program_waste + siphon_quality
    )

    breakdown = {
        "step_penalty": step_penalty,
        "stage_completion": stage_completion,
        "score_gain": score_gain,
        "kills": kills,
        "data_siphon": data_siphon,
        "distance_shaping": distance_shaping,
        "damage_penalty": damage_penalty,
        "hp_recovery": hp_recovery,
        "victory": victory,
        "death_penalty": death_penalty,
        "siphon_death_penalty": siphon_death_penalty,
        "resource_gain": resource_gain,
        "resource_holding": resource_holding,
        "program_waste": program_waste,
        "siphon_quality": siphon_quality,
    }

    return total, breakdown


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
