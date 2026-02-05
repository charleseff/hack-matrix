"""
JAX-only reward parity tests — Phases 1-3.

Tests call `calculate_reward` directly with constructed EnvState objects,
verifying each reward component matches Swift RewardCalculator.swift behavior.

Why direct testing:
- The parity/test_rewards.py tests exercise the full step() pipeline through
  the env interface (set_state → step → check reward), which couples rewards
  to movement, enemy AI, and stage logic.
- These tests isolate the reward function itself, making failures precise:
  a failing test here means the reward formula is wrong, not some upstream bug.

Phase 3 additions:
- BFS distance shaping tests verify pathfinding matches Swift Pathfinding.findDistance()
- Tests cover: block walls, no-path fallback, target-on-block edge case
"""

import jax
import jax.numpy as jnp
import pytest

from hackmatrix.jax_env.pathfinding import bfs_distance
from hackmatrix.jax_env.rewards import (
    ACTION_RESET,
    CREDIT_GAIN_MULTIPLIER,
    CREDIT_HOLDING_MULTIPLIER,
    DISTANCE_SHAPING_COEF,
    ENERGY_GAIN_MULTIPLIER,
    ENERGY_HOLDING_MULTIPLIER,
    RESET_AT_2HP_PENALTY,
    SIPHON_CAUSED_DEATH_PENALTY,
    STEP_PENALTY,
    calculate_reward,
)
from hackmatrix.jax_env.state import (
    ACTION_MOVE_UP,
    ACTION_SIPHON,
    GRID_SIZE,
    MAX_ENEMIES,
    STAGE_COMPLETION_REWARDS,
    Player,
    create_empty_state,
)


def _make_state(**overrides):
    """Build an EnvState with sensible defaults, applying overrides.

    Defaults: player at (0,0), hp=3, exit at (5,5), stage=1, no enemies.
    The 'previous' snapshot mirrors 'current' so deltas start at zero.

    prev_distance_to_exit is auto-computed from previous_player position and
    grid_block_type via BFS, unless explicitly provided. This matches
    save_previous_state() which computes BFS before the action.
    """
    key = jax.random.PRNGKey(0)
    state = create_empty_state(key)

    # Apply player overrides
    player_fields = {}
    prev_fields = {}
    for field in ("row", "col", "hp", "credits", "energy", "data_siphons", "attack_damage", "score"):
        player_fields[field] = jnp.int32(overrides.pop(field, getattr(state.player, field)))
    player = Player(**player_fields)

    # Previous player mirrors current by default, unless explicitly overridden
    for field in ("row", "col", "hp", "credits", "energy", "data_siphons", "attack_damage", "score"):
        key_name = f"prev_player_{field}"
        prev_fields[field] = jnp.int32(overrides.pop(key_name, player_fields[field]))
    previous_player = Player(**prev_fields)

    # Scalar state overrides
    stage = jnp.int32(overrides.pop("stage", 1))
    prev_hp = jnp.int32(overrides.pop("prev_hp", player_fields["hp"]))
    prev_score = jnp.int32(overrides.pop("prev_score", player_fields["score"]))
    prev_credits = jnp.int32(overrides.pop("prev_credits", player_fields["credits"]))
    prev_energy = jnp.int32(overrides.pop("prev_energy", player_fields["energy"]))
    cumulative_reward = jnp.float32(overrides.pop("cumulative_reward", 0.0))

    # Enemy overrides
    enemies = overrides.pop("enemies", state.enemies)
    enemy_mask = overrides.pop("enemy_mask", state.enemy_mask)
    previous_enemy_mask = overrides.pop("previous_enemy_mask", state.previous_enemy_mask)

    # Grid overrides (for BFS pathfinding tests)
    grid_block_type = overrides.pop("grid_block_type", state.grid_block_type)

    # prev_distance_to_exit: auto-compute from previous player pos + grid if not given
    explicit_prev_dist = overrides.pop("prev_distance_to_exit", None)

    if overrides:
        raise ValueError(f"Unknown overrides: {overrides}")

    state = state.replace(
        player=player,
        previous_player=previous_player,
        stage=stage,
        prev_hp=prev_hp,
        prev_score=prev_score,
        prev_credits=prev_credits,
        prev_energy=prev_energy,
        cumulative_reward=cumulative_reward,
        enemies=enemies,
        enemy_mask=enemy_mask,
        previous_enemy_mask=previous_enemy_mask,
        grid_block_type=grid_block_type,
    )

    # Compute prev_distance_to_exit via BFS from previous player position
    prev_dist = (
        jnp.int32(explicit_prev_dist) if explicit_prev_dist is not None
        else bfs_distance(
            previous_player.row, previous_player.col,
            state.exit_row, state.exit_col,
            grid_block_type,
        )
    )
    return state.replace(prev_distance_to_exit=prev_dist)


def _reward(state, *, stage_completed=False, game_won=False, player_died=False, action=0):
    """Call calculate_reward with keyword-friendly booleans."""
    return float(
        calculate_reward(
            state,
            jnp.bool_(stage_completed),
            jnp.bool_(game_won),
            jnp.bool_(player_died),
            jnp.int32(action),
        )
    )


# ---------------------------------------------------------------------------
# Step penalty
# ---------------------------------------------------------------------------

class TestStepPenalty:
    """Every step costs -0.01 to create time pressure."""

    def test_baseline_step_penalty(self):
        """A no-op step (no movement, no events) yields exactly the step penalty."""
        state = _make_state()
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY, abs=1e-6)


# ---------------------------------------------------------------------------
# Stage completion
# ---------------------------------------------------------------------------

class TestStageCompletion:
    """Completing a stage awards STAGE_COMPLETION_REWARDS[stage-1]."""

    @pytest.mark.parametrize("completed_stage,expected", [
        (1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0),
        (5, 16.0), (6, 32.0), (7, 64.0), (8, 100.0),
    ])
    def test_stage_rewards(self, completed_stage, expected):
        """Stage N completion gives the Nth reward from the table."""
        # After advance_stage, state.stage = completed_stage + 1
        state = _make_state(stage=completed_stage + 1)
        r = _reward(state, stage_completed=True)
        assert r == pytest.approx(STEP_PENALTY + expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Score gain
# ---------------------------------------------------------------------------

class TestScoreGain:
    """Score delta * 0.5."""

    def test_score_gain_10_points(self):
        state = _make_state(score=10, prev_score=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + 10 * 0.5, abs=1e-5)

    def test_no_score_change(self):
        state = _make_state(score=5, prev_score=5)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY, abs=1e-5)


# ---------------------------------------------------------------------------
# Kill reward
# ---------------------------------------------------------------------------

class TestKillReward:
    """0.3 per enemy killed (prev active - curr active)."""

    def test_one_kill(self):
        prev_mask = jnp.zeros(20, dtype=jnp.bool_).at[0].set(True)
        curr_mask = jnp.zeros(20, dtype=jnp.bool_)
        state = _make_state(previous_enemy_mask=prev_mask, enemy_mask=curr_mask)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + 0.3, abs=1e-5)

    def test_two_kills(self):
        prev_mask = jnp.zeros(20, dtype=jnp.bool_).at[0].set(True).at[1].set(True)
        curr_mask = jnp.zeros(20, dtype=jnp.bool_)
        state = _make_state(previous_enemy_mask=prev_mask, enemy_mask=curr_mask)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + 0.6, abs=1e-5)


# ---------------------------------------------------------------------------
# Data siphon collection
# ---------------------------------------------------------------------------

class TestDataSiphonCollection:
    """1.0 per data siphon collected."""

    def test_collect_one_siphon(self):
        state = _make_state(data_siphons=1, prev_player_data_siphons=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + 1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Distance shaping
# ---------------------------------------------------------------------------

class TestDistanceShaping:
    """BFS-based distance shaping: +0.05 per cell closer to exit.

    Uses BFS pathfinding (matching Swift Pathfinding.findDistance) instead of
    Manhattan distance. On an empty grid, BFS == Manhattan. With blocks, BFS
    accurately measures walkable distance around obstacles.
    """

    def test_move_one_cell_closer(self):
        """Empty grid: BFS distance == Manhattan. Move (0,0)→(1,0), exit at (5,5)."""
        # BFS(0,0→5,5) = 10, BFS(1,0→5,5) = 9. Delta = +1.
        state = _make_state(row=1, col=0, prev_player_row=0, prev_player_col=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + DISTANCE_SHAPING_COEF, abs=1e-5)

    def test_move_away_no_penalty(self):
        """One-directional: moving away gives 0 distance reward (not negative)."""
        # BFS(1,0→5,5) = 9, BFS(0,0→5,5) = 10. Delta = -1 → clamped to 0.
        state = _make_state(row=0, col=0, prev_player_row=1, prev_player_col=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY, abs=1e-5)

    def test_bfs_with_block_wall(self):
        """BFS correctly routes around a block wall that Manhattan ignores.

        Grid layout (exit at 5,5):
          Row 3 has blocks at cols 0-4 (wall), col 5 is open.
          Player moves from (2,4) to (2,5).

          With blocks: BFS from (2,4) must go around the wall.
          BFS(2,4→5,5) goes: (2,4)→(2,5)→(3,5)→(4,5)→(5,5) = 4
          BFS(2,5→5,5) goes: (2,5)→(3,5)→(4,5)→(5,5) = 3
          Delta = 4 - 3 = 1 → reward = 0.05
        """
        grid_block_type = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        # Wall across row 3, cols 0-4
        grid_block_type = grid_block_type.at[3, 0].set(1)
        grid_block_type = grid_block_type.at[3, 1].set(1)
        grid_block_type = grid_block_type.at[3, 2].set(1)
        grid_block_type = grid_block_type.at[3, 3].set(1)
        grid_block_type = grid_block_type.at[3, 4].set(1)

        state = _make_state(
            row=2, col=5,
            prev_player_row=2, prev_player_col=4,
            grid_block_type=grid_block_type,
        )
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + DISTANCE_SHAPING_COEF, abs=1e-5)

    def test_bfs_no_path_no_reward(self):
        """When BFS finds no path, distance falls back to 0 → delta = 0.

        Matching Swift: findDistance returns nil → ?? 0.
        Both prev and curr distances are 0, so no distance reward.
        """
        grid_block_type = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        # Completely wall off the player with blocks on all sides of row 0
        # Row 1 all blocks → player at (0,x) cannot reach row 2+
        for c in range(GRID_SIZE):
            grid_block_type = grid_block_type.at[1, c].set(1)

        state = _make_state(
            row=0, col=1,
            prev_player_row=0, prev_player_col=0,
            grid_block_type=grid_block_type,
        )
        r = _reward(state)
        # Both BFS distances are -1 → fallback 0. Delta = 0.
        assert r == pytest.approx(STEP_PENALTY, abs=1e-5)

    def test_bfs_target_on_block_still_reachable(self):
        """BFS matches Swift: target cell is checked BEFORE block check.

        Swift's findDistance checks `if newRow == target` before `if cell.hasBlock`,
        so the target is reachable even if it has a block.
        """
        grid_block_type = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        # Put a block on the exit cell (5,5)
        grid_block_type = grid_block_type.at[5, 5].set(1)

        state = _make_state(
            row=5, col=4,
            prev_player_row=4, prev_player_col=4,
            grid_block_type=grid_block_type,
        )
        # BFS(4,4→5,5) = 2, BFS(5,4→5,5) = 1. Delta = 1.
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + DISTANCE_SHAPING_COEF, abs=1e-5)


# ---------------------------------------------------------------------------
# HP change
# ---------------------------------------------------------------------------

class TestHPChange:
    """+1.0 per HP gained, -1.0 per HP lost."""

    def test_take_one_damage(self):
        state = _make_state(hp=2, prev_hp=3)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY - 1.0, abs=1e-5)

    def test_heal_two_hp(self):
        state = _make_state(hp=3, prev_hp=1)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + 2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Victory bonus
# ---------------------------------------------------------------------------

class TestVictoryBonus:
    """500 + score * 100."""

    def test_victory_with_score_10(self):
        state = _make_state(score=10)
        r = _reward(state, game_won=True)
        assert r == pytest.approx(STEP_PENALTY + 500.0 + 10 * 100.0, abs=1e-3)

    def test_victory_with_score_0(self):
        state = _make_state(score=0)
        r = _reward(state, game_won=True)
        assert r == pytest.approx(STEP_PENALTY + 500.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Death penalty (stage-only calculation — key fix in Phase 1)
# ---------------------------------------------------------------------------

class TestDeathPenalty:
    """Death penalty = -0.5 * sum(stage rewards for completed stages).

    Uses stage-only cumulative, NOT the running cumulative_reward field.
    When a player dies, stage has NOT been incremented.
    """

    def test_stage_1_death_zero_penalty(self):
        """Die on stage 1: no stages completed → 0 penalty (just step + HP)."""
        state = _make_state(stage=1, hp=0, prev_hp=1)
        r = _reward(state, player_died=True)
        # penalty = 0 (no completed stages), HP loss = -1.0
        assert r == pytest.approx(STEP_PENALTY - 1.0, abs=1e-5)

    def test_stage_2_death(self):
        """Die on stage 2: completed stage 1 (reward 1.0) → penalty -0.5."""
        state = _make_state(stage=2, hp=0, prev_hp=1)
        r = _reward(state, player_died=True)
        assert r == pytest.approx(STEP_PENALTY - 1.0 - 0.5, abs=1e-5)

    def test_stage_4_death(self):
        """Die on stage 4: completed 1-3 (1+2+4=7) → penalty -3.5."""
        state = _make_state(stage=4, hp=0, prev_hp=1)
        r = _reward(state, player_died=True)
        assert r == pytest.approx(STEP_PENALTY - 1.0 - 3.5, abs=1e-5)

    def test_stage_8_death(self):
        """Die on stage 8: completed 1-7 (1+2+4+8+16+32+64=127) → penalty -63.5."""
        state = _make_state(stage=8, hp=0, prev_hp=1)
        r = _reward(state, player_died=True)
        assert r == pytest.approx(STEP_PENALTY - 1.0 - 63.5, abs=1e-5)

    def test_death_penalty_ignores_cumulative_reward_field(self):
        """Death penalty must NOT use state.cumulative_reward (old bug)."""
        state = _make_state(stage=1, hp=0, prev_hp=1, cumulative_reward=100.0)
        r = _reward(state, player_died=True)
        # Stage 1: no completed stages → penalty 0. If it used cumulative_reward,
        # penalty would be -50.0 which is very different.
        assert r == pytest.approx(STEP_PENALTY - 1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Resource gain (NEW in Phase 1)
# ---------------------------------------------------------------------------

class TestResourceGain:
    """credits_delta * 0.05 + energy_delta * 0.05."""

    def test_gain_credits(self):
        state = _make_state(credits=10, prev_credits=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + 10 * CREDIT_GAIN_MULTIPLIER, abs=1e-5)

    def test_gain_energy(self):
        state = _make_state(energy=8, prev_energy=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + 8 * ENERGY_GAIN_MULTIPLIER, abs=1e-5)

    def test_gain_both(self):
        state = _make_state(credits=5, energy=3, prev_credits=0, prev_energy=0)
        r = _reward(state)
        expected_resource = 5 * CREDIT_GAIN_MULTIPLIER + 3 * ENERGY_GAIN_MULTIPLIER
        assert r == pytest.approx(STEP_PENALTY + expected_resource, abs=1e-5)

    def test_lose_resources_negative_reward(self):
        """Spending resources (e.g. using a program) gives negative resource gain."""
        state = _make_state(credits=0, energy=0, prev_credits=5, prev_energy=3)
        r = _reward(state)
        expected_resource = -5 * CREDIT_GAIN_MULTIPLIER + -3 * ENERGY_GAIN_MULTIPLIER
        assert r == pytest.approx(STEP_PENALTY + expected_resource, abs=1e-5)


# ---------------------------------------------------------------------------
# Resource holding (NEW in Phase 1)
# ---------------------------------------------------------------------------

class TestResourceHolding:
    """(credits * 0.01 + energy * 0.01) only on stage completion."""

    def test_holding_on_stage_complete(self):
        # Stage 2 after advance (completed stage 1)
        state = _make_state(credits=10, energy=5, stage=2)
        r = _reward(state, stage_completed=True)
        stage_reward = float(STAGE_COMPLETION_REWARDS[0])  # stage 1 = 1.0
        holding = 10 * CREDIT_HOLDING_MULTIPLIER + 5 * ENERGY_HOLDING_MULTIPLIER
        assert r == pytest.approx(STEP_PENALTY + stage_reward + holding, abs=1e-5)

    def test_no_holding_without_stage_complete(self):
        """Resource holding should NOT apply during normal steps."""
        state = _make_state(credits=100, energy=50)
        r = _reward(state)
        # Only step penalty — no holding bonus
        assert r == pytest.approx(STEP_PENALTY, abs=1e-5)


# ---------------------------------------------------------------------------
# Program waste penalty (NEW in Phase 1)
# ---------------------------------------------------------------------------

class TestProgramWaste:
    """RESET at 2 HP → -0.3 penalty (wastes 1 HP since max is 3)."""

    def test_reset_at_2hp_penalty(self):
        """Using RESET (action 19) at 2 HP should incur waste penalty."""
        state = _make_state(hp=3, prev_hp=2)
        r = _reward(state, action=int(ACTION_RESET))
        # HP gain = 1 → +1.0, waste penalty = -0.3
        assert r == pytest.approx(STEP_PENALTY + 1.0 + RESET_AT_2HP_PENALTY, abs=1e-5)

    def test_reset_at_1hp_no_penalty(self):
        """Using RESET at 1 HP is efficient — no waste penalty."""
        state = _make_state(hp=3, prev_hp=1)
        r = _reward(state, action=int(ACTION_RESET))
        # HP gain = 2 → +2.0, no waste penalty
        assert r == pytest.approx(STEP_PENALTY + 2.0, abs=1e-5)

    def test_non_reset_action_no_penalty(self):
        """Non-RESET actions should never trigger waste penalty regardless of HP."""
        state = _make_state(hp=3, prev_hp=2)
        r = _reward(state, action=ACTION_MOVE_UP)
        # HP gain = 1 → +1.0, no waste penalty (not RESET action)
        assert r == pytest.approx(STEP_PENALTY + 1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------

class TestCombinedRewards:
    """Multi-component reward scenarios to verify components sum correctly."""

    def test_siphon_with_resources_and_score(self):
        """Siphoning gives score gain + resource gain simultaneously."""
        state = _make_state(
            score=10, prev_score=0,
            credits=5, prev_credits=0,
            energy=3, prev_energy=0,
        )
        r = _reward(state, action=ACTION_SIPHON)
        score_gain = 10 * 0.5
        resource_gain = 5 * CREDIT_GAIN_MULTIPLIER + 3 * ENERGY_GAIN_MULTIPLIER
        assert r == pytest.approx(STEP_PENALTY + score_gain + resource_gain, abs=1e-5)

    def test_stage_complete_with_resources(self):
        """Stage completion includes stage reward + resource holding bonus."""
        state = _make_state(
            stage=3,  # After advance: completed stage 2
            credits=20, energy=10,
        )
        r = _reward(state, stage_completed=True)
        stage_reward = float(STAGE_COMPLETION_REWARDS[1])  # stage 2 = 2.0
        holding = 20 * CREDIT_HOLDING_MULTIPLIER + 10 * ENERGY_HOLDING_MULTIPLIER
        assert r == pytest.approx(STEP_PENALTY + stage_reward + holding, abs=1e-5)


# ---------------------------------------------------------------------------
# Siphon-caused death penalty (NEW in Phase 2)
# ---------------------------------------------------------------------------


def _make_enemy(row, col, spawned_from_siphon=False):
    """Build a single enemy row: [type, row, col, hp, disabled_turns, is_stunned,
    spawned_from_siphon, is_from_scheduled_task].

    Returns int32 array of shape (8,).
    """
    return jnp.array(
        [0, row, col, 1, 0, 0, int(spawned_from_siphon), 0], dtype=jnp.int32,
    )


def _enemies_and_mask(enemy_specs):
    """Build enemies array and mask from a list of (row, col, spawned_from_siphon) tuples.

    Fills remaining slots with zeros and False masks.
    """
    enemies = jnp.zeros((MAX_ENEMIES, 8), dtype=jnp.int32)
    mask = jnp.zeros(MAX_ENEMIES, dtype=jnp.bool_)
    for i, (row, col, siphon) in enumerate(enemy_specs):
        enemies = enemies.at[i].set(_make_enemy(row, col, spawned_from_siphon=siphon))
        mask = mask.at[i].set(True)
    return enemies, mask


class TestSiphonCausedDeathPenalty:
    """Extra -10.0 penalty when player dies to a siphon-spawned, cardinally adjacent enemy.

    Matches Swift: GameState.isAdjacentToPlayer (cardinal only) AND enemy.spawnedFromSiphon.
    This penalty is *in addition to* the regular stage-based death penalty, heavily
    penalizing reckless siphoning that spawns enemies which then kill the player.
    """

    def test_siphon_death_adjacent_enemy(self):
        """Die with a siphon-spawned enemy one cell above → full penalty."""
        # Player at (2, 3), siphon enemy at (3, 3) — cardinal adjacent
        enemies, mask = _enemies_and_mask([(3, 3, True)])
        state = _make_state(
            row=2, col=3, hp=0, prev_hp=1,
            stage=1,
            enemies=enemies, enemy_mask=mask,
        )
        r = _reward(state, player_died=True)
        # step + HP loss + death penalty (stage 1 = 0) + siphon death
        assert r == pytest.approx(
            STEP_PENALTY - 1.0 + SIPHON_CAUSED_DEATH_PENALTY, abs=1e-5,
        )

    def test_non_siphon_death_no_extra_penalty(self):
        """Die to a regular (non-siphon) adjacent enemy → only normal death penalty."""
        enemies, mask = _enemies_and_mask([(3, 3, False)])
        state = _make_state(
            row=2, col=3, hp=0, prev_hp=1,
            stage=1,
            enemies=enemies, enemy_mask=mask,
        )
        r = _reward(state, player_died=True)
        # No siphon penalty — just step + HP loss
        assert r == pytest.approx(STEP_PENALTY - 1.0, abs=1e-5)

    def test_siphon_enemy_not_adjacent(self):
        """Siphon enemy exists but 2+ cells away → no siphon death penalty."""
        # Player at (2, 3), siphon enemy at (4, 3) — distance 2, NOT adjacent
        enemies, mask = _enemies_and_mask([(4, 3, True)])
        state = _make_state(
            row=2, col=3, hp=0, prev_hp=1,
            stage=1,
            enemies=enemies, enemy_mask=mask,
        )
        r = _reward(state, player_died=True)
        assert r == pytest.approx(STEP_PENALTY - 1.0, abs=1e-5)

    def test_siphon_enemy_diagonal_not_adjacent(self):
        """Siphon enemy diagonally adjacent → NOT cardinal, no siphon death penalty.

        Cardinal adjacency requires exactly one axis at distance 1 and the other at 0.
        Diagonal means both axes differ, so isAdjacentToPlayer returns false.
        """
        # Player at (2, 3), siphon enemy at (3, 4) — diagonal
        enemies, mask = _enemies_and_mask([(3, 4, True)])
        state = _make_state(
            row=2, col=3, hp=0, prev_hp=1,
            stage=1,
            enemies=enemies, enemy_mask=mask,
        )
        r = _reward(state, player_died=True)
        assert r == pytest.approx(STEP_PENALTY - 1.0, abs=1e-5)

    def test_siphon_death_with_stage_penalty(self):
        """Siphon death at stage 4 compounds with regular stage death penalty.

        Both penalties stack: -3.5 (stage) + -10.0 (siphon) = -13.5 total penalties.
        """
        enemies, mask = _enemies_and_mask([(2, 3, True)])
        state = _make_state(
            row=2, col=2, hp=0, prev_hp=1,
            stage=4,
            enemies=enemies, enemy_mask=mask,
        )
        r = _reward(state, player_died=True)
        # step + HP loss + stage penalty (stages 1-3: 1+2+4=7 → -3.5) + siphon death
        assert r == pytest.approx(
            STEP_PENALTY - 1.0 - 3.5 + SIPHON_CAUSED_DEATH_PENALTY, abs=1e-5,
        )

    def test_alive_with_adjacent_siphon_enemy_no_penalty(self):
        """Player alive with adjacent siphon enemy → no siphon death penalty.

        The penalty only applies when player_died is True.
        """
        enemies, mask = _enemies_and_mask([(3, 3, True)])
        state = _make_state(
            row=2, col=3, hp=2, prev_hp=3,
            enemies=enemies, enemy_mask=mask,
        )
        r = _reward(state, player_died=False)
        # Just step + HP loss, no death penalties at all
        assert r == pytest.approx(STEP_PENALTY - 1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# BFS pathfinding unit tests (NEW in Phase 3)
# ---------------------------------------------------------------------------


class TestBFSDistance:
    """Direct tests for bfs_distance() matching Swift Pathfinding.findDistance().

    These tests verify the BFS implementation independently of the reward function,
    ensuring the pathfinding algorithm correctly handles obstacles, edge cases,
    and matches Swift's exact behavior.
    """

    def test_same_position_zero(self):
        """Start == target → distance 0."""
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        d = int(bfs_distance(jnp.int32(3), jnp.int32(3), jnp.int32(3), jnp.int32(3), grid))
        assert d == 0

    def test_adjacent_distance_one(self):
        """Cardinal neighbor → distance 1."""
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        d = int(bfs_distance(jnp.int32(2), jnp.int32(3), jnp.int32(3), jnp.int32(3), grid))
        assert d == 1

    def test_empty_grid_manhattan(self):
        """On empty grid, BFS == Manhattan distance."""
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        d = int(bfs_distance(jnp.int32(0), jnp.int32(0), jnp.int32(5), jnp.int32(5), grid))
        assert d == 10  # |5-0| + |5-0|

    def test_block_wall_longer_path(self):
        """Wall of blocks forces BFS path to be longer than Manhattan.

        Grid (6x6), exit at (5,5):
          Row 3 blocked at cols 0-4, col 5 open.
          BFS from (0,0) must go around: right to col 5, then up through (3,5).
          Manhattan(0,0→5,5) = 10
          BFS(0,0→5,5) = 10 (go right 5 to (0,5), up 5 to (5,5)) — same!

        Better test: block row 3 fully except col 0.
          BFS(2,3→4,3): Manhattan = 2. BFS must go left to col 0, up through (3,0), right.
        """
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        # Block row 3, cols 1-5 (leave col 0 open)
        grid = grid.at[3, 1].set(1)
        grid = grid.at[3, 2].set(1)
        grid = grid.at[3, 3].set(1)
        grid = grid.at[3, 4].set(1)
        grid = grid.at[3, 5].set(1)

        # BFS from (2,3) to (4,3): must detour via (3,0)
        # Path: (2,3)→(2,2)→(2,1)→(2,0)→(3,0)→(4,0)→(4,1)→(4,2)→(4,3) = 8
        d = int(bfs_distance(jnp.int32(2), jnp.int32(3), jnp.int32(4), jnp.int32(3), grid))
        assert d == 8  # Manhattan would be 2

    def test_no_path_returns_negative_one(self):
        """Completely blocked target → returns -1."""
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        # Block entire row 1, cutting off row 0 from rows 2-5
        for c in range(GRID_SIZE):
            grid = grid.at[1, c].set(1)

        d = int(bfs_distance(jnp.int32(0), jnp.int32(0), jnp.int32(5), jnp.int32(5), grid))
        assert d == -1

    def test_target_on_block_reachable(self):
        """Target cell with a block is still reachable (Swift checks target before block).

        Swift findDistance: `if newRow == target` is checked before `if cell.hasBlock`,
        so stepping onto the target is always allowed regardless of blocks.
        """
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        grid = grid.at[5, 5].set(1)  # Block on target

        d = int(bfs_distance(jnp.int32(5), jnp.int32(4), jnp.int32(5), jnp.int32(5), grid))
        assert d == 1  # Adjacent, reachable despite block

    def test_corner_to_corner(self):
        """(0,0) to (5,5) on empty grid = 10 steps."""
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        d = int(bfs_distance(jnp.int32(0), jnp.int32(0), jnp.int32(5), jnp.int32(5), grid))
        assert d == 10

    def test_surrounded_start_no_path(self):
        """Start position surrounded by blocks on all sides → no path to distant target."""
        grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        # Surround (2,2) with blocks
        grid = grid.at[1, 2].set(1)  # below
        grid = grid.at[3, 2].set(1)  # above
        grid = grid.at[2, 1].set(1)  # left
        grid = grid.at[2, 3].set(1)  # right

        d = int(bfs_distance(jnp.int32(2), jnp.int32(2), jnp.int32(5), jnp.int32(5), grid))
        assert d == -1
