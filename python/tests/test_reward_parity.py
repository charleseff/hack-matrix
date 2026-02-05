"""
JAX-only reward parity tests — Phase 1.

Tests call `calculate_reward` directly with constructed EnvState objects,
verifying each reward component matches Swift RewardCalculator.swift behavior.

Why direct testing:
- The parity/test_rewards.py tests exercise the full step() pipeline through
  the env interface (set_state → step → check reward), which couples rewards
  to movement, enemy AI, and stage logic.
- These tests isolate the reward function itself, making failures precise:
  a failing test here means the reward formula is wrong, not some upstream bug.
"""

import jax
import jax.numpy as jnp
import pytest

from hackmatrix.jax_env.rewards import (
    ACTION_RESET,
    CREDIT_GAIN_MULTIPLIER,
    CREDIT_HOLDING_MULTIPLIER,
    DISTANCE_SHAPING_COEF,
    ENERGY_GAIN_MULTIPLIER,
    ENERGY_HOLDING_MULTIPLIER,
    RESET_AT_2HP_PENALTY,
    STEP_PENALTY,
    calculate_reward,
)
from hackmatrix.jax_env.state import (
    ACTION_MOVE_UP,
    ACTION_SIPHON,
    STAGE_COMPLETION_REWARDS,
    Player,
    create_empty_state,
)


def _make_state(**overrides):
    """Build an EnvState with sensible defaults, applying overrides.

    Defaults: player at (0,0), hp=3, exit at (5,5), stage=1, no enemies.
    The 'previous' snapshot mirrors 'current' so deltas start at zero.
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

    if overrides:
        raise ValueError(f"Unknown overrides: {overrides}")

    return state.replace(
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
    )


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
    """One-directional: +0.05 per cell closer to exit, no penalty for moving away."""

    def test_move_one_cell_closer(self):
        # Previous: (0,0), distance = 10. Current: (1,0), distance = 9. Delta = +1.
        state = _make_state(row=1, col=0, prev_player_row=0, prev_player_col=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY + DISTANCE_SHAPING_COEF, abs=1e-5)

    def test_move_away_no_penalty(self):
        # Previous: (1,0), distance = 9. Current: (0,0), distance = 10. Delta = -1 → clamped to 0.
        state = _make_state(row=0, col=0, prev_player_row=1, prev_player_col=0)
        r = _reward(state)
        assert r == pytest.approx(STEP_PENALTY, abs=1e-5)


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
