"""
Interface Smoke Tests

These tests verify that both Swift and JAX environments correctly implement
the EnvInterface protocol. They test basic functionality without requiring
set_state() (which JAX doesn't support yet).

Why these tests:
- Verify observation shapes and dtypes match the specification
- Ensure reset(), step(), get_valid_actions() work correctly
- Catch interface mismatches early before comprehensive tests
"""

import numpy as np
import pytest

from .env_interface import (
    Observation,
    StepResult,
    GRID_SIZE,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_SIPHON,
)


# MARK: - Observation Structure Tests

class TestObservationStructure:
    """Verify observation shape and dtype requirements."""

    def test_reset_returns_observation(self, env_smoke):
        """reset() should return an Observation with correct structure."""
        obs = env_smoke.reset()

        assert isinstance(obs, Observation)
        assert obs.player is not None
        assert obs.programs is not None
        assert obs.grid is not None

    def test_player_shape(self, env_smoke):
        """Player state should have shape (10,) and dtype float32."""
        obs = env_smoke.reset()

        assert obs.player.shape == (10,), f"Expected (10,), got {obs.player.shape}"
        assert obs.player.dtype == np.float32, f"Expected float32, got {obs.player.dtype}"

    def test_programs_shape(self, env_smoke):
        """Programs should have shape (23,) and dtype int32."""
        obs = env_smoke.reset()

        assert obs.programs.shape == (23,), f"Expected (23,), got {obs.programs.shape}"
        assert obs.programs.dtype == np.int32, f"Expected int32, got {obs.programs.dtype}"

    def test_grid_shape(self, env_smoke):
        """Grid should have shape (6, 6, 40) and dtype float32."""
        obs = env_smoke.reset()

        expected_shape = (GRID_SIZE, GRID_SIZE, 40)
        assert obs.grid.shape == expected_shape, f"Expected {expected_shape}, got {obs.grid.shape}"
        assert obs.grid.dtype == np.float32, f"Expected float32, got {obs.grid.dtype}"

    def test_player_values_normalized(self, env_smoke):
        """Player state values should be in [0, 1] range."""
        obs = env_smoke.reset()

        assert np.all(obs.player >= 0.0), "Player values should be >= 0"
        assert np.all(obs.player <= 1.0), "Player values should be <= 1"

    def test_programs_binary(self, env_smoke):
        """Programs should be binary (0 or 1)."""
        obs = env_smoke.reset()

        assert np.all((obs.programs == 0) | (obs.programs == 1)), \
            "Programs should only contain 0 or 1"

    def test_grid_values_normalized(self, env_smoke):
        """Grid values should be in [0, 1] range."""
        obs = env_smoke.reset()

        assert np.all(obs.grid >= 0.0), "Grid values should be >= 0"
        assert np.all(obs.grid <= 1.0), "Grid values should be <= 1"


# MARK: - Step Function Tests

class TestStepFunction:
    """Verify step() function behavior."""

    def test_step_returns_step_result(self, env_smoke):
        """step() should return a StepResult."""
        env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()

        if valid_actions:
            result = env_smoke.step(valid_actions[0])
            assert isinstance(result, StepResult)
            assert isinstance(result.observation, Observation)
            assert isinstance(result.reward, float)
            assert isinstance(result.done, bool)
            assert isinstance(result.info, dict)

    def test_step_observation_structure(self, env_smoke):
        """step() observation should have correct structure."""
        env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()

        if valid_actions:
            result = env_smoke.step(valid_actions[0])
            obs = result.observation

            assert obs.player.shape == (10,)
            assert obs.programs.shape == (23,)
            assert obs.grid.shape == (6, 6, 40)

    def test_step_reward_is_float(self, env_smoke):
        """step() reward should be a finite float."""
        env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()

        if valid_actions:
            result = env_smoke.step(valid_actions[0])
            assert np.isfinite(result.reward), "Reward should be finite"


# MARK: - Valid Actions Tests

class TestValidActions:
    """Verify get_valid_actions() behavior."""

    def test_get_valid_actions_returns_list(self, env_smoke):
        """get_valid_actions() should return a list of integers."""
        env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()

        assert isinstance(valid_actions, list)
        assert all(isinstance(a, int) for a in valid_actions)

    def test_valid_actions_in_range(self, env_smoke):
        """All valid action indices should be in [0, 27]."""
        env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()

        assert all(0 <= a <= 27 for a in valid_actions), \
            f"Actions out of range: {[a for a in valid_actions if not 0 <= a <= 27]}"

    def test_valid_actions_unique(self, env_smoke):
        """Valid actions should not contain duplicates."""
        env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()

        assert len(valid_actions) == len(set(valid_actions)), \
            "Valid actions contain duplicates"

    def test_at_least_one_valid_action(self, env_smoke):
        """There should always be at least one valid action after reset."""
        env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()

        assert len(valid_actions) > 0, "No valid actions available after reset"


# MARK: - Reset Tests

class TestReset:
    """Verify reset() behavior."""

    def test_reset_multiple_times(self, env_smoke):
        """reset() should work multiple times without errors."""
        for _ in range(3):
            obs = env_smoke.reset()
            assert isinstance(obs, Observation)

    def test_reset_clears_state(self, env_smoke):
        """reset() should provide a fresh state."""
        # First episode
        obs1 = env_smoke.reset()
        valid_actions = env_smoke.get_valid_actions()
        if valid_actions:
            env_smoke.step(valid_actions[0])

        # Reset and check we get a valid initial state
        obs2 = env_smoke.reset()
        assert isinstance(obs2, Observation)
        assert obs2.player.shape == (10,)


# MARK: - Swift-Only set_state Tests

class TestSetState:
    """Verify set_state() behavior (Swift only)."""

    @pytest.mark.requires_set_state
    def test_set_state_returns_observation(self, swift_env):
        """set_state() should return an Observation."""
        from .env_interface import GameState, PlayerState

        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0)
        )
        obs = swift_env.set_state(state)

        assert isinstance(obs, Observation)
        assert obs.player.shape == (10,)

    @pytest.mark.requires_set_state
    def test_set_state_player_position(self, swift_env):
        """set_state() should correctly set player position."""
        from .env_interface import GameState, PlayerState

        state = GameState(
            player=PlayerState(row=2, col=4, hp=3)
        )
        obs = swift_env.set_state(state)

        # Denormalize to verify: row = normalized * 5, col = normalized * 5
        row = int(round(obs.player[0] * 5))
        col = int(round(obs.player[1] * 5))

        assert row == 2, f"Expected row=2, got {row}"
        assert col == 4, f"Expected col=4, got {col}"

    @pytest.mark.requires_set_state
    def test_set_state_player_resources(self, swift_env):
        """set_state() should correctly set player resources."""
        from .env_interface import GameState, PlayerState

        state = GameState(
            player=PlayerState(row=3, col=3, hp=2, credits=10, energy=5)
        )
        obs = swift_env.set_state(state)

        # Denormalize: hp = normalized * 3, credits = normalized * 50, energy = normalized * 50
        hp = int(round(obs.player[2] * 3))
        credits = int(round(obs.player[3] * 50))
        energy = int(round(obs.player[4] * 50))

        assert hp == 2, f"Expected hp=2, got {hp}"
        assert credits == 10, f"Expected credits=10, got {credits}"
        assert energy == 5, f"Expected energy=5, got {energy}"

    @pytest.mark.requires_set_state
    def test_set_state_with_enemies(self, swift_env):
        """set_state() should correctly set enemies."""
        from .env_interface import GameState, PlayerState, Enemy

        state = GameState(
            player=PlayerState(row=0, col=0),
            enemies=[
                Enemy(type="virus", row=5, col=5, hp=2)
            ]
        )
        obs = swift_env.set_state(state)

        # Check that enemy appears in grid at (5,5)
        # Enemy type one-hot is at channels 0-3 (virus=0)
        assert obs.grid[5, 5, 0] == 1.0, "Virus should be at (5,5)"

    @pytest.mark.requires_set_state
    def test_set_state_with_programs(self, swift_env):
        """set_state() should correctly set owned programs."""
        from .env_interface import GameState, PlayerState, PROGRAM_PUSH, PROGRAM_WAIT

        state = GameState(
            player=PlayerState(row=3, col=3),
            owned_programs=[PROGRAM_PUSH, PROGRAM_WAIT]  # indices 5 and 10
        )
        obs = swift_env.set_state(state)

        # Programs array: index 0 = action 5, index 5 = action 10
        assert obs.programs[0] == 1, "PUSH (action 5) should be owned"
        assert obs.programs[5] == 1, "WAIT (action 10) should be owned"

    @pytest.mark.requires_set_state
    def test_set_state_empty_state(self, swift_env):
        """set_state() should work with minimal state."""
        from .env_interface import GameState, PlayerState

        # Minimal state: just player position
        state = GameState(
            player=PlayerState(row=0, col=0)
        )
        obs = swift_env.set_state(state)

        assert isinstance(obs, Observation)
        # Player should be at (0, 0)
        assert obs.player[0] == 0.0  # row 0 normalized = 0
        assert obs.player[1] == 0.0  # col 0 normalized = 0
