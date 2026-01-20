#!/usr/bin/env python3
"""
Simple smoke test runner that doesn't require pytest.

This script runs basic validation tests to ensure the infrastructure works.
It can be replaced with pytest once the dependency is installed.

Usage:
    cd python && source venv/bin/activate && python tests/run_smoke_tests.py
"""

import sys
import traceback
import numpy as np

# Add parent directory to path
sys.path.insert(0, '.')

from tests.env_interface import (
    GameState,
    PlayerState,
    Enemy,
    Observation,
    PROGRAM_PUSH,
    PROGRAM_WAIT,
)
from tests.swift_env_wrapper import SwiftEnvWrapper


def test_swift_reset():
    """Test that Swift env reset() returns valid observation."""
    print("Testing Swift reset()...", end=" ")

    with SwiftEnvWrapper() as env:
        obs = env.reset()

        assert isinstance(obs, Observation), "reset() should return Observation"
        assert obs.player.shape == (10,), f"player shape: {obs.player.shape}"
        assert obs.programs.shape == (23,), f"programs shape: {obs.programs.shape}"
        assert obs.grid.shape == (6, 6, 40), f"grid shape: {obs.grid.shape}"
        assert obs.player.dtype == np.float32, f"player dtype: {obs.player.dtype}"
        assert obs.programs.dtype == np.int32, f"programs dtype: {obs.programs.dtype}"
        assert obs.grid.dtype == np.float32, f"grid dtype: {obs.grid.dtype}"

    print("PASSED")


def test_swift_step():
    """Test that Swift env step() works."""
    print("Testing Swift step()...", end=" ")

    with SwiftEnvWrapper() as env:
        env.reset()
        valid_actions = env.get_valid_actions()

        assert len(valid_actions) > 0, "Should have valid actions"

        result = env.step(valid_actions[0])

        assert isinstance(result.observation, Observation), "step() should return Observation"
        assert isinstance(result.reward, float), "reward should be float"
        assert isinstance(result.done, bool), "done should be bool"
        assert isinstance(result.info, dict), "info should be dict"

    print("PASSED")


def test_swift_valid_actions():
    """Test that get_valid_actions() returns valid data."""
    print("Testing Swift get_valid_actions()...", end=" ")

    with SwiftEnvWrapper() as env:
        env.reset()
        valid_actions = env.get_valid_actions()

        assert isinstance(valid_actions, list), "Should return list"
        assert all(isinstance(a, int) for a in valid_actions), "All actions should be int"
        assert all(0 <= a <= 27 for a in valid_actions), "Actions in [0, 27]"
        assert len(valid_actions) == len(set(valid_actions)), "No duplicates"

    print("PASSED")


def test_swift_set_state_basic():
    """Test that set_state() works with basic state."""
    print("Testing Swift set_state() basic...", end=" ")

    with SwiftEnvWrapper() as env:
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0)
        )
        obs = env.set_state(state)

        assert isinstance(obs, Observation), "set_state() should return Observation"

        # Check player position (denormalize)
        row = int(round(obs.player[0] * 5))
        col = int(round(obs.player[1] * 5))
        assert row == 3, f"Expected row=3, got {row}"
        assert col == 3, f"Expected col=3, got {col}"

    print("PASSED")


def test_swift_set_state_with_enemy():
    """Test that set_state() correctly places enemies."""
    print("Testing Swift set_state() with enemy...", end=" ")

    with SwiftEnvWrapper() as env:
        state = GameState(
            player=PlayerState(row=0, col=0),
            enemies=[
                Enemy(type="virus", row=5, col=5, hp=2)
            ]
        )
        obs = env.set_state(state)

        # Check enemy at (5,5) - virus type is channel 0
        assert obs.grid[5, 5, 0] == 1.0, f"Virus should be at (5,5), got {obs.grid[5, 5, 0]}"

    print("PASSED")


def test_swift_set_state_with_programs():
    """Test that set_state() correctly sets owned programs."""
    print("Testing Swift set_state() with programs...", end=" ")

    with SwiftEnvWrapper() as env:
        state = GameState(
            player=PlayerState(row=3, col=3),
            owned_programs=[PROGRAM_PUSH, PROGRAM_WAIT]  # indices 5 and 10
        )
        obs = env.set_state(state)

        # Programs array: index 0 = action 5 (PUSH), index 5 = action 10 (WAIT)
        assert obs.programs[0] == 1, f"PUSH should be owned, got {obs.programs[0]}"
        assert obs.programs[5] == 1, f"WAIT should be owned, got {obs.programs[5]}"

    print("PASSED")


def test_swift_set_state_step_sequence():
    """Test that we can set_state() then step()."""
    print("Testing Swift set_state() then step()...", end=" ")

    with SwiftEnvWrapper() as env:
        # Set up state in middle of grid
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0)
        )
        env.set_state(state)

        # Get valid actions and step
        valid_actions = env.get_valid_actions()
        assert len(valid_actions) > 0, "Should have valid actions after set_state"

        result = env.step(valid_actions[0])
        assert isinstance(result.observation, Observation), "Step should work after set_state"

    print("PASSED")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running Smoke Tests (no pytest)")
    print("=" * 60)

    tests = [
        test_swift_reset,
        test_swift_step,
        test_swift_valid_actions,
        test_swift_set_state_basic,
        test_swift_set_state_with_enemy,
        test_swift_set_state_with_programs,
        test_swift_set_state_step_sequence,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
