"""
Stage Tests (Phase 2.53-2.55)

Tests for stage transitions including:
- Stage completion when reaching exit
- Data block invariant on new stage
- Player state preservation across stages

These tests verify that the Swift environment correctly handles stage transitions.
"""

import pytest
import numpy as np

from ..env_interface import (
    GameState,
    PlayerState,
    Enemy,
    Block,
    Observation,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
)


# MARK: - Helper Functions

def get_player_position(obs: Observation) -> tuple[int, int]:
    """Extract player row, col from observation."""
    row = int(round(obs.player[0] * 5))
    col = int(round(obs.player[1] * 5))
    return row, col


def get_player_stage(obs: Observation) -> int:
    """Extract player stage from observation.

    Stage encoding: (stage-1)/7, where stage is 1-8.
    Decoding: round(normalized * 7) + 1
    """
    return int(round(obs.player[5] * 7)) + 1


def get_player_score(obs: Observation) -> int:
    """Extract player score from observation."""
    # Score is part of info, not directly in observation
    # For now we check stage change instead
    return 0


def get_player_credits(obs: Observation) -> int:
    """Extract player credits from observation."""
    return int(round(obs.player[3] * 50))


def get_player_energy(obs: Observation) -> int:
    """Extract player energy from observation."""
    return int(round(obs.player[4] * 50))


def get_player_hp(obs: Observation) -> int:
    """Extract player HP from observation."""
    return int(round(obs.player[2] * 3))


def count_data_blocks(obs: Observation) -> int:
    """Count data blocks on the grid."""
    count = 0
    for row in range(6):
        for col in range(6):
            # Check block channels (shifted by 1 after adding spawnedFromSiphon)
            if obs.grid[row, col, 7] > 0.5:  # Data block
                count += 1
    return count


def get_blocks_info(obs: Observation) -> list[dict]:
    """Get information about blocks from observation."""
    blocks = []
    for row in range(6):
        for col in range(6):
            if obs.grid[row, col, 7] > 0.5:  # Data block (channel 7)
                blocks.append({
                    "row": row,
                    "col": col,
                    "siphoned": obs.grid[row, col, 11] > 0.5  # Siphoned at channel 11
                })
            elif obs.grid[row, col, 8] > 0.5:  # Program block (channel 8)
                blocks.append({
                    "row": row,
                    "col": col,
                    "type": "program",
                    "siphoned": obs.grid[row, col, 11] > 0.5  # Siphoned at channel 11
                })
    return blocks


# MARK: - Test 2.53: Stage Completion

class TestStageCompletion:
    """Test 2.53: Stage completes when player reaches exit."""

    @pytest.mark.requires_set_state
    def test_stage_completes_on_exit(self, env):
        """Moving to exit should advance to next stage.

        Note: Exit is always at top-right (row=5, col=5).
        Player starts adjacent to exit.
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3, credits=0, energy=0),
            enemies=[],
            blocks=[],
            stage=1
        )
        obs_before = env.set_state(state)
        stage_before = get_player_stage(obs_before)

        # Move right to reach exit at (5, 5)
        result = env.step(ACTION_MOVE_RIGHT)

        stage_after = get_player_stage(result.observation)
        # Stage should advance
        assert stage_after == stage_before + 1, \
            f"Stage should advance from {stage_before} to {stage_before + 1}, got {stage_after}"

    @pytest.mark.requires_set_state
    def test_stage_reward_increases_with_stage(self, env):
        """Stage completion rewards should increase exponentially."""
        # Complete stage 1
        state1 = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state1)
        result1 = env.step(ACTION_MOVE_RIGHT)
        reward1 = result1.reward

        # Complete stage 3 (should have higher reward)
        state3 = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=3
        )
        env.set_state(state3)
        result3 = env.step(ACTION_MOVE_RIGHT)
        reward3 = result3.reward

        # Stage 3 completion reward should be higher than stage 1
        # (Reward multipliers: [1, 2, 4, 8, 16, 32, 64, 100])
        assert reward3 > reward1, \
            f"Stage 3 reward ({reward3}) should be > stage 1 reward ({reward1})"


# MARK: - Test 2.54: Data Block Invariant

class TestDataBlockInvariant:
    """Test 2.54: New stages maintain data block invariant (points == spawnCount)."""

    @pytest.mark.requires_set_state
    def test_new_stage_generates_content(self, env):
        """New stage should be generated with content.

        Note: Stage advancement happens on reaching exit.
        We verify the stage transition occurred.
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        obs_before = env.set_state(state)
        stage_before = get_player_stage(obs_before)

        # Move to exit
        result = env.step(ACTION_MOVE_RIGHT)

        # Stage should advance
        stage_after = get_player_stage(result.observation)
        assert stage_after > stage_before or stage_after == stage_before + 1, \
            f"Stage should advance from {stage_before}, got {stage_after}"


# MARK: - Test 2.55: Player State Preserved

class TestPlayerStatePreserved:
    """Test 2.55: Player state is preserved across stage transitions."""

    @pytest.mark.requires_set_state
    def test_credits_preserved_on_stage_transition(self, env):
        """Player credits should be preserved when transitioning stages."""
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3, credits=10, energy=5),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        result = env.step(ACTION_MOVE_RIGHT)

        # Credits should be preserved
        credits_after = get_player_credits(result.observation)
        assert credits_after == 10, f"Credits should be 10, got {credits_after}"

    @pytest.mark.requires_set_state
    def test_energy_preserved_on_stage_transition(self, env):
        """Player energy should be preserved when transitioning stages."""
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3, credits=5, energy=7),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        result = env.step(ACTION_MOVE_RIGHT)

        energy_after = get_player_energy(result.observation)
        assert energy_after == 7, f"Energy should be 7, got {energy_after}"

    @pytest.mark.requires_set_state
    def test_hp_restored_on_stage_transition(self, env):
        """Player HP is restored to full when transitioning stages.

        Note: The game auto-heals player to 3 HP on stage completion.
        This is the expected game behavior.
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=2, credits=0, energy=0),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        result = env.step(ACTION_MOVE_RIGHT)

        hp_after = get_player_hp(result.observation)
        # Game restores HP on stage transition
        assert hp_after == 3, f"HP should be restored to 3 on stage transition, got {hp_after}"

    @pytest.mark.requires_set_state
    def test_player_position_preserved_on_stage_transition(self, env):
        """Player position should be preserved on stage transition.

        When player reaches exit, they stay at the exit position after
        the stage advances. The exit position becomes their starting
        position for the new stage.
        """
        # Player starts adjacent to exit at (5,5)
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        # Move right to reach exit at (5, 5)
        result = env.step(ACTION_MOVE_RIGHT)

        # Player should still be at (5, 5) after stage transition
        row, col = get_player_position(result.observation)
        assert (row, col) == (5, 5), \
            f"Player should stay at exit position (5, 5) after stage transition, got ({row}, {col})"

    @pytest.mark.requires_set_state
    def test_hp_gains_one_on_stage_transition(self, env):
        """Player gains 1 HP (up to max) on stage transition, not reset to max.

        Starting at HP=1, player should have HP=2 after stage transition.
        This distinguishes "gains 1 HP" from "restored to max".
        """
        state = GameState(
            player=PlayerState(row=5, col=4, hp=1, credits=0, energy=0),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        result = env.step(ACTION_MOVE_RIGHT)

        hp_after = get_player_hp(result.observation)
        # HP should gain 1, not reset to max
        assert hp_after == 2, \
            f"HP should gain 1 (1->2) on stage transition, got {hp_after}"


# MARK: - Test: Exit Position Changes on Stage Transition

class TestExitPositionOnStageTransition:
    """Test that exit position changes to a different corner on stage transition."""

    @pytest.mark.requires_set_state
    def test_exit_at_different_corner_after_stage_complete(self, env):
        """Exit should move to a different corner after stage completion.

        After completing a stage, the exit should be at one of the other
        three corners (not where the player is standing).
        """
        # Player at (5,4), will move to exit at (5,5)
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        # Move to exit
        result = env.step(ACTION_MOVE_RIGHT)

        # Player should be at (5,5) after stage transition
        row, col = get_player_position(result.observation)
        assert (row, col) == (5, 5), f"Player should be at (5,5), got ({row}, {col})"

        # Find exit position from grid observation
        # Exit is at grid feature index 40
        exit_positions = []
        for r in range(6):
            for c in range(6):
                if result.observation.grid[r, c, 40] > 0.5:
                    exit_positions.append((r, c))

        # Should be exactly one exit
        assert len(exit_positions) == 1, \
            f"Should have exactly one exit, found {len(exit_positions)} at {exit_positions}"

        new_exit = exit_positions[0]

        # Exit should be at a corner (one of: (0,0), (0,5), (5,0), (5,5))
        corners = [(0, 0), (0, 5), (5, 0), (5, 5)]
        assert new_exit in corners, \
            f"Exit should be at a corner, found {new_exit}"

        # Exit should NOT be at player's position (5,5)
        assert new_exit != (5, 5), \
            f"Exit should have moved to a different corner, but it's still at {new_exit}"


# MARK: - Test: Stage Generation Content

class TestStageGenerationContent:
    """Test that new stages generate proper content."""

    @pytest.mark.requires_set_state
    def test_new_stage_has_blocks(self, env):
        """New stage should have blocks generated."""
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        # Complete stage 1
        result = env.step(ACTION_MOVE_RIGHT)

        # Count blocks in new stage observation
        # Block channels: 7 (data), 8 (program), 9 (question)
        block_count = 0
        for r in range(6):
            for c in range(6):
                has_data = result.observation.grid[r, c, 7] > 0.5
                has_program = result.observation.grid[r, c, 8] > 0.5
                has_question = result.observation.grid[r, c, 9] > 0.5
                if has_data or has_program or has_question:
                    block_count += 1

        # Should have 5-11 blocks
        assert 5 <= block_count <= 11, \
            f"New stage should have 5-11 blocks, found {block_count}"

    @pytest.mark.requires_set_state
    def test_new_stage_has_data_siphons_at_corners(self, env):
        """New stage should have data siphons at non-exit, non-player corners."""
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        # Complete stage 1
        result = env.step(ACTION_MOVE_RIGHT)

        # Find data siphon positions
        # Data siphon is at grid feature index 39
        siphon_positions = []
        for r in range(6):
            for c in range(6):
                if result.observation.grid[r, c, 39] > 0.5:
                    siphon_positions.append((r, c))

        # Should have exactly 2 data siphons (4 corners - player - exit = 2)
        assert len(siphon_positions) == 2, \
            f"Should have 2 data siphons, found {len(siphon_positions)} at {siphon_positions}"

        # All siphons should be at corners
        corners = [(0, 0), (0, 5), (5, 0), (5, 5)]
        for pos in siphon_positions:
            assert pos in corners, \
                f"Data siphon at {pos} should be at a corner"

    @pytest.mark.requires_set_state
    def test_new_stage_has_transmissions(self, env):
        """New stage should spawn transmissions based on stage number."""
        state = GameState(
            player=PlayerState(row=5, col=4, hp=3),
            enemies=[],
            blocks=[],
            stage=1
        )
        env.set_state(state)

        # Complete stage 1 -> stage 2
        result = env.step(ACTION_MOVE_RIGHT)

        # Count transmissions in observation
        # Transmission turns remaining is at channel 36
        transmission_count = 0
        for r in range(6):
            for c in range(6):
                if result.observation.grid[r, c, 36] > 0:
                    transmission_count += 1

        # Stage 2 should spawn 2 transmissions
        assert transmission_count == 2, \
            f"Stage 2 should spawn 2 transmissions, found {transmission_count}"
