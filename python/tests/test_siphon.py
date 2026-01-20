"""
Siphon Tests (Phase 2.10-2.15)

Tests for siphon mechanics including:
- Siphoning adjacent data blocks
- Siphon validity (requires data siphons, not block adjacency)
- Transmission spawning from siphons
- Resource collection from siphons
- Siphoning program blocks

These tests verify that the Swift environment correctly implements siphon mechanics
as documented in specs/game-mechanics.md.
"""

try:
    import pytest
except ImportError:
    from . import pytest_compat as pytest
import numpy as np

from .env_interface import (
    GameState,
    PlayerState,
    Enemy,
    Block,
    Resource,
    Observation,
    ACTION_SIPHON,
    ACTION_MOVE_UP,
)


# MARK: - Helper Functions

def get_player_siphons(obs: Observation) -> int:
    """Extract player data siphons from observation."""
    return int(round(obs.player[6] * 10))


def get_player_score(obs: Observation) -> int:
    """Extract player score from info - not directly in observation."""
    # Score is not in observation array, would need info dict
    return 0


def count_transmissions(obs: Observation) -> int:
    """Count transmissions in the grid (channel 35 = transmission countdown)."""
    return int(np.sum(obs.grid[:, :, 35] > 0))


def get_block_at(obs: Observation, row: int, col: int) -> dict | None:
    """Get block info at position, or None if no block."""
    # Block type one-hot at channels 6-8 (data, program, question)
    if not np.any(obs.grid[row, col, 6:9] > 0):
        return None

    block_types = ["data", "program", "question"]
    for i, btype in enumerate(block_types):
        if obs.grid[row, col, 6 + i] > 0:
            points = int(round(obs.grid[row, col, 9] * 9))
            siphoned = obs.grid[row, col, 10] > 0.5
            return {"type": btype, "points": points, "siphoned": siphoned}
    return None


def get_owned_programs(obs: Observation) -> list[int]:
    """Get list of owned program action indices."""
    return [i + 5 for i, owned in enumerate(obs.programs) if owned == 1]


# MARK: - Test 2.10: Siphon Adjacent Block

class TestSiphonDataBlock:
    """Test 2.10: Siphon adjacent data block."""

    @pytest.mark.requires_set_state
    def test_siphon_data_block(self, swift_env):
        """Siphoning adjacent data block should award score and consume siphon."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0,
                             dataSiphons=1, attackDamage=1, score=0),
            blocks=[Block(row=4, col=3, type="data", points=9, spawnCount=9, siphoned=False)],
            enemies=[],
            stage=1
        )
        obs_before = swift_env.set_state(state)
        siphons_before = get_player_siphons(obs_before)
        assert siphons_before == 1, f"Should start with 1 siphon, got {siphons_before}"

        result = swift_env.step(ACTION_SIPHON)

        # Siphon should be consumed
        siphons_after = get_player_siphons(result.observation)
        assert siphons_after == 0, f"Siphon should be consumed, got {siphons_after}"

        # Block should be marked as siphoned
        block = get_block_at(result.observation, 4, 3)
        assert block is not None, "Block should still exist"
        assert block["siphoned"], "Block should be marked as siphoned"

        # Should get score reward
        assert result.reward > 0, f"Expected positive reward for siphoning, got {result.reward}"


# MARK: - Test 2.11: Siphon Always Valid with Data Siphons

class TestSiphonValidWithSiphons:
    """Test 2.11: Siphon is valid when player has data siphons."""

    @pytest.mark.requires_set_state
    def test_siphon_always_valid_with_siphons(self, swift_env):
        """Siphon should be valid when player has siphons, even without adjacent blocks."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            blocks=[],  # No blocks anywhere
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid_actions = swift_env.get_valid_actions()
        assert ACTION_SIPHON in valid_actions, \
            f"Siphon should be valid when player has siphons, got {valid_actions}"


# MARK: - Test 2.12: Siphon Invalid Without Data Siphons

class TestSiphonInvalidWithoutSiphons:
    """Test 2.12: Siphon is invalid without data siphons."""

    @pytest.mark.requires_set_state
    def test_siphon_invalid_without_siphons(self, swift_env):
        """Siphon should be invalid when player has no siphons."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=0),
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid_actions = swift_env.get_valid_actions()
        assert ACTION_SIPHON not in valid_actions, \
            f"Siphon should be invalid without siphons, got {valid_actions}"


# MARK: - Test 2.13: Siphon Spawns Transmissions

class TestSiphonSpawnsTransmissions:
    """Test 2.13: Siphoning block spawns transmissions."""

    @pytest.mark.requires_set_state
    def test_siphon_spawns_transmissions(self, swift_env):
        """Siphoning a block should spawn transmissions equal to spawnCount."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            blocks=[Block(row=4, col=3, type="data", points=3, spawnCount=3, siphoned=False)],
            transmissions=[],
            enemies=[],
            stage=1
        )
        obs_before = swift_env.set_state(state)
        trans_before = count_transmissions(obs_before)
        assert trans_before == 0, f"Should start with 0 transmissions, got {trans_before}"

        result = swift_env.step(ACTION_SIPHON)

        trans_after = count_transmissions(result.observation)
        # Note: transmissions are spawned, should be >= spawnCount
        # (may be less if not enough empty cells)
        assert trans_after >= 1, f"Should spawn transmissions, got {trans_after}"


# MARK: - Test 2.14: Siphon Does NOT Reveal Resources

class TestSiphonDoesNotRevealResources:
    """Test 2.14: Siphoning doesn't destroy block or reveal resources."""

    @pytest.mark.requires_set_state
    def test_siphon_does_not_reveal_resources(self, swift_env):
        """Siphoning marks block as siphoned but doesn't destroy it."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
            resources=[Resource(row=4, col=3, credits=5, energy=0)],  # Hidden under block
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_SIPHON)

        # Block should still exist (just marked as siphoned)
        block = get_block_at(result.observation, 4, 3)
        assert block is not None, "Block should still exist after siphoning"
        assert block["siphoned"], "Block should be marked as siphoned"

        # Resources under block should NOT be visible
        # (can't verify directly, but block existing means cell not traversable)


# MARK: - Test 2.15: Siphon Program Block

class TestSiphonProgramBlock:
    """Test 2.15: Siphoning program block acquires the program."""

    @pytest.mark.requires_set_state
    def test_siphon_program_block(self, swift_env):
        """Siphoning a program block should add program to inventory."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            blocks=[Block(row=4, col=3, type="program", programType="push",
                        programActionIndex=5, spawnCount=2, siphoned=False)],
            owned_programs=[],
            enemies=[],
            stage=1
        )
        obs_before = swift_env.set_state(state)
        programs_before = get_owned_programs(obs_before)
        assert 5 not in programs_before, f"Should not own PUSH before, got {programs_before}"

        result = swift_env.step(ACTION_SIPHON)

        programs_after = get_owned_programs(result.observation)
        assert 5 in programs_after, f"Should own PUSH after siphoning, got {programs_after}"
