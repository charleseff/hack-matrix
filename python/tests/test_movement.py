"""
Movement Tests (Phase 2.1-2.9)

Tests for player movement mechanics including:
- Basic directional movement
- Edge/wall blocking
- Block collision
- Data siphon collection by walking
- Line-of-sight attacks
- Enemy attacks (adjacent and distant)
- Transmission destruction

These tests verify that the Swift environment correctly implements movement mechanics
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
    Transmission,
    Resource,
    Observation,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    GRID_SIZE,
)


# MARK: - Helper Functions

def get_player_position(obs: Observation) -> tuple[int, int]:
    """Extract player row, col from observation."""
    row = int(round(obs.player[0] * 5))
    col = int(round(obs.player[1] * 5))
    return row, col


def get_player_hp(obs: Observation) -> int:
    """Extract player HP from observation."""
    return int(round(obs.player[2] * 3))


def get_turn(obs: Observation) -> int:
    """Extract turn from info dict or estimate from observation."""
    # Turn is not directly in observation - we track it via step count
    return 0  # This would need to come from info dict


def count_enemies(obs: Observation) -> int:
    """Count enemies in the grid observation."""
    # Enemy type one-hot is at channels 0-3
    return int(np.sum(np.any(obs.grid[:, :, 0:4] > 0, axis=2)))


def get_enemy_at(obs: Observation, row: int, col: int) -> dict | None:
    """Get enemy info at a specific position, or None if no enemy."""
    if not np.any(obs.grid[row, col, 0:4] > 0):
        return None

    enemy_types = ["virus", "daemon", "glitch", "cryptog"]
    for i, etype in enumerate(enemy_types):
        if obs.grid[row, col, i] > 0:
            hp = int(round(obs.grid[row, col, 4] * 3))
            stunned = obs.grid[row, col, 5] > 0.5
            return {"type": etype, "hp": hp, "stunned": stunned}
    return None


# MARK: - Test 2.1: Move to Empty Cell

class TestMoveToEmptyCell:
    """Test 2.1: Move to empty cell (all 4 directions)."""

    @pytest.mark.requires_set_state
    def test_move_up_to_empty_cell(self, swift_env):
        """Moving up from center should increase row by 1."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0,
                             dataSiphons=0, attackDamage=1, score=0),
            enemies=[],
            transmissions=[],
            blocks=[],
            resources=[],
            owned_programs=[],
            stage=1,
            turn=0,
            showActivated=False,
            scheduledTasksDisabled=False
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_UP)

        row, col = get_player_position(result.observation)
        assert row == 4, f"Expected row=4 after moving up, got {row}"
        assert col == 3, f"Column should be unchanged, got {col}"

    @pytest.mark.requires_set_state
    def test_move_down_to_empty_cell(self, swift_env):
        """Moving down from center should decrease row by 1."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_DOWN)

        row, col = get_player_position(result.observation)
        assert row == 2, f"Expected row=2 after moving down, got {row}"
        assert col == 3, f"Column should be unchanged, got {col}"

    @pytest.mark.requires_set_state
    def test_move_left_to_empty_cell(self, swift_env):
        """Moving left should decrease column by 1."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_LEFT)

        row, col = get_player_position(result.observation)
        assert row == 3, f"Row should be unchanged, got {row}"
        assert col == 2, f"Expected col=2 after moving left, got {col}"

    @pytest.mark.requires_set_state
    def test_move_right_to_empty_cell(self, swift_env):
        """Moving right should increase column by 1."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_RIGHT)

        row, col = get_player_position(result.observation)
        assert row == 3, f"Row should be unchanged, got {row}"
        assert col == 4, f"Expected col=4 after moving right, got {col}"


# MARK: - Test 2.2: Move Blocked by Grid Edge

class TestMoveBlockedByEdge:
    """Test 2.2: Movement blocked by grid edges."""

    @pytest.mark.requires_set_state
    def test_move_blocked_by_top_edge(self, swift_env):
        """Moving up from top row should be invalid."""
        state = GameState(
            player=PlayerState(row=5, col=3, hp=3),
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid_actions = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP not in valid_actions, \
            f"Move up should be blocked at top edge, but found in {valid_actions}"

    @pytest.mark.requires_set_state
    def test_move_blocked_by_bottom_edge(self, swift_env):
        """Moving down from bottom row should be invalid."""
        state = GameState(
            player=PlayerState(row=0, col=3, hp=3),
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid_actions = swift_env.get_valid_actions()
        assert ACTION_MOVE_DOWN not in valid_actions, \
            f"Move down should be blocked at bottom edge, but found in {valid_actions}"

    @pytest.mark.requires_set_state
    def test_move_blocked_by_left_edge(self, swift_env):
        """Moving left from left edge should be invalid."""
        state = GameState(
            player=PlayerState(row=3, col=0, hp=3),
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid_actions = swift_env.get_valid_actions()
        assert ACTION_MOVE_LEFT not in valid_actions, \
            f"Move left should be blocked at left edge, but found in {valid_actions}"

    @pytest.mark.requires_set_state
    def test_move_blocked_by_right_edge(self, swift_env):
        """Moving right from right edge should be invalid."""
        state = GameState(
            player=PlayerState(row=3, col=5, hp=3),
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid_actions = swift_env.get_valid_actions()
        assert ACTION_MOVE_RIGHT not in valid_actions, \
            f"Move right should be blocked at right edge, but found in {valid_actions}"


# MARK: - Test 2.3: Move Blocked by Block

class TestMoveBlockedByBlock:
    """Test 2.3: Movement blocked by blocks."""

    @pytest.mark.requires_set_state
    def test_move_blocked_by_block(self, swift_env):
        """Moving toward a block should be invalid."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid_actions = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP not in valid_actions, \
            f"Move up should be blocked by block, but found in {valid_actions}"


# MARK: - Test 2.4: Move Onto Cell with Data Siphon

class TestMoveCollectsDataSiphon:
    """Test 2.4: Moving onto data siphon cell collects it."""

    @pytest.mark.requires_set_state
    def test_move_collects_data_siphon(self, swift_env):
        """Walking onto a data siphon cell should collect it."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0,
                             dataSiphons=0, attackDamage=1, score=0),
            resources=[Resource(row=4, col=3, dataSiphon=True, credits=0, energy=0)],
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        # Check initial siphons
        obs_before = swift_env.set_state(state)
        siphons_before = int(round(obs_before.player[6] * 10))
        assert siphons_before == 0, f"Should start with 0 siphons, got {siphons_before}"

        result = swift_env.step(ACTION_MOVE_UP)

        row, col = get_player_position(result.observation)
        assert row == 4, f"Should move to row 4, got {row}"

        siphons_after = int(round(result.observation.player[6] * 10))
        assert siphons_after == 1, f"Should have 1 siphon after collecting, got {siphons_after}"

        # Data siphon collection gives reward
        assert result.reward >= 0.5, f"Expected positive reward for collecting siphon, got {result.reward}"


# MARK: - Test 2.5: Line-of-Sight Attack on Distant Enemy

class TestLineOfSightAttack:
    """Test 2.5: Line-of-sight attacks on distant enemies."""

    @pytest.mark.requires_set_state
    def test_line_of_sight_attack_distant_enemy(self, swift_env):
        """Moving toward a distant enemy in line of sight should attack, not move."""
        state = GameState(
            player=PlayerState(row=0, col=3, hp=3, credits=0, energy=0,
                             dataSiphons=0, attackDamage=1, score=0),
            enemies=[Enemy(type="virus", row=5, col=3, hp=1, stunned=False)],  # 5 cells away, same column
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_UP)

        # Player should NOT move (attack instead)
        row, col = get_player_position(result.observation)
        assert row == 0, f"Player should stay at row 0 (attacked), got {row}"

        # Enemy with 1 HP should be killed
        enemy_count = count_enemies(result.observation)
        assert enemy_count == 0, f"Enemy should be killed, but found {enemy_count} enemies"

        # Kill reward
        assert result.reward >= 0.2, f"Expected kill reward, got {result.reward}"


# MARK: - Test 2.6: Line-of-Sight Attack on Enemy on Block

class TestLineOfSightAttackOnBlock:
    """Test 2.6: Line-of-sight attack on enemy standing on a block."""

    @pytest.mark.requires_set_state
    def test_line_of_sight_attack_enemy_on_block(self, swift_env):
        """Can attack enemy on block via line-of-sight even though can't walk there."""
        state = GameState(
            player=PlayerState(row=0, col=3, hp=3, attackDamage=1),
            enemies=[Enemy(type="glitch", row=4, col=3, hp=1, stunned=False)],
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_UP)

        # Player should NOT move
        row, col = get_player_position(result.observation)
        assert row == 0, f"Player should stay at row 0, got {row}"

        # Enemy should be killed (1 HP, took 1 damage)
        enemy_count = count_enemies(result.observation)
        assert enemy_count == 0, f"Enemy should be killed, found {enemy_count}"


# MARK: - Test 2.7: Attack and Kill Adjacent Enemy

class TestAttackKillsEnemy:
    """Test 2.7: Moving into adjacent enemy kills it."""

    @pytest.mark.requires_set_state
    def test_attack_kills_enemy(self, swift_env):
        """Moving into adjacent 1-HP enemy should kill it without moving."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, attackDamage=1),
            enemies=[Enemy(type="virus", row=4, col=3, hp=1, stunned=False)],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_UP)

        # Player should NOT move (attack doesn't move player)
        row, col = get_player_position(result.observation)
        assert row == 3, f"Player should stay at row 3, got {row}"

        # Enemy should be killed
        enemy_count = count_enemies(result.observation)
        assert enemy_count == 0, f"Enemy should be killed, found {enemy_count}"

        # Kill reward should be awarded
        assert result.reward >= 0.2, f"Expected kill reward, got {result.reward}"


# MARK: - Test 2.8: Attack Enemy That Survives

class TestAttackEnemySurvives:
    """Test 2.8: Attacking enemy that survives."""

    @pytest.mark.requires_set_state
    def test_attack_damages_enemy(self, swift_env):
        """Attacking 2-HP enemy should damage but not kill it."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, attackDamage=1),
            enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_UP)

        # Player should NOT move
        row, col = get_player_position(result.observation)
        assert row == 3, f"Player should stay at row 3, got {row}"

        # Enemy should still exist with reduced HP
        enemy = get_enemy_at(result.observation, 4, 3)
        assert enemy is not None, "Enemy should still exist"
        assert enemy["hp"] == 1, f"Enemy HP should be 1 after attack, got {enemy['hp']}"


# MARK: - Test 2.9: Attack Transmission

class TestAttackTransmission:
    """Test 2.9: Moving into transmission destroys it."""

    @pytest.mark.requires_set_state
    def test_attack_destroys_transmission(self, swift_env):
        """Moving into transmission should destroy it without moving."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            transmissions=[Transmission(row=4, col=3, turnsRemaining=3, enemyType="virus")],
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        # Check transmission exists before
        obs_before = swift_env.set_state(state)
        trans_before = obs_before.grid[4, 3, 35]  # Transmission countdown channel
        assert trans_before > 0, "Transmission should exist before attack"

        result = swift_env.step(ACTION_MOVE_UP)

        # Player should NOT move
        row, col = get_player_position(result.observation)
        assert row == 3, f"Player should stay at row 3, got {row}"

        # Transmission should be gone
        trans_after = result.observation.grid[4, 3, 35]
        assert trans_after == 0, f"Transmission should be destroyed, countdown={trans_after}"
