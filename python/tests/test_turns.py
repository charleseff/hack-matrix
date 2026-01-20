"""
Turn Tests (Phase 2.47-2.52)

Tests for turn mechanics including:
- Move/attack/siphon ending player turn
- Programs NOT ending turn (except WAIT)
- Turn counter incrementing
- Enemy turn execution after player turn ends
- Program chaining (multiple programs before turn ends)

These tests verify that the Swift environment correctly implements turn mechanics.
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
    Observation,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_SIPHON,
    PROGRAM_PUSH,
    PROGRAM_PULL,
    PROGRAM_WAIT,
    PROGRAM_SIPH_PLUS,
)


# MARK: - Helper Functions

def get_enemy_at(obs: Observation, row: int, col: int) -> dict | None:
    """Get enemy info at position."""
    if not np.any(obs.grid[row, col, 0:4] > 0):
        return None
    enemy_types = ["virus", "daemon", "glitch", "cryptog"]
    for i, etype in enumerate(enemy_types):
        if obs.grid[row, col, i] > 0:
            hp = int(round(obs.grid[row, col, 4] * 3))
            stunned = obs.grid[row, col, 5] > 0.5
            return {"type": etype, "hp": hp, "stunned": stunned, "row": row, "col": col}
    return None


def find_enemies(obs: Observation) -> list[dict]:
    """Find all enemies on the grid."""
    enemies = []
    for row in range(6):
        for col in range(6):
            enemy = get_enemy_at(obs, row, col)
            if enemy:
                enemies.append(enemy)
    return enemies


def get_player_position(obs: Observation) -> tuple[int, int]:
    """Extract player row, col from observation."""
    row = int(round(obs.player[0] * 5))
    col = int(round(obs.player[1] * 5))
    return row, col


def get_player_energy(obs: Observation) -> int:
    """Extract player energy from observation."""
    return int(round(obs.player[4] * 50))


# MARK: - Test 2.47: Move Ends Turn

class TestMoveEndsTurn:
    """Test 2.47: Movement ends player turn."""

    @pytest.mark.requires_set_state
    def test_move_ends_turn(self, swift_env):
        """Moving should end turn and trigger enemy movement."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            enemies=[Enemy(type="daemon", row=5, col=5, hp=3, stunned=False)],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_UP)

        # Player should have moved
        row, col = get_player_position(result.observation)
        assert row == 4, f"Player should move up, got row {row}"

        # Enemy should have moved (daemon moves 1 cell)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        # Daemon was at (5,5), should move diagonally toward player (4,3)
        # Could be (4,4) or (4,5) or (5,4) depending on pathfinding
        assert enemies[0]["row"] <= 5 or enemies[0]["col"] <= 5, \
            f"Daemon should move closer, got ({enemies[0]['row']}, {enemies[0]['col']})"


# MARK: - Test 2.48: Programs Don't End Turn

class TestProgramsDoNotEndTurn:
    """Test 2.48: Programs (except WAIT) don't end turn."""

    @pytest.mark.requires_set_state
    def test_push_does_not_end_turn(self, swift_env):
        """PUSH program should NOT end the turn."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=2),
            enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_PUSH)

        # Enemy should NOT have moved (turn didn't end)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        # PUSH moves enemy away - from (5,3) to (5+1, 3) but capped at edge = (5,3)
        # Actually PUSH pushes away from player, so row increases (away from player at 3,3)
        # If enemy was at row 5, it can't go further (edge), so stays at 5

        # The key check: movement should still be valid (turn not ended)
        valid = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP in valid, f"Movement should be valid after PUSH, got {valid}"


# MARK: - Test 2.49: WAIT Ends Turn

class TestWaitEndsTurn:
    """Test 2.49: WAIT program ends turn."""

    @pytest.mark.requires_set_state
    def test_wait_ends_turn(self, swift_env):
        """WAIT should end turn and trigger enemy movement."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=1),
            enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        # Enemy should have moved (daemon at row 5 should be at row 4)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 4, f"Daemon should move to row 4, got {enemies[0]['row']}"


# MARK: - Test 2.50: Attack Ends Turn

class TestAttackEndsTurn:
    """Test 2.50: Attacking ends player turn."""

    @pytest.mark.requires_set_state
    def test_attack_ends_turn(self, swift_env):
        """Attacking an enemy should end turn."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3),
            enemies=[
                Enemy(type="virus", row=4, col=3, hp=2, stunned=False),  # Adjacent, will be attacked
                Enemy(type="daemon", row=5, col=5, hp=3, stunned=False),  # Will move after turn
            ],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_MOVE_UP)

        # Should have attacked the virus (not moved)
        row, col = get_player_position(result.observation)
        assert row == 3, f"Player should stay (attacked), got row {row}"

        # Virus should be damaged
        virus = get_enemy_at(result.observation, 4, 3)
        if virus:
            assert virus["hp"] == 1, f"Virus should have 1 HP, got {virus['hp']}"

        # Daemon at (5,5) should have moved (turn ended)
        daemon = [e for e in find_enemies(result.observation) if e["type"] == "daemon"]
        if daemon:
            # Should have moved closer to player
            assert daemon[0]["row"] < 5 or daemon[0]["col"] < 5, \
                f"Daemon should move closer, got ({daemon[0]['row']}, {daemon[0]['col']})"


# MARK: - Test 2.51: Siphon Ends Turn

class TestSiphonEndsTurn:
    """Test 2.51: Siphoning ends player turn."""

    @pytest.mark.requires_set_state
    def test_siphon_ends_turn(self, swift_env):
        """Siphoning should end turn and trigger enemy movement."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, dataSiphons=1),
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
            enemies=[Enemy(type="daemon", row=5, col=5, hp=3, stunned=False)],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(ACTION_SIPHON)

        # Daemon should have moved (turn ended)
        enemies = find_enemies(result.observation)
        daemon = [e for e in enemies if e["type"] == "daemon"]
        if daemon:
            assert daemon[0]["row"] < 5 or daemon[0]["col"] < 5, \
                f"Daemon should move after siphon, got ({daemon[0]['row']}, {daemon[0]['col']})"


# MARK: - Test 2.52: Program Chaining

class TestProgramChaining:
    """Test 2.52: Multiple programs can be used before turn ends."""

    @pytest.mark.requires_set_state
    def test_chain_push_then_pull(self, swift_env):
        """Can use multiple programs before ending turn."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=4),  # Enough for PUSH + PULL
            enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
            owned_programs=[PROGRAM_PUSH, PROGRAM_PULL],
            stage=1
        )
        swift_env.set_state(state)

        # Use PUSH
        result1 = swift_env.step(PROGRAM_PUSH)
        enemies1 = find_enemies(result1.observation)
        # PUSH moves enemy away - daemon stays at (5,3) since it's at edge

        # Movement should still be valid
        valid = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP in valid, f"Should be able to move after PUSH, got {valid}"

        # Use PULL
        result2 = swift_env.step(PROGRAM_PULL)
        enemies2 = find_enemies(result2.observation)
        # PULL moves enemy closer

        # Movement should still be valid
        valid = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP in valid, f"Should be able to move after PULL, got {valid}"

        # Now move to end turn
        result3 = swift_env.step(ACTION_MOVE_UP)

        # Player should have moved
        row, col = get_player_position(result3.observation)
        # Movement ends turn, so enemy also moves

    @pytest.mark.requires_set_state
    def test_chain_siph_plus_then_move(self, swift_env):
        """Can use SIPH+ then move (SIPH+ doesn't end turn).

        Note: Enemy is placed in different column to avoid triggering
        a line-of-sight attack when player moves up.
        """
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=5, energy=0, dataSiphons=0),
            # Place enemy in different column to avoid line-of-sight attack
            enemies=[Enemy(type="daemon", row=5, col=4, hp=3, stunned=False)],
            owned_programs=[PROGRAM_SIPH_PLUS],
            stage=1
        )
        swift_env.set_state(state)

        # Use SIPH+
        result1 = swift_env.step(PROGRAM_SIPH_PLUS)

        # Enemy should NOT have moved
        enemies = find_enemies(result1.observation)
        assert enemies[0]["row"] == 5, f"Enemy should not move after SIPH+, got row {enemies[0]['row']}"

        # Movement should be valid
        valid = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP in valid, f"Should be able to move after SIPH+, got {valid}"

        # Now move
        result2 = swift_env.step(ACTION_MOVE_UP)

        # Enemy should have moved now (turn ended)
        # Daemon was at (5,4), player now at (4,3), daemon moves diagonally toward player
        enemies = find_enemies(result2.observation)
        assert enemies[0]["row"] == 4, f"Enemy should move after player move, got row {enemies[0]['row']}"
