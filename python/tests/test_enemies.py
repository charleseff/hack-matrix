"""
Enemy Tests (Phase 2.39-2.46)

Tests for enemy behavior including:
- Enemy spawning from transmissions
- Enemy movement patterns
- Enemy type-specific behaviors (virus speed, glitch on blocks, cryptog visibility)
- Enemy attacks on adjacent player
- Stun mechanics

These tests verify that the Swift environment correctly implements enemy mechanics.
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
    Observation,
    ACTION_MOVE_UP,
    PROGRAM_WAIT,
)


# MARK: - Helper Functions

def get_player_hp(obs: Observation) -> int:
    """Extract player HP from observation."""
    return int(round(obs.player[2] * 3))


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


def count_enemies(obs: Observation) -> int:
    """Count enemies on grid."""
    return len(find_enemies(obs))


def count_transmissions(obs: Observation) -> int:
    """Count transmissions in the grid."""
    return int(np.sum(obs.grid[:, :, 35] > 0))


# MARK: - Test 2.39: Enemy Spawns from Transmission

class TestEnemySpawnsFromTransmission:
    """Test 2.39: Enemies spawn when transmission countdown reaches 0."""

    @pytest.mark.requires_set_state
    def test_enemy_spawns_from_transmission(self, swift_env):
        """When transmission countdown hits 0, enemy should spawn."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, energy=1),
            transmissions=[Transmission(row=5, col=5, turnsRemaining=1, enemyType="virus")],
            enemies=[],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        obs_before = swift_env.set_state(state)
        assert count_enemies(obs_before) == 0, "Should start with no enemies"
        assert count_transmissions(obs_before) == 1, "Should have 1 transmission"

        # Wait ends turn, enemy turn executes, transmission spawns
        result = swift_env.step(PROGRAM_WAIT)

        # Transmission should be gone, enemy should appear
        enemies_after = find_enemies(result.observation)
        assert len(enemies_after) == 1, f"Should have 1 enemy after spawn, got {len(enemies_after)}"
        assert enemies_after[0]["type"] == "virus", f"Should spawn virus, got {enemies_after[0]['type']}"


# MARK: - Test 2.40: Enemy Movement Toward Player

class TestEnemyMovesIntoPlayer:
    """Test 2.40: Enemies move toward the player."""

    @pytest.mark.requires_set_state
    def test_daemon_moves_one_cell(self, swift_env):
        """Daemon should move 1 cell per turn toward player."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, energy=1),
            enemies=[Enemy(type="daemon", row=3, col=0, hp=3, stunned=False)],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        # Daemon should move from (3,0) to (2,0)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 2, f"Daemon should move to row 2, got {enemies[0]['row']}"
        assert enemies[0]["col"] == 0, f"Daemon should stay in col 0, got {enemies[0]['col']}"


# MARK: - Test 2.41: Virus Double-Move

class TestVirusDoubleMoves:
    """Test 2.41: Viruses move 2 cells per turn."""

    @pytest.mark.requires_set_state
    def test_virus_moves_two_cells(self, swift_env):
        """Virus should move 2 cells per turn."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, energy=1),
            enemies=[Enemy(type="virus", row=4, col=0, hp=2, stunned=False)],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        # Virus should move from (4,0) to (2,0) - 2 cells
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 2, f"Virus should move to row 2, got {enemies[0]['row']}"


# MARK: - Test 2.42: Glitch Can Move On Blocks

class TestGlitchMovesOnBlocks:
    """Test 2.42: Glitch can move through/onto blocks."""

    @pytest.mark.requires_set_state
    def test_glitch_moves_onto_block(self, swift_env):
        """Glitch should be able to move onto blocks."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, energy=1),
            enemies=[Enemy(type="glitch", row=3, col=0, hp=2, stunned=False)],
            blocks=[Block(row=2, col=0, type="data", points=5, spawnCount=5, siphoned=False)],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        # Glitch should move onto block at (2,0)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 2, f"Glitch should move to row 2 (onto block), got {enemies[0]['row']}"


# MARK: - Test 2.43: Cryptog Visibility

class TestCryptogVisibility:
    """Test 2.43: Cryptog visibility based on row/column alignment."""

    @pytest.mark.requires_set_state
    def test_cryptog_visible_in_same_row(self, swift_env):
        """Cryptog in same row as player should be visible."""
        state = GameState(
            player=PlayerState(row=3, col=0, hp=3),
            enemies=[Enemy(type="cryptog", row=3, col=5, hp=2, stunned=False)],
            showActivated=False,
            stage=1
        )
        obs = swift_env.set_state(state)

        enemies = find_enemies(obs)
        assert len(enemies) == 1, f"Cryptog in same row should be visible, found {len(enemies)}"

    @pytest.mark.requires_set_state
    def test_cryptog_visible_in_same_column(self, swift_env):
        """Cryptog in same column as player should be visible."""
        state = GameState(
            player=PlayerState(row=0, col=3, hp=3),
            enemies=[Enemy(type="cryptog", row=5, col=3, hp=2, stunned=False)],
            showActivated=False,
            stage=1
        )
        obs = swift_env.set_state(state)

        enemies = find_enemies(obs)
        assert len(enemies) == 1, f"Cryptog in same column should be visible, found {len(enemies)}"

    @pytest.mark.requires_set_state
    def test_cryptog_hidden_different_row_col(self, swift_env):
        """Cryptog in different row AND column should be hidden."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3),
            enemies=[Enemy(type="cryptog", row=5, col=5, hp=2, stunned=False)],
            showActivated=False,
            stage=1
        )
        obs = swift_env.set_state(state)

        enemies = find_enemies(obs)
        assert len(enemies) == 0, f"Cryptog in different row/col should be hidden, found {len(enemies)}"


# MARK: - Test 2.44: Enemy Attack When Adjacent

class TestEnemyAttacksPlayer:
    """Test 2.44: Enemies attack player when adjacent."""

    @pytest.mark.requires_set_state
    def test_adjacent_enemy_attacks_player(self, swift_env):
        """Adjacent enemy should attack player, reducing HP."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=1),
            enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],  # Adjacent
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        hp_after = get_player_hp(result.observation)
        assert hp_after == 2, f"Player should take 1 damage, HP should be 2, got {hp_after}"


# MARK: - Test 2.45: Stunned Enemy Doesn't Move

class TestStunnedEnemyNoMovement:
    """Test 2.45: Stunned enemies don't move or attack."""

    @pytest.mark.requires_set_state
    def test_stunned_enemy_doesnt_move(self, swift_env):
        """Stunned enemy should not move during enemy turn."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, energy=1),
            enemies=[Enemy(type="virus", row=3, col=0, hp=2, stunned=True)],  # Stunned
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        # Stunned enemy should stay at (3,0)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 3, f"Stunned enemy should not move, got row {enemies[0]['row']}"


# MARK: - Test 2.46: Non-stunned Enemies Move After Turn End

class TestEnemiesMoveAfterTurnEnd:
    """Test 2.46: Non-stunned enemies move after player's turn ends."""

    @pytest.mark.requires_set_state
    def test_enemies_move_after_wait(self, swift_env):
        """Non-stunned enemies should move after player waits.

        Note: Use WAIT program instead of movement to avoid line-of-sight
        attacks triggering instead of actual movement.
        """
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, energy=1),
            enemies=[
                Enemy(type="daemon", row=4, col=3, hp=3, stunned=False),  # Will move (not in same row/col)
                Enemy(type="daemon", row=5, col=5, hp=3, stunned=True),   # Won't move (stunned)
            ],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        obs_before = swift_env.set_state(state)
        enemies_before = find_enemies(obs_before)

        # Wait ends turn
        result = swift_env.step(PROGRAM_WAIT)

        enemies_after = find_enemies(result.observation)

        # Non-stunned daemon should have moved (it was at row 4, should be at row 3)
        non_stunned_moved = [e for e in enemies_after if e["row"] < 4 and e["stunned"] == False]
        assert len(non_stunned_moved) >= 1, \
            f"Non-stunned daemon should move closer, before: {enemies_before}, after: {enemies_after}"

        # Stunned daemon at (5,5) should still be at (5,5) but now un-stunned
        stunned_one = [e for e in enemies_after if e["row"] == 5 and e["col"] == 5]
        assert len(stunned_one) == 1, f"Stunned daemon should still be at (5,5), got {stunned_one}"
