"""
Program Tests (Phase 2.16-2.38)

Tests for program mechanics. Each program has:
- Resource costs (credits, energy)
- Applicability conditions
- Primary effects
- Secondary effects (stuns, damage, etc.)

These tests verify that the Swift environment correctly implements program mechanics.
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
    PROGRAM_PUSH,
    PROGRAM_PULL,
    PROGRAM_CRASH,
    PROGRAM_WARP,
    PROGRAM_POLY,
    PROGRAM_WAIT,
    PROGRAM_DEBUG,
    PROGRAM_ROW,
    PROGRAM_COL,
    PROGRAM_UNDO,
    PROGRAM_STEP,
    PROGRAM_SIPH_PLUS,
    PROGRAM_EXCH,
    PROGRAM_SHOW,
    PROGRAM_RESET,
    PROGRAM_CALM,
    PROGRAM_D_BOM,
    PROGRAM_DELAY,
    PROGRAM_ANTI_V,
    PROGRAM_SCORE,
    PROGRAM_REDUC,
    PROGRAM_ATK_PLUS,
    PROGRAM_HACK,
    ACTION_MOVE_UP,
    ACTION_SIPHON,
)


# MARK: - Helper Functions

def get_player_energy(obs: Observation) -> int:
    """Extract player energy from observation."""
    return int(round(obs.player[4] * 50))


def get_player_credits(obs: Observation) -> int:
    """Extract player credits from observation."""
    return int(round(obs.player[3] * 50))


def get_player_hp(obs: Observation) -> int:
    """Extract player HP from observation."""
    return int(round(obs.player[2] * 3))


def get_player_siphons(obs: Observation) -> int:
    """Extract player data siphons from observation."""
    return int(round(obs.player[6] * 10))


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


# MARK: - Test 2.16: PUSH Program

class TestPushProgram:
    """Test 2.16: PUSH program pushes enemies away."""

    @pytest.mark.requires_set_state
    def test_push_enemies_away(self, swift_env):
        """PUSH should move enemies away from player and cost energy."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=2, dataSiphons=0),
            enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        obs = swift_env.set_state(state)
        energy_before = get_player_energy(obs)
        assert energy_before == 2, f"Should have 2 energy, got {energy_before}"

        result = swift_env.step(PROGRAM_PUSH)

        # Energy should be consumed (PUSH costs 0C, 2E)
        energy_after = get_player_energy(result.observation)
        assert energy_after == 0, f"Energy should be 0 after PUSH, got {energy_after}"

        # Enemy should be pushed away (row increases)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, f"Should still have 1 enemy, got {len(enemies)}"
        assert enemies[0]["row"] == 5, f"Enemy should be at row 5, got {enemies[0]['row']}"

    @pytest.mark.requires_set_state
    def test_push_requires_enemies(self, swift_env):
        """PUSH should be invalid without enemies."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),
            enemies=[],  # No enemies
            owned_programs=[PROGRAM_PUSH],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_PUSH not in valid, f"PUSH should be invalid without enemies, got {valid}"


# MARK: - Test 2.17: PULL Program

class TestPullProgram:
    """Test 2.17: PULL program pulls enemies toward player."""

    @pytest.mark.requires_set_state
    def test_pull_enemies_toward(self, swift_env):
        """PULL should move enemies toward player."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=2),
            enemies=[Enemy(type="virus", row=5, col=3, hp=2, stunned=False)],
            owned_programs=[PROGRAM_PULL],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_PULL)

        # Enemy should be pulled closer
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 4, f"Enemy should be at row 4, got {enemies[0]['row']}"


# MARK: - Test 2.20: POLY Program

class TestPolyProgram:
    """Test 2.20: POLY program transforms enemies."""

    @pytest.mark.requires_set_state
    def test_poly_randomizes_enemy_types(self, swift_env):
        """POLY should change enemy type.

        Note: Enemy is placed in same row as player so it remains visible
        even if transformed to cryptog (cryptogs are only visible when in
        same row/column as player or when showActivated=True).
        """
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=1, energy=1),
            # Place enemy in SAME ROW as player to ensure visibility after transformation
            enemies=[Enemy(type="virus", row=3, col=5, hp=2, stunned=False)],
            owned_programs=[PROGRAM_POLY],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_POLY)

        # Enemy type should change (guaranteed different from virus)
        # Enemy stays visible because it's in same row as player
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy (visible because same row as player)"
        assert enemies[0]["type"] != "virus", f"Enemy type should change, still {enemies[0]['type']}"


# MARK: - Test 2.21: WAIT Program

class TestWaitProgram:
    """Test 2.21: WAIT program ends turn."""

    @pytest.mark.requires_set_state
    def test_wait_ends_turn(self, swift_env):
        """WAIT should end turn and cause enemy movement."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=1),
            enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
            owned_programs=[PROGRAM_WAIT],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WAIT)

        # Daemon should move closer (1 cell per turn)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        # Daemon was at row 5, should move to row 4
        assert enemies[0]["row"] == 4, f"Daemon should be at row 4, got {enemies[0]['row']}"


# MARK: - Test 2.27: SIPH+ Program

class TestSiphPlusProgram:
    """Test 2.27: SIPH+ program grants a data siphon."""

    @pytest.mark.requires_set_state
    def test_siph_plus_gains_data_siphon(self, swift_env):
        """SIPH+ should give player a data siphon."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=5, energy=0, dataSiphons=0),
            owned_programs=[PROGRAM_SIPH_PLUS],
            enemies=[],
            stage=1
        )
        obs = swift_env.set_state(state)
        siphons_before = get_player_siphons(obs)
        assert siphons_before == 0, f"Should start with 0 siphons, got {siphons_before}"

        result = swift_env.step(PROGRAM_SIPH_PLUS)

        siphons_after = get_player_siphons(result.observation)
        assert siphons_after == 1, f"Should have 1 siphon after SIPH+, got {siphons_after}"

        # Credits should be consumed (5C cost)
        credits_after = get_player_credits(result.observation)
        assert credits_after == 0, f"Credits should be 0 after SIPH+, got {credits_after}"


# MARK: - Test 2.30: RESET Program

class TestResetProgram:
    """Test 2.30: RESET program restores HP."""

    @pytest.mark.requires_set_state
    def test_reset_restores_hp(self, swift_env):
        """RESET should restore player HP to max."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=1, credits=0, energy=4),
            owned_programs=[PROGRAM_RESET],
            enemies=[],
            stage=1
        )
        obs = swift_env.set_state(state)
        hp_before = get_player_hp(obs)
        assert hp_before == 1, f"Should start with 1 HP, got {hp_before}"

        result = swift_env.step(PROGRAM_RESET)

        hp_after = get_player_hp(result.observation)
        assert hp_after == 3, f"HP should be 3 after RESET, got {hp_after}"

    @pytest.mark.requires_set_state
    def test_reset_requires_low_hp(self, swift_env):
        """RESET should be invalid at full HP."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),  # Full HP
            owned_programs=[PROGRAM_RESET],
            enemies=[],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_RESET not in valid, f"RESET should be invalid at full HP, got {valid}"


# MARK: - Program Chaining Tests

class TestProgramChaining:
    """Test that programs don't end turn (except WAIT)."""

    @pytest.mark.requires_set_state
    def test_program_does_not_end_turn(self, swift_env):
        """Programs (except WAIT) should not end the turn."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=5, energy=0, dataSiphons=0),
            owned_programs=[PROGRAM_SIPH_PLUS],
            enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_SIPH_PLUS)

        # Enemy should NOT have moved (turn didn't end)
        enemies = find_enemies(result.observation)
        assert len(enemies) == 1, "Should have 1 enemy"
        assert enemies[0]["row"] == 5, f"Enemy should still be at row 5, got {enemies[0]['row']}"

        # Movement should still be valid (turn not ended)
        valid = swift_env.get_valid_actions()
        assert ACTION_MOVE_UP in valid, f"Movement should be valid after program, got {valid}"


# MARK: - Test 2.18: CRASH Program

class TestCrashProgram:
    """Test 2.18: CRASH clears surrounding cells."""

    @pytest.mark.requires_set_state
    def test_crash_kills_surrounding_enemies(self, swift_env):
        """CRASH should kill enemies in 8 surrounding cells."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=3, energy=2),
            enemies=[
                Enemy(type="virus", row=4, col=3, hp=1, stunned=False),  # Above
                Enemy(type="virus", row=3, col=4, hp=1, stunned=False),  # Right
            ],
            owned_programs=[PROGRAM_CRASH],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_CRASH)

        # Both enemies should be killed
        enemies = find_enemies(result.observation)
        assert len(enemies) == 0, f"All surrounding enemies should die, found {len(enemies)}"

        # Resources consumed
        credits = get_player_credits(result.observation)
        energy = get_player_energy(result.observation)
        assert credits == 0, f"Credits should be 0, got {credits}"
        assert energy == 0, f"Energy should be 0, got {energy}"

    @pytest.mark.requires_set_state
    def test_crash_requires_targets(self, swift_env):
        """CRASH should be invalid without targets in surrounding cells."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],  # Not adjacent
            blocks=[],  # No blocks
            owned_programs=[PROGRAM_CRASH],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_CRASH not in valid, f"CRASH requires targets, got {valid}"


# MARK: - Test 2.19: WARP Program

class TestWarpProgram:
    """Test 2.19: WARP teleports to and kills enemy."""

    @pytest.mark.requires_set_state
    def test_warp_kills_enemy(self, swift_env):
        """WARP should teleport to an enemy and kill it."""
        state = GameState(
            player=PlayerState(row=0, col=0, hp=3, credits=2, energy=2),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False)],
            owned_programs=[PROGRAM_WARP],
            stage=1
        )
        obs_before = swift_env.set_state(state)

        result = swift_env.step(PROGRAM_WARP)

        # Player should have moved
        row = int(round(result.observation.player[0] * 5))
        col = int(round(result.observation.player[1] * 5))
        # Player warps TO enemy position (5,5)
        assert row == 5 and col == 5, f"Player should warp to enemy position, got ({row}, {col})"

        # Enemy should be killed
        enemies = find_enemies(result.observation)
        assert len(enemies) == 0, f"Enemy should be killed by warp, found {len(enemies)}"

    @pytest.mark.requires_set_state
    def test_warp_requires_targets(self, swift_env):
        """WARP requires enemies or transmissions."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            enemies=[],  # No enemies
            transmissions=[],  # No transmissions
            owned_programs=[PROGRAM_WARP],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_WARP not in valid, f"WARP requires targets, got {valid}"


# MARK: - Test 2.22: DEBUG Program

class TestDebugProgram:
    """Test 2.22: DEBUG damages enemies on blocks."""

    @pytest.mark.requires_set_state
    def test_debug_damages_enemies_on_blocks(self, swift_env):
        """DEBUG should damage and stun enemies standing on blocks."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0),
            enemies=[
                Enemy(type="glitch", row=4, col=4, hp=2, stunned=False),  # On block
                Enemy(type="virus", row=5, col=5, hp=2, stunned=False),   # NOT on block
            ],
            blocks=[
                Block(row=4, col=4, type="data", points=5, spawnCount=5, siphoned=False),
            ],
            owned_programs=[PROGRAM_DEBUG],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_DEBUG)

        # Enemy on block should be damaged
        enemy_on_block = get_enemy_at(result.observation, 4, 4)
        if enemy_on_block:
            assert enemy_on_block["hp"] == 1, f"Enemy on block should take damage, hp={enemy_on_block['hp']}"
            assert enemy_on_block["stunned"], "Enemy on block should be stunned"

        # Enemy not on block should be untouched
        enemy_off_block = get_enemy_at(result.observation, 5, 5)
        assert enemy_off_block is not None, "Enemy not on block should still exist"
        assert enemy_off_block["hp"] == 2, f"Enemy not on block should be unhurt, hp={enemy_off_block['hp']}"

    @pytest.mark.requires_set_state
    def test_debug_requires_enemies_on_blocks(self, swift_env):
        """DEBUG requires enemies on blocks."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],  # Not on block
            blocks=[Block(row=4, col=4, type="data", points=5, spawnCount=5)],  # Block without enemy
            owned_programs=[PROGRAM_DEBUG],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_DEBUG not in valid, f"DEBUG requires enemies on blocks, got {valid}"


# MARK: - Test 2.23: ROW Program

class TestRowProgram:
    """Test 2.23: ROW attacks all enemies in player's row."""

    @pytest.mark.requires_set_state
    def test_row_attacks_enemies_in_row(self, swift_env):
        """ROW should damage all enemies in player's row."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=3, energy=1),
            enemies=[
                Enemy(type="virus", row=3, col=0, hp=1, stunned=False),   # Same row, will die
                Enemy(type="daemon", row=3, col=5, hp=3, stunned=False),  # Same row, survives
                Enemy(type="virus", row=4, col=3, hp=2, stunned=False),   # Different row
            ],
            owned_programs=[PROGRAM_ROW],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_ROW)

        # Enemy at (3,0) should be killed
        enemy_at_3_0 = get_enemy_at(result.observation, 3, 0)
        assert enemy_at_3_0 is None, "1-HP enemy in row should be killed"

        # Enemy at (3,5) should survive but be damaged and stunned
        enemy_at_3_5 = get_enemy_at(result.observation, 3, 5)
        assert enemy_at_3_5 is not None, "3-HP enemy in row should survive"
        assert enemy_at_3_5["hp"] == 2, f"Daemon should be damaged, hp={enemy_at_3_5['hp']}"

        # Enemy at (4,3) should be untouched
        enemy_at_4_3 = get_enemy_at(result.observation, 4, 3)
        assert enemy_at_4_3 is not None, "Enemy not in row should be untouched"
        assert enemy_at_4_3["hp"] == 2, f"Enemy not in row should have full hp={enemy_at_4_3['hp']}"

    @pytest.mark.requires_set_state
    def test_row_requires_enemies_in_row(self, swift_env):
        """ROW requires enemies in player's row."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],  # Different row
            owned_programs=[PROGRAM_ROW],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_ROW not in valid, f"ROW requires enemies in row, got {valid}"


# MARK: - Test 2.24: COL Program

class TestColProgram:
    """Test 2.24: COL attacks all enemies in player's column."""

    @pytest.mark.requires_set_state
    def test_col_attacks_enemies_in_column(self, swift_env):
        """COL should damage all enemies in player's column."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=3, energy=1),
            enemies=[
                Enemy(type="virus", row=0, col=3, hp=1, stunned=False),   # Same col, will die
                Enemy(type="daemon", row=5, col=3, hp=3, stunned=False),  # Same col, survives
                Enemy(type="virus", row=3, col=4, hp=2, stunned=False),   # Different col
            ],
            owned_programs=[PROGRAM_COL],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_COL)

        # Enemy at (0,3) should be killed
        enemy_at_0_3 = get_enemy_at(result.observation, 0, 3)
        assert enemy_at_0_3 is None, "1-HP enemy in col should be killed"

        # Enemy at (5,3) should survive but be damaged
        enemy_at_5_3 = get_enemy_at(result.observation, 5, 3)
        assert enemy_at_5_3 is not None, "3-HP enemy in col should survive"
        assert enemy_at_5_3["hp"] == 2, f"Daemon should be damaged, hp={enemy_at_5_3['hp']}"

        # Enemy at (3,4) should be untouched
        enemy_at_3_4 = get_enemy_at(result.observation, 3, 4)
        assert enemy_at_3_4 is not None, "Enemy not in col should be untouched"
        assert enemy_at_3_4["hp"] == 2, f"Enemy not in col should have full hp={enemy_at_3_4['hp']}"

    @pytest.mark.requires_set_state
    def test_col_requires_enemies_in_column(self, swift_env):
        """COL requires enemies in player's column."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],  # Different col
            owned_programs=[PROGRAM_COL],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_COL not in valid, f"COL requires enemies in column, got {valid}"


# MARK: - Test 2.28: EXCH Program

class TestExchProgram:
    """Test 2.28: EXCH converts credits to energy."""

    @pytest.mark.requires_set_state
    def test_exch_converts_credits_to_energy(self, swift_env):
        """EXCH should convert credits to energy."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=8, energy=0),
            owned_programs=[PROGRAM_EXCH],
            stage=1
        )
        obs_before = swift_env.set_state(state)
        credits_before = get_player_credits(obs_before)
        assert credits_before == 8, f"Should start with 8 credits, got {credits_before}"

        result = swift_env.step(PROGRAM_EXCH)

        # Credits: 8 - 4 (cost) = 4, then converted: 4 - 4 = 0 and +4 energy
        # OR: Credits: 8 - 4 (cost) = 4, and 4 credits become 4 energy
        credits_after = get_player_credits(result.observation)
        energy_after = get_player_energy(result.observation)

        # According to implementation, cost is 4C, and you gain 4E
        assert credits_after == 4, f"Credits should be 4 after EXCH (8 - 4 cost), got {credits_after}"
        assert energy_after == 4, f"Energy should be 4 after EXCH, got {energy_after}"

    @pytest.mark.requires_set_state
    def test_exch_requires_credits(self, swift_env):
        """EXCH requires at least 4 credits (for cost)."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0),  # Only 3 credits
            owned_programs=[PROGRAM_EXCH],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_EXCH not in valid, f"EXCH requires 4+ credits, got {valid}"


# MARK: - Test 2.29: SHOW Program

class TestShowProgram:
    """Test 2.29: SHOW reveals cryptogs and transmission types."""

    @pytest.mark.requires_set_state
    def test_show_reveals_cryptog(self, swift_env):
        """SHOW should make cryptogs visible."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=2, energy=0),
            enemies=[Enemy(type="cryptog", row=5, col=5, hp=2, stunned=False)],  # Hidden (different row/col)
            showActivated=False,
            owned_programs=[PROGRAM_SHOW],
            stage=1
        )

        # Before SHOW, cryptog at (5,5) should be hidden
        obs_before = swift_env.set_state(state)
        enemies_before = find_enemies(obs_before)
        assert len(enemies_before) == 0, f"Cryptog should be hidden before SHOW, found {len(enemies_before)}"

        result = swift_env.step(PROGRAM_SHOW)

        # After SHOW, cryptog should be visible
        enemies_after = find_enemies(result.observation)
        assert len(enemies_after) == 1, f"Cryptog should be visible after SHOW, found {len(enemies_after)}"

    @pytest.mark.requires_set_state
    def test_show_requires_not_activated(self, swift_env):
        """SHOW requires showActivated to be false."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            showActivated=True,  # Already activated
            owned_programs=[PROGRAM_SHOW],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_SHOW not in valid, f"SHOW requires showActivated=false, got {valid}"


# MARK: - Test 2.31: CALM Program

class TestCalmProgram:
    """Test 2.31: CALM disables scheduled spawns."""

    @pytest.mark.requires_set_state
    def test_calm_disables_spawns(self, swift_env):
        """CALM should disable scheduled task spawns."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=2, energy=4),
            scheduledTasksDisabled=False,
            owned_programs=[PROGRAM_CALM],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_CALM)

        # Resources consumed
        credits = get_player_credits(result.observation)
        energy = get_player_energy(result.observation)
        assert credits == 0, f"Credits should be 0 after CALM, got {credits}"
        assert energy == 0, f"Energy should be 0 after CALM, got {energy}"

        # Note: We can't directly check scheduledTasksDisabled in observation,
        # but we can verify the program executed by checking resources consumed

    @pytest.mark.requires_set_state
    def test_calm_requires_not_disabled(self, swift_env):
        """CALM requires scheduledTasksDisabled to be false."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            scheduledTasksDisabled=True,  # Already disabled
            owned_programs=[PROGRAM_CALM],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_CALM not in valid, f"CALM requires scheduledTasksDisabled=false, got {valid}"


# MARK: - Test 2.32: D_BOM Program

class TestDBomProgram:
    """Test 2.32: D_BOM destroys nearest daemon."""

    @pytest.mark.requires_set_state
    def test_d_bom_destroys_daemon(self, swift_env):
        """D_BOM should destroy the nearest daemon."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0),
            enemies=[
                Enemy(type="daemon", row=5, col=5, hp=3, stunned=False),
                Enemy(type="virus", row=4, col=4, hp=2, stunned=False),  # Not a daemon
            ],
            owned_programs=[PROGRAM_D_BOM],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_D_BOM)

        # Daemon should be destroyed
        enemy_at_5_5 = get_enemy_at(result.observation, 5, 5)
        assert enemy_at_5_5 is None, "Daemon should be destroyed by D_BOM"

        # Virus should be affected by splash damage (it's adjacent to daemon at 5,5)
        # (4,4) is adjacent to (5,5) diagonally
        enemy_at_4_4 = get_enemy_at(result.observation, 4, 4)
        if enemy_at_4_4:
            assert enemy_at_4_4["hp"] <= 1, f"Virus should take splash damage, hp={enemy_at_4_4['hp']}"

    @pytest.mark.requires_set_state
    def test_d_bom_requires_daemon(self, swift_env):
        """D_BOM requires a daemon to exist."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            enemies=[Enemy(type="virus", row=5, col=5, hp=2)],  # Not a daemon
            owned_programs=[PROGRAM_D_BOM],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_D_BOM not in valid, f"D_BOM requires daemon, got {valid}"


# MARK: - Test 2.33: DELAY Program

class TestDelayProgram:
    """Test 2.33: DELAY extends transmission timers."""

    @pytest.mark.requires_set_state
    def test_delay_extends_transmissions(self, swift_env):
        """DELAY should add turns to transmission countdowns."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=1, energy=2),
            transmissions=[
                Transmission(row=5, col=5, turnsRemaining=2, enemyType="virus"),
            ],
            owned_programs=[PROGRAM_DELAY],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_DELAY)

        # Transmission countdown should increase by 3
        # Check transmission at (5,5)
        trans_countdown = result.observation.grid[5, 5, 35]  # Transmission countdown channel
        # Original was 2, +3 = 5, normalized by /10 = 0.5
        assert trans_countdown > 0.3, f"Transmission countdown should increase, got {trans_countdown}"

    @pytest.mark.requires_set_state
    def test_delay_requires_transmissions(self, swift_env):
        """DELAY requires transmissions to exist."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            transmissions=[],  # No transmissions
            owned_programs=[PROGRAM_DELAY],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_DELAY not in valid, f"DELAY requires transmissions, got {valid}"


# MARK: - Test 2.34: ANTI-V Program

class TestAntiVProgram:
    """Test 2.34: ANTI-V damages all viruses."""

    @pytest.mark.requires_set_state
    def test_antiv_damages_viruses(self, swift_env):
        """ANTI-V should damage all viruses."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0),
            enemies=[
                Enemy(type="virus", row=4, col=3, hp=1, stunned=False),   # Will die
                Enemy(type="virus", row=5, col=5, hp=2, stunned=False),   # Will survive
                Enemy(type="daemon", row=0, col=0, hp=3, stunned=False),  # Not a virus
            ],
            owned_programs=[PROGRAM_ANTI_V],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_ANTI_V)

        # 1-HP virus should be killed
        enemy_at_4_3 = get_enemy_at(result.observation, 4, 3)
        assert enemy_at_4_3 is None, "1-HP virus should be killed"

        # 2-HP virus should survive with 1 HP
        enemy_at_5_5 = get_enemy_at(result.observation, 5, 5)
        assert enemy_at_5_5 is not None, "2-HP virus should survive"
        assert enemy_at_5_5["hp"] == 1, f"Virus should be damaged, hp={enemy_at_5_5['hp']}"

        # Daemon should be untouched
        enemy_at_0_0 = get_enemy_at(result.observation, 0, 0)
        assert enemy_at_0_0 is not None, "Daemon should be untouched"
        assert enemy_at_0_0["hp"] == 3, f"Daemon should have full hp={enemy_at_0_0['hp']}"

    @pytest.mark.requires_set_state
    def test_antiv_requires_virus(self, swift_env):
        """ANTI-V requires a virus to exist."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10),
            enemies=[Enemy(type="daemon", row=5, col=5, hp=3)],  # Not a virus
            owned_programs=[PROGRAM_ANTI_V],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_ANTI_V not in valid, f"ANTI-V requires virus, got {valid}"


# MARK: - Test 2.35: SCORE Program

class TestScoreProgram:
    """Test 2.35: SCORE gains points equal to stages left."""

    @pytest.mark.requires_set_state
    def test_score_gains_points(self, swift_env):
        """SCORE should give points equal to 8 - current stage."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=0, energy=5, score=0),
            stage=2,  # 8 - 2 = 6 points
            owned_programs=[PROGRAM_SCORE],
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_SCORE)

        # Energy should be consumed
        energy = get_player_energy(result.observation)
        assert energy == 0, f"Energy should be 0 after SCORE, got {energy}"

        # Reward should include score gain (6 * 0.5 = 3.0)
        assert result.reward > 0, f"Should get positive reward from score gain, got {result.reward}"

    @pytest.mark.requires_set_state
    def test_score_requires_not_last_stage(self, swift_env):
        """SCORE requires not being on last stage."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, energy=10),
            stage=8,  # Last stage
            owned_programs=[PROGRAM_SCORE],
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_SCORE not in valid, f"SCORE requires stage < 8, got {valid}"


# MARK: - Test 2.36: REDUC Program

class TestReducProgram:
    """Test 2.36: REDUC reduces block spawn counts."""

    @pytest.mark.requires_set_state
    def test_reduc_reduces_spawn_counts(self, swift_env):
        """REDUC should reduce spawn counts of unsiphoned blocks."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=2, energy=1),
            blocks=[
                Block(row=4, col=3, type="data", points=3, spawnCount=3, siphoned=False),
                Block(row=4, col=4, type="data", points=2, spawnCount=2, siphoned=True),  # Siphoned
            ],
            owned_programs=[PROGRAM_REDUC],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_REDUC)

        # Resources consumed
        credits = get_player_credits(result.observation)
        energy = get_player_energy(result.observation)
        assert credits == 0, f"Credits should be 0, got {credits}"
        assert energy == 0, f"Energy should be 0, got {energy}"

        # Block spawn counts are reduced - can't directly verify in observation
        # but we can verify resources were consumed

    @pytest.mark.requires_set_state
    def test_reduc_requires_unsiphoned_blocks(self, swift_env):
        """REDUC requires unsiphoned blocks with spawnCount > 0."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            blocks=[
                Block(row=4, col=3, type="data", points=0, spawnCount=0, siphoned=False),  # spawnCount=0
            ],
            owned_programs=[PROGRAM_REDUC],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_REDUC not in valid, f"REDUC requires blocks with spawnCount>0, got {valid}"


# MARK: - Test 2.37: ATK+ Program

class TestAtkPlusProgram:
    """Test 2.37: ATK+ increases attack damage."""

    @pytest.mark.requires_set_state
    def test_atkplus_increases_damage(self, swift_env):
        """ATK+ should increase player's attack damage to 2."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=4, energy=4, attackDamage=1),
            owned_programs=[PROGRAM_ATK_PLUS],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_ATK_PLUS)

        # Resources consumed
        credits = get_player_credits(result.observation)
        energy = get_player_energy(result.observation)
        assert credits == 0, f"Credits should be 0, got {credits}"
        assert energy == 0, f"Energy should be 0, got {energy}"

        # Attack damage is in player observation index 7
        attack_damage = int(round(result.observation.player[7] * 2))
        assert attack_damage == 2, f"Attack damage should be 2, got {attack_damage}"

    @pytest.mark.requires_set_state
    def test_atkplus_requires_low_damage(self, swift_env):
        """ATK+ requires attackDamage < 2."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10, attackDamage=2),
            owned_programs=[PROGRAM_ATK_PLUS],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_ATK_PLUS not in valid, f"ATK+ requires damage < 2, got {valid}"


# MARK: - Test 2.38: HACK Program

class TestHackProgram:
    """Test 2.38: HACK damages enemies on siphoned cells."""

    @pytest.mark.requires_set_state
    def test_hack_damages_enemies_on_siphoned_blocks(self, swift_env):
        """HACK should damage enemies standing on siphoned blocks."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=2, energy=2),
            enemies=[
                Enemy(type="virus", row=4, col=3, hp=1, stunned=False),  # On siphoned block
                Enemy(type="virus", row=5, col=5, hp=2, stunned=False),  # Not on siphoned block
            ],
            blocks=[
                Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=True),   # Siphoned
                Block(row=5, col=4, type="data", points=3, spawnCount=3, siphoned=False),  # Not siphoned
            ],
            owned_programs=[PROGRAM_HACK],
            stage=1
        )
        swift_env.set_state(state)

        result = swift_env.step(PROGRAM_HACK)

        # Enemy on siphoned block should be killed
        enemy_at_4_3 = get_enemy_at(result.observation, 4, 3)
        assert enemy_at_4_3 is None, "Enemy on siphoned block should be killed"

        # Enemy not on siphoned block should be untouched
        enemy_at_5_5 = get_enemy_at(result.observation, 5, 5)
        assert enemy_at_5_5 is not None, "Enemy not on siphoned block should survive"
        assert enemy_at_5_5["hp"] == 2, f"Enemy not on siphoned block should be unhurt, hp={enemy_at_5_5['hp']}"

    @pytest.mark.requires_set_state
    def test_hack_requires_siphoned_blocks(self, swift_env):
        """HACK requires siphoned blocks to exist."""
        state = GameState(
            player=PlayerState(row=3, col=3, hp=3, credits=10, energy=10),
            blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],  # Not siphoned
            owned_programs=[PROGRAM_HACK],
            stage=1
        )
        swift_env.set_state(state)

        valid = swift_env.get_valid_actions()
        assert PROGRAM_HACK not in valid, f"HACK requires siphoned blocks, got {valid}"
