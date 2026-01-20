"""
Environment Interface Protocol and Dataclasses

This module defines the contract that both Swift and JAX environments must implement
for parity testing. The interface enables deterministic test setup via set_state().

Why this design:
- Protocol-based interface allows duck typing without inheritance requirements
- Dataclasses provide type-safe, immutable state representations
- set_state() enables precise test preconditions without relying on random generation
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np


# =============================================================================
# State Dataclasses (for set_state)
# =============================================================================

@dataclass
class PlayerState:
    """Player state for test setup."""
    row: int
    col: int
    hp: int = 3
    credits: int = 0
    energy: int = 0
    dataSiphons: int = 0
    attackDamage: int = 1
    score: int = 0


@dataclass
class Enemy:
    """Enemy state for test setup."""
    type: str  # "virus", "daemon", "glitch", "cryptog"
    row: int
    col: int
    hp: int
    stunned: bool = False


@dataclass
class Transmission:
    """Transmission (pending enemy spawn) for test setup."""
    row: int
    col: int
    turnsRemaining: int
    enemyType: str


@dataclass
class Block:
    """Block state for test setup.

    For data blocks: points == spawnCount (invariant)
    For program blocks: programType and programActionIndex identify the program
    """
    row: int
    col: int
    type: str  # "data" or "program"
    # Data block fields
    points: int = 0
    spawnCount: int = 0
    siphoned: bool = False
    # Program block fields
    programType: str | None = None
    programActionIndex: int | None = None


@dataclass
class Resource:
    """Resource pickup on a cell.

    Data siphons are collected by walking onto cells.
    Credits/energy are collected by siphoning blocks.
    """
    row: int
    col: int
    credits: int = 0
    energy: int = 0
    dataSiphon: bool = False


@dataclass
class GameState:
    """Complete game state for set_state().

    This dataclass represents all the information needed to set up
    a specific game scenario for testing.
    """
    player: PlayerState
    enemies: list[Enemy] = field(default_factory=list)
    transmissions: list[Transmission] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)
    owned_programs: list[int] = field(default_factory=list)  # Action indices (5-27)
    stage: int = 1
    turn: int = 0
    showActivated: bool = False
    scheduledTasksDisabled: bool = False


# =============================================================================
# Observation Dataclass
# =============================================================================

@dataclass
class Observation:
    """Observation returned from environment.

    Structure:
    - player: 10-element float32 array with normalized player state
    - programs: 23-element int32 array with binary ownership flags
    - grid: (6, 6, 40) float32 array with cell features
    """
    player: np.ndarray  # shape (10,), dtype float32
    programs: np.ndarray  # shape (23,), dtype int32
    grid: np.ndarray  # shape (6, 6, 40), dtype float32

    def __post_init__(self):
        """Validate observation shapes and dtypes."""
        assert self.player.shape == (10,), f"player shape {self.player.shape} != (10,)"
        assert self.programs.shape == (23,), f"programs shape {self.programs.shape} != (23,)"
        assert self.grid.shape == (6, 6, 40), f"grid shape {self.grid.shape} != (6, 6, 40)"


# =============================================================================
# Step Result
# =============================================================================

@dataclass
class StepResult:
    """Result of executing a step in the environment."""
    observation: Observation
    reward: float
    done: bool
    info: dict


# =============================================================================
# Environment Interface Protocol
# =============================================================================

@runtime_checkable
class EnvInterface(Protocol):
    """Protocol defining the contract for environment implementations.

    Both Swift and JAX environments must implement this interface.
    The set_state() method is critical for deterministic testing.
    """

    def reset(self) -> Observation:
        """Reset the environment to initial state.

        Returns:
            Observation of the initial state
        """
        ...

    def step(self, action: int) -> StepResult:
        """Execute an action in the environment.

        Args:
            action: Action index (0-27)
                - 0-3: Movement (up, down, left, right)
                - 4: Siphon
                - 5-27: Programs

        Returns:
            StepResult containing observation, reward, done flag, and info dict
        """
        ...

    def get_valid_actions(self) -> list[int]:
        """Get list of valid action indices for current state.

        Returns:
            List of valid action indices (subset of 0-27)
        """
        ...

    def set_state(self, state: GameState) -> Observation:
        """Set the complete game state for test setup.

        This method enables deterministic testing by allowing precise
        control over the game state before executing test actions.

        Args:
            state: Complete game state to set

        Returns:
            Observation of the set state
        """
        ...


# =============================================================================
# Constants
# =============================================================================

# Action indices
ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_SIPHON = 4
ACTION_PROGRAM_START = 5  # Programs are indices 5-27

# Program action indices (5-27)
PROGRAM_PUSH = 5
PROGRAM_PULL = 6
PROGRAM_CRASH = 7
PROGRAM_WARP = 8
PROGRAM_POLY = 9
PROGRAM_WAIT = 10
PROGRAM_DEBUG = 11
PROGRAM_ROW = 12
PROGRAM_COL = 13
PROGRAM_UNDO = 14
PROGRAM_STEP = 15
PROGRAM_SIPH_PLUS = 16
PROGRAM_EXCH = 17
PROGRAM_SHOW = 18
PROGRAM_RESET = 19
PROGRAM_CALM = 20
PROGRAM_D_BOM = 21
PROGRAM_DELAY = 22
PROGRAM_ANTI_V = 23
PROGRAM_SCORE = 24
PROGRAM_REDUC = 25
PROGRAM_ATK_PLUS = 26
PROGRAM_HACK = 27

# Grid constants
GRID_SIZE = 6
EXIT_ROW = 5
EXIT_COL = 5

# Player stats
MAX_HP = 3
MAX_ATTACK_DAMAGE = 2

# Enemy types
ENEMY_VIRUS = "virus"
ENEMY_DAEMON = "daemon"
ENEMY_GLITCH = "glitch"
ENEMY_CRYPTOG = "cryptog"


# =============================================================================
# Helper functions
# =============================================================================

def game_state_to_json(state: GameState) -> dict:
    """Convert GameState dataclass to JSON-serializable dict.

    Used by SwiftEnvWrapper to send setState commands.
    """
    return {
        "player": {
            "row": state.player.row,
            "col": state.player.col,
            "hp": state.player.hp,
            "credits": state.player.credits,
            "energy": state.player.energy,
            "dataSiphons": state.player.dataSiphons,
            "attackDamage": state.player.attackDamage,
            "score": state.player.score,
        },
        "enemies": [
            {
                "type": e.type,
                "row": e.row,
                "col": e.col,
                "hp": e.hp,
                "stunned": e.stunned,
            }
            for e in state.enemies
        ],
        "transmissions": [
            {
                "row": t.row,
                "col": t.col,
                "turnsRemaining": t.turnsRemaining,
                "enemyType": t.enemyType,
            }
            for t in state.transmissions
        ],
        "blocks": [
            {
                "row": b.row,
                "col": b.col,
                "type": b.type,
                "points": b.points,
                "spawnCount": b.spawnCount,
                "siphoned": b.siphoned,
                **({"programType": b.programType, "programActionIndex": b.programActionIndex}
                   if b.type == "program" else {}),
            }
            for b in state.blocks
        ],
        "resources": [
            {
                "row": r.row,
                "col": r.col,
                "credits": r.credits,
                "energy": r.energy,
                "dataSiphon": r.dataSiphon,
            }
            for r in state.resources
        ],
        "ownedPrograms": state.owned_programs,
        "stage": state.stage,
        "turn": state.turn,
        "showActivated": state.showActivated,
        "scheduledTasksDisabled": state.scheduledTasksDisabled,
    }
