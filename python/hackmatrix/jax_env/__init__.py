"""
HackMatrix JAX Environment.

A pure functional JAX environment designed for TPU-accelerated training.

Usage:
    from hackmatrix.jax_env import reset, step, get_valid_actions

    key = jax.random.PRNGKey(0)
    state, obs = reset(key)
    state, obs, reward, done, breakdown = step(state, action, key)
    valid_actions = get_valid_actions(state)
"""

# Core environment functions
from .env import (
    batched_get_valid_actions,
    batched_reset,
    batched_step,
    get_valid_actions,
    reset,
    step,
)

# Observation types
from .observation import Observation, get_observation

# State types and helpers
from .state import (
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    # Action indices
    ACTION_MOVE_UP,
    ACTION_PROGRAM_START,
    ACTION_SIPHON,
    BLOCK_DATA,
    # Block types
    BLOCK_EMPTY,
    BLOCK_PROGRAM,
    BLOCK_QUESTION,
    ENEMY_CRYPTOG,
    ENEMY_DAEMON,
    ENEMY_GLITCH,
    ENEMY_INT_TO_TYPE,
    ENEMY_MAX_HP,
    ENEMY_TYPE_TO_INT,
    # Enemy types
    ENEMY_VIRUS,
    # Constants
    GRID_SIZE,
    MAX_ENEMIES,
    MAX_TRANSMISSIONS,
    NUM_ACTIONS,
    NUM_PROGRAMS,
    PLAYER_MAX_HP,
    # Program constants
    PROGRAM_COSTS,
    PROGRAM_WAIT,
    EnvState,
    Player,
    add_enemy,
    add_transmission,
    create_empty_state,
)

__all__ = [
    # Environment functions
    "reset",
    "step",
    "get_valid_actions",
    "batched_reset",
    "batched_step",
    "batched_get_valid_actions",
    # State
    "EnvState",
    "Player",
    "Observation",
    "create_empty_state",
    "get_observation",
    "add_enemy",
    "add_transmission",
    # Constants
    "GRID_SIZE",
    "NUM_ACTIONS",
    "NUM_PROGRAMS",
    "MAX_ENEMIES",
    "MAX_TRANSMISSIONS",
    "PLAYER_MAX_HP",
    "ACTION_MOVE_UP",
    "ACTION_MOVE_DOWN",
    "ACTION_MOVE_LEFT",
    "ACTION_MOVE_RIGHT",
    "ACTION_SIPHON",
    "ACTION_PROGRAM_START",
    "ENEMY_VIRUS",
    "ENEMY_DAEMON",
    "ENEMY_GLITCH",
    "ENEMY_CRYPTOG",
    "ENEMY_TYPE_TO_INT",
    "ENEMY_INT_TO_TYPE",
    "ENEMY_MAX_HP",
    "BLOCK_EMPTY",
    "BLOCK_DATA",
    "BLOCK_PROGRAM",
    "BLOCK_QUESTION",
    "PROGRAM_COSTS",
    "PROGRAM_WAIT",
]
