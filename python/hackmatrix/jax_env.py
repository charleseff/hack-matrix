"""
Pure functional JAX environment for HackMatrix.
Designed for JIT compilation and TPU training with PureJaxRL.

This is a dummy environment that establishes JAX patterns and interface parity
with the Swift-backed gym_env.py. It returns zeroed observations and terminates
with 10% probability - no actual game logic yet.
"""

import jax
import jax.numpy as jnp
from flax import struct

# ---------------------------------------------------------------------------
# State and Observation dataclasses
# ---------------------------------------------------------------------------


@struct.dataclass
class EnvState:
    """Immutable environment state."""

    step_count: jnp.int32


@struct.dataclass
class Observation:
    """Observation returned to agent.

    Matches the structure from gym_env.py for parity testing.
    """

    player_state: jax.Array  # (10,) float32
    programs: jax.Array  # (23,) int32
    grid: jax.Array  # (6, 6, 40) float32


# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

NUM_ACTIONS = 28
GRID_SIZE = 6
GRID_FEATURES = 40
PLAYER_STATE_SIZE = 10
NUM_PROGRAMS = 23


# ---------------------------------------------------------------------------
# Core environment functions (pure, JIT-compatible)
# ---------------------------------------------------------------------------


@jax.jit
def reset(key: jax.Array) -> tuple[EnvState, Observation]:
    """
    Initialize environment state.

    Args:
        key: JAX PRNG key (unused in dummy env, but required for interface)

    Returns:
        (initial_state, initial_observation)
    """
    state = EnvState(step_count=jnp.int32(0))
    obs = _zero_observation()
    return state, obs


@jax.jit
def step(
    state: EnvState, action: jnp.int32, key: jax.Array
) -> tuple[EnvState, Observation, jnp.float32, jnp.bool_]:
    """
    Take one environment step.

    Args:
        state: Current environment state
        action: Action index (0-27)
        key: JAX PRNG key

    Returns:
        (new_state, observation, reward, done)
    """
    new_state = EnvState(step_count=state.step_count + 1)
    obs = _zero_observation()
    reward = jnp.float32(0.0)

    # 10% chance of termination
    done = jax.random.uniform(key) < 0.1

    return new_state, obs, reward, done


@jax.jit
def get_valid_actions(state: EnvState) -> jax.Array:
    """
    Return mask of valid actions.

    Args:
        state: Current environment state

    Returns:
        Boolean array of shape (NUM_ACTIONS,) where True = valid
    """
    # For dummy env: only directional actions (0-3) are valid
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    mask = mask.at[0:4].set(True)
    return mask


def _zero_observation() -> Observation:
    """Create zeroed observation."""
    return Observation(
        player_state=jnp.zeros(PLAYER_STATE_SIZE, dtype=jnp.float32),
        programs=jnp.zeros(NUM_PROGRAMS, dtype=jnp.int32),
        grid=jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_FEATURES), dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Vectorized versions for batch training
# ---------------------------------------------------------------------------

# vmap over batch dimension for parallel environment execution
batched_reset = jax.vmap(reset)
batched_step = jax.vmap(step)
batched_get_valid_actions = jax.vmap(get_valid_actions)
