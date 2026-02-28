"""
Gymnax-compatible wrapper for HackMatrix JAX environment.

This wrapper adapts the HackMatrix JAX environment to the interface
expected by PureJaxRL, which is based on Gymnax conventions.

Key adaptations:
- reset(key, params) -> (obs, state) instead of reset(key) -> (state, obs)
- step returns info dict
- Observations flattened to single array (1545 features)
- Action mask available via get_action_mask(state)
"""

import jax
import jax.numpy as jnp
from flax import struct

from hackmatrix.jax_env import (
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_PROGRAMS,
    EnvState,
    Observation,
    get_valid_actions,
)
from hackmatrix.jax_env import (
    reset as jax_reset,
)
from hackmatrix.jax_env import (
    step as jax_step,
)
from hackmatrix.jax_env.state import GRID_FEATURES, PLAYER_STATE_SIZE

# Observation dimensions
# player_state: 10, programs: 23, grid: 6*6*42 = 1512
OBS_SIZE = PLAYER_STATE_SIZE + NUM_PROGRAMS + GRID_SIZE * GRID_SIZE * GRID_FEATURES  # 1545


@struct.dataclass
class EnvParams:
    """Environment parameters for curriculum learning.

    Default values match the normal (full difficulty) game.
    Curriculum phases override these to create easier training environments.
    """

    starting_data_siphons: jnp.int32 = 0
    starting_credits: jnp.int32 = 0
    starting_energy: jnp.int32 = 0
    transmission_scale: jnp.float32 = 1.0  # multiplier on per-stage transmission count
    siphon_death_penalty: jnp.float32 = -10.0
    distance_shaping_coef: jnp.float32 = 0.05
    data_siphon_reward: jnp.float32 = 1.0


@struct.dataclass
class GymnaxEnvState:
    """Wrapper state that includes the underlying EnvState and extra info."""

    env_state: EnvState
    key: jax.Array


class HackMatrixGymnax:
    """Gymnax-compatible wrapper for HackMatrix environment.

    This wrapper provides the standard Gymnax interface:
    - reset(key, params) -> (obs, state)
    - step(key, state, action, params) -> (obs, state, reward, done, info)
    - get_action_mask(state) -> mask

    Observations are flattened to a 1D array of shape (1545,) for neural network input.
    """

    def __init__(self):
        self.num_actions = NUM_ACTIONS
        self.obs_shape = (OBS_SIZE,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset(self, key: jax.Array, params: EnvParams = None) -> tuple[jax.Array, GymnaxEnvState]:
        """Reset environment and return (obs, state).

        Args:
            key: JAX random key
            params: Environment parameters (curriculum settings)

        Returns:
            obs: Flattened observation array of shape (1545,)
            state: Wrapper state containing EnvState
        """
        params = params if params is not None else self.default_params
        key, reset_key = jax.random.split(key)
        env_state, obs = jax_reset(reset_key, params)

        flat_obs = self._flatten_obs(obs)
        wrapper_state = GymnaxEnvState(env_state=env_state, key=key)

        return flat_obs, wrapper_state

    def step(
        self,
        key: jax.Array,
        state: GymnaxEnvState,
        action: jnp.int32,
        params: EnvParams = None,
    ) -> tuple[jax.Array, GymnaxEnvState, jnp.float32, jnp.bool_, dict]:
        """Execute action and return (obs, state, reward, done, info).

        Args:
            key: JAX random key
            state: Current wrapper state
            action: Action index (0-27)
            params: Environment parameters (curriculum settings)

        Returns:
            obs: Flattened observation array
            state: New wrapper state
            reward: Scalar reward
            done: Episode termination flag
            info: Dictionary with additional info
        """
        params = params if params is not None else self.default_params
        key, step_key = jax.random.split(key)
        env_state, obs, reward, done, breakdown = jax_step(state.env_state, action, step_key, params)

        flat_obs = self._flatten_obs(obs)
        new_state = GymnaxEnvState(env_state=env_state, key=key)

        info = {
            "stage": env_state.stage,
            "score": env_state.player.score,
            "hp": env_state.player.hp,
            **breakdown,
        }

        return flat_obs, new_state, reward, done, info

    def get_action_mask(self, state: GymnaxEnvState) -> jax.Array:
        """Get valid action mask for current state.

        Args:
            state: Wrapper state

        Returns:
            Boolean array of shape (28,) where True = valid action
        """
        return get_valid_actions(state.env_state)

    def _flatten_obs(self, obs: Observation) -> jax.Array:
        """Flatten structured observation to single array.

        Concatenates:
        - player_state: (10,) float32
        - programs: (23,) -> cast to float32
        - grid: (6, 6, 42) -> flatten to (1512,)

        Total: 1545 features
        """
        return jnp.concatenate(
            [
                obs.player_state,  # (10,)
                obs.programs.astype(jnp.float32),  # (23,)
                obs.grid.ravel(),  # (1512,)
            ]
        )

    def observation_space(self, params: EnvParams = None):
        """Return observation space specification."""
        return {
            "shape": self.obs_shape,
            "dtype": jnp.float32,
            "low": 0.0,
            "high": 1.0,
        }

    def action_space(self, params: EnvParams = None):
        """Return action space specification."""
        return {
            "n": self.num_actions,
            "dtype": jnp.int32,
        }


# Vectorized versions for batched training
def make_batched_env(num_envs: int):
    """Create batched reset and step functions.

    Args:
        num_envs: Number of parallel environments

    Returns:
        Tuple of (batched_reset, batched_step, batched_get_mask, env)
        batched_reset(keys, params) and batched_step(keys, states, actions, params)
        accept EnvParams that are broadcast across all envs.
    """
    env = HackMatrixGymnax()

    # Vectorize over the key dimension; params are broadcast (not vmapped)
    batched_reset = jax.vmap(
        lambda key, params: env.reset(key, params),
        in_axes=(0, None),
    )

    batched_step = jax.vmap(
        lambda key, state, action, params: env.step(key, state, action, params),
        in_axes=(0, 0, 0, None),
    )

    batched_get_mask = jax.vmap(env.get_action_mask)

    return batched_reset, batched_step, batched_get_mask, env
