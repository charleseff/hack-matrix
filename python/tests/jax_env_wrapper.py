"""
JAX Environment Wrapper implementing EnvInterface.

This wrapper adapts the pure functional JAX environment to the EnvInterface
protocol for parity testing. Currently a skeleton/stub that returns dummy data.

Why this design:
- set_state() is stubbed because the JAX env doesn't support state injection yet
- Once the full JAX implementation is complete, this wrapper can be updated
- The interface matches SwiftEnvWrapper for test compatibility
"""

import numpy as np
import jax
import jax.numpy as jnp

from .env_interface import (
    EnvInterface,
    GameState,
    Observation,
    StepResult,
    GRID_SIZE,
)

# Import the JAX environment functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hackmatrix import jax_env


class JaxEnvWrapper:
    """JAX environment wrapper implementing EnvInterface.

    This is a skeleton/stub implementation. The JAX environment currently
    returns dummy observations and doesn't support set_state().
    Tests using set_state() will skip or fail for JAX until implemented.
    """

    def __init__(self, seed: int = 0):
        """
        Initialize the JAX environment wrapper.

        Args:
            seed: Random seed for JAX PRNG.
        """
        self.key = jax.random.PRNGKey(seed)
        self.state = None
        self._initialized = False

    def _convert_observation(self, jax_obs: jax_env.Observation) -> Observation:
        """Convert JAX Observation to EnvInterface Observation."""
        return Observation(
            player=np.array(jax_obs.player_state, dtype=np.float32),
            programs=np.array(jax_obs.programs, dtype=np.int32),
            grid=np.array(jax_obs.grid, dtype=np.float32),
        )

    # MARK: - EnvInterface Implementation

    def reset(self) -> Observation:
        """Reset the environment to initial state."""
        self.key, subkey = jax.random.split(self.key)
        self.state, jax_obs = jax_env.reset(subkey)
        self._initialized = True
        return self._convert_observation(jax_obs)

    def step(self, action: int) -> StepResult:
        """Execute an action in the environment."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.key, subkey = jax.random.split(self.key)
        self.state, jax_obs, reward, done = jax_env.step(
            self.state, jnp.int32(action), subkey
        )

        return StepResult(
            observation=self._convert_observation(jax_obs),
            reward=float(reward),
            done=bool(done),
            info={}
        )

    def get_valid_actions(self) -> list[int]:
        """Get list of valid action indices for current state."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        mask = jax_env.get_valid_actions(self.state)
        return [i for i, valid in enumerate(np.array(mask)) if valid]

    def set_state(self, state: GameState) -> Observation:
        """Set the complete game state for test setup.

        STUB: The JAX environment doesn't support state injection yet.
        This method raises NotImplementedError.

        When the full JAX implementation is complete, this will:
        1. Convert GameState to JAX EnvState
        2. Build the observation from the state
        """
        raise NotImplementedError(
            "JAX environment does not yet support set_state(). "
            "Tests requiring set_state() should use the Swift environment only."
        )

    # MARK: - Cleanup

    def close(self):
        """Clean up resources (no-op for JAX)."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
