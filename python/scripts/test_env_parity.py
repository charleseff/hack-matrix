"""
Test that JAX and Swift environments have compatible interfaces.

NOTE: These tests verify structural compatibility (shapes, types, formats),
not behavioral equivalence. The Swift env doesn't support seeding yet, and
both implementations have inherent randomness, so exact value comparisons
are not possible at this stage.

Run with: cd python && source venv/bin/activate && python scripts/test_env_parity.py
"""

import sys
from pathlib import Path

# Add python directory to path for hackmatrix import
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Protocol


# ---------------------------------------------------------------------------
# Common test interface
# ---------------------------------------------------------------------------


class EnvAdapter(Protocol):
    """Minimal interface for parity testing."""

    def reset(self) -> tuple[dict, list[int]]:
        """Returns (observation_dict, valid_actions)"""
        ...

    def step(self, action: int) -> tuple[dict, float, bool, list[int]]:
        """Returns (observation_dict, reward, done, valid_actions)"""
        ...


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class SwiftEnvAdapter:
    """Adapter for Swift-backed HackEnv."""

    def __init__(self):
        from hackmatrix.gym_env import HackEnv

        self.env = HackEnv()

    def reset(self) -> tuple[dict, list[int]]:
        obs, _ = self.env.reset()
        valid = self.env.get_valid_actions()
        return obs, valid

    def step(self, action: int) -> tuple[dict, float, bool, list[int]]:
        obs, reward, done, _, _ = self.env.step(action)
        valid = self.env.get_valid_actions()
        return obs, reward, done, valid

    def close(self):
        self.env.close()


class JaxEnvAdapter:
    """Adapter for pure JAX environment."""

    def __init__(self):
        import jax
        from hackmatrix import jax_env

        self.jax = jax
        self.env = jax_env
        self.state = None
        self.key = jax.random.PRNGKey(0)

    def reset(self) -> tuple[dict, list[int]]:
        self.key, subkey = self.jax.random.split(self.key)
        self.state, obs = self.env.reset(subkey)
        valid_mask = self.env.get_valid_actions(self.state)
        return self._obs_to_dict(obs), self._mask_to_list(valid_mask)

    def step(self, action: int) -> tuple[dict, float, bool, list[int]]:
        self.key, subkey = self.jax.random.split(self.key)
        self.state, obs, reward, done = self.env.step(self.state, action, subkey)
        valid_mask = self.env.get_valid_actions(self.state)
        return (
            self._obs_to_dict(obs),
            float(reward),
            bool(done),
            self._mask_to_list(valid_mask),
        )

    def _obs_to_dict(self, obs) -> dict:
        return {
            "player": np.asarray(obs.player_state),
            "programs": np.asarray(obs.programs),
            "grid": np.asarray(obs.grid),
        }

    def _mask_to_list(self, mask) -> list[int]:
        return [i for i, v in enumerate(np.asarray(mask)) if v]

    def close(self):
        pass  # JAX env has no resources to clean up


# ---------------------------------------------------------------------------
# JAX-only tests (no Swift dependency)
# ---------------------------------------------------------------------------


def test_jax_observation_shapes():
    """Verify JAX env returns correct observation shapes."""
    jax_adapter = JaxEnvAdapter()
    jax_obs, _ = jax_adapter.reset()

    expected_shapes = {
        "player": (10,),
        "programs": (23,),
        "grid": (6, 6, 40),
    }

    for key, expected_shape in expected_shapes.items():
        actual_shape = jax_obs[key].shape
        assert (
            actual_shape == expected_shape
        ), f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}"

    print("[PASS] JAX observation shapes correct")
    jax_adapter.close()


def test_jax_observation_dtypes():
    """Verify JAX env returns correct observation dtypes."""
    jax_adapter = JaxEnvAdapter()
    jax_obs, _ = jax_adapter.reset()

    expected_dtypes = {
        "player": np.float32,
        "programs": np.int32,
        "grid": np.float32,
    }

    for key, expected_dtype in expected_dtypes.items():
        actual_dtype = jax_obs[key].dtype
        assert (
            actual_dtype == expected_dtype
        ), f"Dtype mismatch for {key}: expected {expected_dtype}, got {actual_dtype}"

    print("[PASS] JAX observation dtypes correct")
    jax_adapter.close()


def test_jax_valid_actions_format():
    """Verify JAX valid actions are returned in correct format."""
    jax_adapter = JaxEnvAdapter()
    _, jax_valid = jax_adapter.reset()

    assert isinstance(jax_valid, list), "JAX valid_actions should be list"
    assert all(isinstance(a, int) for a in jax_valid), "JAX actions should be ints"
    assert all(
        0 <= a < 28 for a in jax_valid
    ), "JAX actions should be in range 0-27"
    assert jax_valid == [0, 1, 2, 3], f"Expected [0,1,2,3], got {jax_valid}"

    print("[PASS] JAX valid actions format correct")
    jax_adapter.close()


def test_jax_step_return_types():
    """Verify JAX step() returns correct types."""
    jax_adapter = JaxEnvAdapter()
    _, jax_valid = jax_adapter.reset()

    jax_action = jax_valid[0] if jax_valid else 0
    jax_obs, jax_reward, jax_done, jax_valid = jax_adapter.step(jax_action)

    assert isinstance(
        jax_reward, float
    ), f"JAX reward should be float, got {type(jax_reward)}"
    assert isinstance(
        jax_done, bool
    ), f"JAX done should be bool, got {type(jax_done)}"
    assert isinstance(jax_obs, dict), f"JAX obs should be dict, got {type(jax_obs)}"

    print("[PASS] JAX step return types correct")
    jax_adapter.close()


def test_jax_batched_operations():
    """Verify batched JAX operations work correctly."""
    import jax
    from hackmatrix import jax_env

    batch_size = 4
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    # Test batched reset
    states, obs = jax_env.batched_reset(keys)
    assert obs.player_state.shape == (
        batch_size,
        10,
    ), f"Expected ({batch_size}, 10), got {obs.player_state.shape}"
    assert obs.programs.shape == (
        batch_size,
        23,
    ), f"Expected ({batch_size}, 23), got {obs.programs.shape}"
    assert obs.grid.shape == (
        batch_size,
        6,
        6,
        40,
    ), f"Expected ({batch_size}, 6, 6, 40), got {obs.grid.shape}"

    # Test batched step
    actions = jax.numpy.zeros(batch_size, dtype=jax.numpy.int32)
    step_keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    new_states, new_obs, rewards, dones = jax_env.batched_step(
        states, actions, step_keys
    )

    assert rewards.shape == (batch_size,), f"Expected ({batch_size},), got {rewards.shape}"
    assert dones.shape == (batch_size,), f"Expected ({batch_size},), got {dones.shape}"

    # Test batched get_valid_actions
    valid_masks = jax_env.batched_get_valid_actions(new_states)
    assert valid_masks.shape == (
        batch_size,
        28,
    ), f"Expected ({batch_size}, 28), got {valid_masks.shape}"

    print("[PASS] JAX batched operations correct")


# ---------------------------------------------------------------------------
# Interface parity tests (require Swift env)
# ---------------------------------------------------------------------------


def test_observation_shapes_parity():
    """Verify both envs return same observation shapes."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    swift_obs, _ = swift.reset()
    jax_obs, _ = jax_adapter.reset()

    # Map JAX keys to Swift keys
    key_mapping = {
        "player": "player",
        "programs": "programs",
        "grid": "grid",
    }

    for jax_key, swift_key in key_mapping.items():
        swift_shape = swift_obs[swift_key].shape
        jax_shape = jax_obs[jax_key].shape
        assert (
            swift_shape == jax_shape
        ), f"Shape mismatch for {jax_key}: Swift {swift_shape} vs JAX {jax_shape}"

    print("[PASS] Observation shapes match between Swift and JAX")
    swift.close()
    jax_adapter.close()


def test_observation_dtypes_parity():
    """Verify both envs return same observation dtypes."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    swift_obs, _ = swift.reset()
    jax_obs, _ = jax_adapter.reset()

    key_mapping = {
        "player": "player",
        "programs": "programs",
        "grid": "grid",
    }

    for jax_key, swift_key in key_mapping.items():
        swift_dtype = swift_obs[swift_key].dtype
        jax_dtype = jax_obs[jax_key].dtype
        assert (
            swift_dtype == jax_dtype
        ), f"Dtype mismatch for {jax_key}: Swift {swift_dtype} vs JAX {jax_dtype}"

    print("[PASS] Observation dtypes match between Swift and JAX")
    swift.close()
    jax_adapter.close()


def test_valid_actions_format_parity():
    """Verify valid actions are returned in same format."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    _, swift_valid = swift.reset()
    _, jax_valid = jax_adapter.reset()

    assert isinstance(swift_valid, list), "Swift valid_actions should be list"
    assert isinstance(jax_valid, list), "JAX valid_actions should be list"
    assert all(
        isinstance(a, int) for a in swift_valid
    ), "Swift actions should be ints"
    assert all(isinstance(a, int) for a in jax_valid), "JAX actions should be ints"
    assert all(
        0 <= a < 28 for a in swift_valid
    ), "Swift actions should be in range 0-27"
    assert all(
        0 <= a < 28 for a in jax_valid
    ), "JAX actions should be in range 0-27"

    print("[PASS] Valid actions format matches between Swift and JAX")
    swift.close()
    jax_adapter.close()


def test_step_return_types_parity():
    """Verify step() returns correct types in both envs."""
    swift = SwiftEnvAdapter()
    jax_adapter = JaxEnvAdapter()

    _, swift_valid = swift.reset()
    _, jax_valid = jax_adapter.reset()

    swift_action = swift_valid[0] if swift_valid else 0
    jax_action = jax_valid[0] if jax_valid else 0

    swift_obs, swift_reward, swift_done, swift_valid = swift.step(swift_action)
    jax_obs, jax_reward, jax_done, jax_valid = jax_adapter.step(jax_action)

    assert isinstance(
        swift_reward, float
    ), f"Swift reward should be float, got {type(swift_reward)}"
    assert isinstance(
        jax_reward, float
    ), f"JAX reward should be float, got {type(jax_reward)}"
    assert isinstance(
        swift_done, bool
    ), f"Swift done should be bool, got {type(swift_done)}"
    assert isinstance(
        jax_done, bool
    ), f"JAX done should be bool, got {type(jax_done)}"

    print("[PASS] Step return types match between Swift and JAX")
    swift.close()
    jax_adapter.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_jax_only_tests():
    """Run tests that only require JAX (no Swift binary needed)."""
    print("\n=== JAX-only tests ===\n")
    test_jax_observation_shapes()
    test_jax_observation_dtypes()
    test_jax_valid_actions_format()
    test_jax_step_return_types()
    test_jax_batched_operations()
    print("\nAll JAX-only tests passed!")


def run_parity_tests():
    """Run tests that compare JAX and Swift environments."""
    print("\n=== Swift-JAX parity tests ===\n")
    test_observation_shapes_parity()
    test_observation_dtypes_parity()
    test_valid_actions_format_parity()
    test_step_return_types_parity()
    print("\nAll parity tests passed!")


if __name__ == "__main__":
    # Always run JAX-only tests
    run_jax_only_tests()

    # Only run parity tests if --parity flag is provided
    # (requires Swift binary which may not be available in all environments)
    if "--parity" in sys.argv:
        run_parity_tests()
    else:
        print("\nNote: Skipping Swift-JAX parity tests.")
        print("Run with --parity flag to include them (requires Swift binary).")
