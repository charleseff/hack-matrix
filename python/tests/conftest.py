"""
Pytest configuration and fixtures for environment parity tests.

This module provides fixtures for both Swift and JAX environments,
enabling tests to run against either or both implementations.

Why this design:
- swift_env fixture: Primary test target, uses real game logic
- jax_env fixture: Secondary target, currently returns dummy data
- Fixtures handle setup/teardown automatically via context managers
"""

import pytest
from typing import Generator

from .swift_env_wrapper import SwiftEnvWrapper
from .jax_env_wrapper import JaxEnvWrapper


# MARK: - Swift Environment Fixtures

@pytest.fixture
def swift_env() -> Generator[SwiftEnvWrapper, None, None]:
    """
    Provide a Swift environment wrapper for testing.

    The Swift environment is the reference implementation - all tests
    should pass against it. Uses set_state() for deterministic test setup.

    Yields:
        SwiftEnvWrapper instance that implements EnvInterface
    """
    with SwiftEnvWrapper() as env:
        yield env


@pytest.fixture
def swift_env_debug() -> Generator[SwiftEnvWrapper, None, None]:
    """
    Provide a Swift environment with debug logging enabled.

    Use this fixture when debugging test failures to see Swift's
    internal state transitions.

    Yields:
        SwiftEnvWrapper instance with debug=True
    """
    with SwiftEnvWrapper(debug=True) as env:
        yield env


# MARK: - JAX Environment Fixtures

@pytest.fixture
def jax_env() -> Generator[JaxEnvWrapper, None, None]:
    """
    Provide a JAX environment wrapper for testing.

    The JAX environment is currently a stub returning dummy data.
    Tests using set_state() will fail until JAX implementation is complete.

    Yields:
        JaxEnvWrapper instance that implements EnvInterface
    """
    with JaxEnvWrapper() as env:
        yield env


# MARK: - Parametrized Fixtures

@pytest.fixture(params=["swift"])
def env(request) -> Generator[SwiftEnvWrapper | JaxEnvWrapper, None, None]:
    """
    Parametrized fixture for running tests against Swift only.

    Use this fixture for comprehensive tests that require set_state().
    JAX is excluded because it doesn't support set_state() yet.

    When JAX implementation is complete, add "jax" to params.
    """
    if request.param == "swift":
        with SwiftEnvWrapper() as env:
            yield env
    # Future: Add JAX when set_state() is implemented
    # elif request.param == "jax":
    #     with JaxEnvWrapper() as env:
    #         yield env


@pytest.fixture(params=["swift", "jax"])
def env_smoke(request) -> Generator[SwiftEnvWrapper | JaxEnvWrapper, None, None]:
    """
    Parametrized fixture for smoke tests against both environments.

    Use this for interface compliance tests that don't require set_state().
    These tests verify reset(), step(), get_valid_actions() work correctly.

    Yields:
        Either SwiftEnvWrapper or JaxEnvWrapper
    """
    if request.param == "swift":
        with SwiftEnvWrapper() as env:
            yield env
    elif request.param == "jax":
        with JaxEnvWrapper() as env:
            yield env


# MARK: - Test Markers

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "swift_only: mark test to run only with Swift environment"
    )
    config.addinivalue_line(
        "markers", "jax_only: mark test to run only with JAX environment"
    )
    config.addinivalue_line(
        "markers", "requires_set_state: mark test that requires set_state() support"
    )


# MARK: - Test Collection Hooks

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip JAX set_state tests automatically
    skip_jax_set_state = pytest.mark.skip(
        reason="JAX environment doesn't support set_state() yet"
    )

    for item in items:
        # If test requires set_state and uses JAX fixture, skip it
        if "requires_set_state" in [m.name for m in item.iter_markers()]:
            if "jax" in item.fixturenames or (
                hasattr(item, 'callspec') and
                item.callspec.params.get('env_smoke') == 'jax'
            ):
                item.add_marker(skip_jax_set_state)
