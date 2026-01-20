"""
Compatibility layer for running tests without pytest.

This module provides a minimal pytest-like interface that allows test files
to import and use @pytest.mark decorators without having pytest installed.

Usage in test files:
    try:
        import pytest
    except ImportError:
        from . import pytest_compat as pytest
"""


class _Mark:
    """Compatibility marker that does nothing."""

    def __init__(self, name=None):
        self.name = name

    def __call__(self, *args, **kwargs):
        """When used as decorator @pytest.mark.something"""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Used as @pytest.mark.something without args
            return args[0]
        # Used as @pytest.mark.something(args)
        return lambda fn: fn

    def __getattr__(self, name):
        """Support @pytest.mark.arbitrary_name"""
        return _Mark(name)


class _Pytest:
    """Minimal pytest compatibility."""
    mark = _Mark()


# Global instance
mark = _Pytest.mark


def raises(exception_type):
    """Context manager for expecting exceptions."""
    class RaisesContext:
        def __init__(self, exception_type):
            self.exception_type = exception_type
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exception_type.__name__} but no exception was raised")
            if not issubclass(exc_type, self.exception_type):
                return False  # Re-raise the exception
            self.value = exc_val
            return True  # Suppress the expected exception

    return RaisesContext(exception_type)
