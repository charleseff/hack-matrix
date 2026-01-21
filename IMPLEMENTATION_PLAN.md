# Implementation Plan

**Current Focus:** CI Setup (specs/ci-setup.md)

## Completed Work

### Test Reorganization (Complete)
- ✅ pyproject.toml created with pytest configuration
- ✅ run_all_tests.py deleted
- ✅ Tests reorganized into parity/ and implementation/ subdirectories
- ✅ All Swift tests pass (182 tests)
- ✅ Scheduled task parity tests added
- ✅ get_internal_state() added to interface (Swift and Python)
- ✅ Implementation-level tests for hidden state (test_scheduled_task_internals.py)

### Bug Fixes Applied
- Fixed `test_entities_spawn_over_time` to use `get_internal_state()` for enemy counting
  - **Why:** Cryptogs (25% of scheduled spawns) are invisible in observations when outside player's row/column. Using internal state ensures we count all enemies regardless of visibility.

## Next Steps

1. **CI Setup** (specs/ci-setup.md - Draft)
   - Set up GitHub Actions to run Swift and Python tests
   - Open questions to resolve:
     - Swift version pinning strategy
     - Job structure (single vs multi-job)
     - Caching strategy

2. **JAX Implementation** (specs/jax-implementation.md - Deferred)
   - Will be done after CI is set up
   - Full port of game logic to JAX for TPU training

## Test Status

- **Swift tests:** 182 passed
- **JAX tests:** 144 failed (expected - JAX env doesn't support set_state() yet, deferred to jax-implementation spec)
