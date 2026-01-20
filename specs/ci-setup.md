# CI Setup Spec

## Goal

Set up GitHub Actions CI to run all tests (Swift and Python) on every push and pull request.

## Background

The project has two test suites:
- **Swift tests**: Run via `swift test`, test game logic directly
- **Python tests**: Run via `pytest tests/`, test the Python-Swift bridge and environment behavior

Python tests depend on the Swift binary being built first (`swift build` produces the headless CLI).

## Current Test Commands

```bash
# Swift
swift test

# Python (requires Swift binary)
cd python && source venv/bin/activate && pytest tests/ -v
```

## Requirements

### Must Have

1. Run Swift tests on push/PR
2. Run Python tests on push/PR
3. Fail the build if any tests fail
4. Work on ubuntu-latest (matches dev container)

### Nice to Have

1. Caching for faster builds (Swift packages, pip dependencies)
2. Test result annotations in PR
3. Parallel jobs where possible
4. Badge for README

## Open Questions

> These need to be resolved before implementation.

### 1. Swift Version Pinning

How should we pin the Swift version?
- **Option A**: Use `swift:5.10` Docker image (matches dev container)
- **Option B**: Use `swiftlang/swift` action with version parameter
- **Option C**: Install via `apt` with specific version

Current dev container uses: `swift:5.10-jammy`

### 2. Python Version

Which Python version(s) to test?
- **Option A**: Single version matching dev container (3.11)
- **Option B**: Matrix of versions (3.10, 3.11, 3.12)

### 3. Job Structure

How to structure the workflow?
- **Option A**: Single job - simpler, but slower (sequential)
- **Option B**: Two jobs with artifact passing - parallel Swift tests, Python waits for binary
- **Option C**: Three jobs - Swift build, Swift tests, Python tests (max parallelism)

### 4. Caching Strategy

What to cache?
- Swift Package Manager cache (`~/.cache/org.swift.swiftpm`)
- Python venv or pip cache
- Built Swift binary (if using multi-job)

### 5. Branch Protection

Should we require CI to pass before merging to main?
- Requires repo settings change
- What about the existing `test-reorganization.md` work?

### 6. Dev Container Reuse

Should CI use the dev container definition?
- **Option A**: Yes - use `devcontainers/ci` action, guarantees parity
- **Option B**: No - faster to install dependencies directly

## Proposed Workflow Structure

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  swift-build:
    runs-on: ubuntu-latest
    # Build Swift and upload binary artifact

  swift-test:
    runs-on: ubuntu-latest
    # Run swift test

  python-test:
    runs-on: ubuntu-latest
    needs: swift-build
    # Download binary, run pytest
```

## Dependencies

- Ideally complete after `test-reorganization.md` (so CI runs reorganized tests)
- Can be done in parallel if needed

## Success Criteria

1. [ ] GitHub Actions workflow file created
2. [ ] Swift tests run and pass in CI
3. [ ] Python tests run and pass in CI
4. [ ] CI runs on push to main and on PRs
5. [ ] Caching configured for reasonable build times
6. [ ] Open questions resolved

## References

- [GitHub Actions Swift setup](https://github.com/swift-actions/setup-swift)
- [Dev Container CI action](https://github.com/devcontainers/ci)
- Current dev container: `.devcontainer/Dockerfile`
