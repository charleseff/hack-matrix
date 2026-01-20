# Agent Operational Notes

## Building

```bash
# Headless CLI (for training)
swift build

# Binary location after build
.build/debug/HackMatrix
```

## Running Tests

```bash
# Set the binary path (required if HACKMATRIX_BINARY is set to wrong path)
export HACKMATRIX_BINARY=/workspaces/868-hack-2/.build/debug/HackMatrix

# Run smoke tests (without pytest)
cd python && source venv/bin/activate && python3 tests/run_smoke_tests.py

# Run with pytest (once network is available to install pytest)
cd python && source venv/bin/activate && pytest tests/ -v
```

## Environment Variables

- `HACKMATRIX_BINARY` - Path to Swift binary. Set to `.build/debug/HackMatrix` for debug builds.

## Known Issues

- Network unreachable in dev container - cannot install pytest via pip. Use `run_smoke_tests.py` instead.
