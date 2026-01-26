# HackMatrix

A turn-based tactical game on a 6x6 grid, designed for reinforcement learning research.

## Overview

HackMatrix is a roguelike strategy game where players navigate a grid, fight enemies, collect resources, and execute programs. The game is written in Swift with a Python RL training stack.

**Key features:**
- 6x6 grid with enemies, blocks, transmissions, and resources
- 28-action space: 4 movement directions, siphon, and 23 programs
- Turn-based combat with line-of-sight attacks
- Stage progression with procedural generation

## Architecture

```
Swift (Game Logic)          Python (RL Training)
├── GameState.swift         ├── hack_env.py (Gymnasium wrapper)
├── HeadlessGameCLI.swift   ├── scripts/train.py (SB3 PPO)
└── GameScene.swift (GUI)   └── scripts/train_purejaxrl.py (JAX PPO)
```

**Training modes:**
- **Swift subprocess**: Python communicates via JSON stdin/stdout
- **PureJaxRL**: Full JAX implementation for TPU training (Colab/TRC)

## Quick Start

### Prerequisites

- macOS with Xcode (for GUI)
- Python 3.11+
- Dev Container support (for Linux/headless training)

### Build & Run

```bash
# Build headless (SPM)
swift build

# Build GUI (Xcode)
xcodebuild -scheme HackMatrix -configuration Debug build

# Run tests
swift test
```

### Training

```bash
cd python && source venv/bin/activate

# Test environment
python scripts/test_env.py

# Train with Stable Baselines3
python scripts/train.py --timesteps 1000000

# Train with PureJaxRL (for TPU)
python scripts/train_purejaxrl.py --wandb
```

## Project Structure

```
HackMatrix/          # Swift source (game logic, GUI, CLI)
python/
├── hackmatrix/      # Python package (env, JAX implementation)
├── scripts/         # Training and utility scripts
└── tests/           # Python test suite
specs/               # Design documents and implementation specs
Tests/               # Swift test suite
```

## Documentation

- [CLAUDE.md](./CLAUDE.md) - Development guide and architecture details
- [specs/](./specs/) - Design specs (see [specs/README.md](./specs/README.md))
- [python/README.md](./python/README.md) - RL training documentation

## License

Private repository.
