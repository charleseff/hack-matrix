# Building HackMatrix

HackMatrix uses a hybrid build approach:
- **SPM** (`swift build`): Headless CLI for ML training (macOS + Linux)
- **Xcode** (`xcodebuild`): Full GUI app (macOS only)

## Quick Start

```bash
# Training (headless CLI)
swift build
.build/debug/HackMatrix --headless-cli

# GUI app (macOS)
xcodebuild -scheme HackMatrix -configuration Debug build
open DerivedData/HackMatrix/Build/Products/Debug/HackMatrix.app
```

## Python Integration

Python automatically selects the right binary:
- **Training** (headless): Uses SPM build at `.build/debug/HackMatrix`
- **Visual mode**: Uses Xcode build at `DerivedData/.../HackMatrix.app`

```bash
# Training uses SPM build
cd python && source venv/bin/activate
python scripts/train.py

# Manual play uses Xcode build (GUI)
python scripts/manual_play.py
```

## Linux (Docker)

```bash
# Build headless CLI
docker run --rm -v "$(pwd)":/workspace -w /workspace swift:6.0.3-jammy swift build

# Test
docker run --rm -v "$(pwd)":/workspace -w /workspace swift:6.0.3-jammy \
  bash -c 'echo "{\"action\": \"reset\"}" | .build/debug/HackMatrix'
```

## Architecture

| Component | SPM (swift build) | Xcode |
|-----------|-------------------|-------|
| Game logic | ✓ | ✓ |
| Headless CLI | ✓ | ✓ |
| SwiftUI GUI | - | ✓ |
| SpriteKit rendering | - | ✓ |
| .app bundle | - | ✓ |

SPM excludes `App.swift` and GUI code via conditional compilation. Xcode includes everything and produces a proper .app bundle for macOS GUI.
