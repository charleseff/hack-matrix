# Manual Play Monitor Plan

## Goal
Enable manual gameplay monitoring using existing `--visual-cli` mode with zero Swift changes. Extract observation parsing for reuse and add more debug scenarios.

## Key Insight
**`--visual-cli` already does everything we need!**
- Shows GUI ✅
- Accepts keyboard/mouse input ✅
- Sends observations to stdout after each action ✅
- Uses exact same encoding as training ✅

We just need Python to read stdout without writing to stdin.

## Phase 1: Modify manual_play.py (Read-Only Mode)

**File: `python/scripts/manual_play.py`**

Change from interactive stdin/stdout to passive stdout monitoring with action space display:

```python
"""
Monitor manual gameplay and display observation space in real-time.

Launches the game with --visual-cli and monitors observations as you play.
No commands needed - just play with keyboard/mouse!
"""

import subprocess
import json
import sys
from pathlib import Path


def monitor_gameplay(app_path: str, debug_scenario: bool = False):
    """Launch game and monitor observations from manual play."""

    print("="*80)
    print("MANUAL PLAY MONITOR")
    print("="*80)
    print("\nLaunching game...")
    print("Play with keyboard/mouse. Observations will appear here.\n")

    if debug_scenario:
        print("🎮 Debug scenario enabled - predictable starting state\n")

    print("="*80)

    # Build command
    cmd = [app_path, "--visual-cli"]
    if debug_scenario:
        cmd.append("--debug-scenario")

    # Launch Swift process
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    step = 0

    try:
        # Read stdout line by line (same as gym_env._send_command)
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            # Skip non-JSON lines (debug messages)
            if not line.startswith('{'):
                print(f"[Swift] {line}")
                continue

            try:
                data = json.loads(line)

                # Handle different response types
                if "observation" in data:
                    step += 1
                    print_observation(data, step)

                    # Query valid actions after each observation
                    request_valid_actions(process)

                elif "validActions" in data:
                    print_valid_actions(data["validActions"])

                elif "error" in data:
                    print(f"\n❌ Error: {data['error']}")

            except json.JSONDecodeError as e:
                print(f"[Parse Error] {line[:100]}")

    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")
    finally:
        process.terminate()
        process.wait()


def request_valid_actions(process):
    """Send getValidActions command to Swift."""
    command = json.dumps({"action": "getValidActions"}) + "\n"
    process.stdin.write(command)
    process.stdin.flush()


def print_observation(data: dict, step: int):
    """Pretty-print observation data.

    TODO: This should use the robust observation printer from Phase 2.
    For now, just basic output.
    """

    obs_raw = data.get("observation", {})
    reward = data.get("reward", 0)
    done = data.get("done", False)
    info = data.get("info", {})

    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print('='*80)

    # Player
    player = obs_raw.get("player", {})
    print(f"\n📍 Player: pos=({player.get('row')},{player.get('col')}) "
          f"hp={player.get('hp')} credits={player.get('credits')} energy={player.get('energy')}")

    # Reward
    print(f"💰 Reward: {reward:+.3f}")

    # Reward breakdown
    breakdown = info.get("reward_breakdown", {})
    nonzero = {k: v for k, v in breakdown.items() if v != 0}
    if nonzero:
        print("   Breakdown:", ", ".join(f"{k}:{v:+.2f}" for k, v in nonzero.items()))

    if done:
        print("\n🏁 EPISODE DONE\n")


def print_valid_actions(valid_actions: list):
    """Print valid actions in readable format (from current manual_play.py)."""

    action_names = {
        0: "Up(W)", 1: "Down(S)", 2: "Left(A)", 3: "Right(D)", 4: "Siphon"
    }
    # Add program names for 5-27
    for i in range(5, 28):
        action_names[i] = f"Prog{i}"

    valid_names = [action_names.get(a, f"Action{a}") for a in valid_actions]

    print(f"\n🎮 Valid Actions: {', '.join(valid_names[:8])}")  # Limit display
    if len(valid_names) > 8:
        print(f"                  ... and {len(valid_names) - 8} more")
    print(f"   Indices: {valid_actions}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor manual gameplay")
    parser.add_argument("--debug-scenario", action="store_true",
                       help="Use debug scenario starting state")
    args = parser.parse_args()

    # Path to Swift binary
    binary_path = "../.build/debug/HackMatrix"

    if not Path(binary_path).exists():
        print(f"❌ Binary not found: {binary_path}")
        print("Build first: swift build")
        sys.exit(1)

    monitor_gameplay(binary_path, debug_scenario=args.debug_scenario)
```

**Key Changes:**
- Launch with `--visual-cli` (existing mode)
- Read stdout only (no stdin writes)
- Parse JSON responses (same as gym_env)
- Simple pretty-print (Phase 2 will make this robust)

## Phase 2: Extract Observation Parsing/Printing

**File: `python/hackmatrix/observation_utils.py` (NEW)**

Extract the parsing logic from `gym_env.py` line 344+ into reusable functions:

```python
"""
Utilities for parsing and displaying observations.

Extracts the observation parsing/normalization logic from gym_env
for reuse in monitoring and debugging tools.
"""

import numpy as np
from typing import Dict, Any, Tuple


def parse_observation(obs_raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Parse raw observation dict from Swift into normalized arrays.

    This is the EXACT same logic as gym_env.step() lines 344-362.
    Extracted here for reuse in monitoring tools.

    Args:
        obs_raw: Raw observation dict from Swift JSON response

    Returns:
        Normalized observation dict with:
        - player: np.array (10 values, normalized [0,1])
        - programs: np.array (23 values, binary)
        - grid: np.array (6,6,40, normalized [0,1])
    """

    player_data = obs_raw["player"]

    # Normalize player state to [0, 1]
    player = np.array([
        player_data["row"] / 5.0,           # 0: row (0-5)
        player_data["col"] / 5.0,           # 1: col (0-5)
        player_data["hp"] / 3.0,            # 2: hp (0-3)
        player_data["credits"] / 100.0,     # 3: credits (normalize)
        player_data["energy"] / 100.0,      # 4: energy (normalize)
        player_data["stage"] / 8.0,         # 5: stage (1-8)
        player_data["dataSiphons"] / 5.0,   # 6: siphons (0-5)
        player_data["baseAttack"] / 3.0,    # 7: attack (1-3)
        1.0 if player_data["showActivated"] else 0.0,              # 8: show flag
        1.0 if player_data["scheduledTasksDisabled"] else 0.0,     # 9: calm flag
    ], dtype=np.float32)

    # Programs (binary)
    programs = np.array(obs_raw["programs"], dtype=np.int32)

    # Grid (already normalized by Swift)
    grid_data = obs_raw["grid"]
    grid = np.array([
        [[cell["features"][i] for i in range(40)]
         for cell in row]
        for row in grid_data
    ], dtype=np.float32)

    return {
        "player": player,
        "programs": programs,
        "grid": grid
    }


def print_observation_detailed(obs_raw: Dict[str, Any],
                               reward: float,
                               info: Dict[str, Any],
                               step: int = 0,
                               valid_actions: list = None):
    """Print detailed, human-readable observation data.

    This is the robust printer for validation and debugging.
    Shows ALL observation data in a clear format.

    Args:
        obs_raw: Raw observation dict from Swift
        reward: Total reward for this step
        info: Info dict with reward breakdown
        step: Current step number
        valid_actions: Optional list of valid action indices
    """

    print("\n" + "="*80)
    print(f"STEP {step:4d}")
    print("="*80)

    # Parse observation
    obs = parse_observation(obs_raw)

    # PLAYER STATE
    print("\n📍 PLAYER STATE")
    print("-" * 80)
    player = obs_raw["player"]  # Use raw for display
    print(f"  Position:  row={player['row']:2d}, col={player['col']:2d}")
    print(f"  Health:    {player['hp']}/3")
    print(f"  Resources: {player['credits']:3d} credits, {player['energy']:3d} energy")
    print(f"  Stage:     {player['stage']}/8  (Turn {player.get('turn', 0)})")
    print(f"  Combat:    {player['baseAttack']} attack, {player['dataSiphons']} siphons")
    print(f"  Flags:     Show={'ON' if player['showActivated'] else 'off'}, "
          f"Calm={'ON' if player['scheduledTasksDisabled'] else 'off'}")

    # Normalized values (what ML model sees)
    print(f"\n  Normalized: {obs['player'][:6]}")  # First 6 values

    # PROGRAMS
    print("\n📚 PROGRAMS")
    print("-" * 80)
    owned_indices = [i for i, owned in enumerate(obs['programs']) if owned == 1]
    print(f"  Owned: {len(owned_indices)}/23")
    if owned_indices:
        # Show in groups of 8
        for i in range(0, len(owned_indices), 8):
            chunk = owned_indices[i:i+8]
            print(f"    {chunk}")

    # GRID SUMMARY
    print("\n🗺️  GRID STATE")
    print("-" * 80)
    grid = obs['grid']

    # Count active channels
    active_channels = []
    for ch in range(40):
        if np.any(grid[:, :, ch] > 0):
            active_channels.append(ch)

    print(f"  Active channels: {len(active_channels)}/40")

    # Channel groups
    channel_groups = {
        "Enemies (0-5)": list(range(0, 6)),
        "Blocks (6-10)": list(range(6, 11)),
        "Programs (11-33)": list(range(11, 34)),
        "Transmissions (34-35)": list(range(34, 36)),
        "Resources (36-37)": list(range(36, 38)),
        "Special (38-39)": list(range(38, 40)),
    }

    for group_name, channels in channel_groups.items():
        active_in_group = [ch for ch in channels if ch in active_channels]
        if active_in_group:
            print(f"    {group_name}: {active_in_group}")

    # Grid cell details (entities present)
    enemies = []
    blocks = []
    for row in range(6):
        for col in range(6):
            # Check for enemy (channels 0-3 one-hot)
            if np.any(grid[row, col, 0:4] > 0):
                enemy_type = np.argmax(grid[row, col, 0:4])
                hp = grid[row, col, 4]
                enemies.append((row, col, enemy_type, hp))

            # Check for block (channels 6-8 one-hot)
            if np.any(grid[row, col, 6:9] > 0):
                block_type = np.argmax(grid[row, col, 6:9])
                blocks.append((row, col, block_type))

    if enemies:
        print(f"\n  Enemies: {len(enemies)}")
        for row, col, etype, hp in enemies[:5]:  # Limit to 5
            types = ['virus', 'daemon', 'glitch', 'cryptog']
            print(f"    ({row},{col}): {types[etype]}, hp={hp:.2f}")

    if blocks:
        print(f"\n  Blocks: {len(blocks)}")
        for row, col, btype in blocks[:5]:  # Limit to 5
            types = ['data', 'program', 'question']
            print(f"    ({row},{col}): {types[btype]}")

    # REWARD
    print("\n💰 REWARD")
    print("-" * 80)
    print(f"  Total: {reward:+.4f}")

    breakdown = info.get("reward_breakdown", {})
    nonzero = {k: v for k, v in breakdown.items() if v != 0}

    if nonzero:
        print("  Breakdown:")
        # Group by sign
        positive = {k: v for k, v in nonzero.items() if v > 0}
        negative = {k: v for k, v in nonzero.items() if v < 0}

        if positive:
            print("    ✓ Positive:")
            for k, v in positive.items():
                print(f"      {k:20s}: +{v:.4f}")

        if negative:
            print("    ✗ Negative:")
            for k, v in negative.items():
                print(f"      {k:20s}: {v:.4f}")

    # DONE FLAG
    if info.get("done", False):
        print("\n🏁 EPISODE DONE")

    # VALID ACTIONS
    if valid_actions is not None:
        print("\n🎮 VALID ACTIONS")
        print("-" * 80)

        action_names = {
            0: "Up(W)", 1: "Down(S)", 2: "Left(A)", 3: "Right(D)", 4: "Siphon"
        }
        # Add program names for 5-27
        for i in range(5, 28):
            action_names[i] = f"Prog{i}"

        valid_names = [action_names.get(a, f"Action{a}") for a in valid_actions]

        print(f"  Count: {len(valid_actions)}/28")
        print(f"  Actions: {', '.join(valid_names)}")
        print(f"  Indices: {valid_actions}")

    print("="*80)


def print_observation_compact(obs_raw: Dict[str, Any],
                              reward: float,
                              info: Dict[str, Any],
                              step: int = 0,
                              valid_actions: list = None):
    """Print compact one-line observation summary.

    Useful for watching many steps quickly.
    """

    player = obs_raw["player"]
    breakdown = info.get("reward_breakdown", {})
    nonzero = {k: v for k, v in breakdown.items() if v != 0}

    # Build compact string
    parts = [
        f"Step {step:4d}",
        f"pos=({player['row']},{player['col']})",
        f"hp={player['hp']}",
        f"cr={player['credits']:3d}",
        f"en={player['energy']:3d}",
        f"R={reward:+.3f}",
    ]

    if valid_actions is not None:
        parts.append(f"actions={len(valid_actions)}")

    if nonzero:
        breakdown_str = " ".join(f"{k[:3]}:{v:+.1f}" for k, v in list(nonzero.items())[:3])
        parts.append(f"[{breakdown_str}]")

    print("  |  ".join(parts))
```

**Then update `gym_env.py` to use this:**

```python
# In gym_env.py, line 344:
from hackmatrix.observation_utils import parse_observation

# In step() method:
obs = parse_observation(obs_raw)  # Instead of inline parsing
```

**Then update `manual_play.py` to use it:**

```python
from hackmatrix.observation_utils import print_observation_detailed, print_observation_compact

# Store the last valid actions to pass to printer
last_valid_actions = None

def print_observation(data: dict, step: int):
    """Use the robust printer with valid actions."""
    global last_valid_actions
    print_observation_detailed(
        obs_raw=data["observation"],
        reward=data["reward"],
        info=data["info"],
        step=step,
        valid_actions=last_valid_actions
    )

def print_valid_actions(valid_actions: list):
    """Store valid actions for next observation print."""
    global last_valid_actions
    last_valid_actions = valid_actions
    # Don't print here - will be printed with observation
```

## Phase 3: Add More Debug Scenarios

**File: `HackMatrix/GameState.swift`**

Currently only has one `createDebugScenario()`. Add more:

```swift
// MARK: - Debug Scenarios

/// Create a debug scenario for testing
static func createDebugScenario() -> GameState {
    // Default: existing scenario
    return createDebugScenario(type: .basic)
}

enum DebugScenarioType {
    case basic           // Current default
    case enemyTypes      // One of each enemy type
    case blockTypes      // Different block types and siphoning
    case programs        // Program acquisition and usage
    case transmissions   // Transmission spawning
    case combat          // Combat and damage
    case resources       // Credits and energy
    case stageTransition // Near-exit, stage advancement
}

static func createDebugScenario(type: DebugScenarioType) -> GameState {
    let state = GameState()

    switch type {
    case .basic:
        // Current scenario (existing code)
        state.player.row = 2
        state.player.col = 2
        // ... existing basic scenario

    case .enemyTypes:
        // Spawn one of each enemy type in corners
        state.enemies = [
            Enemy(type: .virus, row: 0, col: 0),
            Enemy(type: .daemon, row: 0, col: 5),
            Enemy(type: .glitch, row: 5, col: 0),
            Enemy(type: .cryptog, row: 5, col: 5),
        ]
        state.player.row = 2
        state.player.col = 2

    case .blockTypes:
        // Place different block types for siphoning
        state.grid.cells[1][1].content = .block(.data(points: 10, transmissionSpawn: 0))
        state.grid.cells[1][2].content = .block(.program(Program(type: .scan), transmissionSpawn: 0))
        state.grid.cells[1][3].content = .block(.question(isData: true, points: 5, program: nil, transmissionSpawn: 1))

        state.grid.cells[2][1].resources = .credits(10)
        state.grid.cells[2][2].resources = .energy(5)

        state.player.row = 2
        state.player.col = 2
        state.player.dataSiphons = 3

    case .programs:
        // Start with some programs owned
        state.ownedPrograms = [
            Program(type: .scan),
            Program(type: .shield),
            Program(type: .overclock),
        ]
        // Place program blocks nearby
        state.grid.cells[1][1].content = .block(.program(Program(type: .reset), transmissionSpawn: 0))
        state.grid.cells[1][2].content = .block(.program(Program(type: .stun), transmissionSpawn: 0))

        state.player.row = 2
        state.player.col = 2
        state.player.dataSiphons = 2

    case .transmissions:
        // Spawn transmissions at different stages
        state.transmissions = [
            Transmission(row: 0, col: 0, turnsUntilSpawn: 1, enemyType: .virus),
            Transmission(row: 0, col: 5, turnsUntilSpawn: 2, enemyType: .daemon),
            Transmission(row: 5, col: 0, turnsUntilSpawn: 3, enemyType: .glitch),
        ]
        state.player.row = 2
        state.player.col = 2

    case .combat:
        // Player near enemies, low HP
        state.player.row = 2
        state.player.col = 2
        state.player.health = .twoHP

        state.enemies = [
            Enemy(type: .virus, row: 2, col: 3),  // Adjacent
            Enemy(type: .daemon, row: 3, col: 2), // Adjacent
        ]

        // Place RESET program for healing
        state.grid.cells[1][1].content = .block(.program(Program(type: .reset), transmissionSpawn: 0))
        state.player.dataSiphons = 1

    case .resources:
        // Lots of resources scattered
        state.grid.cells[1][1].resources = .credits(25)
        state.grid.cells[1][3].resources = .credits(15)
        state.grid.cells[3][1].resources = .energy(20)
        state.grid.cells[3][3].resources = .energy(10)

        state.player.row = 2
        state.player.col = 2
        state.player.credits = 50
        state.player.energy = 30

    case .stageTransition:
        // Near exit, ready to advance
        state.grid.cells[5][5].content = .exit
        state.player.row = 4
        state.player.col = 5
        state.currentStage = 2
    }

    return state
}
```

**Add CLI parameter to select scenario:**

```swift
// In App.swift or main arg parsing:
if CommandLine.arguments.contains("--debug-scenario") {
    // Check for scenario type
    if let typeIndex = CommandLine.arguments.firstIndex(of: "--debug-scenario"),
       typeIndex + 1 < CommandLine.arguments.count {
        let typeArg = CommandLine.arguments[typeIndex + 1]

        let scenarioType: GameState.DebugScenarioType
        switch typeArg {
        case "enemy-types": scenarioType = .enemyTypes
        case "blocks": scenarioType = .blockTypes
        case "programs": scenarioType = .programs
        case "transmissions": scenarioType = .transmissions
        case "combat": scenarioType = .combat
        case "resources": scenarioType = .resources
        case "stage": scenarioType = .stageTransition
        default: scenarioType = .basic
        }

        gameState = GameState.createDebugScenario(type: scenarioType)
    } else {
        gameState = GameState.createDebugScenario()  // Default
    }
}
```

**Usage:**

```bash
# Test different scenarios
python manual_play.py --debug-scenario enemy-types
python manual_play.py --debug-scenario blocks
python manual_play.py --debug-scenario programs
```

## Phase 4: Update CLAUDE.md

Add section explaining the workflow:

```markdown
### Validating Observation Space (Manual Play Monitoring)

Test that observations are correctly encoded/decoded during manual gameplay:

```bash
cd python
source venv/bin/activate
python scripts/manual_play.py

# Or use a debug scenario for specific testing:
python scripts/manual_play.py --debug-scenario blocks
```

The monitor shows the EXACT observation data fed to the ML model, using the same parsing code as training.

**Available Debug Scenarios:**
- `basic` - Default scenario (enemies, blocks, resources)
- `enemy-types` - One of each enemy type
- `blocks` - Different block types for siphoning
- `programs` - Program acquisition and usage
- `transmissions` - Transmission spawning states
- `combat` - Combat situations, low HP
- `resources` - Resource collection
- `stage` - Stage transition testing
```

## Summary

**Phase 1:** Modify manual_play.py (~50 lines changed)
- Read stdout from `--visual-cli` mode
- Request valid actions after each observation
- Display both observation AND action space

**Phase 2:** Extract observation utils (~250 lines new)
- Parse observations (extracted from gym_env)
- Detailed printer showing all observation data
- Valid actions display
- Compact printer for quick monitoring

**Phase 3:** Add debug scenarios (~150 lines new)
- 7 new scenario types for testing edge cases
- CLI parameter to select scenarios

**Phase 4:** Update docs (~20 lines)
- Document workflow in CLAUDE.md

**Total Swift changes:** ~150 lines (all in GameState.swift for scenarios)
**Total Python changes:** ~300 lines (manual_play.py + observation_utils.py)
**Zero duplication:** Reuses all existing encoding/decoding

**What Gets Monitored:**
- ✅ Observation space (player, programs, grid) - 6×6×40 channels
- ✅ Action space (valid actions mask) - 28 actions
- ✅ Reward breakdown (all 14 components)
- ✅ Game state changes in real-time

## Files Modified

- `python/scripts/manual_play.py` - Change to read-only monitoring
- `python/hackmatrix/observation_utils.py` - NEW (extracted from gym_env)
- `python/hackmatrix/gym_env.py` - Import and use observation_utils
- `HackMatrix/GameState.swift` - Add more debug scenarios
- `HackMatrix/App.swift` - Add scenario selection
- `CLAUDE.md` - Document usage

## Testing

1. Build Swift app
2. Run: `python scripts/manual_play.py`
3. Play game with keyboard/mouse
4. Verify detailed observation output appears
5. Try different debug scenarios
6. Verify all observation channels activate
