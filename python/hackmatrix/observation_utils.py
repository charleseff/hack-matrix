"""
Observation parsing and display utilities for HackMatrix.

This module provides robust functions for:
1. Parsing raw JSON observations into numpy arrays (same logic as gym_env)
2. Pretty-printing observations in detailed or compact formats
3. Converting between normalized and raw values

These utilities are shared across:
- gym_env.py (ML training)
- manual_play.py (validation tool)
- Any other scripts that need to display observations
"""

import numpy as np
from typing import Dict, Any, List, Optional


# MARK: Observation Parsing

def parse_observation(obs_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert JSON observation to numpy arrays.

    This is the EXACT same logic as gym_env._observation_to_array().
    Any changes here should be synchronized with gym_env.py.

    Args:
        obs_dict: Raw JSON observation dictionary from Swift

    Returns:
        Dictionary with keys 'player', 'programs', 'grid' containing numpy arrays
    """
    # Player state (10 values, normalized to [0, 1])
    player = np.array([
        obs_dict["playerRow"] / 5.0,                        # 0-5 → 0-1
        obs_dict["playerCol"] / 5.0,                        # 0-5 → 0-1
        obs_dict["playerHP"] / 3.0,                         # 0-3 → 0-1
        min(obs_dict["credits"] / 50.0, 1.0),               # 0-50+ → 0-1 (capped)
        min(obs_dict["energy"] / 50.0, 1.0),                # 0-50+ → 0-1 (capped)
        (obs_dict["stage"] - 1) / 7.0,                      # 1-8 → 0-1
        obs_dict["dataSiphons"] / 10.0,                     # 0-10 → 0-1
        (obs_dict["baseAttack"] - 1) / 1.0,                 # 1-2 → 0-1
        1.0 if obs_dict.get("showActivated", False) else 0.0,          # Binary flag
        1.0 if obs_dict.get("scheduledTasksDisabled", False) else 0.0  # Binary flag
    ], dtype=np.float32)

    # Program inventory (23 values, binary vector)
    programs = np.zeros(23, dtype=np.int32)
    if "ownedPrograms" in obs_dict:
        for action_idx in obs_dict["ownedPrograms"]:
            # Action indices 5-27 are programs
            # Map to array indices 0-22
            if 5 <= action_idx <= 27:
                programs[action_idx - 5] = 1

    # Grid state (6x6x40, normalized to [0, 1])
    grid = np.zeros((6, 6, 40), dtype=np.float32)

    for row_idx, row in enumerate(obs_dict["cells"]):
        for col_idx, cell in enumerate(row):
            features = []

            # Enemy features (6 features) - one-hot type encoding
            if "enemy" in cell:
                enemy = cell["enemy"]
                enemy_type = enemy["type"]
                # One-hot encoding for enemy types
                features.extend([
                    1.0 if enemy_type == "virus" else 0.0,
                    1.0 if enemy_type == "daemon" else 0.0,
                    1.0 if enemy_type == "glitch" else 0.0,
                    1.0 if enemy_type == "cryptog" else 0.0,
                    enemy["hp"] / 3.0,  # 0-3 → 0-1
                    1.0 if enemy["isStunned"] else 0.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Block features (5 features) - one-hot type encoding
            if "block" in cell:
                block = cell["block"]
                block_type = block["blockType"]
                # One-hot encoding for block types
                features.extend([
                    1.0 if block_type == "data" else 0.0,
                    1.0 if block_type == "program" else 0.0,
                    1.0 if block_type == "question" else 0.0,
                    block.get("points", 0) / 9.0,  # 0-9 → 0-1
                    1.0 if block["isSiphoned"] else 0.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Program features (23 features) - one-hot encoding for program type
            program_one_hot = [0.0] * 23
            if "block" in cell:
                block = cell["block"]
                program_idx = block.get("programActionIndex", 0)
                # Action indices 5-27 are programs, map to array indices 0-22
                if 5 <= program_idx <= 27:
                    program_one_hot[program_idx - 5] = 1.0
            features.extend(program_one_hot)

            # Transmission features (2 features)
            transmission_spawncount = 0
            transmission_turns = 0

            if "block" in cell:
                block = cell["block"]
                transmission_spawncount = block.get("transmissionSpawnCount", 0)

            if "transmission" in cell:
                trans = cell["transmission"]
                transmission_turns = trans["turnsUntilSpawn"]

            features.extend([
                transmission_spawncount / 9.0,               # 0-9 → 0-1
                min(transmission_turns / 4.0, 1.0)           # 0-4 → 0-1 (capped)
            ])

            # Resources (2 features)
            features.extend([
                cell.get("credits", 0) / 3.0,    # 0-3 → 0-1
                cell.get("energy", 0) / 3.0      # 0-3 → 0-1
            ])

            # Special cells (2 features)
            features.extend([
                1.0 if cell.get("isDataSiphon", False) else 0.0,
                1.0 if cell.get("isExit", False) else 0.0
            ])

            # Store features for this cell (ensure exactly 40 features)
            grid[row_idx, col_idx, :] = features[:40]

    return {
        "player": player,
        "programs": programs,
        "grid": grid
    }


# MARK: Value Denormalization

def denormalize_player(player: np.ndarray) -> Dict[str, Any]:
    """Convert normalized player array back to raw values."""
    return {
        "row": int(round(player[0] * 5)),
        "col": int(round(player[1] * 5)),
        "hp": int(round(player[2] * 3)),
        "credits": int(round(player[3] * 50)),
        "energy": int(round(player[4] * 50)),
        "stage": int(round(player[5] * 7)) + 1,
        "dataSiphons": int(round(player[6] * 10)),
        "baseAttack": int(round(player[7] * 1)) + 1,
        "showActivated": player[8] > 0.5,
        "scheduledTasksDisabled": player[9] > 0.5
    }


# MARK: Detailed Printer

def print_observation_detailed(
    observation: Dict[str, np.ndarray],
    step: int = 0,
    reward: float = 0.0,
    done: bool = False,
    info: Optional[Dict[str, Any]] = None,
    valid_actions: Optional[List[int]] = None
):
    """
    Print observation in detailed format with full grid analysis.

    Args:
        observation: Parsed observation dictionary (player, programs, grid)
        step: Current step number
        reward: Reward value
        done: Episode done flag
        info: Additional info dict (e.g., reward_breakdown)
        valid_actions: List of valid action indices
    """
    player_raw = denormalize_player(observation["player"])
    programs = observation["programs"]
    grid = observation["grid"]

    # Program names by action index (5-27) - must match ProgramType in Program.swift
    program_names = [
        "PUSH", "PULL", "CRASH", "WARP", "POLY", "WAIT", "DEBUG",
        "ROW", "COL", "UNDO", "STEP", "SIPH+", "EXCH", "SHOW",
        "RESET", "CALM", "D_BOM", "DELAY", "ANTI-V", "SCORE",
        "REDUC", "ATK+", "HACK"
    ]

    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print('='*80)

    # PLAYER STATE
    print(f"\n📍 Player State:")
    print(f"  Position: ({player_raw['row']}, {player_raw['col']})")
    print(f"  HP: {player_raw['hp']}/3  |  Credits: {player_raw['credits']}  |  Energy: {player_raw['energy']}")
    print(f"  Stage: {player_raw['stage']}/8  |  Turn: {info.get('turn', 0) if info else 0}")
    print(f"  Siphons: {player_raw['dataSiphons']}  |  Attack: {player_raw['baseAttack']}")
    print(f"  Show: {'ON' if player_raw['showActivated'] else 'off'}  |  "
          f"Calm: {'ON' if player_raw['scheduledTasksDisabled'] else 'off'}")

    # PROGRAMS
    owned_indices = np.where(programs == 1)[0]
    print(f"\n📚 Programs: {len(owned_indices)}/23 owned")
    if len(owned_indices) > 0:
        owned_names = [program_names[i] for i in owned_indices]
        action_indices = [int(i) + 5 for i in owned_indices]
        print(f"  Owned: {', '.join(owned_names)}")
        print(f"  Action indices: {action_indices}")

    # GRID ANALYSIS
    print(f"\n🗺️  Grid Analysis (6×6×40):")

    # Count entities
    enemy_count = np.sum(np.any(grid[:, :, 0:4] > 0, axis=2))
    block_count = np.sum(np.any(grid[:, :, 6:9] > 0, axis=2))
    transmission_count = np.sum(grid[:, :, 34] > 0)

    print(f"  Enemies: {enemy_count}  |  Blocks: {block_count}  |  Transmissions: {transmission_count}")

    # Show active grid channels
    active_channels = []
    for ch in range(40):
        if np.any(grid[:, :, ch] != 0):
            active_channels.append(ch)

    channel_names = {
        0: "Virus", 1: "Daemon", 2: "Glitch", 3: "Cryptog",
        4: "Enemy HP", 5: "Enemy Stunned",
        6: "Data Block", 7: "Program Block", 8: "Question Block",
        9: "Block Points", 10: "Block Siphoned",
        # Channels 11-33: Program type one-hot (action indices 5-27)
        **{11 + i: f"Program: {program_names[i]}" for i in range(23)},
        34: "Siphon Spawn Cost", 35: "Transmission Countdown",
        36: "Credits", 37: "Energy",
        38: "Data Siphon Cell", 39: "Exit Cell"
    }

    # Denormalization multipliers for channels that represent counts
    # Maps channel -> multiplier to convert normalized [0,1] back to raw value
    channel_denorm = {
        4: 3,    # Enemy HP: 0-3
        9: 9,    # Block Points: 0-9
        34: 9,   # Siphon Spawn Cost: 1-9 (stored as 0-9 normalized)
        35: 4,   # Transmission Countdown: 0-4 turns
        36: 3,   # Credits: 0-3
        37: 3,   # Energy: 0-3
    }

    if active_channels:
        print(f"  Active Channels ({len(active_channels)}/40):")
        for ch in active_channels:
            name = channel_names.get(ch, f"Channel {ch}")
            # Show which cells have non-zero values for this channel
            nonzero_cells = []
            for r in range(6):
                for c in range(6):
                    val = grid[r, c, ch]
                    if val != 0:
                        # Denormalize if this channel has a multiplier
                        if ch in channel_denorm:
                            raw_val = int(round(val * channel_denorm[ch]))
                            nonzero_cells.append(f"({r},{c})={raw_val}")
                        else:
                            nonzero_cells.append(f"({r},{c})={val:.2f}")
            cells_str = ", ".join(nonzero_cells[:8])  # First 8 cells
            if len(nonzero_cells) > 8:
                cells_str += f", ...+{len(nonzero_cells)-8} more"
            print(f"    - {ch:2d}: {name:24s} → {cells_str}")

    # REWARD
    print(f"\n💰 Reward: {reward:+.4f}")

    if info and "reward_breakdown" in info:
        breakdown = info["reward_breakdown"]
        nonzero = {k: v for k, v in breakdown.items() if v != 0}

        if nonzero:
            # Group by sign for clarity
            positive = {k: v for k, v in nonzero.items() if v > 0}
            negative = {k: v for k, v in nonzero.items() if v < 0}

            if positive:
                print("  ✓ Positive:")
                for k, v in positive.items():
                    print(f"      {k:20s}: +{v:.4f}")

            if negative:
                print("  ✗ Negative:")
                for k, v in negative.items():
                    print(f"      {k:20s}: {v:.4f}")

    # VALID ACTIONS
    if valid_actions is not None:
        print(f"\n🎮 Valid Actions: {len(valid_actions)}/28")

        # Map action indices to names
        action_names = {
            0: "Up(W)", 1: "Down(S)", 2: "Left(A)", 3: "Right(D)", 4: "Siphon"
        }
        for i in range(5, 28):
            action_names[i] = f"Prog{i-5}"

        valid_names = [action_names.get(a, f"Action{a}") for a in valid_actions]

        # Print first 10, then indicate if more
        if len(valid_names) <= 10:
            print(f"  {', '.join(valid_names)}")
        else:
            print(f"  {', '.join(valid_names[:10])}")
            print(f"  ... and {len(valid_names) - 10} more")

        print(f"  Indices: {valid_actions}")

    # DONE FLAG
    if done:
        print("\n🏁 EPISODE DONE")

    print("="*80)


# MARK: Compact Printer

def print_observation_compact(
    observation: Dict[str, np.ndarray],
    step: int = 0,
    reward: float = 0.0,
    valid_actions: Optional[List[int]] = None
):
    """
    Print observation in compact single-line format.

    Args:
        observation: Parsed observation dictionary (player, programs, grid)
        step: Current step number
        reward: Reward value
        valid_actions: List of valid action indices
    """
    player_raw = denormalize_player(observation["player"])
    programs = observation["programs"]
    grid = observation["grid"]

    # Count entities
    enemy_count = np.sum(np.any(grid[:, :, 0:4] > 0, axis=2))
    block_count = np.sum(np.any(grid[:, :, 6:9] > 0, axis=2))
    owned_programs = len(np.where(programs == 1)[0])

    print(f"Step {step:4d} | "
          f"Pos:({player_raw['row']},{player_raw['col']}) "
          f"HP:{player_raw['hp']} "
          f"$:{player_raw['credits']:2d} "
          f"⚡:{player_raw['energy']:2d} "
          f"Stage:{player_raw['stage']} | "
          f"Enemies:{enemy_count} "
          f"Blocks:{block_count} "
          f"Programs:{owned_programs}/23 | "
          f"Reward:{reward:+.3f} | "
          f"Actions:{len(valid_actions) if valid_actions else '?'}/28")


# MARK: Grid Visualization

def print_grid_map(grid: np.ndarray, player_pos: tuple):
    """
    Print ASCII map of the grid showing entities.

    Args:
        grid: Grid observation (6x6x40)
        player_pos: (row, col) tuple for player position
    """
    print("\n🗺️  Grid Map:")
    print("   " + " ".join(str(i) for i in range(6)))

    for row in range(6):
        row_str = f"{row}  "
        for col in range(6):
            if (row, col) == player_pos:
                cell_char = "@"  # Player
            elif grid[row, col, 39] > 0:
                cell_char = "X"  # Exit
            elif grid[row, col, 38] > 0:
                cell_char = "S"  # Data siphon
            elif np.any(grid[row, col, 0:4] > 0):
                cell_char = "E"  # Enemy
            elif np.any(grid[row, col, 6:9] > 0):
                cell_char = "B"  # Block
            elif grid[row, col, 34] > 0:
                cell_char = "T"  # Transmission
            elif grid[row, col, 36] > 0 or grid[row, col, 37] > 0:
                cell_char = "$"  # Resources
            else:
                cell_char = "."  # Empty

            row_str += cell_char + " "

        print(row_str)

    print("\nLegend: @ = Player, E = Enemy, B = Block, T = Transmission")
    print("        S = Data Siphon, X = Exit, $ = Resources, . = Empty")
