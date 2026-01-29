"""
Inspect observation space structure and verify all game state is captured.
"""

import json
import os
import sys

# Add python directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hackmatrix import HackEnv

# Create environment
env = HackEnv(debug_scenario=True)

# Reset and get initial observation
obs, info = env.reset()

# Get the raw JSON response to see what's actually being sent
response = env._send_command({"action": "reset"})

print("=" * 80)
print("RAW JSON OBSERVATION FROM SWIFT:")
print("=" * 80)
print(json.dumps(response["observation"], indent=2))

print("\n" + "=" * 80)
print("CHECKING BLOCK FEATURES:")
print("=" * 80)

# Check a few cells to see what block information is available
for row_idx, row in enumerate(response["observation"]["cells"]):
    for col_idx, cell in enumerate(row):
        if "block" in cell:
            block = cell["block"]
            print(f"\nCell ({row_idx}, {col_idx}) has block:")
            print(f"  blockType: {block.get('blockType')}")
            print(f"  points: {block.get('points')}")
            print(f"  programType: {block.get('programType')}")  # Is this present?
            print(f"  transmissionSpawnCount: {block.get('transmissionSpawnCount')}")
            print(f"  isSiphoned: {block.get('isSiphoned')}")

print("\n" + "=" * 80)
print("PYTHON NUMPY OBSERVATION SHAPE:")
print("=" * 80)
print(f"player: {obs['player'].shape} (dtype={obs['player'].dtype}) = {obs['player']}")
print(f"  Player values in [0,1]: {obs['player'].min():.3f} to {obs['player'].max():.3f}")
print(f"programs: {obs['programs'].shape} (dtype={obs['programs'].dtype})")
print(f"  Owned programs (action indices): {list(np.where(obs['programs'] == 1)[0] + 5)}")
print(f"grid: {obs['grid'].shape} (dtype={obs['grid'].dtype})")
print(f"  Grid values in [0,1]: {obs['grid'].min():.3f} to {obs['grid'].max():.3f}")

# Check grid features with new one-hot encoding
print("\n" + "=" * 80)
print("SAMPLE GRID FEATURES (NEW ONE-HOT ENCODING):")
print("=" * 80)
for row_idx in range(6):
    for col_idx in range(6):
        cell_features = obs["grid"][row_idx, col_idx, :]

        # Check if this cell has any interesting features
        has_enemy = np.any(cell_features[0:4] > 0)
        has_block = np.any(cell_features[6:9] > 0)
        has_program = np.any(cell_features[11:37] > 0)
        has_transmission = cell_features[37] > 0 or cell_features[38] > 0

        if has_enemy or has_block or has_program or has_transmission:
            print(f"\nCell ({row_idx}, {col_idx}) grid features (43 total):")
            print(f"  Enemy one-hot [0-3]: {cell_features[0:4]} (virus, daemon, glitch, cryptog)")
            print(f"  Enemy hp [4]: {cell_features[4]:.3f}")
            print(f"  Enemy stunned [5]: {cell_features[5]:.0f}")
            print(f"  Block one-hot [6-8]: {cell_features[6:9]} (data, program, question)")
            print(f"  Block points [9]: {cell_features[9]:.3f}")
            print(f"  Block siphoned [10]: {cell_features[10]:.0f}")
            if has_program:
                prog_idx = np.where(cell_features[11:37] > 0)[0]
                if len(prog_idx) > 0:
                    action_idx = prog_idx[0] + 5
                    print(f"  Program one-hot [11-36]: action index {action_idx}")
            print(f"  Transmission spawncount [37]: {cell_features[37]:.3f}")
            print(f"  Transmission turns [38]: {cell_features[38]:.3f}")
            print(f"  Credits [39]: {cell_features[39]:.3f}")
            print(f"  Energy [40]: {cell_features[40]:.3f}")
            print(f"  Is data siphon [41]: {cell_features[41]:.0f}")
            print(f"  Is exit [42]: {cell_features[42]:.0f}")

env.close()
