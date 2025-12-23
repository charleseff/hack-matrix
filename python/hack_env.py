"""
Gymnasium environment wrapper for HackMatrix game.
Communicates with Swift game via JSON over subprocess.
"""

import json
import os
import subprocess
import sys
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

# Default app path relative to this file (Xcode default location)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_APP_PATH = os.path.join(_SCRIPT_DIR, "..", "DerivedData", "HackMatrix", "Build", "Products", "Debug", "HackMatrix.app", "Contents", "MacOS", "HackMatrix")


class HackEnv(gym.Env):
    """HackMatrix Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self, app_path: str = _DEFAULT_APP_PATH, visual: bool = False, debug_scenario: bool = False, debug: bool = False, info: bool = False):
        """
        Initialize the environment.

        Args:
            app_path: Path to the Swift executable
            visual: If True, launch GUI with animations (visual CLI mode)
            debug_scenario: If True, use debug scenario (fixed stage layout)
            debug: If True, enable verbose debug logging
            info: If True, enable info-level logging (less verbose than debug)
        """
        super().__init__()

        self.app_path = app_path
        self.visual = visual
        self.debug_scenario = debug_scenario
        self.debug = debug
        self.info = info
        self.process: Optional[subprocess.Popen] = None

        # Action space: 28 discrete actions (4 moves + 1 siphon + 23 programs)
        self.action_space = spaces.Discrete(28)

        # Observation space: Multi-part observation
        # We'll use a Dict space with the main components
        self.observation_space = spaces.Dict({
            # Player state (10 values, normalized to [0, 1])
            # [row, col, hp, credits, energy, stage, siphons, attack, showActivated, scheduledTasksDisabled]
            "player": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(10,),
                dtype=np.float32
            ),

            # Program inventory (26 values, binary vector)
            # Indices 0-25 correspond to program action indices 5-30
            "programs": spaces.Box(
                low=0,
                high=1,
                shape=(26,),
                dtype=np.int32
            ),

            # Grid state: 6x6x43 (43 features per cell, normalized to [0, 1])
            # Enemy: one-hot types (4) + hp + stunned = 6
            # Block: one-hot types (3) + points + siphoned = 5
            # Program: one-hot (26) + transmission_spawncount + transmission_turns = 28
            # Resources: credits + energy = 2
            # Special: is_data_siphon + is_exit = 2
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6, 6, 43),
                dtype=np.float32
            )
        })

        self._start_process()

    def _start_process(self):
        """Start the Swift subprocess."""
        if self.process is not None:
            self.process.terminate()
            self.process.wait()

        flag = "--visual-cli" if self.visual else "--headless-cli"

        # Open log file for Swift stderr (debug output)
        mode = "visual" if self.visual else "headless"
        stderr_log = open(f"/tmp/swift_{mode}.log", "w")

        # Build command with optional debug scenario and logging flags
        cmd = [self.app_path, flag]
        if self.debug_scenario:
            cmd.append("--debug-scenario")
        if self.debug:
            cmd.append("--debug")
        elif self.info:
            cmd.append("--info")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_log,  # Write Swift debug output to log file
            text=True,
            bufsize=1
        )

        # Give GUI time to initialize if in visual mode
        if self.visual:
            import time
            time.sleep(2.0)

    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the Swift process and get response."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Swift process not running")

        # Send command
        json_str = json.dumps(command) + "\n"
        self.process.stdin.write(json_str)
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from Swift process")

        response = json.loads(response_line)

        if "error" in response:
            raise RuntimeError(f"Swift error: {response['error']}")

        return response

    def _observation_to_array(self, obs_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert JSON observation to numpy arrays."""
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

        # Program inventory (26 values, binary vector)
        programs = np.zeros(26, dtype=np.int32)
        if "ownedPrograms" in obs_dict:
            for action_idx in obs_dict["ownedPrograms"]:
                # Action indices 5-30 are programs
                # Map to array indices 0-25
                if 5 <= action_idx <= 30:
                    programs[action_idx - 5] = 1

        # Grid state (6x6x43, normalized to [0, 1])
        grid = np.zeros((6, 6, 43), dtype=np.float32)

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

                # Program features (26 features) - one-hot encoding for program type
                program_one_hot = [0.0] * 26
                if "block" in cell:
                    block = cell["block"]
                    program_idx = block.get("programActionIndex", 0)
                    # Action indices 5-30 are programs, map to array indices 0-25
                    if 5 <= program_idx <= 30:
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

                grid[row_idx, col_idx, :] = features[:43]

        return {
            "player": player,
            "programs": programs,
            "grid": grid
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        response = self._send_command({"action": "reset"})
        observation = self._observation_to_array(response["observation"])

        return observation, {"action_mask": self._get_action_mask()}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step."""
        response = self._send_command({
            "action": "step",
            "actionIndex": int(action)
        })

        observation = self._observation_to_array(response["observation"])
        reward = float(response["reward"])
        terminated = bool(response["done"])
        truncated = False
        info = response.get("info", {})

        # Add action mask for next state
        if not terminated:
            info["action_mask"] = self._get_action_mask()

        return observation, reward, terminated, truncated, info

    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices."""
        response = self._send_command({"action": "getValidActions"})
        return response["validActions"]

    def _get_action_mask(self) -> np.ndarray:
        """Get action mask for MaskablePPO (boolean array of valid actions)."""
        valid_actions = self.get_valid_actions()
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        mask[valid_actions] = True

        # Debug logging - only when debug mode enabled and action space is restricted
        if self.debug and len(valid_actions) < 28:
            print(f"[HackEnv] Valid action indices from Swift: {valid_actions}", file=sys.stderr)
            print(f"[HackEnv] Action mask created: {np.where(mask)[0].tolist()}", file=sys.stderr)
            sys.stderr.flush()

        return mask

    def close(self):
        """Clean up resources."""
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
