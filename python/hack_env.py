"""
Gymnasium environment wrapper for HackMatrix game.
Communicates with Swift game via JSON over subprocess.
"""

import json
import os
import subprocess
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

# Default app path relative to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_APP_PATH = os.path.join(_SCRIPT_DIR, "..", "DerivedData", "Build", "Products", "Debug", "HackMatrix.app", "Contents", "MacOS", "HackMatrix")


class HackEnv(gym.Env):
    """HackMatrix Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self, app_path: str = _DEFAULT_APP_PATH, visual: bool = False):
        """
        Initialize the environment.

        Args:
            app_path: Path to the Swift executable
            visual: If True, launch GUI with animations (visual CLI mode)
        """
        super().__init__()

        self.app_path = app_path
        self.visual = visual
        self.process: Optional[subprocess.Popen] = None

        # Action space: 31 discrete actions (4 moves + 1 siphon + 26 programs)
        self.action_space = spaces.Discrete(31)

        # Observation space: Multi-part observation
        # We'll use a Dict space with the main components
        self.observation_space = spaces.Dict({
            # Player state (9 values)
            "player": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 1, 0, 0, 1]),  # row, col, hp, credits, energy, stage, turn, siphons, attack
                high=np.array([5, 5, 3, 999, 999, 8, 9999, 99, 2]),
                dtype=np.int32
            ),

            # Grid state: 6x6x20 (20 features per cell)
            # Features: enemy_type(4), enemy_hp(1), enemy_stunned(1),
            #           block_type(3), block_points(1), block_siphoned(1),
            #           transmission_turns(1), transmission_type(4),
            #           credits(1), energy(1), is_siphon(1), is_exit(1)
            "grid": spaces.Box(
                low=0, high=999,
                shape=(6, 6, 20),
                dtype=np.int32
            ),

            # Flags (2 values)
            "flags": spaces.Box(
                low=0, high=1,
                shape=(1,),  # showActivated
                dtype=np.int32
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

        self.process = subprocess.Popen(
            [self.app_path, flag, "--debug-scenario"],
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
        # Player state
        player = np.array([
            obs_dict["playerRow"],
            obs_dict["playerCol"],
            obs_dict["playerHP"],
            obs_dict["credits"],
            obs_dict["energy"],
            obs_dict["stage"],
            obs_dict["turn"],
            obs_dict["dataSiphons"],
            obs_dict["baseAttack"]
        ], dtype=np.int32)

        # Grid state (6x6x20)
        grid = np.zeros((6, 6, 20), dtype=np.int32)

        for row_idx, row in enumerate(obs_dict["cells"]):
            for col_idx, cell in enumerate(row):
                features = []

                # Enemy features (6)
                if "enemy" in cell:
                    enemy_types = {"virus": 0, "daemon": 1, "glitch": 2, "cryptog": 3}
                    enemy = cell["enemy"]
                    features.extend([
                        enemy_types.get(enemy["type"], 0),
                        enemy["hp"],
                        1 if enemy["isStunned"] else 0
                    ])
                else:
                    features.extend([0, 0, 0])

                # Block features (5)
                if "block" in cell:
                    block_types = {"data": 1, "program": 2, "question": 3}
                    block = cell["block"]
                    features.extend([
                        block_types.get(block["blockType"], 0),
                        block.get("points", 0),
                        1 if block["isSiphoned"] else 0
                    ])
                else:
                    features.extend([0, 0, 0])

                # Transmission features (2)
                if "transmission" in cell:
                    trans = cell["transmission"]
                    features.append(trans["turnsUntilSpawn"])
                else:
                    features.append(0)

                # Resources (2)
                features.extend([
                    cell.get("credits", 0),
                    cell.get("energy", 0)
                ])

                # Special cells (2)
                features.extend([
                    1 if cell.get("isDataSiphon", False) else 0,
                    1 if cell.get("isExit", False) else 0
                ])

                # Pad to 20 features
                while len(features) < 20:
                    features.append(0)

                grid[row_idx, col_idx, :] = features[:20]

        # Flags
        flags = np.array([
            1 if obs_dict["showActivated"] else 0,
        ], dtype=np.int32)

        return {
            "player": player,
            "grid": grid,
            "flags": flags
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
