"""
Swift Environment Wrapper implementing EnvInterface.

This wrapper communicates with the Swift game via JSON over subprocess,
implementing the EnvInterface protocol for parity testing.

Why this design:
- Reuses subprocess communication pattern from gym_env.py
- Adds set_state() capability for deterministic test setup
- Converts observations to Observation dataclass format
"""

import json
import os
import subprocess
from typing import Any, Optional

import numpy as np

from .env_interface import (
    EnvInterface,
    GameState,
    Observation,
    StepResult,
    game_state_to_json,
)

# Paths relative to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_APP_PATH = os.environ.get(
    "HACKMATRIX_BINARY",
    os.path.join(_SCRIPT_DIR, "..", "..", ".build", "debug", "HackMatrix")
)


class SwiftEnvWrapper:
    """Swift environment wrapper implementing EnvInterface."""

    def __init__(self, app_path: str | None = None, debug: bool = False):
        """
        Initialize the Swift environment wrapper.

        Args:
            app_path: Path to Swift executable. If None, uses default.
            debug: Enable debug logging in Swift process.
        """
        self.app_path = app_path or _DEFAULT_APP_PATH
        self.debug = debug
        self.process: Optional[subprocess.Popen] = None
        self.stderr_log = None
        self._start_process()

    def _start_process(self):
        """Start the Swift subprocess."""
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

        if self.stderr_log is not None:
            self.stderr_log.close()
            self.stderr_log = None

        self.stderr_log = open("/tmp/swift_test.log", "w")

        cmd = [self.app_path, "--headless-cli"]
        if self.debug:
            cmd.append("--debug")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_log,
            text=True,
            bufsize=1
        )

    def _send_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Send a command to the Swift process and get response."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Swift process not running")

        json_str = json.dumps(command) + "\n"
        self.process.stdin.write(json_str)
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from Swift process")

        response = json.loads(response_line)

        if "error" in response:
            raise RuntimeError(f"Swift error: {response['error']}")

        return response

    def _parse_observation(self, obs_dict: dict[str, Any]) -> Observation:
        """Convert JSON observation to Observation dataclass."""
        # Player state (10 values, normalized to [0, 1])
        player = np.array([
            obs_dict["playerRow"] / 5.0,
            obs_dict["playerCol"] / 5.0,
            obs_dict["playerHP"] / 3.0,
            min(obs_dict["credits"] / 50.0, 1.0),
            min(obs_dict["energy"] / 50.0, 1.0),
            (obs_dict["stage"] - 1) / 7.0,
            obs_dict["dataSiphons"] / 10.0,
            (obs_dict["baseAttack"] - 1) / 1.0,
            1.0 if obs_dict.get("showActivated", False) else 0.0,
            1.0 if obs_dict.get("scheduledTasksDisabled", False) else 0.0
        ], dtype=np.float32)

        # Program inventory (23 values, binary vector)
        programs = np.zeros(23, dtype=np.int32)
        if "ownedPrograms" in obs_dict:
            for action_idx in obs_dict["ownedPrograms"]:
                if 5 <= action_idx <= 27:
                    programs[action_idx - 5] = 1

        # Grid state (6x6x40, normalized to [0, 1])
        grid = np.zeros((6, 6, 40), dtype=np.float32)

        for row_idx, row in enumerate(obs_dict["cells"]):
            for col_idx, cell in enumerate(row):
                features = []

                # Enemy features (6 features)
                if "enemy" in cell:
                    enemy = cell["enemy"]
                    enemy_type = enemy["type"]
                    features.extend([
                        1.0 if enemy_type == "virus" else 0.0,
                        1.0 if enemy_type == "daemon" else 0.0,
                        1.0 if enemy_type == "glitch" else 0.0,
                        1.0 if enemy_type == "cryptog" else 0.0,
                        enemy["hp"] / 3.0,
                        1.0 if enemy["isStunned"] else 0.0
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                # Block features (5 features)
                if "block" in cell:
                    block = cell["block"]
                    block_type = block["blockType"]
                    features.extend([
                        1.0 if block_type == "data" else 0.0,
                        1.0 if block_type == "program" else 0.0,
                        1.0 if block_type == "question" else 0.0,
                        block.get("points", 0) / 9.0,
                        1.0 if block["isSiphoned"] else 0.0
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

                # Program features (23 features)
                program_one_hot = [0.0] * 23
                if "block" in cell:
                    block = cell["block"]
                    program_idx = block.get("programActionIndex", 0)
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
                    transmission_spawncount / 9.0,
                    min(transmission_turns / 4.0, 1.0)
                ])

                # Resources (2 features)
                features.extend([
                    cell.get("credits", 0) / 3.0,
                    cell.get("energy", 0) / 3.0
                ])

                # Special cells (2 features)
                features.extend([
                    1.0 if cell.get("isDataSiphon", False) else 0.0,
                    1.0 if cell.get("isExit", False) else 0.0
                ])

                grid[row_idx, col_idx, :] = features[:40]

        return Observation(player=player, programs=programs, grid=grid)

    # MARK: - EnvInterface Implementation

    def reset(self) -> Observation:
        """Reset the environment to initial state."""
        response = self._send_command({"action": "reset"})
        return self._parse_observation(response["observation"])

    def step(self, action: int) -> StepResult:
        """Execute an action in the environment."""
        response = self._send_command({
            "action": "step",
            "actionIndex": int(action)
        })

        observation = self._parse_observation(response["observation"])
        reward = float(response["reward"])
        done = bool(response["done"])
        info = response.get("info", {})

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )

    def get_valid_actions(self) -> list[int]:
        """Get list of valid action indices for current state."""
        response = self._send_command({"action": "getValidActions"})
        return response["validActions"]

    def set_state(self, state: GameState) -> Observation:
        """Set the complete game state for test setup."""
        state_json = game_state_to_json(state)
        response = self._send_command({
            "action": "setState",
            "state": state_json
        })
        return self._parse_observation(response["observation"])

    # MARK: - Cleanup

    def close(self):
        """Clean up resources."""
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

        if self.stderr_log is not None:
            self.stderr_log.close()
            self.stderr_log = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
