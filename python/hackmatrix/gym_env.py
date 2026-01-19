"""
Gymnasium environment wrapper for HackMatrix game.
Communicates with Swift game via JSON over subprocess.
"""

# MARK: Imports

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .observation_utils import denormalize_player, parse_observation

# MARK: Constants

# App paths relative to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SPM build - headless CLI only (for training)
# Check HACKMATRIX_BINARY env var first (for Docker), then fall back to relative path
_DEFAULT_APP_PATH = os.environ.get(
    "HACKMATRIX_BINARY",
    os.path.join(_SCRIPT_DIR, "..", "..", ".build", "debug", "HackMatrix")
)
# Xcode build - full GUI app (for visual mode)
_XCODE_APP_PATH = os.path.join(_SCRIPT_DIR, "..", "..", "DerivedData", "HackMatrix", "Build", "Products", "Debug", "HackMatrix.app", "Contents", "MacOS", "HackMatrix")


# MARK: HackEnv Class

class HackEnv(gym.Env):
    """HackMatrix Gymnasium environment."""

    metadata = {"render_modes": []}

    # MARK: Initialization

    def __init__(self, app_path: str = None, visual: bool = False, debug_scenario: bool = False, debug: bool = False, info: bool = False):
        """
        Initialize the environment.

        Args:
            app_path: Path to the Swift executable. If None, uses:
                      - Xcode build (GUI app) for visual mode
                      - SPM build (headless CLI) for training mode
            visual: If True, launch GUI with animations (visual CLI mode)
            debug_scenario: If True, use debug scenario (fixed stage layout)
            debug: If True, enable verbose debug logging
            info: If True, enable info-level logging (less verbose than debug)
        """
        super().__init__()

        # Select appropriate binary based on mode
        if app_path is None:
            if visual:
                self.app_path = _XCODE_APP_PATH
            else:
                self.app_path = _DEFAULT_APP_PATH
        else:
            self.app_path = app_path
        self.visual = visual
        self.debug_scenario = debug_scenario
        self.debug = debug
        self.info = info
        self.process: Optional[subprocess.Popen] = None
        self.stderr_log = None  # Track file handle for cleanup

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

            # Program inventory (23 values, binary vector)
            # Indices 0-22 correspond to program action indices 5-27
            "programs": spaces.Box(
                low=0,
                high=1,
                shape=(23,),
                dtype=np.int32
            ),

            # Grid state: 6x6x40 (40 features per cell, normalized to [0, 1])
            # Enemy: one-hot types (4) + hp + stunned = 6
            # Block: one-hot types (3) + points + siphoned = 5
            # Program: one-hot (23) + transmission_spawncount + transmission_turns = 25
            # Resources: credits + energy = 2
            # Special: is_data_siphon + is_exit = 2
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6, 6, 40),
                dtype=np.float32
            )
        })

        self._start_process()
        self._reset_episode_stats()

    # MARK: Episode Stats Tracking

    def _reset_episode_stats(self):
        """Reset episode statistics for tracking."""
        self._episode_reward_breakdown = defaultdict(float)
        self._episode_action_counts = Counter()
        self._episode_programs_used = 0
        self._episode_highest_stage = 1
        self._episode_steps = 0

    # MARK: Process Management

    def _start_process(self):
        """Start the Swift subprocess."""
        # Close old process and file handle if they exist
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

        if self.stderr_log is not None:
            self.stderr_log.close()
            self.stderr_log = None

        flag = "--visual-cli" if self.visual else "--headless-cli"

        # Open log file for Swift stderr (debug output)
        mode = "visual" if self.visual else "headless"
        self.stderr_log = open(f"/tmp/swift_{mode}.log", "w")

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
            stderr=self.stderr_log,  # Write Swift debug output to log file
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

    # MARK: Observation Conversion

    def _observation_to_array(self, obs_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert JSON observation to numpy arrays.

        This method now delegates to the shared parse_observation() function
        in observation_utils.py to ensure consistency across all tools.
        """
        return parse_observation(obs_dict)

    # MARK: Environment Interface

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self._reset_episode_stats()

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

        # Track episode stats
        self._episode_steps += 1

        # Track action distribution
        if action < 4:
            self._episode_action_counts["move"] += 1
        elif action == 4:
            self._episode_action_counts["siphon"] += 1
        else:
            self._episode_action_counts["program"] += 1
            self._episode_programs_used += 1

        # Track highest stage from player observation (denormalize: stage = normalized * 7 + 1)
        stage = int(observation["player"][5] * 7 + 1)
        self._episode_highest_stage = max(self._episode_highest_stage, stage)

        # Accumulate reward breakdown from Swift
        if "reward_breakdown" in info:
            for key, value in info["reward_breakdown"].items():
                self._episode_reward_breakdown[key] += value

        # On episode end, add summary to info
        if terminated:
            info["episode_stats"] = {
                "reward_breakdown": dict(self._episode_reward_breakdown),
                "programs_used": self._episode_programs_used,
                "highest_stage": self._episode_highest_stage,
                "action_counts": dict(self._episode_action_counts),
                "steps": self._episode_steps,
            }

        # Add action mask for next state
        if not terminated:
            info["action_mask"] = self._get_action_mask()

        # Visual CLI mode: Print action, reward, and key observation values
        if self.visual:
            action_names = ["Up", "Down", "Left", "Right", "Siphon"] + [f"Prog{i}" for i in range(5, 28)]
            action_name = action_names[action] if action < len(action_names) else f"Unknown({action})"

            # Decode player state for display
            player = observation["player"]
            print(f"\n{'='*60}")
            print(f"ACTION: {action_name} (index={action})")
            print(f"REWARD: {reward:+.3f}")
            print(f"Player: row={player[0]:.2f} col={player[1]:.2f} hp={player[2]:.2f} "
                  f"credits={player[3]:.2f} energy={player[4]:.2f} stage={player[5]:.2f}")
            print(f"Status: siphons={player[6]:.2f} attack={player[7]:.2f} "
                  f"show={int(player[8])} calm={int(player[9])}")
            owned_programs = np.where(observation["programs"] == 1)[0] + 5
            print(f"Owned Programs: {list(owned_programs)}")
            if terminated:
                print(f"TERMINATED: {'Victory!' if player[5] > 0.875 else 'Defeated'}")
            print(f"{'='*60}\n")
            sys.stdout.flush()

        return observation, reward, terminated, truncated, info

    # MARK: Action Mask

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

    # MARK: Cleanup

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
