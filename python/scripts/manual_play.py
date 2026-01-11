#!/usr/bin/env python3
"""
Monitor manual gameplay and display observation space in real-time.

Launches the game with --visual-cli and monitors observations as you play.
No commands needed - just play with keyboard/mouse in the GUI!

This validates that observation encoding/decoding works correctly across
all game scenarios by showing the EXACT data fed to the ML model.
"""

import subprocess
import json
import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from hackmatrix.observation_utils import parse_observation, print_observation_detailed


def monitor_gameplay(app_path: str, debug_scenario: bool = False):
    """Launch game and monitor observations from manual play."""

    print("="*80)
    print("MANUAL PLAY MONITOR - Observation & Action Space Validator")
    print("="*80)
    print("\nLaunching game in visual-cli mode...")
    print("🎮 Play with keyboard/mouse in the GUI window")
    print("📊 Observations will appear here in real-time\n")

    if debug_scenario:
        print("🔬 Debug scenario enabled - predictable starting state\n")

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
    pending_observation = None  # Store observation until we get valid actions
    total_reward = 0.0
    cumulative_breakdown = {}  # Track cumulative reward by component

    try:
        # Send initial reset command to start the game
        print("Sending reset command to start game...\n")
        reset_cmd = json.dumps({"action": "reset"}) + "\n"
        process.stdin.write(reset_cmd)
        process.stdin.flush()

        # Read stdout line by line (same as gym_env._send_command)
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            # Debug: show raw line (can be disabled later)
            print(f"[DEBUG] Received line: {line[:100]}", file=sys.stderr)

            # Skip non-JSON lines (debug messages, go to stderr output)
            if not line.startswith('{'):
                # Print Swift debug output
                if line:
                    print(f"[Swift] {line}", file=sys.stderr)
                continue

            try:
                data = json.loads(line)

                # Debug: show what type of message we received
                msg_type = "unknown"
                if "observation" in data:
                    msg_type = "observation"
                elif "validActions" in data:
                    msg_type = "validActions"
                elif "error" in data:
                    msg_type = "error"
                print(f"[DEBUG] Message type: {msg_type}", file=sys.stderr)

                # Handle different response types
                if "observation" in data:
                    # Store observation, request valid actions (same flow as training)
                    step += 1
                    pending_observation = (data, step)
                    request_valid_actions(process)

                elif "validActions" in data:
                    # Now we have valid actions, print the pending observation
                    valid_actions = data["validActions"]
                    print(f"[DEBUG] Got {len(valid_actions)} valid actions", file=sys.stderr)

                    if pending_observation:
                        obs_data, obs_step = pending_observation

                        # Accumulate rewards
                        reward = obs_data.get("reward", 0)
                        total_reward += reward

                        # Accumulate breakdown
                        breakdown = obs_data.get("info", {}).get("reward_breakdown", {})
                        for key, value in breakdown.items():
                            cumulative_breakdown[key] = cumulative_breakdown.get(key, 0) + value

                        print_observation(obs_data, obs_step, valid_actions, total_reward, cumulative_breakdown)
                        pending_observation = None

                elif "error" in data:
                    print(f"\n❌ Error: {data['error']}")

            except json.JSONDecodeError as e:
                print(f"[Parse Error] {e}: {line[:100]}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")
    finally:
        process.terminate()
        process.wait()


def request_valid_actions(process):
    """Send getValidActions command to Swift."""
    command = json.dumps({"action": "getValidActions"}) + "\n"
    try:
        process.stdin.write(command)
        process.stdin.flush()
    except BrokenPipeError:
        pass  # Process terminated


def print_observation(data: dict, step: int, valid_actions: list = None,
                       total_reward: float = 0.0, cumulative_breakdown: dict = None):
    """Pretty-print observation data using robust observation_utils."""

    try:
        obs_raw = data.get("observation", {})
        reward = data.get("reward", 0)
        done = data.get("done", False)
        info = data.get("info", {})

        # Parse observation using the EXACT same logic as gym_env
        observation = parse_observation(obs_raw)

        # Use detailed printer from observation_utils
        print_observation_detailed(
            observation=observation,
            step=step,
            reward=reward,
            done=done,
            info=info,
            valid_actions=valid_actions
        )

        # Print cumulative rewards
        print(f"\n📈 Cumulative: {total_reward:+.4f}")
        if cumulative_breakdown:
            nonzero = {k: v for k, v in cumulative_breakdown.items() if v != 0}
            if nonzero:
                # Sort by absolute value, descending
                sorted_items = sorted(nonzero.items(), key=lambda x: abs(x[1]), reverse=True)
                for k, v in sorted_items[:8]:  # Top 8 contributors
                    sign = "+" if v > 0 else ""
                    print(f"    {k:20s}: {sign}{v:.4f}")
                if len(sorted_items) > 8:
                    print(f"    ... and {len(sorted_items) - 8} more")
        print("="*80)

        sys.stdout.flush()

    except Exception as e:
        print(f"[ERROR in print_observation] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor manual gameplay and validate observation/action space"
    )
    parser.add_argument(
        "--debug-scenario",
        action="store_true",
        help="Use debug scenario starting state"
    )
    args = parser.parse_args()

    # Path to Swift binary
    binary_path = "../DerivedData/HackMatrix/Build/Products/Debug/HackMatrix.app/Contents/MacOS/HackMatrix"

    if not Path(binary_path).exists():
        print(f"❌ Binary not found: {binary_path}")
        print("\nBuild first:")
        print("  xcodebuild -scheme HackMatrix -configuration Debug")
        sys.exit(1)

    monitor_gameplay(binary_path, debug_scenario=args.debug_scenario)
