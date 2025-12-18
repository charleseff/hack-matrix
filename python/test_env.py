"""
Test script for the 868-HACK environment.
"""

from hack_env import HackEnv
import numpy as np
import time
import sys


def test_basic_functionality(visual=False, steps=5, delay=0.0):
    """
    Test basic environment functionality.

    Args:
        visual: If True, launch GUI and animate actions
        steps: Number of random steps to take
        delay: Seconds to wait between steps (useful for visual mode)
    """
    mode = "visual CLI" if visual else "headless"
    print(f"Testing 868-HACK environment ({mode} mode)...")

    # Create environment
    env = HackEnv(visual=visual)
    print("✓ Environment created")

    # Reset
    obs, info = env.reset()
    print(f"✓ Environment reset")
    print(f"  Player state: {obs['player']}")
    print(f"  Grid shape: {obs['grid'].shape}")
    print(f"  Flags: {obs['flags']}")

    # Get valid actions
    valid_actions = env.get_valid_actions()
    print(f"✓ Got {len(valid_actions)} valid actions: {valid_actions}")

    # Take random steps
    for i in range(steps):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("  No valid actions, game likely over")
            break

        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Step {i+1}: action={action}, reward={reward}, done={terminated}")

        if delay > 0:
            time.sleep(delay)

        if terminated:
            print("  Game ended!")
            break

    # Cleanup
    env.close()
    print("✓ Environment closed")
    print("\nAll tests passed!")


if __name__ == "__main__":
    # Check for --visual flag
    visual = "--visual" in sys.argv
    steps = 20 if visual else 5
    delay = 0.5 if visual else 0.0

    test_basic_functionality(visual=visual, steps=steps, delay=delay)