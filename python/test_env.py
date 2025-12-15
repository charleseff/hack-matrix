"""
Test script for the 868-HACK environment.
"""

from hack_env import HackEnv
import numpy as np


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing 868-HACK environment...")

    # Create environment
    env = HackEnv()
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

    # Take a few random steps
    for i in range(5):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("  No valid actions, game likely over")
            break

        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Step {i+1}: action={action}, reward={reward}, done={terminated}")

        if terminated:
            print("  Game ended!")
            break

    # Cleanup
    env.close()
    print("✓ Environment closed")
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_basic_functionality()