#!/usr/bin/env python3
"""
Manual play script for WOR-27: Verify rewards and observations.

Launches the game in visual CLI mode where you can:
1. See the GUI and play the game
2. Input actions via terminal
3. View rewards and observations after each action

Use this to verify rewards are sensible and observations match game state.
"""

from hackmatrix import HackEnv
import sys


def print_action_menu():
    """Print available actions."""
    print("\n" + "="*70)
    print("ACTIONS:")
    print("  0-3: Move (0=Up, 1=Down, 2=Left, 3=Right)")
    print("  4: Siphon")
    print("  5-30: Programs (check owned programs in output)")
    print()
    print("COMMANDS:")
    print("  v: Show valid actions")
    print("  q: Quit")
    print("="*70)


def main():
    """Run manual play session."""
    print("=" * 70)
    print("WOR-27: Manual Verification - Visual CLI Mode")
    print("=" * 70)
    print("\nLaunching game in visual mode...")
    print("You'll see:")
    print("  1. GUI window (for visual feedback)")
    print("  2. Terminal output (showing rewards & observations)")
    print()

    # Create environment in visual mode
    env = HackEnv(visual=True, info=True)

    # Reset environment
    obs, info = env.reset()
    print("\n✓ Game started!")
    print_action_menu()

    step_count = 0

    while True:
        # Get valid actions
        valid_actions = env.get_valid_actions()

        # Get user input
        try:
            user_input = input("\nEnter action: ").strip().lower()

            if user_input == 'q':
                print("Quitting...")
                break

            if user_input == 'v':
                print(f"Valid actions: {valid_actions}")
                continue

            if user_input == '':
                continue

            # Parse action
            try:
                action = int(user_input)
            except ValueError:
                print(f"Invalid input: '{user_input}'. Enter a number, 'v', or 'q'.")
                continue

            # Validate action
            if action not in valid_actions:
                print(f"❌ Action {action} is NOT valid!")
                print(f"   Valid actions: {valid_actions}")
                continue

            # Take action (visual CLI debug output will print automatically)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Check if game ended
            if terminated or truncated:
                print("\n" + "="*70)
                print("GAME ENDED")
                print(f"Total steps: {step_count}")
                print("="*70)

                choice = input("\nPlay again? (y/n): ").strip().lower()
                if choice == 'y':
                    obs, info = env.reset()
                    step_count = 0
                    print("\n✓ Game reset!")
                    print_action_menu()
                else:
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Quitting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

    env.close()
    print("\n✓ Session ended.")


if __name__ == "__main__":
    main()
