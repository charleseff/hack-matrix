#!/usr/bin/env python3
"""
Watch a trained MaskablePPO agent play HackMatrix in visual mode.
"""

import sys
import argparse
from pathlib import Path
from sb3_contrib import MaskablePPO
from hack_env import HackEnv


def watch_agent(model_path: str, episodes: int = 3, max_steps: int = 500, debug: bool = False, info: bool = False):
    """
    Load a trained agent and watch it play in visual mode.

    Args:
        model_path: Path to the saved model (e.g., "./models/maskable_ppo_20241218_120000/best_model.zip")
        episodes: Number of episodes to watch
        max_steps: Maximum steps per episode
        debug: Enable verbose debug logging
        info: Enable info-level logging (less verbose)
    """
    print(f"Loading model from: {model_path}")
    if debug:
        print("⚠️  Debug mode: ENABLED")
    elif info:
        print("ℹ️  Info mode: ENABLED")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("\nAvailable models:")
        models_dir = Path("models")
        if models_dir.exists():
            for run_dir in sorted(models_dir.iterdir(), reverse=True):
                if run_dir.is_dir() and run_dir.name.startswith("maskable_ppo_"):
                    print(f"  {run_dir}/")
                    for model_file in run_dir.glob("*.zip"):
                        print(f"    - {model_file.name}")
        return

    # Load model
    model = MaskablePPO.load(model_path)
    print("✓ Model loaded successfully!\n")

    # Create visual environment
    print("Starting visual environment...")
    env = HackEnv(visual=True, debug=debug, info=info)

    try:
        for episode in range(episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{episodes}")
            print('='*60)

            obs, info = env.reset()
            action_mask = info.get("action_mask")

            episode_reward = 0
            step_count = 0
            done = False

            while not done and step_count < max_steps:
                # Get action from model (with action masking)
                action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                action_mask = info.get("action_mask")

                episode_reward += reward
                step_count += 1
                done = terminated or truncated

            print(f"\n{'─'*60}")
            print(f"Episode {episode + 1} Results:")
            print(f"  Steps taken: {step_count}")
            print(f"  Total reward: {episode_reward:.3f}")

            # Try to get additional info if available
            try:
                # Info might not have these fields depending on when episode ended
                if hasattr(env, 'unwrapped'):
                    game_state = env.unwrapped._get_game_state()
                    if game_state:
                        print(f"  Final score: {game_state.get('score', 'N/A')}")
                        print(f"  Stage reached: {game_state.get('stage', 'N/A')}")
            except:
                pass

            print('─'*60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        env.close()
        print("\n✓ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch trained MaskablePPO agent play HackMatrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch best model from latest run
  python watch_agent.py models/maskable_ppo_20251218_120000/best_model.zip

  # Watch specific checkpoint
  python watch_agent.py models/maskable_ppo_20251218_120000/maskable_ppo_hack_100000_steps.zip

  # Watch 5 episodes
  python watch_agent.py models/maskable_ppo_20251218_120000/best_model.zip --episodes 5

  # Auto-find latest model
  python watch_agent.py --latest
        """
    )

    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        help="Path to saved model (.zip file)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to watch (default: 3)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically use the latest best_model.zip"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Enable info-level logging (important events only)"
    )

    args = parser.parse_args()

    # Handle --latest flag
    if args.latest:
        models_dir = Path("models")
        if not models_dir.exists():
            print("Error: models/ directory not found")
            sys.exit(1)

        # Find latest run directory
        run_dirs = sorted(
            [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("maskable_ppo_")],
            reverse=True
        )

        if not run_dirs:
            print("Error: No model directories found in models/")
            sys.exit(1)

        latest_run = run_dirs[0]

        # Try best_model.zip first, then interrupted_model.zip, then latest checkpoint
        candidates = [
            latest_run / "best_model.zip",
            latest_run / "interrupted_model.zip",
        ]

        # Add latest checkpoint
        checkpoints = sorted(latest_run.glob("maskable_ppo_hack_*_steps.zip"), reverse=True)
        if checkpoints:
            candidates.append(checkpoints[0])

        model_path = None
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                break

        if not model_path:
            print(f"Error: No model files found in {latest_run}")
            sys.exit(1)

        print(f"Using latest model: {model_path}\n")
        args.model_path = str(model_path)

    # Check if model_path provided
    if not args.model_path:
        parser.print_help()
        print("\nError: model_path is required (or use --latest)")
        sys.exit(1)

    watch_agent(args.model_path, args.episodes, args.max_steps, args.debug, args.info)
