#!/usr/bin/env python3
"""
Watch a trained JAX/PureJaxRL agent play HackMatrix in visual mode.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hackmatrix import HackEnv
from hackmatrix.purejaxrl.checkpointing import load_params_npz, unflatten_params
from hackmatrix.purejaxrl.masked_ppo import ActorCritic
from hackmatrix.jax_env import NUM_ACTIONS


def flatten_observation(obs_dict: dict) -> jnp.ndarray:
    """Convert HackEnv dict observation to flat array for JAX network.

    HackEnv returns:
        player: (10,) float32
        programs: (23,) int32
        grid: (6, 6, 42) float32

    JAX network expects: (1545,) = 10 + 23 + 1512
    """
    player = obs_dict["player"].astype(np.float32)
    programs = obs_dict["programs"].astype(np.float32)
    grid = obs_dict["grid"].flatten().astype(np.float32)

    return jnp.concatenate([player, programs, grid])


def watch_agent(
    model_path: str,
    episodes: int = 3,
    max_steps: int = 500,
    hidden_dim: int = 256,
    num_layers: int = 2,
    debug: bool = False,
    info: bool = False,
):
    """
    Load a trained JAX agent and watch it play in visual mode.

    Args:
        model_path: Path to the saved params (.npz file)
        episodes: Number of episodes to watch
        max_steps: Maximum steps per episode
        hidden_dim: Hidden layer dimension (must match training)
        num_layers: Number of hidden layers (must match training)
        debug: Enable verbose debug logging
        info: Enable info-level logging (less verbose)
    """
    print(f"Loading JAX model from: {model_path}")
    if debug:
        print("Debug mode: ENABLED")
    elif info:
        print("Info mode: ENABLED")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("\nLooking for .npz files in python/checkpoints/:")
        checkpoints_dir = Path("python/checkpoints")
        if checkpoints_dir.exists():
            for npz_file in sorted(checkpoints_dir.glob("*.npz"), reverse=True):
                print(f"  - {npz_file}")
        return

    # Load parameters
    print("Loading parameters...")
    flat_params = load_params_npz(model_path)
    params = unflatten_params(flat_params)
    print(f"Loaded {len(flat_params)} parameter arrays")

    # Create network with same architecture as training
    obs_dim = 10 + 23 + (6 * 6 * 42)  # 1545
    network = ActorCritic(
        action_dim=NUM_ACTIONS,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    # Verify params match network by doing a test forward pass
    dummy_obs = jnp.zeros((obs_dim,))
    try:
        logits, value = network.apply(params, dummy_obs)
        print(f"Network initialized: {logits.shape[0]} actions, value={value:.3f}")
    except Exception as e:
        print(f"Error: Parameters don't match network architecture: {e}")
        print(f"  Expected: hidden_dim={hidden_dim}, num_layers={num_layers}")
        print("  Try adjusting --hidden-dim and --num-layers to match training config")
        return

    # JIT compile the forward pass for speed
    @jax.jit
    def get_action(params, obs, action_mask):
        """Get greedy action from network with masking."""
        logits, value = network.apply(params, obs)
        # Mask invalid actions
        masked_logits = jnp.where(action_mask, logits, -1e9)
        action = jnp.argmax(masked_logits)
        return action, value

    print("Model loaded successfully!\n")

    # Create visual environment
    print("Starting visual environment...")
    env = HackEnv(visual=True, debug=debug, info=info)

    try:
        for episode in range(episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{episodes}")
            print("=" * 60)

            obs_dict, info_dict = env.reset()
            obs = flatten_observation(obs_dict)
            action_mask = jnp.array(info_dict.get("action_mask", np.ones(NUM_ACTIONS, dtype=bool)))

            episode_reward = 0
            step_count = 0
            done = False

            while not done and step_count < max_steps:
                # Get action from JAX model (greedy/deterministic)
                action, value = get_action(params, obs, action_mask)
                action = int(action)

                # Take step
                obs_dict, reward, terminated, truncated, info_dict = env.step(action)
                obs = flatten_observation(obs_dict)
                action_mask = jnp.array(info_dict.get("action_mask", np.ones(NUM_ACTIONS, dtype=bool)))

                episode_reward += reward
                step_count += 1
                done = terminated or truncated

            print(f"\n{'-'*60}")
            print(f"Episode {episode + 1} Results:")
            print(f"  Steps taken: {step_count}")
            print(f"  Total reward: {episode_reward:.3f}")

            # Show episode stats if available
            if "episode_stats" in info_dict:
                stats = info_dict["episode_stats"]
                print(f"  Highest stage: {stats.get('highest_stage', 'N/A')}")
                print(f"  Programs used: {stats.get('programs_used', 'N/A')}")

            print("-" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        env.close()
        print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch trained JAX/PureJaxRL agent play HackMatrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch model from checkpoints
  python watch_jax_agent.py python/checkpoints/final_params.npz

  # Watch with custom architecture
  python watch_jax_agent.py python/checkpoints/final_params.npz --hidden-dim 512 --num-layers 3

  # Watch 5 episodes
  python watch_jax_agent.py python/checkpoints/final_params.npz --episodes 5

  # Auto-find latest checkpoint
  python watch_jax_agent.py --latest
        """,
    )

    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        help="Path to saved params (.npz file)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to watch (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension (default: 256, must match training)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2, must match training)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically use the latest .npz file from python/checkpoints/",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Enable info-level logging (important events only)",
    )

    args = parser.parse_args()

    # Handle --latest flag
    if args.latest:
        checkpoints_dir = Path("python/checkpoints")
        if not checkpoints_dir.exists():
            print("Error: python/checkpoints/ directory not found")
            sys.exit(1)

        # Find npz files
        npz_files = list(checkpoints_dir.glob("*.npz"))
        if not npz_files:
            print("Error: No .npz files found in python/checkpoints/")
            sys.exit(1)

        # Prefer final_params.npz, otherwise use most recent
        final_params = checkpoints_dir / "final_params.npz"
        if final_params.exists():
            model_path = final_params
        else:
            model_path = sorted(npz_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        print(f"Using checkpoint: {model_path}\n")
        args.model_path = str(model_path)

    # Check if model_path provided
    if not args.model_path:
        parser.print_help()
        print("\nError: model_path is required (or use --latest)")
        sys.exit(1)

    watch_agent(
        args.model_path,
        args.episodes,
        args.max_steps,
        args.hidden_dim,
        args.num_layers,
        args.debug,
        args.info,
    )
