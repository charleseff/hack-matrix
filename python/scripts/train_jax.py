"""
TPU-optimized training using pure JAX environment + PureJaxRL.

This is a skeleton script that demonstrates the JAX training pattern.
The actual training loop with PureJaxRL would be integrated here.

Run with: cd python && source venv/bin/activate && python scripts/train_jax.py
"""

import sys
from pathlib import Path

# Add python directory to path for hackmatrix import
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from typing import NamedTuple

from hackmatrix import jax_env


class Transition(NamedTuple):
    """Single environment transition for trajectory storage."""

    obs: jax_env.Observation
    action: jnp.int32
    reward: jnp.float32
    done: jnp.bool_
    next_obs: jax_env.Observation


def make_train(config: dict):
    """
    Create JIT-compiled training function.

    Args:
        config: Training configuration dict

    Returns:
        JIT-compiled train function
    """
    num_steps = config.get("num_steps", 100)

    def train(key: jax.Array):
        """
        Run training loop.

        Args:
            key: JAX PRNG key

        Returns:
            Training metrics dict
        """
        # Initialize
        key, env_key = jax.random.split(key)
        env_state, obs = jax_env.reset(env_key)

        # Collect some transitions (placeholder for actual training)
        total_reward = jnp.float32(0.0)
        total_steps = 0

        def step_fn(carry, _):
            """Single environment step."""
            env_state, obs, key, total_reward = carry
            key, action_key, step_key = jax.random.split(key, 3)

            # Get valid actions and sample uniformly (placeholder for policy)
            valid_mask = jax_env.get_valid_actions(env_state)
            valid_indices = jnp.where(valid_mask, size=4)[0]
            action_idx = jax.random.randint(action_key, (), 0, 4)
            action = valid_indices[action_idx]

            # Take step
            new_state, new_obs, reward, done = jax_env.step(
                env_state, action, step_key
            )

            # Reset if done
            key, reset_key = jax.random.split(key)
            reset_state, reset_obs = jax_env.reset(reset_key)
            env_state = jax.lax.cond(
                done, lambda: reset_state, lambda: new_state
            )
            obs = jax.lax.cond(done, lambda: reset_obs, lambda: new_obs)

            total_reward = total_reward + reward

            return (env_state, obs, key, total_reward), reward

        # Run for num_steps
        (final_state, final_obs, _, total_reward), rewards = jax.lax.scan(
            step_fn,
            (env_state, obs, key, total_reward),
            None,
            length=num_steps,
        )

        return {
            "total_reward": total_reward,
            "mean_reward": jnp.mean(rewards),
            "final_step_count": final_state.step_count,
        }

    return train


def main():
    """Main entry point for JAX training script."""
    print("=" * 60)
    print("JAX Environment Training Script")
    print("=" * 60)

    # Check available devices
    devices = jax.devices()
    print(f"\nJAX devices: {devices}")
    print(f"Default backend: {jax.default_backend()}")

    # Configuration
    config = {
        "num_steps": 1000,
    }
    print(f"\nConfig: {config}")

    # Create and JIT-compile training function
    print("\nCompiling training function...")
    train_fn = make_train(config)
    train_fn = jax.jit(train_fn)

    # Run training
    print("Running training...")
    key = jax.random.PRNGKey(42)
    result = train_fn(key)

    # Print results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Total reward: {float(result['total_reward']):.4f}")
    print(f"Mean reward: {float(result['mean_reward']):.6f}")
    print(f"Final step count: {int(result['final_step_count'])}")

    print("\nTraining complete!")
    print("\nNote: This is a skeleton script with random action selection.")
    print("PureJaxRL integration would replace the random policy with PPO.")


if __name__ == "__main__":
    main()
