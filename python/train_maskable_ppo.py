"""
Train a MaskablePPO agent to play 868-HACK with action masking.
"""

import os
from datetime import datetime
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from hack_env import HackEnv


def mask_fn(env: HackEnv) -> np.ndarray:
    """Return action mask for current state."""
    return env._get_action_mask()


def make_env():
    """Create and wrap the environment."""
    env = HackEnv()
    return ActionMasker(env, mask_fn)


def train(
        total_timesteps: int = 1_000_000,
        save_freq: int = 10_000,
        eval_freq: int = 5_000,
        log_dir: str = "./logs",
        model_dir: str = "./models"
):
    """
    Train a MaskablePPO agent with action masking.

    Args:
        total_timesteps: Total number of timesteps to train for
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        log_dir: Directory for TensorBoard logs
        model_dir: Directory to save models
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"maskable_ppo_{timestamp}")
    run_model_dir = os.path.join(model_dir, f"maskable_ppo_{timestamp}")
    os.makedirs(run_model_dir, exist_ok=True)

    print("Creating environment...")
    env = DummyVecEnv([make_env])

    print("Creating eval environment...")
    eval_env = DummyVecEnv([make_env])

    print("Initializing MaskablePPO agent...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=run_log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=run_model_dir,
        name_prefix="maskable_ppo_hack",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Logs: {run_log_dir}")
    print(f"Models: {run_model_dir}")
    print("\nTo monitor training, run:")
    print(f"  tensorboard --logdir {log_dir}")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        final_model_path = os.path.join(run_model_dir, "final_model")
        model.save(final_model_path)
        print(f"\nTraining complete! Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupted_model_path = os.path.join(run_model_dir, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for 868-HACK")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total timesteps to train (default: 1,000,000)")
    parser.add_argument("--save-freq", type=int, default=10_000,
                        help="Save checkpoint every N steps (default: 10,000)")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Evaluate every N steps (default: 5,000)")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Directory to save models")

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )
