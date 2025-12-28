"""
Train a MaskablePPO agent to play HackMatrix with action masking.
"""

# MARK: Imports

import os
from datetime import datetime
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from hackmatrix import HackEnv


# MARK: Helper Functions

def mask_fn(env: HackEnv) -> np.ndarray:
    """Return action mask for current state."""
    return env._get_action_mask()


# MARK: Training Function

def train(
        total_timesteps: int = 1_000_000,
        save_freq: int = 10_000,
        eval_freq: int = 5_000,
        log_dir: str = "./logs",
        model_dir: str = "./models",
        resume_path: str = None,
        debug: bool = False,
        info: bool = False,
        num_envs: int = 1
):
    """
    Train a MaskablePPO agent with action masking.

    Args:
        total_timesteps: Total number of timesteps to train for
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        log_dir: Directory for TensorBoard logs
        model_dir: Directory to save models
        resume_path: Path to checkpoint to resume from
        debug: Enable verbose debug logging
        info: Enable info-level logging (less verbose)
        num_envs: Number of parallel environments (1=single, 4-8=parallel)
    """
    # MARK: Setup Directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # When resuming, reuse the existing run directory to keep TensorBoard graphs continuous
    if resume_path:
        # Extract run directory from resume path (e.g., models/maskable_ppo_20241223_120000/checkpoint.zip)
        resume_dir = os.path.dirname(resume_path)
        run_name = os.path.basename(resume_dir)
        run_log_dir = os.path.join(log_dir, run_name)
        run_model_dir = resume_dir
        print(f"Resuming run: {run_name}")
    else:
        # New run - create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_dir = os.path.join(log_dir, f"maskable_ppo_{timestamp}")
        run_model_dir = os.path.join(model_dir, f"maskable_ppo_{timestamp}")
        os.makedirs(run_model_dir, exist_ok=True)

    # MARK: Configure Logging
    if debug:
        print("⚠️  Debug mode: ENABLED (verbose logging)")
    elif info:
        print("ℹ️  Info mode: ENABLED (important events only)")

    if num_envs > 1:
        print(f"🚀 Parallel mode: {num_envs} environments")
    else:
        print("📍 Single environment mode")

    # MARK: Create Environments

    def make_env():
        """Create and wrap the environment."""
        env = HackEnv(debug=debug, info=info)
        env = ActionMasker(env, mask_fn)  # ActionMasker needs access to HackEnv
        env = Monitor(env)  # Monitor goes on the outside to track episode statistics
        return env

    print("Creating environment...")
    if num_envs > 1:
        # Use SubprocVecEnv for parallel environments (faster on multi-core CPUs)
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
    else:
        # Use DummyVecEnv for single environment (easier debugging)
        env = DummyVecEnv([make_env])

    print("Creating eval environment...")
    # Always use single environment for evaluation (deterministic)
    eval_env = DummyVecEnv([make_env])

    # MARK: Initialize Model

    if resume_path:
        print(f"Resuming training from: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=env,
            verbose=1,
            tensorboard_log=run_log_dir
        )
        # Increase exploration when resuming
        model.ent_coef = 0.3  # Increased from 0.1 for more exploration
        print(f"Model loaded successfully! Entropy coefficient set to {model.ent_coef}")
    else:
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
            ent_coef=0.3,  # High exploration to prevent entropy collapse (increased from 0.1)
        )

    # MARK: Setup Callbacks

    # Adjust save_freq for parallel environments (save_freq is per-environment)
    # With 4 envs and save_freq=10000, we save every 10000/(4 envs) = 2500 steps per env
    checkpoint_save_freq = max(save_freq // num_envs, 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_path=run_model_dir,
        name_prefix="maskable_ppo_hack",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # Adjust eval_freq for parallel environments
    checkpoint_eval_freq = max(eval_freq // num_envs, 1)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=checkpoint_eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        use_masking=True  # Enable action masking during evaluation
    )

    # MARK: Training Loop

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

    except RuntimeError as e:
        print(f"\n❌ Training crashed: {e}")
        print("Swift process likely crashed - check logs for details")
        crash_model_path = os.path.join(run_model_dir, "crash_recovery_model")
        model.save(crash_model_path)
        print(f"✓ Model saved to: {crash_model_path}")
        print(f"\nTo resume training:")
        print(f"  python scripts/train.py --resume {crash_model_path} --timesteps <remaining>")
        raise  # Re-raise to show full traceback

    finally:
        # Close environments gracefully (suppress errors from dead subprocesses)
        try:
            env.close()
        except (BrokenPipeError, EOFError):
            pass  # Subprocesses already dead after Ctrl-C

        try:
            eval_env.close()
        except (BrokenPipeError, EOFError):
            pass  # Subprocesses already dead after Ctrl-C


# MARK: Command-Line Interface

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for HackMatrix")
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g., './models/maskable_ppo_20241218_120000/maskable_ppo_hack_100000_steps.zip')")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging (Swift + Python)")
    parser.add_argument("--info", action="store_true",
                        help="Enable info-level logging (important events only)")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (1=single, 4-8=parallel, default: 1)")

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        resume_path=args.resume,
        debug=args.debug,
        info=args.info,
        num_envs=args.num_envs
    )
