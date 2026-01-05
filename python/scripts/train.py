"""
Train a MaskablePPO agent to play HackMatrix with action masking.
"""

# MARK: Imports

import hashlib
import os
import uuid
from datetime import datetime
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers import TimeLimit

import wandb
from wandb.integration.sb3 import WandbCallback

from hackmatrix import HackEnv
from hackmatrix.training_db import TrainingDB


# MARK: Helper Functions

def mask_fn(env: HackEnv) -> np.ndarray:
    """Return action mask for current state."""
    return env._get_action_mask()


# MARK: Custom Callbacks

class EpisodeStatsCallback(BaseCallback):
    """Callback to log episode statistics to W&B and SQLite."""

    def __init__(self, run_id: str, wandb_enabled: bool = True, db_path: str = "training_history.db", verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.run_id = run_id
        self.wandb_enabled = wandb_enabled
        self.db = TrainingDB(db_path)

    def _on_step(self) -> bool:
        # Check for episode ends in info buffer
        for info in self.locals.get("infos", []):
            if "episode_stats" in info:
                stats = info["episode_stats"]
                self.episode_count += 1
                breakdown = stats["reward_breakdown"]

                # Log to W&B (if enabled)
                if self.wandb_enabled:
                    wandb.log({
                        "episode/count": self.episode_count,
                        "episode/reward_stage": breakdown.get("stage", 0),
                        "episode/reward_kills": breakdown.get("kills", 0),
                        "episode/reward_distance": breakdown.get("distance", 0),
                        "episode/reward_score": breakdown.get("score", 0),
                        "episode/reward_dataSiphon": breakdown.get("dataSiphon", 0),
                        "episode/reward_victory": breakdown.get("victory", 0),
                        "episode/reward_death": breakdown.get("death", 0),
                        "episode/programs_used": stats["programs_used"],
                        "episode/highest_stage": stats["highest_stage"],
                        "episode/steps": stats["steps"],
                        "episode/action_moves": stats["action_counts"].get("move", 0),
                        "episode/action_siphons": stats["action_counts"].get("siphon", 0),
                        "episode/action_programs": stats["action_counts"].get("program", 0),
                    }, step=self.num_timesteps)

                # Log to SQLite
                self.db.log_episode(
                    run_id=self.run_id,
                    episode_num=self.episode_count,
                    timestep=self.num_timesteps,
                    stats=stats
                )

                # Console log with percentages (using absolute values so they sum to 100%)
                total_reward = sum(breakdown.values())
                abs_total = sum(abs(v) for v in breakdown.values()) or 1  # Avoid div by zero
                def pct(key, label=None):
                    label = label or key
                    return f"{label}:{breakdown.get(key,0)/abs_total*100:.0f}%"
                # print(f"Ep {self.episode_count} | R={total_reward:.2f} | "
                #       f"{pct('stage')} {pct('kills')} {pct('distance', 'dist')} {pct('score')} {pct('dataSiphon', 'siphon')} {pct('death')} | "
                #       f"S{stats['highest_stage']} | {stats['steps']} steps | "
                #       f"prog:{stats['programs_used']} moves:{stats['action_counts'].get('move',0)} "
                #       f"siph:{stats['action_counts'].get('siphon',0)}")

        return True

    def _on_training_end(self) -> None:
        """Close database connection when training ends."""
        self.db.close()


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
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        no_wandb: bool = False
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
        max_episode_steps: Maximum steps per episode before truncation (default: 1000)
        no_wandb: Disable Weights & Biases logging
    """
    # MARK: Setup Directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # When resuming, reuse the existing run directory to keep TensorBoard graphs continuous
    if resume_path:
        # Extract run directory from resume path (e.g., models/hackmatrix-dec29-25-1/checkpoint.zip)
        resume_dir = os.path.dirname(resume_path)
        run_name = os.path.basename(resume_dir)
        run_log_dir = os.path.join(log_dir, run_name)
        run_model_dir = resume_dir
        print(f"Resuming run: {run_name}")
    else:
        # New run - create friendly name like hackmatrix-dec29-25-1
        date_prefix = datetime.now().strftime("hackmatrix-%b%d-%y").lower()  # e.g., hackmatrix-dec29-25

        # Find next available number
        existing = [d for d in os.listdir(model_dir) if d.startswith(date_prefix)] if os.path.exists(model_dir) else []
        next_num = 1
        for name in existing:
            try:
                num = int(name.split("-")[-1])
                next_num = max(next_num, num + 1)
            except ValueError:
                pass

        run_name = f"{date_prefix}-{next_num}"
        run_log_dir = os.path.join(log_dir, run_name)
        run_model_dir = os.path.join(model_dir, run_name)
        os.makedirs(run_model_dir, exist_ok=True)

    # MARK: Model Hyperparameters (defined once, used everywhere)
    model_config = {
        "learning_rate": 3e-4,
        "n_steps": 4096,  # Increased from 2048 for better value estimates
        "batch_size": 64,
        "n_epochs": 20,  # Increased from 10 to train value function more
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.3,
        "vf_coef": 1.0,  # Increased from 0.5 to prioritize value function learning
        "policy_kwargs": {
            "net_arch": {
                "pi": [256, 256, 128],  # Policy: 3 layers (more capacity for complex decisions)
                "vf": [256, 256, 128],  # Value: 3 layers (better state evaluation)
            }
        }
    }

    # MARK: Initialize W&B
    # Derive run_id from run_name so it's consistent across resumes
    run_id = hashlib.md5(run_name.encode()).hexdigest()[:8]
    wandb_enabled = not no_wandb

    if wandb_enabled:
        wandb.init(
            project="hackmatrix",
            entity="charles-team",
            name=run_name,
            id=run_id,
            resume="allow",  # Create if new, resume if exists
            save_code=True,  # Save code snapshot
            config={
                # Model hyperparameters
                **model_config,

                # Reward structure
                "reward_stage_multipliers": [1, 2, 4, 8, 16, 32, 64, 100],
                "reward_score_multiplier": 0.5,
                "reward_kill": 0.3,
                "reward_data_siphon": 1.0,
                "reward_distance_shaping": 0.05,
                "reward_victory_base": 500,
                "reward_victory_score_mult": 100,
                "reward_death_penalty_pct": 0.5,

                # Environment
                "num_envs": num_envs,
                "max_episode_steps": max_episode_steps,
                "total_timesteps": total_timesteps,
            },
            sync_tensorboard=True,
        )
        print("📊 W&B logging: ENABLED")
    else:
        print("📊 W&B logging: DISABLED (use without --no-wandb to enable)")

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
        env = ActionMasker(env, mask_fn)  # ActionMasker needs direct access to HackEnv
        env = TimeLimit(env, max_episode_steps=max_episode_steps)  # Truncate long episodes
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
            **model_config  # Use shared config defined above
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

    # Create callbacks
    episode_stats_callback = EpisodeStatsCallback(run_id=run_id, wandb_enabled=wandb_enabled)

    callbacks = [checkpoint_callback, eval_callback, episode_stats_callback]
    if wandb_enabled:
        wandb_callback = WandbCallback(
            model_save_path=run_model_dir,
            verbose=2,
        )
        callbacks.append(wandb_callback)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
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

        # Finish W&B run
        if wandb_enabled:
            wandb.finish()


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
    parser.add_argument("--max-episode-steps", type=int, default=1000,
                        help="Maximum steps per episode before truncation (default: 1000)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")

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
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        no_wandb=args.no_wandb
    )
