#!/usr/bin/env python3
"""
PureJaxRL Training Script for HackMatrix.

This script trains an action-masked PPO agent on the HackMatrix environment
using pure JAX for maximum performance on TPU/GPU.

Features:
- Chunked training loop for real-time WandB logging
- Auto-generated run names with resume support
- Checkpoint artifacts upload to WandB

Usage:
    python scripts/train_purejaxrl.py
    python scripts/train_purejaxrl.py --num-envs 512 --total-timesteps 100000000
    python scripts/train_purejaxrl.py --no-wandb  # disable wandb logging
    python scripts/train_purejaxrl.py --resume checkpoints/run-name/checkpoint_100.pkl

Example TPU usage:
    python scripts/train_purejaxrl.py --num-envs 2048 --total-timesteps 1000000000
"""

# MARK: Imports

import argparse
import os
import signal
import sys
import time

# Enable JAX compilation cache (must be set before importing jax)
# This caches compiled XLA programs to disk, speeding up subsequent runs
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    cache_dir = os.path.join(os.path.dirname(__file__), "..", ".jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir

import jax

# Add python directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hackmatrix.purejaxrl import (
    HackMatrixGymnax,
    TrainConfig,
    get_device_config,
    make_chunked_train,
)
from hackmatrix.purejaxrl.checkpointing import (
    save_checkpoint,
    save_params_npz,
)
from hackmatrix.purejaxrl.config import auto_tune_for_device
from hackmatrix.purejaxrl.logging import TrainingLogger, print_config
from hackmatrix.run_utils import (
    derive_run_id,
    generate_run_name,
    get_run_name_from_checkpoint_dir,
)


# MARK: Argument Parsing


def parse_args():
    parser = argparse.ArgumentParser(description="Train HackMatrix agent with PureJaxRL")

    # Environment
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Number of parallel environments (default: 256)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="Steps per rollout (default: 128)",
    )

    # Training duration
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps (default: 10M)",
    )

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coef")
    parser.add_argument(
        "--ent-coef", type=float, default=0.1, help="Entropy coef (0.1+ recommended)"
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

    # Network
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--project", type=str, default="hackmatrix", help="WandB project name")
    parser.add_argument("--entity", type=str, default="charles-team", help="WandB entity/team")
    parser.add_argument(
        "--run-name", type=str, default=None, help="WandB run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional suffix for auto-generated run name (e.g., 'test' -> hackmatrix-jax-jan25-26-1-test)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (e.g., 'checkpoints/hackmatrix-jax-feb01-26-1/checkpoint_40.pkl')",
    )

    # Checkpointing
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Save every N updates (default: use --save-interval-minutes instead)",
    )
    parser.add_argument(
        "--save-interval-minutes",
        type=float,
        default=10.0,
        help="Save checkpoint every N minutes (default: 10). Ignored if --save-interval is set.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--no-artifact", action="store_true", help="Disable checkpoint artifact uploads to WandB"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--auto-tune", action="store_true", help="Auto-tune config for device")

    return parser.parse_args()


# MARK: Main


def main():
    args = parse_args()

    # Print device info
    device_info = get_device_config()
    print(f"\nDevice: {device_info['device_type'].upper()}")
    print(f"Device count: {device_info['device_count']}")
    print(f"Backend: {device_info['backend']}")

    # Create config
    config = TrainConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        log_interval=args.log_interval,
        save_interval=args.save_interval if args.save_interval is not None else 1_000_000,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    # Auto-tune for device if requested
    if args.auto_tune:
        config = auto_tune_for_device(config)
        print("Config auto-tuned for device")

    # Check that total_timesteps is sufficient for meaningful training
    min_updates = 3
    min_timesteps = config.batch_size * min_updates
    if config.total_timesteps < min_timesteps:
        print(f"\nWarning: total_timesteps ({config.total_timesteps:,}) is very small")
        print(f"   batch_size = {config.batch_size:,} (num_envs × num_steps)")
        print(f"   Increasing to {min_timesteps:,} for at least {min_updates} updates")
        config = TrainConfig(**{**config.__dict__, "total_timesteps": min_timesteps})

    print_config(config)

    # Create environment
    env = HackMatrixGymnax()
    key = jax.random.PRNGKey(args.seed)

    # Determine run name and checkpoint directory
    # Structure: checkpoint_dir/run_name/checkpoint_*.pkl
    if args.resume:
        # Extract run directory from checkpoint file path
        # e.g., 'checkpoints/hackmatrix-jax-feb01-26-1/checkpoint_40.pkl' -> 'checkpoints/hackmatrix-jax-feb01-26-1'
        run_checkpoint_dir = os.path.dirname(args.resume)
        run_name = get_run_name_from_checkpoint_dir(run_checkpoint_dir)
        if run_name:
            print(f"Resuming run: {run_name}")
            print(f"From checkpoint: {args.resume}")
        else:
            print(f"Warning: Could not extract run name from {args.resume}")
            print("Using directory name as run name")
            run_name = os.path.basename(os.path.normpath(run_checkpoint_dir))
    else:
        # New run - generate name or use provided
        run_name = args.run_name
        if run_name is None:
            run_name = generate_run_name(
                base_dir=config.checkpoint_dir,
                prefix="hackmatrix-jax",
                run_suffix=args.run_suffix,
            )
            print(f"Generated run name: {run_name}")
        run_checkpoint_dir = os.path.join(config.checkpoint_dir, run_name)
        os.makedirs(run_checkpoint_dir, exist_ok=True)

    # Derive consistent run_id from run_name for wandb resume
    run_id = derive_run_id(run_name) if run_name else None

    # Build wandb config dict
    wandb_config = {
        # Training config
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "total_timesteps": config.total_timesteps,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "num_minibatches": config.num_minibatches,
        "update_epochs": config.update_epochs,
        "clip_eps": config.clip_eps,
        "vf_coef": config.vf_coef,
        "ent_coef": config.ent_coef,
        "max_grad_norm": config.max_grad_norm,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "seed": config.seed,
        # Device info
        "device_type": device_info["device_type"],
        "device_count": device_info["device_count"],
        "backend": device_info["backend"],
    }

    # Initialize logger
    logger = TrainingLogger(
        use_wandb=not args.no_wandb,
        project_name=args.project,
        entity=args.entity,
        run_name=run_name,
        run_id=run_id,
        resume_run=bool(args.resume),  # Just indicate whether resuming, run_id handles the ID
        config=wandb_config,
        upload_artifacts=not args.no_artifact,
    )

    # Check for checkpoint to resume from
    checkpoint_path = None
    resume_from_checkpoint = False
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint_path = args.resume  # Pass the specific file path
            resume_from_checkpoint = True
            # Load last_logged_step from checkpoint to avoid wandb step conflicts
            import pickle

            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
                last_logged_step = ckpt.get("last_logged_step", ckpt["step"])
                logger.set_resume_step(last_logged_step)
                print(f"Setting wandb resume step to {last_logged_step} (from checkpoint)")
        else:
            print(f"Warning: Checkpoint file not found: {args.resume}")
            print("Starting fresh training")

    # Track last checkpoint (step or time based)
    # Only use resume_step if we actually loaded a checkpoint
    effective_start_step = logger.resume_step if resume_from_checkpoint else 0
    last_checkpoint_step = effective_start_step
    last_checkpoint_time = time.time()
    use_time_based = args.save_interval is None
    save_interval_seconds = args.save_interval_minutes * 60

    if use_time_based:
        print(f"Checkpointing every {args.save_interval_minutes} minutes")
    else:
        print(f"Checkpointing every {args.save_interval} updates")

    # Track latest state for save-on-interrupt
    latest_state = {"runner_state": None, "step": 0}

    def handle_interrupt(signum, frame):
        """Save checkpoint on Ctrl+C."""
        print("\n\nInterrupt received, saving checkpoint...")
        if latest_state["runner_state"] is not None:
            interrupted_path = os.path.join(run_checkpoint_dir, "interrupted_params.npz")
            save_params_npz(latest_state["runner_state"].train_state.params, interrupted_path)
            print(f"Saved interrupted checkpoint to {interrupted_path}")
            # Also save full checkpoint for proper resume
            save_checkpoint(
                latest_state["runner_state"].train_state,
                run_checkpoint_dir,
                latest_state["step"],
                logger=logger if not args.no_artifact else None,
                last_logged_step=logger.last_logged_step,
            )
        else:
            print("No state to save yet (training hasn't started)")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    def log_callback(metrics: dict, step: int):
        """Callback for logging metrics after each chunk."""
        logger.log_metrics(metrics, step)

    def checkpoint_callback(runner_state, step: int):
        """Callback for checkpointing (time-based or step-based)."""
        nonlocal last_checkpoint_step, last_checkpoint_time

        # Always update latest state for interrupt handler
        latest_state["runner_state"] = runner_state
        latest_state["step"] = step

        should_save = False
        if use_time_based:
            elapsed = time.time() - last_checkpoint_time
            should_save = elapsed >= save_interval_seconds
        else:
            should_save = step - last_checkpoint_step >= args.save_interval

        if should_save:
            save_checkpoint(
                runner_state.train_state,
                run_checkpoint_dir,
                step,
                logger=logger if not args.no_artifact else None,
                last_logged_step=logger.last_logged_step,
            )
            last_checkpoint_step = step
            last_checkpoint_time = time.time()

    train_fn = make_chunked_train(
        config,
        env,
        chunk_size=config.log_interval,
        log_fn=log_callback,
        checkpoint_fn=checkpoint_callback,
        start_step=effective_start_step,
        checkpoint_path=checkpoint_path,
    )

    print("Compiling training function (first chunk)...")
    start_time = time.time()
    final_state, all_metrics, _ = train_fn(key)  # _ is last_logged_step (already used)
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    # Print final metrics
    total_steps = config.num_updates * config.batch_size
    print("\nFinal metrics:")
    for key_name in ["total_loss", "pg_loss", "vf_loss", "entropy", "mean_reward"]:
        if key_name in all_metrics:
            val = all_metrics[key_name]
            if hasattr(val, "__len__"):
                print(f"  {key_name}: {float(val[-1]):.4f}")
            else:
                print(f"  {key_name}: {float(val):.4f}")

    # Save final checkpoint
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    final_params_path = os.path.join(run_checkpoint_dir, "final_params.npz")
    save_params_npz(final_state.train_state.params, final_params_path)

    # Upload final checkpoint as artifact if enabled
    if not args.no_wandb and not args.no_artifact:
        logger.log_checkpoint_artifact(final_params_path, config.num_updates, "final-model")

    logger.finish()

    print(f"\nTraining complete! Total timesteps: {total_steps:,}")


if __name__ == "__main__":
    main()
