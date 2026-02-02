#!/usr/bin/env python3
"""
PureJaxRL Training Script for HackMatrix.

This script trains an action-masked PPO agent on the HackMatrix environment
using pure JAX for maximum performance on TPU/GPU.

Features:
- Chunked training loop for real-time WandB logging
- Auto-generated run names with resume support
- Checkpoint artifacts upload to WandB
- Performance benchmarking mode

Usage:
    python scripts/train_purejaxrl.py
    python scripts/train_purejaxrl.py --num-envs 512 --total-timesteps 100000000
    python scripts/train_purejaxrl.py --wandb --project hackmatrix
    python scripts/train_purejaxrl.py --wandb --resume-run abc123

Example TPU usage:
    python scripts/train_purejaxrl.py --num-envs 2048 --total-timesteps 1000000000
"""

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
    make_train,
)
from hackmatrix.purejaxrl.checkpointing import (
    get_checkpoint_steps,
    save_checkpoint,
    save_params_npz,
)
from hackmatrix.purejaxrl.config import auto_tune_for_device
from hackmatrix.purejaxrl.logging import TrainingLogger, generate_run_name, print_config


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
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
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
        "--resume-run",
        type=str,
        default=None,
        help="WandB run ID to resume (loads checkpoint + continues wandb logging)",
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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark (compare wandb vs no-wandb overhead)",
    )
    parser.add_argument(
        "--monolithic",
        action="store_true",
        help="Use monolithic training loop (no real-time logging, for comparison)",
    )

    return parser.parse_args()


def run_benchmark(config: TrainConfig, env: HackMatrixGymnax, key: jax.Array):
    """Run performance benchmark comparing monolithic vs chunked training.

    Args:
        config: Training configuration
        env: Environment instance
        key: JAX random key

    Returns:
        overhead_percent: Percentage overhead of chunked vs monolithic
    """
    print("\n" + "=" * 50)
    print("Performance Benchmark")
    print("=" * 50)

    # Use fewer updates for benchmark (100 updates = ~10 chunks of 10)
    benchmark_config = TrainConfig(
        **{**config.__dict__, "total_timesteps": config.batch_size * 100}
    )
    print(f"Benchmark config: {benchmark_config.num_updates} updates")

    # Monolithic timing
    print("\nRunning monolithic training...")
    train_mono = make_train(benchmark_config, env)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # Warmup (compile) - block until complete
    print("  Warmup (compiling)...")
    warmup_state, _ = train_mono(key1)
    jax.tree.map(lambda x: x.block_until_ready(), warmup_state)

    # Timed run with fresh key
    print("  Timing monolithic...")
    start = time.time()
    state, _ = train_mono(key2)
    # Block until computation actually completes
    jax.tree.map(lambda x: x.block_until_ready(), state)
    mono_time = time.time() - start
    print(f"  Monolithic: {mono_time:.2f}s")

    # Chunked timing (with dummy log_fn to simulate overhead)
    print("\nRunning chunked training...")

    def dummy_log(metrics, step):
        pass  # Just the callback overhead

    train_chunked = make_chunked_train(benchmark_config, env, log_fn=dummy_log)

    # Warmup (compile) - chunked compiles on first run
    print("  Warmup (compiling)...")
    warmup_state2, _ = train_chunked(key3)
    jax.tree.map(lambda x: x.block_until_ready(), warmup_state2.train_state)

    # Timed run
    print("  Timing chunked...")
    start = time.time()
    state2, _ = train_chunked(key4)
    # Block until computation actually completes
    jax.tree.map(lambda x: x.block_until_ready(), state2.train_state)
    chunked_time = time.time() - start
    print(f"  Chunked: {chunked_time:.2f}s")

    # Calculate overhead
    if mono_time > 0.01:  # Avoid division by near-zero
        overhead = (chunked_time - mono_time) / mono_time * 100
        print(f"\nOverhead: {overhead:.2f}%")

        if overhead < 5.0:  # Relaxed from 1% since CPU has more overhead
            print("PASS: Overhead acceptable")
        else:
            print("Note: Some overhead expected on CPU. GPU/TPU should be <1%")
    else:
        print("\nMonolithic time too short for accurate measurement")
        overhead = 0

    print("=" * 50)
    return overhead


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

    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(config, env, key)
        return

    # Generate run name if not provided (for wandb)
    run_name = args.run_name
    run_id = args.resume_run  # Use resume_run as run_id if resuming
    if args.wandb and run_name is None and not args.resume_run:
        run_name, run_id = generate_run_name(
            checkpoint_dir=config.checkpoint_dir,
            run_suffix=args.run_suffix,
        )
        print(f"Generated run name: {run_name}")

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
        use_wandb=args.wandb,
        project_name=args.project,
        entity=args.entity,
        run_name=run_name,
        run_id=run_id,
        resume_run=args.resume_run,
        config=wandb_config,
        upload_artifacts=not args.no_artifact,
    )

    # Choose training mode
    if args.monolithic:
        # Use original monolithic training (no real-time logging)
        print("\nUsing monolithic training (no real-time logging)")
        train_fn = make_train(config, env)

        print("Compiling training function...")
        start_compile = time.time()
        final_state, all_metrics = train_fn(key)
        compile_time = time.time() - start_compile
        print(f"Compilation + training completed in {compile_time:.1f}s")

        # Log final metrics only
        final_metrics = {
            "total_loss": float(all_metrics["total_loss"][-1]),
            "pg_loss": float(all_metrics["pg_loss"][-1]),
            "vf_loss": float(all_metrics["vf_loss"][-1]),
            "entropy": float(all_metrics["entropy"][-1]),
            "mean_reward": float(all_metrics["mean_reward"][-1]),
        }
        logger.log_metrics(final_metrics, config.num_updates)

    else:
        # Use chunked training with real-time logging
        print("\nUsing chunked training (real-time logging enabled)")

        # Check for checkpoint to resume from
        checkpoint_path = None
        if args.resume_run:
            checkpoint_steps = get_checkpoint_steps(config.checkpoint_dir)
            if checkpoint_steps:
                checkpoint_path = config.checkpoint_dir
                print(f"Found checkpoints at steps: {checkpoint_steps}")
                print(f"Will resume from latest checkpoint (step {max(checkpoint_steps)})")
            else:
                print(
                    f"Warning: --resume-run specified but no checkpoints found in {config.checkpoint_dir}"
                )
                print("Starting fresh training (wandb run will still be resumed)")

        # Track last checkpoint (step or time based)
        last_checkpoint_step = logger.resume_step if args.resume_run else 0
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
                interrupted_path = os.path.join(config.checkpoint_dir, "interrupted_params.npz")
                save_params_npz(latest_state["runner_state"].train_state.params, interrupted_path)
                print(f"Saved interrupted checkpoint to {interrupted_path}")
                # Also save full checkpoint for proper resume
                save_checkpoint(
                    latest_state["runner_state"].train_state,
                    config.checkpoint_dir,
                    latest_state["step"],
                    logger=logger if not args.no_artifact else None,
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
                    config.checkpoint_dir,
                    step,
                    logger=logger if not args.no_artifact else None,
                )
                last_checkpoint_step = step
                last_checkpoint_time = time.time()

        train_fn = make_chunked_train(
            config,
            env,
            chunk_size=config.log_interval,
            log_fn=log_callback,
            checkpoint_fn=checkpoint_callback,
            start_step=logger.resume_step,
            checkpoint_path=checkpoint_path,
        )

        print("Compiling training function (first chunk)...")
        start_time = time.time()
        final_state, all_metrics = train_fn(key)
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
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    final_params_path = os.path.join(config.checkpoint_dir, "final_params.npz")
    save_params_npz(final_state.train_state.params, final_params_path)

    # Upload final checkpoint as artifact if enabled
    if args.wandb and not args.no_artifact:
        logger.log_checkpoint_artifact(final_params_path, config.num_updates, "final-model")

    logger.finish()

    print(f"\nTraining complete! Total timesteps: {total_steps:,}")


if __name__ == "__main__":
    main()
