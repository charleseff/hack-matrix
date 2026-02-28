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
import jax.numpy as jnp

# Add python directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hackmatrix.purejaxrl import (
    HackMatrixGymnax,
    RunnerState,
    TrainConfig,
    get_device_config,
    make_chunked_train,
)
from hackmatrix.purejaxrl.checkpointing import (
    save_checkpoint,
    save_params_npz,
    save_phase_snapshot,
)
from hackmatrix.purejaxrl.env_wrapper import EnvParams
from hackmatrix.purejaxrl.config import auto_scale_for_batch_size, auto_tune_for_device
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
        "--ent-coef", type=float, default=0.15, help="Entropy coef (0.15 maintains exploration)"
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
    parser.add_argument(
        "--rewind",
        action="store_true",
        help="When resuming, rewind wandb history to checkpoint step (overwrites later data)",
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

    # Curriculum learning
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Enable curriculum learning (3-phase progressive difficulty)",
    )
    parser.add_argument(
        "--curriculum-start-phase", type=int, default=None, choices=[1, 2, 3],
        help="Override starting curriculum phase (1-3). Use with --resume to rollback to a phase.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--auto-tune", action="store_true", help="Auto-tune config for device")

    return parser.parse_args()


# MARK: Curriculum Phases


CURRICULUM_PHASES = {
    1: EnvParams(
        starting_data_siphons=jnp.int32(2),
        starting_credits=jnp.int32(3),
        starting_energy=jnp.int32(3),
        transmission_scale=jnp.float32(0.375),  # [0,0,1,1,2,2,3,3] ≈ 0.375 * [1..8]
        siphon_death_penalty=jnp.float32(-2.0),
        distance_shaping_coef=jnp.float32(0.025),
        data_siphon_reward=jnp.float32(2.0),
    ),
    2: EnvParams(
        starting_data_siphons=jnp.int32(1),
        starting_credits=jnp.int32(1),
        starting_energy=jnp.int32(1),
        transmission_scale=jnp.float32(0.625),  # [1,1,2,2,3,3,4,4] ≈ 0.625 * [1..8]
        siphon_death_penalty=jnp.float32(-5.0),
        distance_shaping_coef=jnp.float32(0.04),
        data_siphon_reward=jnp.float32(1.5),
    ),
    3: EnvParams(),  # Full difficulty (all defaults)
}

# Phase transition thresholds
PHASE_TRANSITION = {
    1: {"return_threshold": -1.0, "consecutive_required": 20, "fallback_updates": 300},
    2: {"return_threshold": -1.2, "consecutive_required": 20, "fallback_updates": 300},
}


def check_phase_transition(
    phase: int,
    consecutive_above: int,
    updates_in_phase: int,
    mean_return: float,
) -> tuple[bool, int]:
    """Check if current phase should transition to next.

    Returns:
        (should_transition, updated_consecutive_count)
    """
    if phase >= 3:
        return False, 0

    thresholds = PHASE_TRANSITION[phase]
    new_consecutive = consecutive_above + 1 if mean_return > thresholds["return_threshold"] else 0
    should_transition = (
        new_consecutive >= thresholds["consecutive_required"]
        or updates_in_phase >= thresholds["fallback_updates"]
    )
    return should_transition, new_consecutive


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

    # Auto-scale num_minibatches for large batch sizes
    # This maintains consistent minibatch_size (and gradient noise) regardless of num_envs
    # Skip if user explicitly set num_minibatches to a non-default value
    if args.num_minibatches == 4:  # Only auto-scale if using default
        config = auto_scale_for_batch_size(config)

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

    # Determine curriculum phase
    current_phase = 1 if args.curriculum else 3
    if args.curriculum_start_phase is not None:
        current_phase = args.curriculum_start_phase
        if not args.curriculum:
            print("Note: --curriculum-start-phase implies --curriculum")
            args.curriculum = True

    env_params = CURRICULUM_PHASES[current_phase] if args.curriculum else EnvParams()

    if args.curriculum:
        print(f"\nCurriculum learning enabled, starting at Phase {current_phase}")
        print(f"  transmission_scale={float(env_params.transmission_scale):.3f}")
        print(f"  starting_siphons={int(env_params.starting_data_siphons)}")
        print(f"  distance_shaping={float(env_params.distance_shaping_coef):.3f}")
        print(f"  siphon_death_penalty={float(env_params.siphon_death_penalty):.1f}")
        print(f"  data_siphon_reward={float(env_params.data_siphon_reward):.1f}")

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
            suffix = "curriculum" if args.curriculum else args.run_suffix
            run_name = generate_run_name(
                base_dir=config.checkpoint_dir,
                prefix="hackmatrix-jax",
                run_suffix=suffix,
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
        # Curriculum
        "curriculum": args.curriculum,
        "curriculum_start_phase": current_phase,
    }

    # Check for checkpoint to resume from (before logger init for rewind support)
    checkpoint_path = None
    resume_from_checkpoint = False
    checkpoint_step = None
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint_path = args.resume
            resume_from_checkpoint = True
            # Load step from checkpoint
            import pickle

            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
                checkpoint_step = ckpt.get("last_logged_step", ckpt["step"])
        else:
            print(f"Warning: Checkpoint file not found: {args.resume}")
            print("Starting fresh training")

    # Initialize logger (with rewind_step if --rewind flag is set)
    rewind_step = checkpoint_step if (args.rewind and checkpoint_step is not None) else None
    logger = TrainingLogger(
        use_wandb=not args.no_wandb,
        project_name=args.project,
        entity=args.entity,
        run_name=run_name,
        run_id=run_id,
        resume_run=bool(args.resume),
        rewind_step=rewind_step,
        config=wandb_config,
        upload_artifacts=not args.no_artifact,
    )

    # Set resume step if resuming without rewind
    if resume_from_checkpoint and checkpoint_step is not None and not args.rewind:
        logger.set_resume_step(checkpoint_step)
        print(f"Setting wandb resume step to {checkpoint_step} (from checkpoint)")

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
                curriculum_phase=current_phase if args.curriculum else None,
            )
        else:
            print("No state to save yet (training hasn't started)")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    # Curriculum tracking state
    consecutive_above_threshold = 0
    updates_in_current_phase = 0

    def log_callback(metrics: dict, step: int):
        """Callback for logging metrics after each chunk."""
        if args.curriculum:
            metrics["curriculum/phase"] = current_phase
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
                curriculum_phase=current_phase if args.curriculum else None,
            )
            last_checkpoint_step = step
            last_checkpoint_time = time.time()

    if not args.curriculum:
        # Non-curriculum: use the existing single-pass training
        train_fn = make_chunked_train(
            config,
            env,
            chunk_size=config.log_interval,
            log_fn=log_callback,
            checkpoint_fn=checkpoint_callback,
            start_step=effective_start_step,
            checkpoint_path=checkpoint_path,
            env_params=env_params,
        )

        print("Compiling training function (first chunk)...")
        start_time = time.time()
        final_state, all_metrics, _ = train_fn(key)
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
    else:
        # Curriculum: manual chunk loop with phase transitions
        from hackmatrix.purejaxrl.training_loop import init_runner_state, make_train_chunk

        chunk_size = config.log_interval
        train_chunk_fn = make_train_chunk(config, env, chunk_size)

        print("Initializing training state...")
        runner_state, last_logged_step = init_runner_state(
            config, env, key, start_step=effective_start_step,
            checkpoint_path=checkpoint_path, env_params=env_params,
        )
        if last_logged_step > 0 and not args.rewind:
            logger.set_resume_step(last_logged_step)

        num_updates = config.num_updates
        num_chunks = num_updates // chunk_size

        print(f"Compiling training function (first chunk of Phase {current_phase})...")
        start_time = time.time()

        for chunk_idx in range(num_chunks):
            runner_state, chunk_metrics = train_chunk_fn(runner_state)

            chunk_metrics_np = jax.tree.map(lambda x: jax.device_get(x), chunk_metrics)
            step = int(runner_state.update_step)

            # Log metrics
            from hackmatrix.purejaxrl.training_loop import aggregate_chunk_metrics
            aggregated = aggregate_chunk_metrics(chunk_metrics_np)
            log_callback(aggregated, step)
            checkpoint_callback(runner_state, step)

            # Check phase transition (using aggregated mean_episode_return)
            mean_return = aggregated.get("mean_episode_return", -999.0)
            updates_in_current_phase += chunk_size

            should_transition, consecutive_above_threshold = check_phase_transition(
                current_phase,
                consecutive_above_threshold,
                updates_in_current_phase,
                mean_return,
            )

            if should_transition:
                # Save phase snapshot before transitioning
                print(f"\n{'='*60}")
                print(f"Phase {current_phase} complete at step {step}!")
                print(f"  mean_return={mean_return:.3f}, updates_in_phase={updates_in_current_phase}")
                save_phase_snapshot(
                    runner_state.train_state,
                    run_checkpoint_dir,
                    step,
                    current_phase,
                    last_logged_step=logger.last_logged_step,
                    logger=logger if not args.no_artifact else None,
                )

                # Advance phase
                current_phase += 1
                env_params = CURRICULUM_PHASES[current_phase]
                consecutive_above_threshold = 0
                updates_in_current_phase = 0

                print(f"Transitioning to Phase {current_phase}")
                print(f"  transmission_scale={float(env_params.transmission_scale):.3f}")
                print(f"  starting_siphons={int(env_params.starting_data_siphons)}")
                print(f"  distance_shaping={float(env_params.distance_shaping_coef):.3f}")
                print(f"  siphon_death_penalty={float(env_params.siphon_death_penalty):.1f}")
                print(f"  data_siphon_reward={float(env_params.data_siphon_reward):.1f}")
                print(f"{'='*60}\n")

                # Swap env_params in runner state (environments will use new params on next reset)
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=runner_state.env_state,
                    obs=runner_state.obs,
                    key=runner_state.key,
                    update_step=runner_state.update_step,
                    episode_returns=runner_state.episode_returns,
                    env_params=env_params,
                )

                # Recompile chunk function with new env_params
                # (env_params is now carried in RunnerState and used by _update_step)
                # No recompile needed since env_params flows through RunnerState

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        final_state = runner_state

    # Print final metrics
    total_steps = config.num_updates * config.batch_size
    print(f"\nTraining complete! Total timesteps: {total_steps:,}")
    if args.curriculum:
        print(f"Final curriculum phase: {current_phase}")

    # Save final checkpoint
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    final_params_path = os.path.join(run_checkpoint_dir, "final_params.npz")
    save_params_npz(final_state.train_state.params, final_params_path)

    # Upload final checkpoint as artifact if enabled
    if not args.no_wandb and not args.no_artifact:
        logger.log_checkpoint_artifact(final_params_path, config.num_updates, "final-model")

    logger.finish()


if __name__ == "__main__":
    main()
