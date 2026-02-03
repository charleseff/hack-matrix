"""
Logging utilities for PureJaxRL training.

Provides console output and optional WandB integration with:
- Full config logging
- Auto-generated run names with resume support
- Checkpoint artifact uploads
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def generate_run_name(
    checkpoint_dir: str = "checkpoints",
    run_suffix: str | None = None,
) -> tuple[str, str]:
    """Generate auto-incrementing run name and ID.

    Format: hackmatrix-jax-{month}{day}-{year}-{N}[-suffix]
    Example: hackmatrix-jax-jan25-26-1, hackmatrix-jax-jan25-26-2-bignet

    Args:
        checkpoint_dir: Directory to scan for existing runs
        run_suffix: Optional suffix (e.g., 'test' -> hackmatrix-jax-jan25-26-1-test)

    Returns:
        run_name: Generated run name
        run_id: MD5-derived run ID for resume support
    """
    # Generate date prefix: hackmatrix-jax-jan25-26
    date_prefix = datetime.now().strftime("hackmatrix-jax-%b%d-%y").lower()

    # Find next available number by scanning checkpoint_dir
    next_num = 1
    if os.path.exists(checkpoint_dir):
        existing = [d for d in os.listdir(checkpoint_dir) if d.startswith(date_prefix)]
        if existing:
            # Extract numbers from existing run names
            nums = []
            for name in existing:
                parts = name.replace(date_prefix + "-", "").split("-")
                if parts and parts[0].isdigit():
                    nums.append(int(parts[0]))
            if nums:
                next_num = max(nums) + 1

    run_name = f"{date_prefix}-{next_num}"
    if run_suffix:
        run_name = f"{run_name}-{run_suffix}"

    # Derive run_id from run_name for consistent resume across Colab disconnects
    run_id = hashlib.md5(run_name.encode()).hexdigest()[:8]

    return run_name, run_id


@dataclass
class TrainingLogger:
    """Logger for training progress with enhanced WandB support.

    Features:
    - Console output with timing info
    - Full hyperparameter config logging to WandB
    - Auto-generated run names with resume support
    - Checkpoint artifact uploads
    """

    use_wandb: bool = False
    project_name: str = "hackmatrix"
    entity: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    resume_run: bool = False
    config: dict[str, Any] | None = None
    upload_artifacts: bool = True
    _wandb_run: Any = None
    _start_time: float = field(default=0.0, init=False)
    _last_log_time: float = field(default=0.0, init=False)
    _total_steps: int = field(default=0, init=False)
    _resume_step: int = field(default=0, init=False)

    def __post_init__(self):
        self._start_time = time.time()
        self._last_log_time = self._start_time

        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB with full configuration."""
        try:
            import wandb

            # Always use run_id for consistent wandb run identification
            # resume="allow" creates if new, resumes if exists
            resume_mode = "allow" if self.run_id else None

            self._wandb_run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=self.run_name,
                id=self.run_id,
                resume=resume_mode,
                config=self.config,
            )

            if self._wandb_run and self.config:
                print(f"WandB run initialized: {self._wandb_run.url}")

            # Track resume step for continued logging
            if self.resume_run and self._wandb_run:
                self._resume_step = self._get_last_step()
            else:
                self._resume_step = 0

        except ImportError:
            print("Warning: wandb not installed, disabling wandb logging")
            self.use_wandb = False
        except Exception as e:
            print(f"Warning: wandb init failed ({e}), disabling wandb logging")
            self.use_wandb = False

    def _get_last_step(self) -> int:
        """Get the last logged step from a resumed wandb run."""
        if not self._wandb_run:
            return 0
        try:
            # Get the last step from the run's history
            if hasattr(self._wandb_run, "summary") and "_step" in self._wandb_run.summary:
                return int(self._wandb_run.summary["_step"])
            return 0
        except Exception:
            return 0

    @property
    def resume_step(self) -> int:
        """Get the step to resume from (0 if not resuming)."""
        return self._resume_step

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "train",
    ):
        """Log training metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step (update number)
            prefix: Prefix for metric names
        """
        current_time = time.time()
        elapsed = current_time - self._start_time

        # Calculate steps per second based on update steps
        sps = step / elapsed if elapsed > 0 else 0

        # Console output
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(f"[Update {step:6d}] [{elapsed:.1f}s] [{sps:.1f} updates/s] {', '.join(metric_strs)}")

        # WandB logging
        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics[f"{prefix}/updates_per_second"] = sps
            wandb_metrics[f"{prefix}/update_step"] = step
            wandb.log(wandb_metrics, step=step)

        self._last_log_time = current_time

    def log_eval(self, metrics: dict[str, float], step: int):
        """Log evaluation metrics.

        Args:
            metrics: Evaluation metrics
            step: Current training step
        """
        print(f"[Eval @ {step}] " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics.items()))

        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)

    def log_checkpoint_artifact(
        self,
        checkpoint_path: str,
        step: int,
        artifact_name: str | None = None,
    ):
        """Upload checkpoint as WandB artifact.

        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step when checkpoint was saved
            artifact_name: Optional custom artifact name
        """
        if not self.use_wandb or not self.upload_artifacts:
            return

        if self._wandb_run is None:
            return

        try:
            import wandb

            name = artifact_name or f"checkpoint-{step}"
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            print(f"Uploaded checkpoint artifact: {name}")

        except Exception as e:
            print(f"Warning: Failed to upload artifact ({e})")

    def finish(self):
        """Clean up logging resources."""
        total_time = time.time() - self._start_time
        print(f"\nTraining completed in {total_time:.1f}s")

        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.finish()


def format_number(n: float) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    else:
        return f"{n:.0f}"


def print_config(config):
    """Print training configuration."""
    print("\n" + "=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"  num_envs: {config.num_envs}")
    print(f"  num_steps: {config.num_steps}")
    print(f"  total_timesteps: {format_number(config.total_timesteps)}")
    print(f"  num_updates: {config.num_updates}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  minibatch_size: {config.minibatch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  gamma: {config.gamma}")
    print(f"  gae_lambda: {config.gae_lambda}")
    print(f"  clip_eps: {config.clip_eps}")
    print(f"  vf_coef: {config.vf_coef}")
    print(f"  ent_coef: {config.ent_coef}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print("=" * 50 + "\n")
