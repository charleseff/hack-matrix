"""
Training configuration for PureJaxRL.

Provides TrainConfig dataclass with PPO hyperparameters and
device detection utilities for CPU/GPU/TPU.
"""

from dataclasses import dataclass

import jax


@dataclass
class TrainConfig:
    """Training configuration with PPO hyperparameters.

    Default values are tuned for HackMatrix environment.
    """

    # Environment
    num_envs: int = 256
    num_steps: int = 128  # Steps per rollout

    # Training duration
    total_timesteps: int = 10_000_000

    # PPO hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_eps: float = 0.2  # PPO clipping epsilon
    vf_coef: float = 0.5  # Value function loss coefficient
    ent_coef: float = 0.1  # Entropy bonus (0.1+ prevents collapse)
    max_grad_norm: float = 0.5  # Gradient clipping

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2

    # Logging
    log_interval: int = 10  # Log every N updates
    eval_interval: int = 100  # Evaluate every N updates

    # Checkpointing
    save_interval: int = 1000  # Save every N updates
    checkpoint_dir: str = "checkpoints"

    # Random seed
    seed: int = 0

    # Derived values (computed)
    @property
    def num_updates(self) -> int:
        """Total number of PPO updates."""
        return self.total_timesteps // (self.num_envs * self.num_steps)

    @property
    def minibatch_size(self) -> int:
        """Size of each minibatch."""
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @property
    def batch_size(self) -> int:
        """Total batch size per update."""
        return self.num_envs * self.num_steps


def get_device_config() -> dict:
    """Detect available devices and return configuration.

    Returns:
        Dictionary with device info:
        - device_type: "cpu", "gpu", or "tpu"
        - device_count: Number of devices
        - backend: JAX backend name
    """
    devices = jax.devices()
    device = devices[0]
    device_type = device.platform

    return {
        "device_type": device_type,
        "device_count": len(devices),
        "backend": jax.default_backend(),
        "devices": devices,
    }


def auto_scale_for_batch_size(config: TrainConfig) -> TrainConfig:
    """Auto-scale training parameters when using large batch sizes.

    Problem:
        When num_envs increases (e.g., 256 -> 2048), batch_size grows proportionally.
        With the default num_minibatches=4, minibatch_size also grows 8x.
        Larger minibatches produce less noisy gradients, which can:
        - Reduce beneficial exploration noise
        - Cause the policy to converge to worse local optima
        - Make the same learning rate effectively more aggressive

    Solution:
        Scale num_minibatches proportionally with batch_size to maintain a
        consistent minibatch_size. This preserves the gradient noise characteristics
        that work well with smaller num_envs.

    Reference configuration (known to work well):
        - num_envs=256, num_steps=128 -> batch_size=32,768
        - num_minibatches=4 -> minibatch_size=8,192
        - update_epochs=4 -> 16 gradient steps per outer update

    With this auto-scaling for num_envs=2048:
        - batch_size=262,144 (8x larger)
        - num_minibatches=32 (scaled 8x) -> minibatch_size=8,192 (same!)
        - update_epochs=4 -> 128 gradient steps (8x, but also 8x more data)

    The gradient-steps-per-sample ratio stays constant, maintaining similar
    training dynamics regardless of num_envs.

    Args:
        config: Base configuration

    Returns:
        Configuration with scaled num_minibatches for large batches
    """
    # Reference values from the known-good 256-env configuration
    REFERENCE_BATCH_SIZE = 256 * 128  # 32,768
    REFERENCE_NUM_MINIBATCHES = 4
    REFERENCE_MINIBATCH_SIZE = REFERENCE_BATCH_SIZE // REFERENCE_NUM_MINIBATCHES  # 8,192

    current_batch_size = config.num_envs * config.num_steps

    # Only scale if batch is larger than reference
    if current_batch_size <= REFERENCE_BATCH_SIZE:
        return config

    # Calculate scale factor (how many times larger is current batch?)
    scale_factor = current_batch_size // REFERENCE_BATCH_SIZE

    # Scale num_minibatches to maintain similar minibatch_size
    # This keeps gradient noise characteristics consistent
    new_num_minibatches = config.num_minibatches * scale_factor

    # Sanity check: cap at reasonable value to avoid memory issues
    # 64 minibatches with 4 epochs = 256 gradient steps max
    new_num_minibatches = min(new_num_minibatches, 64)

    # Verify the resulting minibatch_size is reasonable
    new_minibatch_size = current_batch_size // new_num_minibatches

    print(f"\nAuto-scaling for large batch:")
    print(f"  batch_size: {current_batch_size:,} ({scale_factor}x reference)")
    print(f"  num_minibatches: {config.num_minibatches} -> {new_num_minibatches}")
    print(f"  minibatch_size: {current_batch_size // config.num_minibatches:,} -> {new_minibatch_size:,}")
    print(f"  gradient_steps: {config.num_minibatches * config.update_epochs} -> {new_num_minibatches * config.update_epochs}")

    return TrainConfig(
        num_envs=config.num_envs,
        num_steps=config.num_steps,
        total_timesteps=config.total_timesteps,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        num_minibatches=new_num_minibatches,
        update_epochs=config.update_epochs,
        clip_eps=config.clip_eps,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        max_grad_norm=config.max_grad_norm,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        save_interval=config.save_interval,
        checkpoint_dir=config.checkpoint_dir,
        seed=config.seed,
    )


def auto_tune_for_device(config: TrainConfig) -> TrainConfig:
    """Adjust config based on detected device.

    TPU: Increase parallelism
    GPU: Moderate parallelism
    CPU: Reduce parallelism for memory

    Args:
        config: Base configuration

    Returns:
        Tuned configuration for detected device
    """
    device_info = get_device_config()
    device_type = device_info["device_type"]
    device_count = device_info["device_count"]

    if device_type == "tpu":
        # TPU: maximize parallelism
        return TrainConfig(
            num_envs=config.num_envs * device_count,
            num_steps=config.num_steps,
            total_timesteps=config.total_timesteps,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            num_minibatches=config.num_minibatches * device_count,
            update_epochs=config.update_epochs,
            clip_eps=config.clip_eps,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            save_interval=config.save_interval,
            checkpoint_dir=config.checkpoint_dir,
            seed=config.seed,
        )
    elif device_type == "gpu":
        # GPU: moderate settings
        return config
    else:
        # CPU: reduce for memory
        return TrainConfig(
            num_envs=min(config.num_envs, 64),
            num_steps=config.num_steps,
            total_timesteps=config.total_timesteps,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            num_minibatches=min(config.num_minibatches, 2),
            update_epochs=config.update_epochs,
            clip_eps=config.clip_eps,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            save_interval=config.save_interval,
            checkpoint_dir=config.checkpoint_dir,
            seed=config.seed,
        )
