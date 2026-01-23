"""
PureJaxRL Integration for HackMatrix.

This package provides Gymnax-compatible environment wrapper and
action-masked PPO implementation for JAX-based training.

Usage:
    from hackmatrix.purejaxrl import HackMatrixGymnax, make_train, TrainConfig

    config = TrainConfig()
    env = HackMatrixGymnax()
    train_fn = make_train(config, env)
    train_state = train_fn(jax.random.PRNGKey(0))
"""

from .env_wrapper import HackMatrixGymnax, EnvParams
from .masked_ppo import ActorCritic, Transition, masked_categorical, MaskedCategorical
from .config import TrainConfig, get_device_config
from .train import make_train

__all__ = [
    # Environment
    "HackMatrixGymnax",
    "EnvParams",
    # PPO
    "ActorCritic",
    "Transition",
    "masked_categorical",
    "MaskedCategorical",
    # Training
    "TrainConfig",
    "get_device_config",
    "make_train",
]
