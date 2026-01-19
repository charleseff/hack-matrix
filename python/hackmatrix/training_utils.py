"""Shared utility functions for training scripts."""

import numpy as np
from gymnasium.wrappers import TimeLimit
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from hackmatrix import HackEnv


def mask_fn(env: HackEnv) -> np.ndarray:
    """Return action mask for current state.

    Shared across all scripts that use action masking.
    """
    return env._get_action_mask()


def make_env(debug: bool = False, info: bool = False, max_episode_steps: int = 1000):
    """Create and wrap HackMatrix environment.

    Standard wrapper stack:
    - HackEnv: Core game logic
    - ActionMasker: Enforce valid actions via mask_fn
    - TimeLimit: Truncate long episodes
    - Monitor: Track episode statistics

    Args:
        debug: Enable verbose debug logging
        info: Enable info-level logging
        max_episode_steps: Maximum steps before truncation

    Returns:
        Wrapped environment ready for training/evaluation
    """
    env = HackEnv(debug=debug, info=info)
    env = ActionMasker(env, mask_fn)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env
