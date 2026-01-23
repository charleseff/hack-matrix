"""
Checkpointing utilities for PureJaxRL training.

Provides save/load functionality for model parameters.
Uses simple pickle/numpy format for portability.
"""

import os
import pickle
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState


def save_checkpoint(
    train_state: TrainState,
    path: str,
    step: int,
    metrics: Optional[Dict[str, float]] = None,
):
    """Save training checkpoint.

    Args:
        train_state: Current training state
        path: Directory to save checkpoint
        step: Current training step
        metrics: Optional metrics to save with checkpoint
    """
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "step": step,
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
        "metrics": metrics or {},
    }

    checkpoint_path = os.path.join(path, f"checkpoint_{step}.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    # Also save params as numpy for easy loading
    params_path = os.path.join(path, f"params_{step}.npz")
    flat_params = flatten_params(train_state.params)
    np.savez(params_path, **flat_params)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    path: str,
    train_state: TrainState,
    step: Optional[int] = None,
) -> tuple[TrainState, int, Dict[str, float]]:
    """Load training checkpoint.

    Args:
        path: Directory containing checkpoints
        train_state: Training state to restore into
        step: Specific step to load (None = latest)

    Returns:
        train_state: Restored training state
        step: Training step
        metrics: Saved metrics
    """
    if step is None:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(path) if f.startswith("checkpoint_")]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {path}")
        steps = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
        step = max(steps)

    checkpoint_path = os.path.join(path, f"checkpoint_{step}.pkl")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Restore params and opt_state
    train_state = train_state.replace(
        params=checkpoint["params"],
        opt_state=checkpoint["opt_state"],
    )

    print(f"Loaded checkpoint from {checkpoint_path}")
    return train_state, checkpoint["step"], checkpoint["metrics"]


def save_params_npz(params: Dict[str, Any], path: str):
    """Save model parameters as numpy .npz file.

    This format is easy to load for inference in other frameworks.

    Args:
        params: Model parameters (nested dict)
        path: Path to save .npz file
    """
    flat_params = flatten_params(params)
    np.savez(path, **flat_params)
    print(f"Saved parameters to {path}")


def load_params_npz(path: str) -> Dict[str, np.ndarray]:
    """Load model parameters from .npz file.

    Args:
        path: Path to .npz file

    Returns:
        Flattened parameter dictionary
    """
    with np.load(path) as data:
        return dict(data)


def flatten_params(params: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    """Flatten nested parameter dict to flat dict with dotted keys.

    Args:
        params: Nested parameter dictionary
        prefix: Key prefix for recursion

    Returns:
        Flat dictionary with numpy arrays
    """
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_params(value, full_key))
        else:
            flat[full_key] = np.array(value)
    return flat


def unflatten_params(flat_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Unflatten dotted key dict back to nested dict.

    Args:
        flat_params: Flat parameter dictionary

    Returns:
        Nested parameter dictionary
    """
    nested = {}
    for key, value in flat_params.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = jnp.array(value)
    return nested


def infer_architecture(flat_params: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Infer network architecture from saved parameters.

    Args:
        flat_params: Flattened parameter dictionary from load_params_npz

    Returns:
        Dict with 'hidden_dim' and 'num_layers'
    """
    # Find all Dense layer kernels
    dense_kernels = {}
    for key, value in flat_params.items():
        if "Dense_" in key and ".kernel" in key:
            # Extract layer number from key like "params.Dense_0.kernel"
            parts = key.split(".")
            for part in parts:
                if part.startswith("Dense_"):
                    layer_num = int(part.split("_")[1])
                    dense_kernels[layer_num] = value.shape
                    break

    if not dense_kernels:
        raise ValueError("No Dense layers found in parameters")

    # hidden_dim is the output dim of Dense_0
    hidden_dim = dense_kernels[0][1]

    # num_layers = total Dense layers - 2 (actor and critic heads)
    num_layers = len(dense_kernels) - 2

    return {"hidden_dim": hidden_dim, "num_layers": num_layers}


def get_checkpoint_steps(path: str) -> list:
    """Get list of available checkpoint steps.

    Args:
        path: Checkpoint directory

    Returns:
        Sorted list of step numbers
    """
    if not os.path.exists(path):
        return []

    checkpoints = [f for f in os.listdir(path) if f.startswith("checkpoint_")]
    steps = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
    return sorted(steps)
