"""
Shared utilities for training run management.

Used by both train.py (SB3) and train_purejaxrl.py (PureJaxRL).
"""

import hashlib
import os
from datetime import datetime


def generate_run_name(
    base_dir: str,
    prefix: str = "hackmatrix",
    run_suffix: str | None = None,
) -> str:
    """Generate auto-incrementing run name.

    Format: {prefix}-{month}{day}-{year}-{N}[-suffix]
    Example: hackmatrix-jan25-26-1, hackmatrix-jax-jan25-26-2-bignet

    Args:
        base_dir: Directory to scan for existing runs (e.g., 'checkpoints', 'models')
        prefix: Name prefix (e.g., 'hackmatrix', 'hackmatrix-jax')
        run_suffix: Optional suffix (e.g., 'test' -> hackmatrix-jan25-26-1-test)

    Returns:
        Generated run name
    """
    # Generate date prefix: {prefix}-jan25-26
    date_prefix = datetime.now().strftime(f"{prefix}-%b%d-%y").lower()

    # Find next available number by scanning base_dir for subdirectories
    next_num = 1
    if os.path.exists(base_dir):
        existing = [d for d in os.listdir(base_dir) if d.startswith(date_prefix)]
        if existing:
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

    return run_name


def derive_run_id(run_name: str) -> str:
    """Derive consistent run ID from run name.

    Uses MD5 hash so the same run_name always produces the same ID,
    enabling wandb resume across disconnects.

    Args:
        run_name: The run name (e.g., 'hackmatrix-jax-jan25-26-1')

    Returns:
        8-character hex ID
    """
    return hashlib.md5(run_name.encode()).hexdigest()[:8]


def get_run_name_from_checkpoint_dir(checkpoint_path: str) -> str | None:
    """Extract run name from checkpoint directory path.

    Assumes structure: {base_dir}/{run_name}/checkpoint_*.pkl

    Args:
        checkpoint_path: Path to checkpoint dir (e.g., 'checkpoints/hackmatrix-jax-jan25-26-1')

    Returns:
        Run name if valid structure, None otherwise
    """
    # Get the directory name
    run_name = os.path.basename(os.path.normpath(checkpoint_path))

    # Validate it looks like a run name (has date pattern)
    if run_name and "-" in run_name and any(c.isdigit() for c in run_name):
        return run_name
    return None


def find_latest_run_dir(base_dir: str) -> str | None:
    """Find the most recently modified run directory.

    Args:
        base_dir: Base checkpoint directory (e.g., 'checkpoints')

    Returns:
        Path to latest run dir, or None if no runs exist
    """
    if not os.path.exists(base_dir):
        return None

    # Get all subdirectories that look like run names
    run_dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and "-" in name:
            # Check if it has checkpoint files
            has_checkpoints = any(f.endswith(".pkl") for f in os.listdir(path))
            if has_checkpoints:
                run_dirs.append(path)

    if not run_dirs:
        return None

    # Return most recently modified
    return max(run_dirs, key=os.path.getmtime)
