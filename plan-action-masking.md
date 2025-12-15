# Action Masking Implementation Plan

## Task Checklist

### Phase 1: Complete Environment Action Masking
- ☐ Implement `_get_action_mask()` method in `hack_env.py`
- ☐ Update `step()` to return action mask in info dict
- ☐ Remove DEBUG logging statements

### Phase 2: MaskablePPO Training
- ☐ Add `sb3-contrib` to `requirements.txt`
- ☐ Create `train_maskable_ppo.py` with MaskablePPO algorithm
- ☐ Configure MaskablePPO to use action masks from environment

---

## Phase 1: Complete Environment Action Masking

**Affected files:**
- `python/hack_env.py`

**Changes:**
- Add `_get_action_mask()` helper that converts valid actions list to boolean numpy array
- Update `step()` to include action mask in returned info dict
- Remove DEBUG print statements from `reset()` and `step()`

### Implementation Details

#### Add `_get_action_mask()` method (line ~220, before `close()`)

```python
def _get_action_mask(self) -> np.ndarray:
    """Get action mask for MaskablePPO (boolean array of valid actions)."""
    valid_actions = self.get_valid_actions()
    mask = np.zeros(self.action_space.n, dtype=np.bool_)
    mask[valid_actions] = True
    return mask
```

#### Update `step()` method (line 199-214)

Replace:
```python
def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
    """Execute one step."""
    print(f"DEBUG: step() called with action={action}, type={type(action)}")  # DEBUG
    response = self._send_command({
        "action": "step",
        "actionIndex": int(action)
    })
    print(f"DEBUG: step() response received")  # DEBUG

    observation = self._observation_to_array(response["observation"])
    reward = float(response["reward"])
    terminated = bool(response["done"])
    truncated = False
    info = response.get("info", {})

    return observation, reward, terminated, truncated, info
```

With:
```python
def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
    """Execute one step."""
    response = self._send_command({
        "action": "step",
        "actionIndex": int(action)
    })

    observation = self._observation_to_array(response["observation"])
    reward = float(response["reward"])
    terminated = bool(response["done"])
    truncated = False
    info = response.get("info", {})

    # Add action mask for next state
    if not terminated:
        info["action_mask"] = self._get_action_mask()

    return observation, reward, terminated, truncated, info
```

#### Remove DEBUG logging from `reset()` (line 185-197)

Replace:
```python
def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Reset the environment."""
    print(f"DEBUG: reset() called")  # DEBUG
    super().reset(seed=seed)

    response = self._send_command({"action": "reset"})
    print(f"DEBUG: reset() response received")  # DEBUG
    observation = self._observation_to_array(response["observation"])

    # Get action mask for MaskablePPO
    action_mask = self._get_action_mask()

    return observation, {"action_mask": action_mask}
```

With:
```python
def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Reset the environment."""
    super().reset(seed=seed)

    response = self._send_command({"action": "reset"})
    observation = self._observation_to_array(response["observation"])

    return observation, {"action_mask": self._get_action_mask()}
```

---

## Phase 2: MaskablePPO Training

**Affected files:**
- `python/requirements.txt` (add dependency)
- `python/train_maskable_ppo.py` (new file)

**Changes:**
- Add sb3-contrib package for MaskablePPO implementation
- Create training script using MaskablePPO with action masking enabled

### Implementation Details

#### Update `requirements.txt`

Add after stable-baselines3:
```
sb3-contrib>=2.0.0
```

#### Create `train_maskable_ppo.py`

New file with MaskablePPO configuration:

```python
"""
Train a MaskablePPO agent to play 868-HACK with action masking.
"""

import os
from datetime import datetime

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from hack_env import HackEnv


def make_env():
    """Create and wrap the environment."""
    return HackEnv()


def train(
        total_timesteps: int = 1_000_000,
        save_freq: int = 10_000,
        eval_freq: int = 5_000,
        log_dir: str = "./logs",
        model_dir: str = "./models"
):
    """
    Train a MaskablePPO agent with action masking.

    Args:
        total_timesteps: Total number of timesteps to train for
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        log_dir: Directory for TensorBoard logs
        model_dir: Directory to save models
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"maskable_ppo_{timestamp}")
    run_model_dir = os.path.join(model_dir, f"maskable_ppo_{timestamp}")
    os.makedirs(run_model_dir, exist_ok=True)

    print("Creating environment...")
    env = DummyVecEnv([make_env])

    print("Creating eval environment...")
    eval_env = DummyVecEnv([make_env])

    print("Initializing MaskablePPO agent...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=run_log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=run_model_dir,
        name_prefix="maskable_ppo_hack",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Logs: {run_log_dir}")
    print(f"Models: {run_model_dir}")
    print("\nTo monitor training, run:")
    print(f"  tensorboard --logdir {log_dir}")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
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

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for 868-HACK")
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

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )
```
