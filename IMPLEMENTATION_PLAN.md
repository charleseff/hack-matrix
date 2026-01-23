# PureJaxRL Integration Implementation Plan

## Current State

The PureJaxRL integration is complete for Phases 1-6 (CPU testing). All core functionality is implemented and tested.

### Test Results
- **12/12 PureJaxRL tests passing** (`python/tests/test_purejaxrl.py`)
- **350/350 total tests passing** (all tests pass!)
- **CPU training verified** (both quick and moderate-length runs successful)

### Completed
- [x] JAX environment implementation (`python/hackmatrix/jax_env/`)
- [x] All 154 parity tests pass
- [x] Environment supports `reset()`, `step()`, `get_valid_actions()`
- [x] Batched versions available via `jax.vmap`
- [x] Core dependencies installed (flax, optax, chex, jax, jaxlib)
- [x] `distrax` package for masked categorical distribution
- [x] Gymnax-compatible wrapper
- [x] Action-masked PPO implementation
- [x] Training infrastructure (config, logging, checkpointing)
- [x] Main training script (`train_purejaxrl.py`)
- [x] Smoke tests

## Implementation Tasks

### Phase 1: Dependencies and Setup - COMPLETE

- [x] **1.1 Add distrax to requirements.txt**
  - File: `python/requirements.txt`
  - Added: `distrax>=0.1.3`

### Phase 2: Environment Wrapper - COMPLETE

- [x] **2.1 Create purejaxrl package structure**
  - Created: `python/hackmatrix/purejaxrl/__init__.py`

- [x] **2.2 Implement Gymnax-compatible wrapper**
  - File: `python/hackmatrix/purejaxrl/env_wrapper.py`
  - Components:
    - `EnvParams` empty dataclass
    - `HackMatrixGymnax` class with Gymnax interface:
      - `reset(key, params) -> (obs, state)`
      - `step(key, state, action, params) -> (obs, state, reward, done, info)`
      - `get_action_mask(state) -> action_mask`
    - `_flatten_obs()` to convert structured observation to flat array
      - Shape: (10 + 23 + 6*6*42,) = (1545,)

### Phase 3: Action-Masked PPO - COMPLETE

- [x] **3.1 Implement masked categorical distribution**
  - File: `python/hackmatrix/purejaxrl/masked_ppo.py`
  - `masked_categorical(logits, mask)` using distrax
  - Masks invalid actions by setting logits to -1e9

- [x] **3.2 Implement ActorCritic network**
  - File: `python/hackmatrix/purejaxrl/masked_ppo.py`
  - Flax nn.Module with configurable hidden layers
  - Outputs: (logits, value) tuple

- [x] **3.3 Implement Transition dataclass**
  - File: `python/hackmatrix/purejaxrl/masked_ppo.py`
  - Fields: obs, action, reward, done, log_prob, value, action_mask

- [x] **3.4 Implement PPO loss function**
  - File: `python/hackmatrix/purejaxrl/masked_ppo.py`
  - Clipped surrogate loss
  - Value function loss
  - Entropy bonus
  - Uses stored action_mask for consistent log_prob

- [x] **3.5 Implement training step**
  - File: `python/hackmatrix/purejaxrl/masked_ppo.py`
  - `_env_step()` - collect transition with action mask
  - `_update_epoch()` - minibatch PPO updates
  - GAE computation

### Phase 4: Training Infrastructure - COMPLETE

- [x] **4.1 Implement training configuration**
  - File: `python/hackmatrix/purejaxrl/config.py`
  - `TrainConfig` dataclass with PPO hyperparameters
  - `get_device_config()` for CPU/GPU/TPU detection
  - `auto_tune_for_device()` to adjust num_envs/num_steps

- [x] **4.2 Implement training loop**
  - File: `python/hackmatrix/purejaxrl/train.py`
  - `make_train(config, env)` returns JIT-compiled train function
  - Rollout collection
  - PPO updates
  - Logging hooks

- [x] **4.3 Add logging utilities**
  - File: `python/hackmatrix/purejaxrl/logging.py`
  - Episode return tracking
  - WandB integration (optional)
  - Console progress output

- [x] **4.4 Add checkpointing**
  - File: `python/hackmatrix/purejaxrl/checkpointing.py`
  - Save/load model parameters with orbax or simple pickle

### Phase 5: Training Script - COMPLETE

- [x] **5.1 Create main training script**
  - File: `python/scripts/train_purejaxrl.py`
  - CLI argument parsing
  - Config loading
  - Training execution
  - Results reporting

- [x] **5.2 Add smoke test**
  - File: `python/tests/test_purejaxrl.py`
  - Test env wrapper shapes
  - Test PPO compiles and runs 1 step
  - Test action masking works correctly

### Phase 6: TPU Deployment - IN PROGRESS

- [x] **6.1 Test on CPU locally** - COMPLETE (verified with multiple test runs)
- [x] **6.2 Test on GPU (if available)** - N/A (no GPU in current environment)
- [ ] **6.3 Test on TPU (Google TRC)** - PENDING (awaiting TRC project approval)
- [x] **6.4 Document TPU-specific setup** - COMPLETE (`python/docs/TPU_SETUP.md`)

## File Structure

```
python/hackmatrix/
├── jax_env/           # JAX environment (complete)
│   ├── __init__.py
│   ├── state.py
│   ├── env.py
│   └── ...
└── purejaxrl/         # PureJaxRL integration (complete)
    ├── __init__.py
    ├── env_wrapper.py   # Gymnax-compatible wrapper
    ├── masked_ppo.py    # Action-masked PPO
    ├── config.py        # Training configuration
    ├── train.py         # Training loop
    ├── logging.py       # Logging utilities
    └── checkpointing.py # Model checkpointing

python/scripts/
├── train_jax.py         # Existing skeleton (will be replaced/renamed)
└── train_purejaxrl.py   # Main training entry point

python/tests/
└── test_purejaxrl.py    # Smoke tests (12/12 passing)
```

## Key Technical Details

### Observation Flattening

```python
def _flatten_obs(obs: Observation) -> jax.Array:
    """Flatten structured observation to single array."""
    return jnp.concatenate([
        obs.player_state,            # (10,)
        obs.programs.astype(jnp.float32),  # (23,)
        obs.grid.ravel(),            # (6*6*42 = 1512,)
    ])  # Total: 1545
```

### Masked Categorical Distribution

```python
def masked_categorical(logits, mask):
    """Sample from categorical with invalid actions masked."""
    masked_logits = jnp.where(mask, logits, -1e9)
    return distrax.Categorical(logits=masked_logits)
```

### Training Config Defaults

```python
@dataclass
class TrainConfig:
    num_envs: int = 256
    num_steps: int = 128
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    num_layers: int = 2
```

## Success Criteria

1. **Training runs on CPU** - COMPLETE (Development and testing without TPU)
2. **Training runs on GPU** - PENDING (Local GPU acceleration if available)
3. **Training runs on TPU** - PENDING (Production training on Google TRC)
4. **Action masking works** - COMPLETE (Invalid actions never selected)
5. **Simple CLI interface** - COMPLETE (Single script to run training)

## Dependencies

```
distrax>=0.1.3        # JAX probability distributions (installed)
```

## Future Work

Phase 6 (TPU Deployment) remains as future work:
- Test on different hardware configurations (CPU, GPU, TPU)
- Optimize for TPU-specific requirements
- Document TPU setup process for Google TRC
- Performance benchmarking across platforms

## Notes

- The spec mentions 42 grid features per cell in the wrapper, but the actual JAX env uses 42 features (see `observation.py:GRID_FEATURES`). The spec also mentions 1545 total obs size (10+23+1512) which matches.
- PureJaxRL does not include action masking natively - we implement it by masking logits before categorical sampling
- The existing `train_jax.py` skeleton can be kept as reference or removed after `train_purejaxrl.py` is complete
