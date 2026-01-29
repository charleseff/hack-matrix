# Wandb Integration for PureJaxRL

**Status:** Complete

## Overview

Add Weights & Biases integration to `scripts/train_purejaxrl.py` for experiment tracking on Google Colab and TPU Research Cloud (TRC).

## Goals

1. Real-time metric logging during training (not just at the end)
2. Feature parity with `scripts/train.py` wandb integration
3. Minimal performance overhead (<1%)
4. Works seamlessly on Colab and TRC

## Non-Goals

- Video/gif logging (headless environment, no renderer)
- Hyperparameter sweeps (future work)

## Changes

### 1. Delete `scripts/train_jax.py`

Skeleton script superseded by `train_purejaxrl.py`. Remove it.

### 2. Chunked Training Loop

Restructure training to enable logging between chunks:

```python
# Current: single jax.lax.scan for ALL updates
final_state, all_metrics = jax.lax.scan(_update_step, runner_state, None, length=num_updates)

# New: outer Python loop with inner JIT-compiled chunks
CHUNK_SIZE = 10  # updates per chunk (configurable via --log-interval)

for chunk in range(num_updates // CHUNK_SIZE):
    runner_state, chunk_metrics = train_chunk(runner_state)  # JIT-compiled
    log_metrics_to_wandb(chunk_metrics, step=chunk * CHUNK_SIZE)
```

**What is a chunk?**

| Term | Definition | Default |
|------|------------|---------|
| Step | 1 action in 1 env | - |
| Batch | `num_envs × num_steps` steps | 256 × 128 = 32,768 steps |
| Update | 1 PPO update (consumes 1 batch) | 32,768 steps |
| Chunk | Updates between wandb logs | 10 updates = 327,680 steps |

With default settings, 1 chunk ≈ 328K steps ≈ 2K episode completions ≈ 1-5 seconds wall time.

This keeps the hot loop fully JIT-compiled while allowing Python-side logging between chunks.

**Performance validation:** Add a `--benchmark` flag that runs with and without wandb to measure overhead. Target: <1% slowdown.

### 3. Wandb Features to Implement

#### 3.1 Config Logging

Log all hyperparameters at init:

```python
wandb.init(
    project="hackmatrix",
    entity="charles-team",
    name=run_name,
    id=run_id,  # Derived from run_name for resume support
    resume="allow",
    config={
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
        # Device info
        "device_type": device_info["device_type"],
        "device_count": device_info["device_count"],
        "backend": device_info["backend"],
    },
)
```

#### 3.2 Training Metrics (per chunk)

Log after each chunk completes:

| Metric | Description |
|--------|-------------|
| `train/total_loss` | Combined PPO loss |
| `train/pg_loss` | Policy gradient loss |
| `train/vf_loss` | Value function loss |
| `train/entropy` | Policy entropy |
| `train/mean_reward` | Mean reward in chunk |
| `train/mean_episode_length` | Mean episode length |
| `train/steps_per_second` | Training throughput |
| `train/update_step` | Current update number |

#### 3.3 Run Naming and Resume Support

Auto-generated run names with consistent IDs for Colab disconnect recovery:

```python
# Run naming: hackmatrix-jax-jan25-26-1 (similar to train.py but with -jax- prefix)
date_prefix = datetime.now().strftime("hackmatrix-jax-%b%d-%y").lower()

# Auto-increment: find next available number by scanning checkpoint_dir
existing = [d for d in os.listdir(checkpoint_dir) if d.startswith(date_prefix)]
next_num = max([int(name.split("-")[-1]) for name in existing], default=0) + 1
run_name = f"{date_prefix}-{next_num}"  # e.g., hackmatrix-jax-jan25-26-1

# Optional user suffix (e.g., --run-suffix bignet -> hackmatrix-jax-jan25-26-1-bignet)
if run_suffix:
    run_name = f"{run_name}-{run_suffix}"

# Derive run_id from run_name for consistent resume across Colab disconnects
run_id = hashlib.md5(run_name.encode()).hexdigest()[:8]

wandb.init(
    id=run_id,
    resume="allow",  # Resume if exists, create if new
    ...
)
```

#### 3.4 Checkpoint Artifacts

Save checkpoints to wandb for persistence (Colab local storage is ephemeral):

```python
# After saving local checkpoint
artifact = wandb.Artifact(f"checkpoint-{update_step}", type="model")
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)
```

### 4. CLI Changes

Update argument parser:

```python
# Existing (keep)
parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
parser.add_argument("--project", type=str, default="hackmatrix", help="WandB project")
parser.add_argument("--run-name", type=str, default=None, help="WandB run name")

# New
parser.add_argument("--entity", type=str, default="charles-team", help="WandB entity/team")
parser.add_argument("--run-suffix", type=str, default=None, help="Optional suffix (e.g., 'test' -> hackmatrix-jax-jan25-26-1-test)")
parser.add_argument("--resume-run", type=str, default=None, help="WandB run ID to resume")
parser.add_argument("--benchmark", action="store_true", help="Run perf benchmark (wandb vs no-wandb)")
parser.add_argument("--no-artifact", action="store_true", help="Disable checkpoint artifacts")
```

### 5. File Changes

| File | Change |
|------|--------|
| `scripts/train_jax.py` | Delete |
| `scripts/train_purejaxrl.py` | Add chunked loop, wandb integration |
| `hackmatrix/purejaxrl/train.py` | Refactor to return chunk-based training function |
| `hackmatrix/purejaxrl/logging.py` | Enhance with artifact support, resume logic |

## Testing

1. **Unit test:** Verify chunked training produces same results as monolithic (deterministic with same seed)
2. **Perf test:** `--benchmark` flag measures overhead, assert <1%
3. **Integration test:** Manual run on Colab, verify metrics appear in wandb dashboard

## Firewall Notes (Dev Container Only)

For local development, add to `.devcontainer/init-firewall.sh`:

```bash
# Wandb
"api.wandb.ai"
"cdn.wandb.ai"
```

Not needed for Colab/TRC (no firewall).

## Open Questions

None - all decisions made in discussion.
