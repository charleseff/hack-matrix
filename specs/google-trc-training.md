# Google TRC Training Spec

**Status:** Active

## Overview

Get HackMatrix agent training running on Google TPU Research Cloud (TRC). This spec focuses on **infrastructure** - getting training working end-to-end and validating the checkpoint works locally.

## Goals

1. Successfully run training on TPU via TRC
2. Validate checkpoint works with `watch_jax_agent.py` on macOS
3. Document the TRC workflow for future training runs

## Non-Goals (Future Spec)

- Hyperparameter tuning
- Training to competent gameplay
- Reward shaping experiments

## Prerequisites

- [x] PureJaxRL integration complete ([purejaxrl-integration.md](./purejaxrl-integration.md))
- [x] Wandb integration complete ([wandb-purejaxrl.md](./wandb-purejaxrl.md))
- [ ] TRC project approved and TPU quota available

## Setup Steps

### 1. TRC Project Setup

1. Apply for TRC access at https://sites.research.google/trc/about/
2. Create GCP project linked to TRC
3. Enable TPU API
4. Set up billing (TRC credits)

### 2. TPU VM Creation

```bash
# Create TPU VM (v4-8 recommended for initial testing)
gcloud compute tpus tpu-vm create hackmatrix-train \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-vm-v4-base

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b
```

### 3. Environment Setup on TPU VM

```bash
# Clone repo
git clone https://github.com/charleseff/hack-matrix.git
cd hack-matrix

# Install JAX for TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install dependencies
cd python && pip install -r requirements.txt

# Authenticate wandb
wandb login
```

### 4. Training Command

```bash
# Start training with wandb logging
python scripts/train_purejaxrl.py \
  --wandb \
  --project hackmatrix \
  --entity charles-team \
  --total-timesteps 100000000 \
  --num-envs 2048 \
  --num-steps 256
```

## Training Configuration

### Hyperparameters for TPU

| Parameter | CPU | TPU (v4-8) | Notes |
|-----------|-----|------------|-------|
| `num_envs` | 32 | 2048 | Scale with TPU cores |
| `num_steps` | 64 | 256 | Longer rollouts |
| `hidden_dim` | 256 | 512 | Larger network |
| `num_layers` | 2 | 3 | Deeper network |
| `learning_rate` | 2.5e-4 | 1e-4 | Lower for larger batches |

### Training Targets (This Spec)

| Milestone | Timesteps | Expected Outcome |
|-----------|-----------|------------------|
| Sanity check | 1M | Training runs, metrics logged |
| Validation run | 5-10M | Enough to test checkpoint → watch_jax_agent.py flow |

Longer training runs (50M+) are out of scope for this spec.

## Monitoring

- Wandb dashboard: `hackmatrix` project
- Key metrics to watch:
  - `train/mean_reward` - Should trend upward
  - `train/mean_episode_length` - Should increase
  - `train/entropy` - Should decrease gradually (not collapse)
  - `train/steps_per_second` - TPU throughput

## Checkpointing

| Checkpoint | Frequency | Purpose |
|------------|-----------|---------|
| Periodic | Every 5M timesteps (~10 min on TPU) | Recovery from disconnects |
| Final | End of training | Best model for validation |

Both saved as wandb artifacts. To resume from disconnect:

```bash
python scripts/train_purejaxrl.py \
  --wandb \
  --resume-run <run-id>
```

CLI flags:
- `--save-interval 5000000` - Steps between periodic saves
- `--no-artifact` - Disable wandb uploads (faster iteration)

## Cost Estimation

| TPU Type | $/hour | 100M steps (~10 hours) |
|----------|--------|------------------------|
| v4-8 | TRC credits | Free (TRC) |
| v4-32 | TRC credits | Free (TRC) |

## Open Questions

1. **TPU type**: Start with v4-8, scale up if needed
2. **Zone availability**: us-central2-b is default, may need to try other zones

## Success Criteria

1. [ ] TPU VM created and accessible
2. [ ] Training runs on TPU without errors (1M+ timesteps)
3. [ ] Wandb metrics streaming correctly
4. [ ] Checkpoint downloads successfully to macOS
5. [ ] `watch_jax_agent.py` runs with downloaded checkpoint
6. [ ] Agent takes valid actions (even if not smart)

## Visual Validation (macOS)

After training, download checkpoint and watch agent play:

```bash
# Download checkpoint from wandb
wandb artifact get charles-team/hackmatrix/checkpoint-<step>:latest -o python/checkpoints/

# Or scp from TPU
gcloud compute tpus tpu-vm scp hackmatrix-train:~/hack-matrix/python/checkpoints/final_params.npz \
  python/checkpoints/ --zone=us-central2-b

# Watch agent play (visual mode)
cd python && source venv-macos/bin/activate
python scripts/watch_jax_agent.py checkpoints/final_params.npz
```

## References

- [Google TRC Documentation](https://sites.research.google/trc/about/)
- [JAX TPU Setup](https://jax.readthedocs.io/en/latest/installation.html#google-cloud-tpu)
- [Training Reference](./training-reference.md)
- [watch_jax_agent.py](../python/scripts/watch_jax_agent.py) - Visual validation script
