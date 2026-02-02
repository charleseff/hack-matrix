# Google TRC Training Spec

**Status:** In Progress

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
- [x] TRC project approved and TPU quota available

## TRC Quota Details

**Project:** `hack-matrix`

| Zone | TPU Type | Chips | Scheduling | API Name | Status |
|------|----------|-------|------------|----------|--------|
| us-central2-b | v4 | 32 | spot | v4-8 | **Working** |
| us-central2-b | v4 | 32 | on-demand | v4-8 | Working |
| us-central1-a | v5e | 64 | spot | v5litepod-* | Quota not applied |
| europe-west4-a | v6e | 64 | spot | v6e-* | Untested |
| europe-west4-b | v5e | 64 | spot | v5litepod-* | Untested |
| us-east1-d | v6e | 64 | spot | v6e-* | Untested |

**Note:** v5e is called `v5litepod` in the API. The v5e quota in us-central1-a was not properly applied despite TRC email confirmation.

## Quick Start (Verified Working)

```bash
# 1. Create TPU VM (v4-8, spot, free under TRC)
gcloud compute tpus tpu-vm create hackmatrix-train \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base \
  --spot

# 2. Setup environment
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
git clone https://github.com/charleseff/hack-matrix.git
cd hack-matrix/python && pip install -r requirements.txt
"

# 3. Run training (short test)
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="
cd hack-matrix/python
export PATH=\$HOME/.local/bin:\$PATH
python3 scripts/train_purejaxrl.py \
  --total-timesteps 1000000 \
  --num-envs 256 \
  --num-steps 128 \
  --checkpoint-dir checkpoints
"

# 4. Download checkpoint
gcloud compute tpus tpu-vm scp hackmatrix-train:~/hack-matrix/python/checkpoints/final_params.npz \
  python/checkpoints/ --zone=us-central2-b

# 5. Delete TPU (important!)
gcloud compute tpus tpu-vm delete hackmatrix-train --zone=us-central2-b --quiet
```

## Long-Running Training

For training runs that may exceed SSH timeout limits, use tmux to detach the process:

```bash
# Start training in tmux session
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="
tmux new-session -d -s training '
cd hack-matrix/python
export PATH=\$HOME/.local/bin:\$PATH
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
python3 scripts/train_purejaxrl.py \
  --wandb \
  --total-timesteps 100000000 \
  --num-envs 2048 \
  --num-steps 256 \
  --checkpoint-dir checkpoints
'
echo 'Training started in tmux session: training'
"

# Check progress (reattach to tmux)
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="tmux attach -t training"

# Or just check if still running
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="tmux list-sessions"
```

## JAX Compilation Caching

JAX JIT compilation takes ~200s on first run. Enable persistent caching to skip this on subsequent runs:

```bash
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
```

The cache persists across Python restarts as long as:
- Same JAX version
- Same model architecture (num_envs, hidden_dim, etc.)
- Same TPU type

Add this to your training commands or `~/.bashrc` on the TPU VM.

## Persistent TPU VM (Recommended)

Since on-demand v4-8 is free under TRC quota, keep the TPU running to avoid 15+ minute setup times.

### Reconnect to TPU

```bash
# SSH into the TPU VM
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b

# Or run a single command
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="<command>"
```

### Check TPU Status

```bash
# List all TPU VMs
gcloud compute tpus tpu-vm list --zone=us-central2-b

# Check if TPU is running
gcloud compute tpus tpu-vm describe hackmatrix-train --zone=us-central2-b --format="value(state)"
```

### Reattach to Training Session

```bash
# List tmux sessions
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="tmux list-sessions"

# Reattach to training session (interactive)
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b
# Then inside the VM:
tmux attach -t training
```

### Update Code on TPU

```bash
# Pull latest changes
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="cd hack-matrix && git pull"
```

### When to Delete

- Before TRC trial ends on **February 28, 2026** (to avoid charges)
- If TPU becomes unresponsive
- When switching to a different TPU type/zone

```bash
gcloud compute tpus tpu-vm delete hackmatrix-train --zone=us-central2-b --quiet
```

## Training Configuration

### Hyperparameters for TPU

| Parameter | CPU | TPU (v4-8) | Notes |
|-----------|-----|------------|-------|
| `num_envs` | 32 | 256-2048 | Scale with TPU cores |
| `num_steps` | 64 | 128-256 | Longer rollouts |
| `hidden_dim` | 256 | 256-512 | Larger network |
| `num_layers` | 2 | 2-3 | Deeper network |
| `learning_rate` | 2.5e-4 | 2.5e-4 | May need tuning |

### Verified Training Run (1M steps)

```
Device: TPU (4 cores)
num_envs: 256
num_steps: 128
batch_size: 32768
Training time: 593s (~10 min)
Final entropy: 1.22 (healthy)
Final mean_reward: -0.077
```

**Note:** First chunk takes ~200s for JAX JIT compilation. Subsequent chunks are faster.

### Training Targets

| Milestone | Timesteps | Expected Time | Expected Outcome |
|-----------|-----------|---------------|------------------|
| Sanity check | 1M | ~10 min | Training runs, metrics logged |
| Validation run | 10M | ~1 hour | Test checkpoint → watch_jax_agent.py flow |
| Real training | 100M+ | ~10 hours | Agent learns meaningful behavior |

## Monitoring

- Wandb dashboard: `hackmatrix` project
- Key metrics to watch:
  - `train/mean_reward` - Should trend upward
  - `train/mean_episode_length` - Should increase
  - `train/entropy` - Should decrease gradually (not collapse to 0)
  - `train/steps_per_second` - TPU throughput

## Checkpointing

| Checkpoint | Frequency | Purpose |
|------------|-----------|---------|
| Periodic | Every 5M timesteps | Recovery from disconnects |
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

## Speeding Up TPU Setup

Setup takes ~15 minutes (JAX + deps). Options to reduce this:

### Option 1: Custom TPU VM Image
```bash
# After setting up a TPU VM with all deps:
# (This requires stopping the TPU first)
gcloud compute images create hackmatrix-tpu-image \
  --source-disk=<tpu-vm-disk> \
  --source-disk-zone=us-central2-b

# Create new TPUs from custom image:
gcloud compute tpus tpu-vm create my-tpu \
  --accelerator-type=v4-8 \
  --zone=us-central2-b \
  --vm-image=hackmatrix-tpu-image \
  --spot
```

### Option 2: Startup Script
```bash
gcloud compute tpus tpu-vm create my-tpu \
  --accelerator-type=v4-8 \
  --zone=us-central2-b \
  --spot \
  --metadata=startup-script='#!/bin/bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
git clone https://github.com/charleseff/hack-matrix.git
cd hack-matrix/python && pip install -r requirements.txt'
```

### Option 3: Cache a Base Image (Recommended)
Create and keep a pre-baked VM image with JAX + deps already installed, then clone from it for new TPUs. This avoids repeating package installs on every run.

```bash
# 1) Create a TPU VM and install deps once (see Quick Start setup)
# 2) Stop the TPU VM, then create a reusable image
gcloud compute images create hackmatrix-tpu-base \
  --source-disk=<tpu-vm-disk> \
  --source-disk-zone=us-central2-b

# 3) Create new TPUs from the cached image
gcloud compute tpus tpu-vm create hackmatrix-train \
  --accelerator-type=v4-8 \
  --zone=us-central2-b \
  --vm-image=hackmatrix-tpu-base \
  --spot
```

**Notes:** Keep a single cached image per TPU type/zone. Refresh it when JAX/Python deps change.

## Cost

| TPU Type | Scheduling | Cost |
|----------|------------|------|
| v4-8 | spot | Free (TRC quota) |
| v4-8 | on-demand | Free (TRC quota) |

**TRC Trial End Date:** February 28, 2026

**Recommendation:** Keep on-demand TPU running during active development to avoid 15+ min setup overhead. Delete before trial ends. Spot instances may be preempted unexpectedly.

## Success Criteria

1. [x] TPU VM created and accessible
2. [x] Training runs on TPU without errors (1M+ timesteps)
3. [x] Wandb metrics streaming correctly
4. [x] Checkpoint downloads successfully to local machine
5. [ ] `watch_jax_agent.py` runs with downloaded checkpoint
6. [ ] Agent takes valid actions (even if not smart)

## Visual Validation (macOS)

After training, download checkpoint and watch agent play:

```bash
# Download checkpoint from TPU
gcloud compute tpus tpu-vm scp hackmatrix-train:~/hack-matrix/python/checkpoints/final_params.npz \
  python/checkpoints/ --zone=us-central2-b

# Watch agent play (visual mode)
cd python && source venv-macos/bin/activate
python scripts/watch_jax_agent.py checkpoints/final_params.npz
```

## Troubleshooting

### "User does not have permission" error
Your TRC quota may not be applied for that accelerator type/zone. Check:
```bash
gcloud alpha services quota list --service=tpu.googleapis.com --consumer=projects/hack-matrix --format=json
```
Look for `producerOverride` entries - those indicate TRC quota.

### SSH/gcloud auth errors
Add oauth2.googleapis.com to firewall (dev container only):
```bash
for ip in $(dig +short oauth2.googleapis.com | grep -E '^[0-9]'); do
  sudo ipset add allowed-domains "$ip"
done
```

### Training killed mid-run
Use tmux or nohup for long training runs (see Long-Running Training section).

## References

- [Google TRC Documentation](https://sites.research.google/trc/about/)
- [JAX TPU Setup](https://jax.readthedocs.io/en/latest/installation.html#google-cloud-tpu)
- [Cloud TPU JAX Guide](https://cloud.google.com/tpu/docs/run-calculation-jax)
- [Training Reference](./training-reference.md)
- [watch_jax_agent.py](../python/scripts/watch_jax_agent.py) - Visual validation script
