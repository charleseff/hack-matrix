# TPU Training Spec

**Status:** Active

## Overview

PureJaxRL training on Google TPU Research Cloud (TRC). Goal: 10x throughput improvement over baseline.

**TRC Details:**
- Project: `hack-matrix` | Zone: `us-central2-b` | TPU: v4-8 (free under quota)
- Trial ends: **February 28, 2026**

## Training Commands

```bash
# Fast config (recommended)
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
cd ~/hack-matrix/python
python3 scripts/train_purejaxrl.py \
  --wandb \
  --total-timesteps 100000000 \
  --num-envs 2048 \
  --num-steps 256 \
  --checkpoint-dir checkpoints
```

**Checkpointing:**
- Saves every 10 minutes (override: `--save-interval-minutes N` or `--save-interval N` for update-based)
- Location: `checkpoints/{run_name}/checkpoint_{step}.pkl`
- Resume: `--resume checkpoints/run-name/checkpoint_40.pkl`
- Disable wandb uploads: `--no-artifact`

**Resume works across different `num_envs`/`num_steps`** (model weights are batch-size independent).

## Performance

| Config | num_envs | num_steps | Batch Size | Throughput | 100M Time |
|--------|----------|-----------|------------|------------|-----------|
| Baseline | 256 | 128 | 32k | ~1,800/sec | ~10 hours |
| **Fast** | 2048 | 256 | 524k | ~15-30k/sec | ~1-2 hours |

## Learning Rate Findings

**Use `--lr 2.5e-4` (default). Do NOT increase with larger batches.**

| Run | LR | Final Reward | Entropy | KL | Clip Frac |
|-----|-----|--------------|---------|-----|-----------|
| [feb02-26-1](https://wandb.ai/charles-team/hackmatrix/runs/b0cf9e12) | **2.5e-4** | **-0.060** | **1.65** | 0.055 | 0.22 |
| [feb03-26-2](https://wandb.ai/charles-team/hackmatrix/runs/401d3405) | 3e-4 | -0.088 | 1.30 | 0.10 | 0.40-0.45 |

**Signs of LR too high:** KL > 0.08, clip fraction > 0.35, entropy collapse, oscillating reward.

**Why:** Large batches (524k) produce less noisy gradients. PPO clipping already limits update size—additional LR scaling is harmful.

**Healthy metrics:** KL ~0.05-0.06, clip fraction ~0.22, entropy stable or increasing, steady reward improvement.

## Optimizations

### 1. Parallel Environments (High Impact)
`--num-envs 2048` — TPU v4-8 underutilized at 256. 8x envs ≈ 8x throughput.

### 2. Longer Rollouts (Medium Impact)
`--num-steps 256` — Better advantage estimation, complete episodes. HackMatrix episodes ~20 steps, so 256 is reasonable.

### 3. JAX Compilation Cache
`export JAX_COMPILATION_CACHE_DIR=~/.jax_cache` — Saves ~200s on restart.

### 4. Larger TPU (Not Yet Tested)

| TPU | Chips | Relative Compute |
|-----|-------|------------------|
| v4-8 | 4 | 1x (current) |
| v4-16 | 8 | 2x |
| v4-32 | 16 | 4x |
| v4-64 | 32 | 8x (max quota) |

May need pmap/pjit changes for multi-host.

### 5. bfloat16 (Not Implemented)
~1.5-2x potential. Lower priority than parallelism.

## Monitoring

Wandb project: `hackmatrix`

| Metric | Healthy Range | Problem Sign |
|--------|---------------|--------------|
| `mean_reward` | Trending up | Flat/oscillating |
| `entropy` | Stable or increasing | Collapse to 0 |
| `approx_kl` | 0.01-0.06 | > 0.08 |
| `clip_fraction` | < 0.30 | > 0.40 |

## Remote Access

For running commands from local machine via gcloud:

```bash
# SSH with tmux for long runs
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b --command="
tmux new-session -d -s training 'cd ~/hack-matrix/python && python3 scripts/train_purejaxrl.py --wandb ...'
"

# Reattach
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b
tmux attach -t training

# Download checkpoint
gcloud compute tpus tpu-vm scp hackmatrix-train:~/hack-matrix/python/checkpoints/run/checkpoint.pkl \
  ./checkpoints/ --zone=us-central2-b
```

## Visual Validation (macOS)

```bash
cd python && source venv-macos/bin/activate
python scripts/watch_jax_agent.py checkpoints/checkpoint.pkl
```

## Implementation Plan

### Phase 1: Quick Wins
- [x] Test `--num-envs 2048`
- [x] Test `--num-envs 2048 --num-steps 256`
- [ ] Verify training stability with larger batch
- [ ] Benchmark throughput improvement

### Phase 2: Larger TPU
- [ ] Test v4-16 with current code
- [ ] Check JAX auto-scaling across chips
- [ ] Benchmark v4-16 vs v4-8

### Phase 3: Mixed Precision (Optional)
- [ ] Implement bfloat16 networks
- [ ] Verify stability and benchmark

## Success Criteria

- [ ] 10x throughput (1,800 → 18,000+ steps/sec)
- [ ] 100M timesteps in < 2 hours
- [ ] Training stability maintained
- [ ] `watch_jax_agent.py` works with checkpoint
