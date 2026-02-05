# TPU Training Spec

**Status:** Active

## Overview

PureJaxRL training on Google TPU Research Cloud (TRC). Goal: 10x throughput improvement over baseline.

**TRC Details:**
- Project: `hack-matrix` | Zone: `us-central2-b` | TPU: v4-8 (free under quota)
- Trial ends: **February 28, 2026**

## Training Commands

```bash
# Recommended config (auto-scales for large batches)
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
cd ~/hack-matrix/python
python3 scripts/train_purejaxrl.py \
  --total-timesteps 100000000 \
  --num-envs 2048 \
  --num-steps 128 \
  --checkpoint-dir checkpoints
```

**Notes:**
- Wandb logging is enabled by default (disable with `--no-wandb`)
- `num_minibatches` auto-scales with batch size to maintain consistent training dynamics
- Override auto-scaling by explicitly setting `--num-minibatches N`

**Checkpointing:**
- Saves every 10 minutes (override: `--save-interval-minutes N` or `--save-interval N` for update-based)
- Location: `checkpoints/{run_name}/checkpoint_{step}.pkl`
- Resume: `--resume checkpoints/run-name/checkpoint_40.pkl`
- Disable wandb uploads: `--no-artifact`

**⚠️ Resume Caveat:** Model weights transfer across different `num_envs`, but optimizer state (Adam momentum) may be poorly calibrated for the new batch size. For best results when changing `num_envs`, start fresh rather than resuming.

## Large Batch Training Problem (Solved)

### The Issue

When scaling `num_envs` from 256 to 2048, training degraded despite higher throughput:

| num_envs | batch_size | minibatch_size | Best Return |
|----------|------------|----------------|-------------|
| 256 | 32k | 8,192 | **-1.55** |
| 2048 | 262k | 65,536 | -1.65 (worse) |

**Root cause:** Two problems occur with large batches:
1. **Minibatch size**: With default `num_minibatches=4`, minibatch_size grows 8x. Larger minibatches produce less noisy gradients, reducing beneficial exploration.
2. **Gradient steps**: If we only scale `num_minibatches` up (to fix #1), total gradient steps per update increases 8x (16 → 128), causing the policy to change too much per update (high KL, high clip fraction).

### The Solution

Auto-scale **both** `num_minibatches` (up) and `update_epochs` (down) to maintain:
- Consistent minibatch_size (~8,192) — preserves gradient noise
- Bounded gradient steps (~32) — prevents policy overshooting

```
For num_envs=2048:
  batch_size: 262,144 (8x reference)
  num_minibatches: 4 -> 32 (scaled up)
  minibatch_size: 65,536 -> 8,192 (preserved!)
  update_epochs: 4 -> 1 (scaled down)
  gradient_steps: 16 -> 32 (only 2x, acceptable)
```

This is implemented in `auto_scale_for_batch_size()` and called automatically when using default settings.

## Performance

| Config | num_envs | num_steps | Batch Size | Throughput | 100M Time |
|--------|----------|-----------|------------|------------|-----------|
| Baseline | 256 | 128 | 32k | ~1,800/sec | ~10 hours |
| **Fast** | 2048 | 128 | 262k | ~12,500/sec | ~2 hours |

**Note:** `num_steps=256` increases batch further but wasn't tested with auto-scaling fix.

## Learning Rate Findings

**Use `--lr 2.5e-4` (default) with auto-scaled minibatches.**

### Feb 4 Experiments (2048 envs, 262k batch)

| LR | num_minibatches | Result |
|----|-----------------|--------|
| 2.5e-4 | 4 (default) | KL rose to 0.09, clip to 0.38, return degraded |
| 1.25e-4 | 4 (default) | Healthy metrics but stagnant at -1.65 return |
| 2.5e-4 | 32 (auto-scaled) | **Testing** |

**Signs of LR too high:** KL > 0.08, clip fraction > 0.35, entropy collapse, oscillating reward.

**Healthy metrics:** KL ~0.05-0.06, clip fraction ~0.22-0.30, entropy stable or increasing.

## Optimizations

### 1. Auto-Scaled Batch Parameters (Critical for Large Batches)
Automatically enabled when using default settings. Maintains:
- ~8,192 minibatch_size (scales `num_minibatches` up)
- ~32 gradient steps (scales `update_epochs` down)

### 2. Parallel Environments (High Impact)
`--num-envs 2048` — TPU v4-8 underutilized at 256. 8x envs ≈ 7x throughput.

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
| `mean_episode_return` | Trending up | Flat/oscillating |
| `entropy` | Stable or increasing | Collapse to 0 |
| `approx_kl` | 0.01-0.06 | > 0.08 |
| `clip_fraction` | < 0.30 | > 0.35 |

## Remote Access

```bash
# SSH with tmux for long runs
gcloud compute tpus tpu-vm ssh hackmatrix-train --zone=us-central2-b
tmux new-session -s training
cd ~/hack-matrix/python && python3 scripts/train_purejaxrl.py ...

# Reattach later
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

### Phase 1: Large Batch Training
- [x] Test `--num-envs 2048`
- [x] Identify large-batch training degradation issue
- [x] Implement `auto_scale_for_batch_size()` — scale `num_minibatches`
- [x] Extend auto-scaling to also scale `update_epochs` (fix gradient steps)
- [ ] **Verify complete auto-scaling fix produces good training**
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
- [ ] Training stability: KL < 0.06, clip_fraction < 0.30
- [ ] Return reaches -1.55 or better (matching 256-env baseline)
- [ ] `watch_jax_agent.py` works with checkpoint

## Appendix: Key Learnings

1. **Don't resume across batch size changes** — Optimizer state becomes miscalibrated
2. **Minibatch size matters more than batch size** — Gradient noise helps exploration
3. **Gradient steps matter as much as minibatch size** — More steps = larger policy change per update
4. **PPO doesn't follow linear LR scaling** — Larger batches need same or lower LR
5. **Auto-scaling must adjust both dimensions** — Scale `num_minibatches` up AND `update_epochs` down
