# Training Speedups Spec

**Status:** Active
**Depends on:** [google-trc-training.md](./google-trc-training.md)

## Overview

Optimize PureJaxRL training throughput on TPU. Current baseline on v4-8 with default settings achieves ~1,800 timesteps/sec. Goal is 10-20x improvement.

## Current Baseline

| Setting | Value | Throughput |
|---------|-------|------------|
| num_envs | 256 | ~1,800 steps/sec |
| num_steps | 128 | |
| batch_size | 32,768 | |
| 100M timesteps | ~10 hours | |

## Proposed Changes

### 1. Increase Parallel Environments (High Impact)

**Change:** `num_envs` 256 → 2048

**Expected speedup:** ~8x

**Why it works:** TPU v4-8 has massive parallelism capacity. 256 envs underutilizes it. Each env runs independently, so 8x envs = 8x data collection throughput.

**Tradeoffs:**
- More memory usage (should still fit on v4-8)
- Larger batch size requires **keeping LR at 2.5e-4 or lower** (see [Learning Rate Findings](#learning-rate-findings))

**Implementation:** CLI flag only, no code changes.

```bash
--num-envs 2048
```

### 2. Longer Rollouts (Medium Impact)

**Change:** `num_steps` 128 → 256

**Expected speedup:** ~1.2x (better TPU utilization)

**Expected learning benefit:** Moderate improvement in advantage estimation

**What is a rollout?**
A rollout is a sequence of (state, action, reward, next_state) transitions collected by running the policy. With `num_steps=128`, each environment collects 128 consecutive steps before a gradient update.

**Why longer rollouts help:**
1. **Better advantage estimation**: GAE looks at future rewards to estimate "was this action good?" Longer horizon = more accurate signal.
2. **Complete episodes**: Short rollouts may cut episodes mid-game, biasing value estimates.
3. **Longer-term credit assignment**: Agent can learn that actions many steps ago affected current reward.

**Tradeoffs:**
- More memory per rollout
- Very long rollouts use "stale" policy (collected before gradient update)
- For HackMatrix with ~20 step episodes, 256 is reasonable. Diminishing returns beyond.

**Implementation:** CLI flag only.

```bash
--num-steps 256
```

### 3. JAX Compilation Caching (One-time Savings)

**Change:** Enable persistent JIT cache

**Expected speedup:** Saves ~200s on each restart (not per-step improvement)

**Implementation:** Environment variable.

```bash
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
```

### 4. Use Larger TPU (Potentially High Impact)

**Current:** v4-8 (4 chips, 8 TensorCores)

**Available under TRC quota:** Up to v4-64 (32 chips, 64 TensorCores)

| API Name | Chips | TensorCores | Relative Compute |
|----------|-------|-------------|------------------|
| v4-8 | 4 | 8 | 1x (current) |
| v4-16 | 8 | 16 | 2x |
| v4-32 | 16 | 32 | 4x |
| v4-64 | 32 | 64 | 8x (max quota) |

**Expected speedup:** Potentially 2-8x more, but requires investigation.

**Challenges:**
- Multi-host training may need code changes (data parallelism across hosts)
- JAX pmap/pjit configuration
- May hit diminishing returns if bottlenecked elsewhere

**Implementation:** Requires investigation. Create TPU with different accelerator-type:

```bash
gcloud compute tpus tpu-vm create hackmatrix-train \
  --zone=us-central2-b \
  --accelerator-type=v4-32 \  # or v4-64
  --version=tpu-ubuntu2204-base
```

**Status:** Not yet tested. Start with v4-8 optimizations first.

### 5. Mixed Precision / bfloat16 (Medium Impact)

**Change:** Use bfloat16 for forward/backward pass, float32 for optimizer state

**Expected speedup:** ~1.5-2x (TPU v4 optimized for bfloat16)

**Implementation:** Code changes required in network and training loop.

```python
# Example (not yet implemented)
import jax.numpy as jnp
policy_dtype = jnp.bfloat16
```

**Status:** Not yet implemented. Lower priority than parallelism changes.

## Recommended Fast Config

Combining changes 1-3 (no code changes required):

```bash
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
"
```

**Checkpointing:** Saves every 10 minutes by default (`--save-interval-minutes 10`). Use `--save-interval N` to save every N updates instead. Checkpoints are stored in per-run directories: `checkpoints/{run_name}/checkpoint_{step}.pkl`.

**Resuming:** Use `--resume <checkpoint-file>` to load a specific checkpoint and continue training. Example: `--resume checkpoints/hackmatrix-jax-feb01-26-1/checkpoint_40.pkl`. Works even with different `num_envs`/`num_steps` (model weights and optimizer state are independent of batch size).

**Expected performance:**
- Batch size: 2048 × 256 = 524,288 (16x baseline)
- Throughput: ~15,000-30,000 steps/sec (10-15x baseline)
- 100M timesteps: ~1-2 hours (vs ~10 hours baseline)

## Learning Rate Findings

**TL;DR:** Use `--lr 2.5e-4` (default). Do NOT increase LR with larger batches.

### Experiment Comparison

| Run | LR | Final Reward | Entropy | KL | Clip Frac |
|-----|-----|--------------|---------|-----|-----------|
| [hackmatrix-jax-feb02-26-1](https://wandb.ai/charles-team/hackmatrix/runs/b0cf9e12) | **2.5e-4** | **-0.060** | **1.65** | 0.055 | 0.22 |
| [hackmatrix-jax-feb03-26-2](https://wandb.ai/charles-team/hackmatrix/runs/401d3405) | 3e-4 | -0.088 | 1.30 | 0.10 | 0.40-0.45 |

### Signs of LR Too High (3e-4 run)

1. **High KL divergence (~0.10)**: Policy updates too aggressive. PPO target is 0.01-0.02.
2. **High clip fraction (~0.40-0.45)**: Nearly half of updates clipped → wasted computation.
3. **Low entropy (~1.3)**: Premature convergence, can't explore out of local minimum.
4. **Oscillating reward**: No steady improvement, bounces around.

### Why Lower LR Works Better

With large batches (524k samples/update), gradients are less noisy. Each update has more impact, so smaller steps are needed. This is counterintuitive—common advice is "scale LR with batch size"—but PPO's clipping mechanism already limits update size, making additional LR scaling harmful.

### Healthy Training Metrics (2.5e-4 run)

- KL stable at 0.05-0.06
- Clip fraction ~0.22 (acceptable)
- Entropy *increases* over training (1.12 → 1.65) as agent learns confident exploration
- Steady reward improvement (-0.10 → -0.06)

## Implementation Plan

### Phase 1: Quick Wins (No Code Changes)
- [x] Test `--num-envs 2048` alone
- [x] Test `--num-envs 2048 --num-steps 256` combined
- [ ] Verify training stability with larger batch (see [Learning Rate Findings](#learning-rate-findings))
- [ ] Benchmark throughput improvement
- [ ] Update google-trc-training.md with fast config

### Phase 2: Larger TPU Investigation
- [ ] Test v4-16 creation with current code
- [ ] Check if JAX auto-scales across chips
- [ ] If not, investigate pmap/pjit changes
- [ ] Benchmark v4-16 vs v4-8

### Phase 3: Mixed Precision (Optional)
- [ ] Implement bfloat16 policy/value networks
- [ ] Verify training stability
- [ ] Benchmark speedup

## Success Criteria

1. [ ] 10x throughput improvement (1,800 → 18,000+ steps/sec)
2. [ ] 100M timesteps completes in < 2 hours
3. [ ] Training stability maintained (no divergence, healthy entropy)
4. [ ] Documented fast config in google-trc-training.md

## References

- [JAX Mixed Precision](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)
- [TPU Performance Guide](https://cloud.google.com/tpu/docs/performance-guide)
- [PureJaxRL Scaling](https://github.com/luchris429/purejaxrl)
