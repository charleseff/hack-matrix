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
- Larger batch size may need learning rate tuning

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

**Checkpointing:** Saves every 10 minutes by default (`--save-interval-minutes 10`). Use `--save-interval N` to save every N updates instead.

**Resuming:** Use `--resume-run <wandb-run-id>` to load the latest checkpoint and continue training. Works even with different `num_envs`/`num_steps` (model weights and optimizer state are independent of batch size).

**Expected performance:**
- Batch size: 2048 × 256 = 524,288 (16x baseline)
- Throughput: ~15,000-30,000 steps/sec (10-15x baseline)
- 100M timesteps: ~1-2 hours (vs ~10 hours baseline)

## Implementation Plan

### Phase 1: Quick Wins (No Code Changes)
- [ ] Test `--num-envs 2048` alone
- [ ] Test `--num-envs 2048 --num-steps 256` combined
- [ ] Verify training stability with larger batch
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
