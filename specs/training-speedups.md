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
  --lr 6.25e-5 \
  --ent-coef 0.15 \
  --checkpoint-dir checkpoints
```

**Notes:**
- Wandb logging is enabled by default (disable with `--no-wandb`)
- `num_minibatches` and `update_epochs` auto-scale with batch size to maintain consistent training dynamics
- Use `--lr 6.25e-5` for large batches (quarter default) — compensates for 4x gradient steps with epoch floor 2
- `--ent-coef 0.15` maintains exploration of rare actions (siphon, programs) that would otherwise be pruned by entropy collapse
- Override auto-scaling by explicitly setting `--num-minibatches N` or `--update-epochs N`

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
  update_epochs: 4 -> 2 (scaled down, floor=2)
  gradient_steps: 16 -> 64 (4x reference)
```

The epoch floor of 2 ensures rare strategic events (siphon/program decisions) get at least 2 gradient passes over the data before it's discarded. With `--lr 6.25e-5`, the per-update policy change stays bounded.

This is implemented in `auto_scale_for_batch_size()` and called automatically when using default settings.

## Performance

| Config | num_envs | num_steps | Batch Size | Throughput | 100M Time |
|--------|----------|-----------|------------|------------|-----------|
| Baseline | 256 | 128 | 32k | ~1,800/sec | ~10 hours |
| **Fast** | 2048 | 128 | 262k | ~12,500/sec | ~2 hours |

**Note:** `num_steps=256` increases batch further but wasn't tested with auto-scaling fix.

## Learning Rate Findings

**Use `--lr 1.25e-4` with auto-scaled batch parameters (2048 envs).**

### Experiment History

| Date | LR | Config | Result |
|------|-----|--------|--------|
| Feb 4 | 2.5e-4 | 4 minibatches, 4 epochs | KL 0.09+, clip 0.38+, degraded |
| Feb 4 | 1.25e-4 | 4 minibatches, 4 epochs | Healthy metrics, stagnant -1.65 |
| Feb 4 | 2.5e-4 | 32 minibatches, 4 epochs | KL 0.10+, clip 0.45+, degraded |
| Feb 5 | 2.5e-4 | 32 minibatches, 1 epoch | Early healthy, degraded by update 150 |
| Feb 5 | 1.25e-4 | 32 minibatches, 1 epoch | **Healthy metrics, testing reward fixes** |

**Key insight:** With 32 gradient steps (vs 16 baseline), halving LR compensates for the 2x policy change.

**Signs of LR too high:** KL > 0.08, clip fraction > 0.35, entropy collapse, oscillating reward.

**Healthy metrics:** KL ~0.03-0.05, clip fraction ~0.20-0.25, entropy stable or increasing.

## Reward Function Fixes (Feb 5)

### The Oscillation Problem

Even with healthy PPO metrics, the agent learned to oscillate left-right forever:
- `mean_episode_length` kept increasing (17 → 20+)
- `mean_episode_return` flatlined (~-1.35)
- Agent survives but never reaches the exit

**Root cause:** No cost to stalling. The agent found a "safe" policy that avoids death without making progress.

### Fix 1: One-Directional Distance Shaping

Changed from bidirectional to one-directional:
```
Before: +0.05 for closer, -0.05 for farther (oscillation = 0 net)
After:  +0.05 for closer, 0.00 for farther (can't farm by oscillating)
```

### Fix 2: Step Penalty

Added `-0.01` per step to create time pressure:
```
Move toward exit:  +0.05 - 0.01 = +0.04
Move away:          0.00 - 0.01 = -0.01
Oscillate 100x:                  = -1.00 penalty
```

The fastest path to the exit is now optimal. Stalling bleeds reward.

**Implementation:** Both fixes applied to `rewards.py` (JAX) and `RewardCalculator.swift`.

## Optimizations

### 1. Auto-Scaled Batch Parameters (Critical for Large Batches)
Automatically enabled when using default settings. Maintains:
- ~8,192 minibatch_size (scales `num_minibatches` up)
- Epoch floor of 2 (rare events get 2 passes minimum)
- ~64 gradient steps at 2048 envs (compensated by halved LR)

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
| `mean_episode_length` | Stable or decreasing | Continuously increasing (oscillation) |
| `entropy` | Stable or increasing | Collapse to 0 |
| `approx_kl` | 0.01-0.06 | > 0.08 |
| `clip_fraction` | < 0.30 | > 0.35 |

**Oscillation Detection:** If `mean_episode_length` keeps rising but `mean_episode_return` flatlines, the agent is stalling (moving left-right forever) instead of progressing to the exit.

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
- [x] Lower LR to 1.25e-4 for 2x gradient steps
- [ ] **Verify PPO metrics stay healthy through full training**
- [ ] Benchmark throughput improvement

### Phase 1b: Reward Function Fixes
- [x] Identify oscillation problem (agent stalls instead of progressing)
- [x] Implement one-directional distance shaping (JAX + Swift)
- [x] Implement step penalty -0.01 (JAX + Swift)
- [ ] **Verify agent learns to reach exit instead of oscillating**

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
- [ ] Episode length stabilizes (not continuously increasing = no oscillation)
- [ ] `watch_jax_agent.py` shows agent reaching exit, not oscillating

## Appendix: Key Learnings

### PPO Scaling
1. **Don't resume across batch size changes** — Optimizer state becomes miscalibrated
2. **Minibatch size matters more than batch size** — Gradient noise helps exploration
3. **Gradient steps matter as much as minibatch size** — More steps = larger policy change per update
4. **PPO doesn't follow linear LR scaling** — Larger batches need same or lower LR
5. **Auto-scaling must adjust both dimensions** — Scale `num_minibatches` up AND `update_epochs` down
6. **LR scales inversely with gradient steps** — 2x steps → 0.5x LR

### Reward Shaping
7. **Bidirectional shaping enables reward hacking** — Agent can oscillate for zero net reward
8. **One-directional shaping removes equilibrium** — Only progress gives reward, but no cost to stall
9. **Step penalty creates urgency** — Small per-step cost makes stalling expensive
10. **Watch episode length vs return** — If length rises but return flatlines, agent is stalling
