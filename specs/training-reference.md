# RL Training Reference

Reference documentation for training HackMatrix agents with reinforcement learning.

**Two training paths exist:**

| Path | Status | Environment | Accelerator | Description |
|------|--------|-------------|-------------|-------------|
| **PureJaxRL (JAX)** | **Active** | JAX env (`jax_env/`) | TPU / GPU / CPU | Full game logic in JAX, vectorized rollouts, no subprocess overhead |
| SB3/Swift | Legacy | Gym env (`gym_env.py`) → Swift subprocess | GPU (PyTorch) | MaskablePPO via Stable-Baselines3, Swift binary for game logic |

The PureJaxRL path is ~100x faster on TPU due to fully vectorized rollouts with no Python↔Swift IPC.

---

## PureJaxRL Training (Active)

### Quick Start

```bash
# Dev container (venv auto-activated)
python python/scripts/train_purejaxrl.py

# macOS
cd python && source venv-macos/bin/activate && python scripts/train_purejaxrl.py

# With common options
python python/scripts/train_purejaxrl.py \
  --total-timesteps 1000000000 \
  --num-envs 2048 \
  --num-steps 128 \
  --lr 6.25e-5 \
  --ent-coef 0.15 \
  --log-interval 5 \
  --checkpoint-dir checkpoints
```

### CLI Arguments

**Environment:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-envs` | 256 | Parallel environments (2048 recommended on TPU) |
| `--num-steps` | 128 | Steps per rollout |
| `--total-timesteps` | 10,000,000 | Total training timesteps |

**PPO Hyperparameters:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 2.5e-4 | Learning rate (use 6.25e-5 with 2048 envs) |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--num-minibatches` | 4 | Minibatches per epoch (auto-scales for large batches) |
| `--update-epochs` | 4 | PPO epochs per update (auto-scales, floor of 2) |
| `--clip-eps` | 0.2 | PPO clip epsilon |
| `--vf-coef` | 0.5 | Value loss coefficient |
| `--ent-coef` | 0.15 | Entropy coefficient (prevents entropy collapse) |
| `--max-grad-norm` | 0.5 | Gradient clipping |

**Network:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | 256 | Hidden layer dimension |
| `--num-layers` | 2 | Number of hidden layers |

**Logging & Checkpointing:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-interval` | 10 | Log every N updates |
| `--no-wandb` | false | Disable WandB |
| `--project` | "hackmatrix" | WandB project name |
| `--entity` | "charles-team" | WandB entity |
| `--run-name` | auto-generated | WandB run name |
| `--run-suffix` | — | Suffix for auto-generated run name |
| `--save-interval-minutes` | 10.0 | Time-based checkpointing |
| `--save-interval` | — | Step-based checkpointing (overrides time-based) |
| `--checkpoint-dir` | "checkpoints" | Checkpoint output directory |
| `--no-artifact` | false | Disable checkpoint uploads to WandB |

**Resume & Misc:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--resume` | — | Path to checkpoint `.pkl` file |
| `--rewind` | false | Rewind WandB history to checkpoint step on resume |
| `--seed` | 0 | Random seed |
| `--auto-tune` | false | Auto-tune config for detected device |

### Auto-Scaling

When `num_envs` increases beyond the reference 256, `auto_scale_for_batch_size()` adjusts minibatches and epochs to maintain consistent gradient noise:

```
batch_size = num_envs × num_steps
minibatch_size ≈ 8,192 (target)
gradient_steps ≈ 16 (target, floor of 2 epochs)
```

Example at 2048 envs:
```
batch_size = 2048 × 128 = 262,144
num_minibatches = 4 × 8 = 32
update_epochs = max(2, 4 // 8) = 2
gradient_steps = 32 × 2 = 64
```

When epochs increase, halve the LR to compensate: `2x epochs → 0.5x LR`.

### Resume from Checkpoint

```bash
# Resume training (auto-detects run name from path)
python python/scripts/train_purejaxrl.py \
  --resume checkpoints/run-name/checkpoint_100.pkl

# Resume and rewind WandB history to checkpoint step
python python/scripts/train_purejaxrl.py \
  --resume checkpoints/run-name/checkpoint_100.pkl \
  --rewind
```

Checkpoint files saved per step:
- `checkpoint_{step}.pkl` — full state (params + optimizer + metrics + last_logged_step)
- `params_{step}.npz` — params only (for inference / `watch_jax_agent.py`)

Training saves a checkpoint on Ctrl+C (interrupt safety).

### Watch Trained Agent

```bash
# Watch model play (requires Xcode GUI build on macOS)
python python/scripts/watch_jax_agent.py checkpoints/run-name/params_100.npz

# Watch N episodes
python python/scripts/watch_jax_agent.py checkpoints/run-name/params_100.npz --episodes 5

# Auto-find latest checkpoint
python python/scripts/watch_jax_agent.py --latest
```

Architecture is auto-detected from parameter shapes. Override with `--hidden-dim` / `--num-layers` if needed.

### W&B Metrics

Training logs to WandB project `hackmatrix`. Metrics are logged every `log_interval` updates.

**PPO Training Metrics** (`train/` prefix):

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| `train/entropy` | > 1.0 | Approaching 0 = entropy collapse |
| `train/mean_episode_return` | Climbing steadily | Flatline for extended period |
| `train/mean_episode_length` | Increasing | Stuck at low value |
| `train/pg_loss` | Non-zero | Near zero = no learning |
| `train/approx_kl` | < 0.06 | > 0.06 = policy changing too fast |
| `train/clip_frac` | < 0.30 | > 0.30 = clipping too aggressively |
| `train/mean_reward` | — | Mean step reward across rollout |
| `train/num_episodes` | — | Completed episodes per update |

**Reward Breakdown** (`reward/` prefix, 15 metrics):

| Metric | Description |
|--------|-------------|
| `reward/step_penalty` | -0.01 per step (sanity check: should be ~-0.01) |
| `reward/stage_completion` | Stage advance bonus [1,2,4,...,100] |
| `reward/score_gain` | Points from siphon (delta × 0.5) |
| `reward/kills` | 0.3 per enemy killed |
| `reward/data_siphon` | 1.0 flat for data siphon collection |
| `reward/distance_shaping` | +0.05 per cell closer (BFS path) |
| `reward/victory` | Game win (500 + score × 100) |
| `reward/death_penalty` | -0.5 × stage_cumulative |
| `reward/resource_gain` | Credits/energy acquired (delta × 0.05) |
| `reward/resource_holding` | Resources on stage complete (curr × 0.01) |
| `reward/damage_penalty` | -1.0 per HP lost |
| `reward/hp_recovery` | +1.0 per HP gained |
| `reward/siphon_quality` | -0.5 × missed resources (suboptimal siphon) |
| `reward/siphon_death_penalty` | -10.0 for death to siphon enemy |
| `reward/program_waste` | -0.3 for RESET at 2 HP |

**Action Fractions** (`actions/` prefix):

| Metric | Description |
|--------|-------------|
| `actions/move_frac` | Fraction of move actions (should sum to 1.0) |
| `actions/siphon_frac` | Fraction of siphon actions |
| `actions/program_frac` | Fraction of program actions |

**Stage Stats** (`stats/` prefix):

| Metric | Description |
|--------|-------------|
| `stats/highest_stage` | Max stage reached across all envs in rollout |

### Running and Monitoring

**Check for existing training process:**
```bash
pgrep -af train_purejaxrl
```

**Check device availability:**
```bash
python -c "import jax; print(jax.devices())"
```

**Launch in background with log file:**
```bash
python python/scripts/train_purejaxrl.py [args] > /tmp/training.log 2>&1 &
```

**Monitor output:**
```bash
tail -20 /tmp/training.log
```

**Stop training:**
```bash
kill $(pgrep -f train_purejaxrl)
```

**JAX compilation cache** is auto-enabled via `JAX_COMPILATION_CACHE_DIR`. First run compiles for ~5-20 minutes; subsequent runs reuse the cache.

**CPU-only mode** (when TPU is occupied):
```bash
JAX_PLATFORMS=cpu python python/scripts/train_purejaxrl.py --num-envs 64 --no-wandb
```

### Entropy Collapse

**Symptoms:**
- `train/entropy` approaching 0
- `train/approx_kl` = 0
- `train/mean_episode_return` flatlined
- `train/clip_frac` = 0

**Solutions:**
1. Stop training immediately
2. Increase `--ent-coef` (try 0.15 or 0.2)
3. Start fresh — do NOT resume from collapsed model

**Prevention:**
- Use `--ent-coef >= 0.1` (default 0.15)
- Monitor entropy regularly

### Key Files

| File | Purpose |
|------|---------|
| `python/scripts/train_purejaxrl.py` | Main training script |
| `python/scripts/watch_jax_agent.py` | Watch trained agent play (visual mode) |
| `python/hackmatrix/purejaxrl/config.py` | TrainConfig + auto-scaling + device tuning |
| `python/hackmatrix/purejaxrl/training_loop.py` | JIT-compiled chunked training loop |
| `python/hackmatrix/purejaxrl/logging.py` | WandB + console logging |
| `python/hackmatrix/purejaxrl/checkpointing.py` | Checkpoint save/load |
| `python/hackmatrix/purejaxrl/env_wrapper.py` | Gymnax-style env wrapper (obs flattening, info dict) |
| `python/hackmatrix/jax_env/env.py` | Pure JAX game environment |
| `python/hackmatrix/jax_env/rewards.py` | Reward calculation with 15-component breakdown |

---

## SB3/Swift Training (Legacy)

The SB3 path uses Stable-Baselines3 MaskablePPO with a Swift subprocess for game logic. It is slower than PureJaxRL but useful for debugging reward/observation issues since it runs through the canonical Swift `GameState`.

### Commands

```bash
# Build Swift binary (headless mode)
./swift-build

# Start training
cd python && source venv-macos/bin/activate
python scripts/train.py --timesteps 100000000 --save-freq 50000

# Resume from checkpoint
python scripts/train.py --resume ./models/MODEL_DIR/best_model.zip

# Train without W&B logging
python scripts/train.py --no-wandb

# Watch trained agent play (requires Xcode GUI build)
python scripts/watch_trained_agent.py --model ./models/MODEL_DIR/best_model.zip
```

### Configuration

```python
learning_rate=3e-4
n_steps=2048
batch_size=64
n_epochs=10
gamma=0.99
gae_lambda=0.95
clip_range=0.2
ent_coef=0.1
```

### Architecture

```
train.py → gym_env.py (Gymnasium) → Swift subprocess (--headless-cli)
  └── MaskablePPO (SB3)              └── GameState.tryExecuteAction()
```

Rollout collection is CPU-bound (Swift subprocess). Training updates use GPU (PyTorch/CUDA).

### Key Files

| File | Purpose |
|------|---------|
| `python/scripts/train.py` | SB3 training script |
| `python/scripts/watch_trained_agent.py` | Watch SB3 model play |
| `python/hackmatrix/gym_env.py` | Gymnasium environment (Swift bridge) |
| `python/hackmatrix/training_config.py` | SB3 hyperparameter config |

---

## Troubleshooting

### Rewards Seem Wrong

**PureJaxRL:** Check the `reward/` breakdown in WandB. Verify `reward/step_penalty ≈ -0.01` as a sanity check. Run parity tests: `JAX_PLATFORMS=cpu python -m pytest python/tests/test_reward_parity.py -v`

**SB3/Swift:** Rebuild Swift binary after reward changes (`./swift-build`). Check episode info breakdown.

### Model Won't Load

- Observation space changed = incompatible checkpoints. Must retrain from scratch.
- Architecture mismatch: `watch_jax_agent.py` auto-detects from param shapes, but override with `--hidden-dim` / `--num-layers` if needed.

### JIT Compilation Hangs

The JAX compilation cache (`JAX_COMPILATION_CACHE_DIR`) prevents recompilation. If compilation hangs on first run, ensure the pathfinding module uses wavefront BFS (not while_loop-based BFS).

### TPU Occupied

Use `JAX_PLATFORMS=cpu` for tests and short runs when the TPU is busy with a training process.

## Key Learnings

1. **Entropy collapse is real** — use `ent_coef >= 0.1` (0.15 recommended)
2. **Rare actions need multiple epochs** — auto-scaling floors at 2 epochs to amplify sparse signals
3. **LR scales inversely with gradient steps** — 2x epochs → 0.5x LR
4. **Observation completeness matters** — missing features prevent learning
5. **Reward shaping helps** — intermediate rewards (distance, resource gain) guide learning
6. **Credit assignment is hard** — long episodes (8 stages) slow learning
7. **Don't reward exploits** — enemy farming breaks training
8. **Compilation cache saves time** — first JAX compile takes 5-20 min, cache makes reruns instant
