# Training Tuning Spec

**Status:** Draft
**Depends on:** [training-speedups.md](./training-speedups.md), [reward-parity.md](./reward-parity.md), [game-mechanics.md](./game-mechanics.md)

## Problem

The agent learns a simple "move toward exit, avoid dying" policy but never learns strategic use of siphons or programs. After 370M+ timesteps (1400 updates) in the previous run, episode return plateaued at ~-1.56 with no sign of improvement. The current run shows similar early dynamics.

### Why strategic actions are hard to learn

The 28-action space has a steep rarity hierarchy:

| Action Type | Actions | Availability | Learning Signal |
|-------------|---------|--------------|-----------------|
| Movement | 4 (up/down/left/right) | Almost always | Dense (distance shaping +0.05/cell) |
| Siphon | 1 | When data siphons held | Medium (resource gain 0.05/unit, but costs a turn) |
| Programs | 23 | Requires ownership + resources + applicability | Sparse, delayed, indirect |

An untrained agent with siphon available will use it immediately and unstrategically. The same applies to programs. The agent needs to learn *when* each action should be used — but three factors conspire against this:

1. **Rare encounters**: Programs are available on ~5-10% of steps. Most episodes end before meaningful program decisions arise.
2. **Delayed consequences**: Using RESET at the wrong HP, or siphoning at the wrong position, has consequences that only manifest turns later (death, missed resources).
3. **Dominant dense signal**: Distance shaping (+0.05/cell closer) overwhelms the sparse program/siphon signals. The agent can get decent reward by ignoring programs entirely.

### Identified bottlenecks

#### 1. Auto-scaling reduces update_epochs to 1

The `auto_scale_for_batch_size()` function scales `update_epochs` from 4 down to 1 at 2048 envs. This means every training sample is processed exactly once per update — then discarded.

For common events (movement), this is fine — there are thousands per batch. But for rare events (siphon at suboptimal position, RESET at 2 HP, strategic program chains), a single gradient step produces a weak signal that's easily overwhelmed by the dominant movement gradients.

With 2+ epochs, rare event samples get reprocessed, amplifying their signal relative to noise. This is the standard PPO trade-off: more epochs = better sample efficiency but more policy change per update.

**Current auto-scaling math:**
```
batch_size = 2048 × 128 = 262,144 (8x reference)
num_minibatches = 4 × 8 = 32
update_epochs = max(1, 4 // 8) = 1
gradient_steps = 32 × 1 = 32
```

**Key concern:** The auto-scaling was designed to bound *total gradient steps* (~32) to prevent policy overshooting. But 1 epoch is the floor, and it sacrifices sample efficiency for rare events.

#### 2. Entropy is collapsing early

| Run | Update 100 Entropy | Typical Valid Actions |  Max Entropy (uniform over valid) |
|-----|-------------------|---------------------|----------------------------------|
| Feb 5 (prev) | 2.14 | 4-5 | ~1.6 |
| Feb 6 (curr) | 1.35 | 4-5 | ~1.6 |

The current run's entropy (1.35) is closer to max but trending in the same direction. When the policy becomes too peaked on movement directions, rare actions (siphon, programs) get negligible probability mass even when valid. The agent stops exploring them, creating a self-reinforcing loop.

`ent_coef=0.1` is the current setting. For environments with highly variable action availability (4 actions most steps, 5-10 occasionally), a higher coefficient helps maintain exploration on the occasional steps where more options exist.

#### 3. JIT compilation bottleneck (FIXED)

The BFS pathfinding used `jax.lax.while_loop` with nested `jax.lax.scan`, causing XLA compilation to hang for 20+ minutes (or indefinitely). This was replaced with wavefront BFS using `jax.lax.scan` with fixed iteration count — compilation now completes normally.

**Already applied:** `python/hackmatrix/jax_env/pathfinding.py` refactored from queue-based to wavefront BFS.

## Proposed Changes

### Change 1: Raise auto-scaling epoch floor from 1 to 2

**File:** `python/hackmatrix/purejaxrl/config.py`

Change `auto_scale_for_batch_size()` to floor `update_epochs` at 2 instead of 1.

**New auto-scaling math:**
```
batch_size = 262,144 (unchanged)
num_minibatches = 32 (unchanged)
update_epochs = max(2, 4 // 8) = 2
gradient_steps = 32 × 2 = 64 (was 32)
```

**Trade-off:** 64 gradient steps is 4x the reference 16 (was 2x). This increases policy change per update, which could cause instability. Mitigated by:
- Reducing learning rate proportionally (see Change 2)
- Monitoring KL divergence and clip fraction
- The 2048-env batch provides very stable gradients

**Why 2, not 3 or 4:** 2 epochs doubles the signal for rare events while keeping gradient steps at a manageable 64. At 3 epochs (96 steps) or 4 (128 steps), we risk the overshooting that the auto-scaling was designed to prevent.

### Change 2: Halve learning rate to compensate for doubled epochs

**File:** command-line argument or training script default

With 64 gradient steps (2x more than before), the effective per-update policy change doubles. Halving the learning rate compensates:

```
Before: lr=1.25e-4, 32 gradient steps → effective change ∝ 1.25e-4 × 32 = 0.004
After:  lr=6.25e-5, 64 gradient steps → effective change ∝ 6.25e-5 × 64 = 0.004
```

Same effective policy change per update, but with 2x more passes over the data — amplifying rare event signals without destabilizing training.

**Note:** This follows the established pattern from the training-speedups spec: "LR scales inversely with gradient steps — 2x steps → 0.5x LR."

### Change 3: Increase entropy coefficient from 0.1 to 0.15

**File:** `python/hackmatrix/purejaxrl/config.py` (default) or command-line

Higher entropy bonus keeps the policy more exploratory when rare actions become available. The goal: when siphon or a program enters the valid action mask, the policy assigns it meaningful probability mass instead of near-zero.

**Why 0.15, not higher:** At `ent_coef=0.2+`, entropy loss dominates the objective on steps with many valid actions (e.g., 10+ valid when multiple programs are available). The agent may never commit to any action, preventing learning. 0.15 is a moderate increase that biases toward exploration without overwhelming the policy gradient.

**Monitoring:** If entropy exceeds 2.5+ consistently, or if the policy appears random (no improvement in return), ent_coef is too high.

## Implementation Plan

### Phase 1: Hyperparameter Changes

1. **Modify `auto_scale_for_batch_size()`** — change epoch floor from 1 to 2
2. **Update training-speedups.md** — document new auto-scaling behavior and LR guidance
3. **Update recommended training command** — use `--lr 6.25e-5 --ent-coef 0.15`

### Phase 2: Validation Run

1. **Launch training** with new parameters:
   ```bash
   python3 scripts/train_purejaxrl.py \
     --total-timesteps 1000000000 \
     --num-envs 2048 \
     --num-steps 128 \
     --lr 6.25e-5 \
     --ent-coef 0.15 \
     --log-interval 5 \
     --checkpoint-dir checkpoints
   ```
2. **Monitor early metrics** (first 50 updates):
   - `approx_kl` should stay < 0.06 (was ~0.04 with previous config)
   - `clip_frac` should stay < 0.30
   - `entropy` should be higher than previous run at same update count
3. **Monitor medium-term** (200-500 updates):
   - `mean_episode_length` should increase (surviving longer)
   - `mean_episode_return` should trend upward, not plateau
   - `entropy` should remain > 1.0 (not collapsing)

### Phase 3: Evaluate Strategic Learning (500+ updates)

Success signals that the agent is learning beyond "move toward exit":
- Episodes where siphon is available but NOT used on the first step it appears
- `mean_episode_length` > 30 (surviving long enough for program decisions)
- `mean_episode_return` trending above -1.5 (previous plateau)
- Non-zero program usage in watched replays (`watch_jax_agent.py`)

## Files

| File | Change |
|------|--------|
| `python/hackmatrix/purejaxrl/config.py` | Raise epoch floor to 2 in `auto_scale_for_batch_size()`, update `ent_coef` default to 0.15 |
| `python/hackmatrix/jax_env/pathfinding.py` | Already done: wavefront BFS replacing while_loop |
| `specs/training-speedups.md` | Update auto-scaling docs, LR guidance, recommended command |

## Success Criteria

- [ ] Auto-scaling produces `update_epochs=2` at 2048 envs
- [ ] PPO metrics remain healthy (KL < 0.06, clip_frac < 0.30) with new config
- [ ] Entropy remains above 1.0 through 500+ updates (no premature collapse)
- [ ] `mean_episode_return` improves beyond -1.5 plateau within 1000 updates
- [ ] Agent shows evidence of strategic action use in visual replays

## Future Considerations

### Curriculum Learning

If hyperparameter tuning alone doesn't produce strategic learning, curriculum learning is the next lever. Options:

1. **Stage curriculum**: Start training on later stages where programs matter more, gradually introduce earlier stages
2. **Simplified boards**: Train with fewer enemies or more resources to give the agent breathing room to explore siphon/program strategies
3. **Action-focused curriculum**: Temporarily increase probability of siphon/program availability to accelerate learning for those actions

This is a larger architectural change (requires environment modifications) and should be a separate spec if pursued.

### Network Architecture

The current network (256-dim, 2 layers, MLP) may lack capacity to represent the conditional value of programs (e.g., "RESET is valuable at 1 HP but wasteful at 2 HP"). Attention-based or recurrent architectures could help, but are a larger change with unclear payoff. Defer unless the simpler changes don't work.
