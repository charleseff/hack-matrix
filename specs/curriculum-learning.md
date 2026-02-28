# Curriculum Learning Spec

**Status:** Draft
**Depends on:** [training-tuning.md](./training-tuning.md), [game-mechanics.md](./game-mechanics.md)

## Problem

After extensive hyperparameter tuning (6 training runs, 475+ updates each), the agent reliably discovers siphon strategies — transient spikes growing from 0.21% to 2.51% siphon usage — but cannot stabilize them. Returns plateau at ~-2.0 regardless of configuration.

### Root cause: insufficient learning signal density

The core issue is not PPO stability (solved by 32 gradient steps) or exploration (entropy stays healthy at 1.2-1.4). It's that strategic actions are too rare for the agent to learn from:

| Metric | Expected | Actual |
|--------|----------|--------|
| Steps with siphon available | 5-10% | ~0.01% |
| Siphon actions per rollout (262K steps) | ~13,000-26,000 | ~26 |
| Program actions per rollout | ~1,000+ | ~0 |

The agent must: (1) navigate to a corner to pick up a data siphon, (2) navigate to a block, (3) siphon, (4) survive the spawned enemies — all within a short episode. Most episodes end before step 2.

Meanwhile, distance shaping (+0.05/cell) provides ~6,000+ positive signals per rollout, drowning out the ~26 siphon signals.

### Why hyperparameter tuning can't fix this

The "siphon spike" pattern observed across all runs follows a consistent cycle:
1. Agent stumbles into siphoning through entropy-driven exploration
2. Value function assigns high value to siphon states (based on small sample)
3. Policy overweights siphon, dies to spawned enemies (siphon death penalty: -10.0)
4. Value function corrects downward, policy abandons siphon
5. Return recovers to baseline (~-2.0)

This cycle repeats with growing amplitude but never converges because the agent never accumulates enough successful siphon experiences to learn *when* siphoning is safe.

## Proposed Solution: Curriculum Phases

Modify the environment to create three training phases, each exposing the agent to strategic actions at increasing difficulty. Phase transitions happen automatically based on training metrics.

### Phase 1: Siphon School

Maximize siphon/program exposure with minimal danger. The agent should learn that siphoning and programs are valuable actions.

| Parameter | Normal Value | Phase 1 Value | Rationale |
|-----------|-------------|---------------|-----------|
| Starting data siphons | 0 | 2 | Siphon available immediately, no corner navigation required |
| Transmission count | [1,2,3,4,5,6,7,8] per stage | [0,0,1,1,2,2,3,3] | Far fewer enemies, siphoning is safer |
| Starting credits | 0 | 3 | Programs immediately usable |
| Starting energy | 0 | 3 | Programs immediately usable |
| Siphon death penalty | -10.0 | -2.0 | Reduce harshness of failed siphons |
| Distance shaping coef | 0.05 | 0.025 | Strategic signals less drowned out |
| Data siphon collection reward | 1.0 | 2.0 | Stronger signal for collecting siphons |

**Expected behavior:** Agent learns siphon → resource gain → program use chain. Siphon fraction should reach 5%+ within 200 updates.

### Phase 2: Increasing Pressure

Partially restore difficulty. The agent should learn *when* to siphon (not just that siphoning exists).

| Parameter | Phase 1 Value | Phase 2 Value |
|-----------|---------------|---------------|
| Starting data siphons | 2 | 1 |
| Transmission count | [0,0,1,1,2,2,3,3] | [1,1,2,2,3,3,4,4] |
| Starting credits | 3 | 1 |
| Starting energy | 3 | 1 |
| Siphon death penalty | -2.0 | -5.0 |
| Distance shaping coef | 0.025 | 0.04 |
| Data siphon collection reward | 2.0 | 1.5 |

**Expected behavior:** Agent retains siphon usage but becomes selective. Siphon fraction may decrease from Phase 1 peak but stays above 1%. Episodes survive longer as agent learns to avoid siphoning near enemies.

### Phase 3: Full Game

Restore all parameters to normal values. The agent should apply learned strategies under real game conditions.

| Parameter | Phase 2 Value | Phase 3 (Normal) |
|-----------|---------------|------------------|
| Starting data siphons | 1 | 0 |
| Transmission count | [1,1,2,2,3,3,4,4] | [1,2,3,4,5,6,7,8] |
| Starting credits | 1 | 0 |
| Starting energy | 1 | 0 |
| Siphon death penalty | -5.0 | -10.0 |
| Distance shaping coef | 0.04 | 0.05 |
| Data siphon collection reward | 1.5 | 1.0 |

**Expected behavior:** Agent navigates to corners for siphons, uses programs strategically. Return exceeds -1.5.

### Phase Transitions

Transitions are based on return thresholds with update-count fallbacks:

| Transition | Primary Trigger | Fallback Trigger |
|------------|----------------|------------------|
| Phase 1 → 2 | mean_episode_return > -1.0 for 20 consecutive updates | After 300 updates in Phase 1 |
| Phase 2 → 3 | mean_episode_return > -1.2 for 20 consecutive updates | After 300 updates in Phase 2 |

The return thresholds are lower (easier) than Phase 3's success criteria because the simplified environments have higher achievable returns. The fallback ensures training doesn't stall indefinitely in early phases.

### Phase Snapshots and Rollback

Each phase transition automatically saves a labeled snapshot checkpoint before changing environment parameters. This allows rolling back to the start of any phase without losing all prior training.

**Automatic behavior:**
- When transitioning from Phase N to Phase N+1, save `phase_N_complete.pkl` before applying new env params
- These snapshots are separate from the regular time-based checkpoints and never overwritten

**CLI ergonomics:**

```bash
# Resume from the start of Phase 2 (using Phase 1's final weights)
python python/scripts/train_purejaxrl.py \
  --resume checkpoints/run-name/phase_1_complete.pkl \
  --curriculum --curriculum-start-phase 2

# Resume from the start of Phase 3
python python/scripts/train_purejaxrl.py \
  --resume checkpoints/run-name/phase_2_complete.pkl \
  --curriculum --curriculum-start-phase 3

# Resume from any regular checkpoint within a phase
python python/scripts/train_purejaxrl.py \
  --resume checkpoints/run-name/checkpoint_450.pkl \
  --curriculum --curriculum-start-phase 2
```

The `--curriculum-start-phase` flag overrides the automatic phase detection, forcing training to begin at the specified phase regardless of the checkpoint's original phase. This enables:
- **Rollback one phase**: Load the snapshot from the previous phase boundary and restart
- **Retry a phase with different params**: Tweak phase parameters, reload the snapshot, retrain
- **Skip phases**: If Phase 1 converges quickly, jump straight to Phase 3 to test

**Snapshot contents** are identical to regular checkpoints (params, opt_state, step, metrics) plus the phase number and env params that produced them. The checkpoint file records `curriculum_phase` so that resuming without `--curriculum-start-phase` automatically continues in the correct phase.

## Architecture Changes

### EnvParams (currently empty stub)

The `EnvParams` dataclass in `env_wrapper.py` is the natural insertion point. Currently empty and required by the Gymnax interface, it gets passed to `reset()` and `step()` but ignored. Adding curriculum fields here requires threading params through to the underlying JAX env functions.

### Curriculum parameters to add to EnvParams

```python
@struct.dataclass
class EnvParams:
    starting_data_siphons: jnp.int32 = 0
    starting_credits: jnp.int32 = 0
    starting_energy: jnp.int32 = 0
    transmission_scale: jnp.float32 = 1.0       # multiplier on per-stage transmission count
    siphon_death_penalty: jnp.float32 = -10.0
    distance_shaping_coef: jnp.float32 = 0.05
    data_siphon_reward: jnp.float32 = 1.0
```

`transmission_scale` is a float multiplier applied to the per-stage transmission count array, then rounded. This avoids hardcoding 8 separate counts and allows smooth interpolation.

### RunnerState changes

`env_params` must be carried in `RunnerState` so the Python training loop can update them between JIT-compiled chunks. The update step function must read params from the runner state rather than from a captured closure.

### Files affected

| File | Change |
|------|--------|
| `python/hackmatrix/purejaxrl/env_wrapper.py` | Add fields to `EnvParams`, thread params through `reset()` and `step()` |
| `python/hackmatrix/jax_env/env.py` | Accept params in `reset()` and `step()`, pass to stage gen and rewards |
| `python/hackmatrix/jax_env/stage.py` | Use params for starting siphons/resources, transmission scaling |
| `python/hackmatrix/jax_env/rewards.py` | Use params for distance_shaping_coef, siphon_death_penalty, data_siphon reward |
| `python/hackmatrix/purejaxrl/training_loop.py` | Add `env_params` to `RunnerState`, thread through update step |
| `python/hackmatrix/purejaxrl/checkpointing.py` | Save/load `curriculum_phase` and `env_params` in checkpoints; add `save_phase_snapshot()` |
| `python/scripts/train_purejaxrl.py` | Add `--curriculum`, `--curriculum-start-phase` flags; phase transition logic with auto-snapshots |

## Success Criteria

- [ ] Phase 1 produces consistent siphon usage (siphon_frac > 3% sustained for 50+ updates)
- [ ] Phase 1 produces non-zero program usage (program_frac > 0.1%)
- [ ] Agent retains siphon behavior through Phase 2 transition (siphon_frac stays > 0.5%)
- [ ] Phase 3 mean_episode_return exceeds -1.5 (the plateau from training-tuning)
- [ ] Phase 3 siphon_frac > 0.5% sustained (not transient spikes)

## Training Validation

### Run command

```bash
python python/scripts/train_purejaxrl.py \
  --total-timesteps 1000000000 \
  --num-envs 2048 \
  --num-steps 128 \
  --lr 6.25e-5 \
  --ent-coef 0.15 \
  --num-minibatches 16 \
  --update-epochs 2 \
  --log-interval 5 \
  --checkpoint-dir checkpoints \
  --save-interval-minutes 10 \
  --curriculum
```

### Metrics to monitor

**Phase 1 (first ~300 updates):**
- `actions/siphon_frac` — should climb above 3% within 200 updates
- `actions/program_frac` — should become non-zero
- `reward/data_siphon` — should be consistently positive
- `mean_episode_return` — should improve faster than non-curriculum baseline

**Phase transitions:**
- Watch for return regression at each transition (expected, should recover within 50 updates)
- `approx_kl` — should not spike above 0.08 during transitions

**Phase 3 (after ~600 updates):**
- `mean_episode_return` — target > -1.5
- `actions/siphon_frac` — target > 0.5% sustained
- `stats/highest_stage` — should exceed 5 regularly

### Success thresholds

- Phase 1 siphon_frac > 3% by update 200
- Phase 3 return > -1.5 by update 1000
- No entropy collapse below 1.0 at any point

### Expected duration

~4-6 hours for full curriculum (900+ updates across all phases)

### Early stopping signals

- Phase 1 siphon_frac stays < 0.5% after 200 updates → curriculum params too conservative, simplify further
- KL divergence > 0.10 sustained → learning rate too high for curriculum transitions
- Entropy < 0.8 → policy collapsing, increase ent_coef

## Future Considerations

### Program curriculum (separate spec required)

This spec's Phases 1-3 focus on **siphon** learning — teaching the agent to collect and use data siphons strategically. Program learning is a fundamentally harder problem that this curriculum alone will not solve.

Why programs need their own curriculum:
- **23 individual programs**, each with completely different effects (damage, healing, movement, status, area control)
- Each program has unique **preconditions** (resource costs, target requirements, HP thresholds for value)
- Programs interact with each other — **chaining** (e.g., stun then attack) requires learning multi-step strategies
- The value of a program is **highly contextual**: RESET at 1 HP is life-saving, at full HP is wasteful; SHOW is valuable early in a stage, useless when enemies are already revealed
- Unlike siphon (1 action to learn), programs require learning **when NOT to use** each of the 23 options

After Phases 1-3 produce consistent siphon usage, a separate "Program Curriculum" spec should address:
1. **Program grouping**: Cluster the 23 programs by learning difficulty (simple direct-damage programs first, complex conditional programs later)
2. **Forced program availability**: Give the agent specific programs and resources to learn one cluster at a time
3. **Program-specific reward shaping**: Reward efficient program usage (e.g., using RESET at low HP, not high HP)
4. **Phased program introduction**: Unlock program clusters over training, similar to how Phases 1-3 gradually restore siphon difficulty

This is deferred because the siphon curriculum is prerequisite — programs require resources gained from siphoning, so the agent must learn siphoning first.

### Score maximization curriculum

The victory bonus is `500 + score * 100`, where score comes from siphoning data blocks (BLOCK_DATA). Phases 1-3 teach the agent *that siphoning exists* and *when to siphon*, but not *which blocks to siphon for maximum score*. A future curriculum phase could:
- Amplify `score_gain` reward multiplier to teach data block prioritization
- Give the agent pre-siphoned high-value data blocks to learn score association
- Train specifically on "win with highest possible score" objective

This is deferred because it's the endgame optimization — the agent must first reliably win games (requires all prior curriculum phases).

### Curriculum scheduling alternatives

Linear interpolation between phases (rather than discrete jumps) could reduce transition shock. Each parameter would smoothly transition over 50-100 updates. This adds complexity but may prevent the return regression expected at transitions.

### Network architecture

If curriculum learning produces strategic behavior that the MLP (256-dim, 2 layers) can't retain across phases, consider:
- Larger hidden dim (512)
- Attention over grid features (the 6x6x42 grid has spatial structure an MLP ignores)
- Separate value heads for movement vs strategic actions

### Additional curriculum levers not used

These were considered but deferred to keep the initial curriculum simple:
- **Scheduled task interval**: Increasing from 12 to 20+ would reduce enemy pressure further (redundant with transmission_scale)
- **Enemy type bias**: Removing viruses (speed 2) would make the environment safer (overlaps with fewer transmissions)
- **Extra data siphon placement**: Placing siphons on non-corner cells (structural change to board generation)
- **Block count range**: Reducing from 5-11 to 2-5 (fewer obstacles, but also fewer siphon targets)
