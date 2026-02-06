# JAX Training Metrics Spec

**Status:** Complete
**Depends on:** [reward-parity.md](./reward-parity.md), [training-speedups.md](./training-speedups.md)

## Goal

Add per-update reward breakdown and behavioral metrics to PureJaxRL wandb logging, matching the diagnostic visibility the SB3/Swift training path has. Currently the JAX path logs only a single scalar reward with no visibility into *which* reward components are driving learning.

## Current State

### SB3/Swift path logs (per episode):
- 14-field reward breakdown (stage, kills, distance, score, dataSiphon, victory, death, resourceGain, resourceHolding, damagePenalty, hpRecovery, siphonQuality, programWaste, siphonDeathPenalty)
- Behavioral stats: action counts (move/siphon/program), programs_used, highest_stage, steps
- All goes to wandb + SQLite

### PureJaxRL (JAX) path logs (per update):
- Single scalar `mean_reward` (no breakdown)
- `mean_episode_return`, `mean_episode_length`, `num_episodes`
- PPO loss metrics (kl, clip_frac, entropy, losses)
- Minimal info: stage, score, hp (not logged to wandb)

## Design

### Approach: Per-Update Averages

Compute the mean of each reward component across all `num_steps * num_envs` transitions in the rollout batch. This is the natural fit for JAX vectorized training -- no per-env episode accumulators needed.

For behavioral stats, compute action type fractions and max stage from the rollout batch.

### Wandb Metric Names

**Reward breakdown** (`reward/` prefix, 15 metrics):

| Wandb Key | Source | Description |
|-----------|--------|-------------|
| `reward/step_penalty` | `-0.01` per step | Time pressure |
| `reward/stage_completion` | `[1,2,4,...,100]` | Stage advance bonus |
| `reward/score_gain` | `delta * 0.5` | Points from siphon |
| `reward/kills` | `0.3` per kill | Enemy elimination |
| `reward/data_siphon` | `1.0` flat | Data siphon collection |
| `reward/distance_shaping` | `+0.05` per cell closer | BFS path progress |
| `reward/victory` | `500 + score * 100` | Game win bonus |
| `reward/death_penalty` | `-0.5 * stage_cumulative` | Death cost |
| `reward/resource_gain` | `delta * 0.05` | Credits/energy acquired |
| `reward/resource_holding` | `curr * 0.01` | Resources on stage complete |
| `reward/damage_penalty` | `-1.0` per HP lost | HP loss (negative component) |
| `reward/hp_recovery` | `+1.0` per HP gained | HP gained (positive component) |
| `reward/siphon_quality` | `-0.5 * missed` | Suboptimal siphon position |
| `reward/siphon_death_penalty` | `-10.0` flat | Death to siphon enemy |
| `reward/program_waste` | `-0.3` | RESET at 2 HP |

**Note:** The current code computes `hp_delta * 1.0` as a single term. The breakdown splits this into `damage_penalty` (negative part) and `hp_recovery` (positive part) to match Swift's separate tracking.

**Behavioral stats** (`actions/` and `stats/` prefixes):

| Wandb Key | Computation | Description |
|-----------|-------------|-------------|
| `actions/move_frac` | `(action < 4).mean()` | Fraction of move actions |
| `actions/siphon_frac` | `(action == 4).mean()` | Fraction of siphon actions |
| `actions/program_frac` | `(action >= 5).mean()` | Fraction of program actions |
| `stats/highest_stage` | `stage.max()` | Max stage across all envs in rollout |

### Data Flow

```
calculate_reward() -> (total, breakdown_dict)
    |
env.py:step() -> (state, obs, reward, done, breakdown)
    |
env_wrapper.py:step() -> (obs, state, reward, done, info={...breakdown})
    |
training_loop.py: scan collects infos alongside transitions
    |
After rollout: mean(breakdown), action fractions from transitions
    |
logging.py -> wandb with reward/, actions/, stats/ prefixes
```

### Key Design Decisions

1. **`calculate_reward` returns `(total, breakdown_dict)`** -- breakdown is a flat dict of `jnp.float32` scalars. All 15 components are named keys. Values sum to total.

2. **`env.py:step()` returns 5-tuple** -- adds `reward_breakdown` as 5th return value. This is a breaking change to all callers.

3. **`env_wrapper.py` merges breakdown into info dict** -- the existing info dict (`stage`, `score`, `hp`) gets the 15 breakdown keys added flat.

4. **Training loop collects infos in scan** -- the `_env_step` scan returns `(transition, infos)` instead of just `transition`. Infos have shape `{key: (num_steps, num_envs)}`.

5. **Action stats computed from transition actions** -- no env changes needed, just `(actions < 4).mean()` etc. after rollout.

6. **Pre-prefixed metric keys** -- training loop builds final wandb key names (e.g., `reward/kills`). `log_metrics` passes keys with `/` through without adding the `train/` prefix.

7. **WandB only** -- no SQLite database. WandB provides history, charts, and export.

### Invariants

- Sum of all 15 breakdown values must equal the total reward (within float tolerance)
- `move_frac + siphon_frac + program_frac == 1.0`
- `reward/step_penalty` should be consistently ~`-0.01` (sanity check)

## Files Affected

| File | Change |
|------|--------|
| `python/hackmatrix/jax_env/rewards.py` | Return `(total, breakdown_dict)` instead of scalar |
| `python/hackmatrix/jax_env/env.py` | Return 5-tuple `(state, obs, reward, done, breakdown)` |
| `python/hackmatrix/purejaxrl/env_wrapper.py` | Merge breakdown into info dict |
| `python/hackmatrix/purejaxrl/training_loop.py` | Collect infos in scan, aggregate breakdown + action stats |
| `python/hackmatrix/purejaxrl/logging.py` | Handle pre-prefixed metric keys (skip `train/` for `reward/`, `actions/`, `stats/`) |
| `python/tests/test_purejaxrl.py` | Update for 5-tuple step return |
| `python/tests/test_reward_parity.py` | Update for `(total, breakdown)` return |

## Testing

### Unit tests

- `test_reward_breakdown_keys` -- verify `calculate_reward` returns all 15 breakdown keys
- `test_reward_breakdown_sum` -- verify breakdown values sum to total reward
- `test_breakdown_through_env_step` -- verify env.step returns breakdown dict
- `test_breakdown_through_wrapper` -- verify wrapper includes breakdown in info

### Integration test

- `test_training_metrics_logged` -- run 1 chunk of training, verify metrics dict contains all `reward_*`, `move_frac`, `siphon_frac`, `program_frac`, `highest_stage` keys

### Manual validation

- Run training for ~50 updates with wandb enabled
- Verify all 15 reward breakdown charts appear in wandb dashboard
- Verify action fraction charts appear
- Verify `reward/step_penalty` is consistently ~`-0.01` (sanity check)
- Verify breakdown values sum to `train/mean_reward` (within float tolerance)

## Success Criteria

- [ ] All 15 reward breakdown components visible as separate wandb charts under `reward/`
- [ ] Action fractions visible under `actions/`
- [ ] `stats/highest_stage` visible in wandb
- [ ] Breakdown values sum to `train/mean_reward` (sanity check)
- [ ] No regression in training throughput (< 5% overhead from breakdown tracking)
- [ ] All existing tests pass with updated signatures
