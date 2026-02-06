# Implementation Plan

**Specs:** [jax-training-metrics.md](specs/jax-training-metrics.md) (Active), [training-tuning.md](specs/training-tuning.md) (Draft)

## Completed

### Phase 1: Reward Breakdown (jax-training-metrics) — DONE

All items implemented and tested:

- [x] **1.1** `calculate_reward()` returns `(total, breakdown_dict)` with 15 keys
- [x] **1.2** `env.py:step()` returns 5-tuple `(state, obs, reward, done, breakdown)`
- [x] **1.3** `batched_step = jax.vmap(step)` auto-adapted
- [x] **1.4** `env_wrapper.py:step()` merges breakdown into info dict
- [x] **1.5** `training_loop.py:_env_step()` scan returns `(transition, infos)`
- [x] **1.6** Breakdown aggregation: `jnp.mean()` per key → `reward/` prefix
- [x] **1.7** Action fractions: `actions/move_frac`, `actions/siphon_frac`, `actions/program_frac`, `stats/highest_stage`
- [x] **1.8** `logging.py:log_metrics()` passes pre-prefixed keys (containing `/`) through as-is
- [x] **1.9** `evaluate()` handles 5-tuple step return

### Phase 2: Test Updates (jax-training-metrics) — DONE

- [x] **2.1** `test_reward_parity.py` `_reward()` helper unpacks tuple; added `_reward_with_breakdown()`
- [x] **2.2** `TestRewardBreakdownKeys` — verifies all 15 keys present
- [x] **2.3** `TestRewardBreakdownSum` — verifies sum equals total in multiple scenarios + HP split
- [x] **2.4** `TestBreakdownThroughEnvStep` — verifies env.step returns 5-tuple with breakdown
- [x] **2.5** `TestBreakdownThroughWrapper` — verifies wrapper info dict has all breakdown keys
- [x] **2.6** `jax_env_wrapper.py` updated for 5-tuple (parity tests)
- [x] **2.7** `TestTrainingMetrics` — integration test verifying all `reward/`, `actions/`, `stats/` keys in training output

### Phase 4: Validation (Phases 1-2) — DONE

- [x] **4.1** All 111 JAX tests pass (76 reward parity + 24 purejaxrl + 12 implementation - 1 placeholder)
- [x] **4.2** Training compiles and runs with breakdown (verified via integration test)

## Remaining Work

### Phase 3: Hyperparameter Tuning (training-tuning)

Lower priority — spec is Draft status.

- [ ] **3.1** Change epoch floor in `config.py:auto_scale_for_batch_size()` from `max(1, ...)` to `max(2, ...)`
- [ ] **3.2** Update `ent_coef` default from `0.1` to `0.15` in `config.py`
- [ ] **3.3** Update `--ent-coef` default in `train_purejaxrl.py` from `0.1` to `0.15`
- [ ] **3.4** Update `training-speedups.md` — document new auto-scaling behavior (epoch floor 2), LR guidance (`6.25e-5`), and recommended training command

### Phase 4: Manual Validation

- [ ] **4.3** (Manual) Run training with wandb, verify 15 reward charts + action fraction charts appear

## Files Changed

| File | Change |
|------|--------|
| `python/hackmatrix/jax_env/rewards.py` | Returns `(total, breakdown_dict)` with 15 keys |
| `python/hackmatrix/jax_env/env.py` | Returns 5-tuple, batched_step auto-adapted |
| `python/hackmatrix/jax_env/__init__.py` | Updated docstring for 5-tuple |
| `python/hackmatrix/purejaxrl/env_wrapper.py` | Merges breakdown into info dict |
| `python/hackmatrix/purejaxrl/training_loop.py` | Collects infos in scan, aggregates breakdown + action stats |
| `python/hackmatrix/purejaxrl/logging.py` | Pre-prefixed keys pass through without `train/` prefix |
| `python/tests/test_reward_parity.py` | Updated for 2-tuple return, added 13 breakdown tests |
| `python/tests/test_purejaxrl.py` | Added `TestTrainingMetrics` integration test |
| `python/tests/jax_env_wrapper.py` | Updated for 5-tuple step return |

## Key Design Decisions

1. **Breakdown dict in JAX scan**: The breakdown dict is collected alongside transitions in the scan. Each value is a scalar per step per env, producing shape `(num_steps, num_envs)` after scan — same as rewards. Lightweight.

2. **HP split is cosmetic**: The actual reward calculation computes `damage_penalty = min(hp_delta, 0) * 1.0` and `hp_recovery = max(hp_delta, 0) * 1.0` separately. Sum equals `hp_delta * 1.0`.

3. **Pre-prefixed keys**: The training loop builds `reward/kills`, `actions/move_frac` etc. directly. The logger detects `/` in key names and passes them through without adding `train/`. This avoids `train/reward/kills` nesting.

4. **Phase 3 is separate**: The training-tuning changes (epoch floor, ent_coef) are independent of the metrics work. They can be implemented and tested separately, and the spec is still Draft.
