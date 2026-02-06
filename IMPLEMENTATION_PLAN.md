# Implementation Plan

**Specs:** [jax-training-metrics.md](specs/jax-training-metrics.md) (Active), [training-tuning.md](specs/training-tuning.md) (Draft)

## Completed

### Phase 1: Reward Breakdown (jax-training-metrics) — DONE

- [x] **1.1-1.9** `calculate_reward()` returns `(total, breakdown_dict)` with 15 keys, flowing through env.step (5-tuple), env_wrapper (merged into info dict), training_loop (scan collection, `reward/` prefix aggregation, action stats), and logging (pre-prefixed key passthrough).

### Phase 2: Test Updates (jax-training-metrics) — DONE

- [x] **2.1-2.7** All existing tests updated for 2-tuple/5-tuple returns. Added: `TestRewardBreakdownKeys`, `TestRewardBreakdownSum`, `TestBreakdownThroughEnvStep`, `TestBreakdownThroughWrapper`, `TestTrainingMetrics` integration test.

### Phase 3: Hyperparameter Tuning (training-tuning) — DONE

- [x] **3.1** Epoch floor raised from 1 to 2 in `auto_scale_for_batch_size()` — verified: 2048 envs → `update_epochs=2`
- [x] **3.2** `ent_coef` default updated from `0.1` to `0.15` in `config.py`
- [x] **3.3** `--ent-coef` default updated from `0.1` to `0.15` in `train_purejaxrl.py`
- [x] **3.4** `training-speedups.md` updated: epoch floor 2, LR guidance `6.25e-5`, recommended command with `--ent-coef 0.15`

### Phase 4: Validation — DONE (automated)

- [x] **4.1** All 99 JAX tests pass (76 reward parity + 24 purejaxrl including integration test)
- [x] **4.2** Training compiles and runs with breakdown (verified via `TestTrainingMetrics`)
- [x] Auto-scale at 2048 envs produces `update_epochs=2`, `ent_coef=0.15`

## Remaining Work

### Manual Validation Only

- [ ] **4.3** (Manual) Run training with wandb, verify 15 reward charts + action fraction charts appear
- [ ] (Manual) Monitor `approx_kl < 0.06`, `clip_frac < 0.30`, `entropy > 1.0` through 500+ updates

## Key Design Decisions

1. **Breakdown dict in JAX scan**: Collected alongside transitions. Shape `(num_steps, num_envs)` per key — lightweight.
2. **HP split is cosmetic**: `damage_penalty = min(hp_delta, 0)`, `hp_recovery = max(hp_delta, 0)`. Sum unchanged.
3. **Pre-prefixed keys**: Logger detects `/` in key names and passes through without adding `train/`.
4. **Epoch floor 2**: Rare events get 2 gradient passes. Compensated by halved LR (`6.25e-5`).
