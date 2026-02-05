# Reward Parity Implementation Plan

**Spec:** [specs/reward-parity.md](specs/reward-parity.md)
**Goal:** Bring JAX `rewards.py` to full parity with Swift `RewardCalculator.swift`

## Current State Assessment

### What exists (12/13 reward components at parity)

| Component | JAX File | Status |
|-----------|----------|--------|
| Step penalty (-0.01) | `rewards.py:69` | At parity |
| Stage completion ([1..100]) | `rewards.py:75-80` | At parity |
| Score gain (delta * 0.5) | `rewards.py:83` | At parity |
| Kill reward (0.3/kill) | `rewards.py:86-90` | At parity |
| Data siphon collection (1.0) | `rewards.py:93-96` | At parity |
| Distance shaping (0.05/cell) | `rewards.py:98-109` | **BUG**: Manhattan, should be BFS |
| HP change (±1.0/HP) | `rewards.py:112-113` | At parity |
| Victory bonus (500 + score*100) | `rewards.py:116-121` | At parity |
| Death penalty (stage-only) | `rewards.py:127-132` | At parity (Phase 1) |
| Siphon-caused death (-10.0) | `rewards.py:134-142` | **NEW** in Phase 2 |
| Resource gain (0.05/unit) | `rewards.py:144-151` | At parity (Phase 1) |
| Resource holding (0.01/unit on stage complete) | `rewards.py:153-162` | At parity (Phase 1) |
| Program waste (-0.3 RESET@2HP) | `rewards.py:164-172` | At parity (Phase 1) |

### What's still missing (1 component)

| Component | Swift Reference | Difficulty |
|-----------|----------------|------------|
| Siphon quality (penalty for suboptimal position) | `RewardCalculator.swift:211-219` | High |

### What doesn't exist yet

| File | Purpose | Required Phase |
|------|---------|----------------|
| `python/hackmatrix/jax_env/pathfinding.py` | BFS distance on 6x6 grid | Phase 3 |
| `python/hackmatrix/jax_env/siphon_quality.py` | Siphon optimality checker | Phase 4 |

### Tests

- `python/tests/test_reward_parity.py` — 42 JAX-only reward unit tests (Phase 1+2 complete)
- `python/tests/parity/test_rewards.py` — 14 parity tests via env interface (Swift+JAX)
- `python/tests/test_purejaxrl.py` — 23 PureJaxRL integration tests

## Spec Discrepancies

### 1. BFS obstacle definition
**Spec says:** "Obstacles: unsiphoned blocks, possibly enemies on blocks"
**Swift says:** `Pathfinding.findDistance` skips cells where `cell.hasBlock` is true — this includes **all blocks** (siphoned AND unsiphoned). Enemies are NOT obstacles in `findDistance`.
**Resolution:** Use all blocks (siphoned + unsiphoned) as obstacles in JAX BFS. Match Swift exactly.

### 2. Siphon quality — yield includes resources from non-block cells
**Spec says:** "compute (credits, energy, block_values, programs) for a given position"
**Swift says:** `calculateSiphonYieldAt` counts resources (credits/energy) from **non-block cells** in the siphon cross, plus data block values and programs from **unsiphoned blocks**.
**Resolution:** JAX implementation must include grid resources (not just block resources) in yield calculation.

### 3. `isAdjacentToPlayer` for siphon death — cardinal only
**Swift:** `abs(rowDiff) == 1 && colDiff == 0 || rowDiff == 0 && abs(colDiff) == 1` — **cardinal adjacency only**.
**Resolution:** Match Swift: Manhattan distance == 1 with one axis zero.

## Implementation Tasks

### Phase 1: Simple Reward Additions + Death Penalty Fix — COMPLETE

- [x] **1.1** Extend `calculate_reward` signature to accept `action: jnp.int32`
- [x] **1.2** Add resource gain reward
- [x] **1.3** Add resource holding bonus (stage completion only)
- [x] **1.4** Add program waste penalty (RESET at 2 HP)
- [x] **1.5** Fix death penalty to use stage-only calculation
- [x] **1.6** Update `env.py` to pass `action` to `calculate_reward`
- [x] **1.7** Write Phase 1 parity tests (36 tests, all passing)

**Learnings from Phase 1:**
- Stage 8 completion sets `state.stage = 9` after `advance_stage`. The stage reward guard needed `state.stage <= 9` (not `<= 8`) to include the final stage reward alongside the victory bonus.
- `JAX_PLATFORMS=cpu` is required for running tests when TPU is occupied by a training process.
- Death penalty implementation uses `jnp.arange(8)` mask approach to compute stage-only cumulative without loops — clean JIT-compatible pattern.

### Phase 2: Siphon-Caused Death Penalty — COMPLETE

- [x] **2.1** Implement siphon death detection in `calculate_reward`
  - File: `python/hackmatrix/jax_env/rewards.py`
  - `_any_adjacent_siphon_enemy(state)` scans enemy array for active enemies with `spawned_from_siphon == 1` AND cardinal adjacency
  - Applies `-10.0` extra penalty on top of regular stage death penalty
  - Uses vectorized `jnp.any()` over masked enemy array — JIT-compatible

- [x] **2.2** Write Phase 2 parity tests (6 tests, all passing)
  - File: `python/tests/test_reward_parity.py`
  - Tests: siphon death adjacent, non-siphon death, non-adjacent siphon enemy, diagonal (not cardinal), compound with stage penalty, alive with adjacent siphon enemy

**Learnings from Phase 2:**
- Enemy array column layout: `[type, row, col, hp, disabled_turns, is_stunned, spawned_from_siphon, is_from_scheduled_task]` — column 6 is the siphon flag.
- Cardinal adjacency (matching Swift `isAdjacentToPlayer`): `(|row_diff| == 1 & col_diff == 0) | (row_diff == 0 & |col_diff| == 1)`. Diagonal does NOT count.

### Phase 3: Distance Shaping Fix (BFS)

- [ ] **3.1** Implement JIT-compatible BFS in new module
  - File: `python/hackmatrix/jax_env/pathfinding.py` (new)
  - BFS on 6x6 grid with fixed iteration count (max 36 iterations)
  - Obstacles: ALL blocks (siphoned + unsiphoned) — `grid_block_type != 0`
  - Returns distance (int32), or -1 if no path
  - Must use `jax.lax.while_loop` or `jax.lax.fori_loop` for JIT compatibility

- [ ] **3.2** Replace Manhattan distance in `calculate_reward`
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Import BFS function from `pathfinding.py`
  - Compute `prev_distance` and `curr_distance` using BFS instead of Manhattan
  - Fallback: if BFS returns -1 (no path), use Manhattan as fallback

- [ ] **3.3** Write Phase 3 parity tests
  - File: `python/tests/test_reward_parity.py`
  - 5 test cases: no obstacles, block wall, no path, death, stage complete

### Phase 4: Siphon Quality Penalty

- [ ] **4.1** Implement siphon yield computation
  - File: `python/hackmatrix/jax_env/siphon_quality.py` (new)
  - `compute_siphon_yield(state, row, col)` → (credits, energy, block_values_sorted, programs)
  - Cross pattern: center + 4 cardinal = 5 cells
  - Credits/energy from non-block cells
  - Block values from unsiphoned data blocks
  - Programs from unsiphoned program blocks

- [ ] **4.2** Implement grid-wide optimality check
  - File: `python/hackmatrix/jax_env/siphon_quality.py`
  - `check_siphon_optimality(state)` → (is_optimal, missed_credits, missed_energy)
  - For all 36 grid positions, compute yield
  - Filter: position must not be on a block (any block, siphoned or not)
  - Set equality: sort block values arrays and compare element-wise
  - Strict dominance: `>= credits AND >= energy AND (> credits OR > energy)`

- [ ] **4.3** Integrate into `calculate_reward`
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Only evaluate when `action == ACTION_SIPHON` (action 4)
  - Penalty: `siphonSuboptimalPenalty * missedValue` where `missedValue = missedCredits * 0.05 + missedEnergy * 0.05`

- [ ] **4.4** Write Phase 4 parity tests
  - 4 test cases: optimal, suboptimal, different blocks, non-siphon action

## Success Criteria

- [x] Resource gain, resource holding, program waste implemented
- [x] Death penalty uses stage-only calculation (matches Swift)
- [x] Siphon-caused death penalty implemented
- [ ] Distance shaping uses BFS pathfinding (matches Swift)
- [ ] Siphon quality penalty implemented
- [x] Phase 1+2 parity tests pass (42/42)
- [x] Existing `test_purejaxrl.py` tests still pass (23/23)
- [x] All JAX parity tests still pass (154/154)
- [ ] Training runs without regression (healthy PPO metrics)

## Open Questions

1. ~~**Siphon quality sign bug**~~ **FIXED** — Removed double-negation in `RewardCalculator.swift:217`. Now `siphonSuboptimalPenalty * missedValue` = `-0.5 * missedValue` (correct negative penalty). Test added in `Tests/HackMatrixTests/RewardCalculatorTests.swift`.

2. ~~**Death penalty at stage 1**~~ **RESOLVED** — 0 is correct. No stages completed = no penalty. Spec and comments updated to match.
