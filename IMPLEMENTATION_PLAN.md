# Reward Parity Implementation Plan

**Spec:** [specs/reward-parity.md](specs/reward-parity.md)
**Goal:** Bring JAX `rewards.py` to full parity with Swift `RewardCalculator.swift`

## Current State Assessment

### What exists (8/13 reward components at parity)

| Component | JAX File | Status |
|-----------|----------|--------|
| Step penalty (-0.01) | `rewards.py:38` | At parity |
| Stage completion ([1..100]) | `rewards.py:41-46` | At parity |
| Score gain (delta * 0.5) | `rewards.py:48-50` | At parity |
| Kill reward (0.3/kill) | `rewards.py:52-58` | At parity |
| Data siphon collection (1.0) | `rewards.py:60-67` | At parity |
| Distance shaping (0.05/cell) | `rewards.py:69-82` | **BUG**: Manhattan, should be BFS |
| HP change (±1.0/HP) | `rewards.py:84-86` | At parity (combined, Swift separates for display only) |
| Victory bonus (500 + score*100) | `rewards.py:88-94` | At parity |
| Death penalty (-cumulative*0.5) | `rewards.py:96-102` | **BUG**: uses all cumulative, should use stage-only |

### What's missing (5 components)

| Component | Swift Reference | Difficulty |
|-----------|----------------|------------|
| Resource gain (credits_delta * 0.05 + energy_delta * 0.05) | `RewardCalculator.swift:191-194` | Low |
| Resource holding (credits * 0.01 + energy * 0.01, stage-only) | `RewardCalculator.swift:198-201` | Low |
| Program waste (-0.3 for RESET at 2 HP) | `RewardCalculator.swift:222` | Low |
| Siphon-caused death (-10.0 extra) | `RewardCalculator.swift:225` | Medium |
| Siphon quality (penalty for suboptimal position) | `RewardCalculator.swift:211-219` | High |

### What doesn't exist yet

| File | Purpose | Required Phase |
|------|---------|----------------|
| `python/hackmatrix/jax_env/pathfinding.py` | BFS distance on 6x6 grid | Phase 3 |
| `python/hackmatrix/jax_env/siphon_quality.py` | Siphon optimality checker | Phase 4 |
| `python/tests/test_reward_parity.py` | JAX-only reward parity tests | All phases |

### Existing tests

- `python/tests/parity/test_rewards.py` — Tests Swift env rewards (14 tests, `@pytest.mark.requires_set_state`). These test the Swift binary via JSON protocol, NOT the JAX implementation directly.
- `python/tests/test_purejaxrl.py` — Tests JAX env step/reset/GAE, but doesn't test individual reward components.
- **No JAX-specific reward unit tests exist.** The new `test_reward_parity.py` will test `calculate_reward` directly by constructing `EnvState` objects.

## Spec Discrepancies

### 1. BFS obstacle definition
**Spec says:** "Obstacles: unsiphoned blocks, possibly enemies on blocks"
**Swift says:** `Pathfinding.findDistance` skips cells where `cell.hasBlock` is true — this includes **all blocks** (siphoned AND unsiphoned). Enemies are NOT obstacles in `findDistance` (they are only in `findPath` for enemy movement).
**Resolution:** Use all blocks (siphoned + unsiphoned) as obstacles in JAX BFS. Match Swift exactly.

### 2. Siphon quality — yield includes resources from non-block cells
**Spec says:** "compute (credits, energy, block_values, programs) for a given position"
**Swift says:** `calculateSiphonYieldAt` counts resources (credits/energy) from **non-block cells** in the siphon cross, plus data block values and programs from **unsiphoned blocks**.
**Resolution:** JAX implementation must include grid resources (not just block resources) in yield calculation.

### 3. `isAdjacentToPlayer` for siphon death — cardinal only
**Swift:** `abs(rowDiff) == 1 && colDiff == 0 || rowDiff == 0 && abs(colDiff) == 1` — **cardinal adjacency only** (not diagonal).
**Resolution:** Match Swift: Manhattan distance == 1 with one axis zero.

## Implementation Tasks

### Phase 1: Simple Reward Additions + Death Penalty Fix

- [ ] **1.1** Extend `calculate_reward` signature to accept `action: jnp.int32`
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Add `action` parameter after `player_died`

- [ ] **1.2** Add resource gain reward
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Formula: `(player.credits - prev_credits) * 0.05 + (player.energy - prev_energy) * 0.05`
  - State fields `prev_credits` and `prev_energy` already tracked

- [ ] **1.3** Add resource holding bonus (stage completion only)
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Formula: `(player.credits * 0.01 + player.energy * 0.01) * stage_completed`
  - Gate on `stage_completed` flag

- [ ] **1.4** Add program waste penalty (RESET at 2 HP)
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Condition: `action == 19 && prev_hp == 2` → penalty `-0.3`
  - Action 19 = `ACTION_PROGRAM_START + PROGRAM_RESET` = `5 + 14 = 19`

- [ ] **1.5** Fix death penalty to use stage-only calculation
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Replace `-state.cumulative_reward * 0.5` with:
    ```
    stage_cumulative = sum(STAGE_COMPLETION_REWARDS[0 : stage-1])
    penalty = -stage_cumulative * 0.5
    ```
  - Note: `state.stage` is the *next* stage (already incremented if stage completed this step).
    When player dies, stage has NOT been incremented. So `state.stage - 1` gives completed stages.
    - Stage 1 death: `stage=1`, completed stages = 0, penalty = 0 → but Swift gives -0.5
    - Wait — Swift: `currentStage` is the stage you're ON (1-8). `0..<(currentStage - 1)` means stages completed before this one.
    - Stage 1 death in Swift: `0..<0` = empty, `cumulativeReward = 0`, penalty = `0`.
    - **But spec says** stage 1 death = -0.5. Let me re-check...
    - Spec table says: "Stage 1 death: Swift penalty = -0.5"
    - Swift code: stageRewards = [1,2,4,8,16,32,64,100], `for i in 0..<(currentStage - 1)`. At stage 1: `0..<0` = no iterations = 0. Penalty = 0.
    - **The spec table appears wrong for stage 1.** Swift code gives 0 penalty at stage 1. Trust the code.
    - At stage 3: `0..<2` → stageRewards[0] + stageRewards[1] = 1 + 2 = 3 → penalty = -1.5. Spec says -1.5 too ✓.

- [ ] **1.6** Update `env.py` to pass `action` to `calculate_reward`
  - File: `python/hackmatrix/jax_env/env.py`
  - Change line 133: `calculate_reward(state, stage_complete, game_won, player_died)` → add `action`

- [ ] **1.7** Write Phase 1 parity tests
  - File: `python/tests/test_reward_parity.py` (new)
  - Tests construct `EnvState` directly, call `calculate_reward`, assert exact values
  - ~15 test cases from spec (step penalty, stage completion, score gain, kills, data siphon, HP change, victory, resource gain, resource holding, program waste, death penalty fix)

### Phase 2: Siphon-Caused Death Penalty

- [ ] **2.1** Implement siphon death detection in `calculate_reward`
  - File: `python/hackmatrix/jax_env/rewards.py`
  - When `player_died`: scan `state.enemies` for any active enemy with `spawned_from_siphon == 1` AND cardinally adjacent to player (Manhattan distance 1, not diagonal)
  - Apply `-10.0` penalty
  - Adjacent check: `(|enemy_row - player_row| == 1 AND enemy_col == player_col) OR (enemy_row == player_row AND |enemy_col - player_col| == 1)`
  - Use `jnp.any()` over masked enemy array

- [ ] **2.2** Write Phase 2 parity tests
  - File: `python/tests/test_reward_parity.py`
  - 3 test cases: siphon death, non-siphon death, non-adjacent siphon enemy

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
  - **Edge cases:**
    - On death: distance reward should be 0 (already handled by one-directional capping)
    - On stage complete: player is at exit, distance = 0, so old_dist * 0.05 is the reward

- [ ] **3.3** Write Phase 3 parity tests
  - File: `python/tests/test_reward_parity.py`
  - 5 test cases: no obstacles, block wall, no path, death, stage complete

### Phase 4: Siphon Quality Penalty

- [ ] **4.1** Implement siphon yield computation
  - File: `python/hackmatrix/jax_env/siphon_quality.py` (new)
  - `compute_siphon_yield(state, row, col)` → (credits, energy, block_values_sorted, programs)
  - Cross pattern: center + 4 cardinal = 5 cells
  - Credits/energy from non-block cells (grid_resources_credits, grid_resources_energy)
  - Block values from unsiphoned data blocks (grid_block_type == 1 AND NOT grid_block_siphoned)
  - Programs from unsiphoned program blocks (grid_block_type == 2 AND NOT grid_block_siphoned)
  - Skip already-siphoned blocks
  - Must handle cells outside grid bounds (clip or skip)

- [ ] **4.2** Implement grid-wide optimality check
  - File: `python/hackmatrix/jax_env/siphon_quality.py`
  - `check_siphon_optimality(state)` → (is_optimal, missed_credits, missed_energy)
  - For all 36 grid positions, compute yield
  - Filter: position must not be on a block (grid_block_type[r,c] == 0 or block is siphoned? — check Swift: can't siphon FROM block position)
  - Actually Swift checks: `if case .block = grid.cells[row][col].content` → returns empty yield. So any cell with a block (siphoned or not) is not a valid siphon position.
  - Set equality: sort block values arrays and compare element-wise (fixed-size padded arrays)
  - Strict dominance: `>= credits AND >= energy AND (> credits OR > energy)`
  - Return max missed values across all strictly-better positions

- [ ] **4.3** Integrate into `calculate_reward`
  - File: `python/hackmatrix/jax_env/rewards.py`
  - Only evaluate when `action == ACTION_SIPHON` (action 4)
  - Penalty: `siphonSuboptimalPenalty * missedValue` where `missedValue = missedCredits * 0.05 + missedEnergy * 0.05`
  - Note Swift formula: `breakdown.siphonQuality = -siphonSuboptimalPenalty * missedValue`
    where `siphonSuboptimalPenalty = -0.5`, so result = `0.5 * missedValue` (positive, then negated as quality penalty)
    Actually: `-(-0.5) * missedValue = 0.5 * missedValue`. But this is a **positive** value assigned to `siphonQuality`. Since `siphonQuality` is summed into `total`, this would REWARD suboptimal siphons. That seems wrong.
    Wait — looking again: `missedValue = missedCredits * 0.05 + missedEnergy * 0.05`. This is positive.
    `siphonSuboptimalPenalty = -0.5` (negative constant).
    `breakdown.siphonQuality = -siphonSuboptimalPenalty * missedValue = -(-0.5) * positive = +0.5 * positive`.
    This gives a **positive** siphon quality reward for suboptimal siphons? That's a bug in Swift.
    Actually reading more carefully: the field is named `siphonQuality` and is meant to be a penalty. The double negation likely means the formula should be:
    `siphonQuality = siphonSuboptimalPenalty + 0.5 * missedValue` or similar.
    **This needs clarification from the spec/user.** For now, follow Swift code exactly: `siphonQuality = -siphonSuboptimalPenalty * missedValue = 0.5 * missedValue`.
    Actually wait — re-reading the spec: "Penalty = 0.5 * missedValue (note: double-negation in Swift code)". So the intended penalty is `-(0.5 * missedValue)` = a negative number. The Swift code produces `+0.5 * missedValue` which is positive — this IS a bug in Swift where the signs cancel out. The net effect in the total reward is `+0.5 * missedValue` which rewards suboptimal siphons.
    **Decision:** Flag this to the user. The spec says "penalty" but Swift code gives a positive reward. Match Swift exactly for now (parity goal), then fix both together.

- [ ] **4.4** Write Phase 4 parity tests
  - File: `python/tests/test_reward_parity.py`
  - 4 test cases: optimal, suboptimal, different blocks, non-siphon action

## File Change Summary

| File | Change Type | Phase |
|------|-------------|-------|
| `python/hackmatrix/jax_env/rewards.py` | Modify: add 5 components, fix death penalty, add action param | 1-4 |
| `python/hackmatrix/jax_env/env.py` | Modify: pass `action` to `calculate_reward` | 1 |
| `python/hackmatrix/jax_env/pathfinding.py` | **New**: BFS distance (JIT-compatible) | 3 |
| `python/hackmatrix/jax_env/siphon_quality.py` | **New**: siphon optimality checker | 4 |
| `python/tests/test_reward_parity.py` | **New**: ~27 JAX reward parity tests | 1-4 |

## Success Criteria

- [ ] All 5 missing reward components implemented in JAX
- [ ] Death penalty uses stage-only calculation (matches Swift)
- [ ] Distance shaping uses BFS pathfinding (matches Swift)
- [ ] All parity tests pass
- [ ] Existing `test_purejaxrl.py` tests still pass
- [ ] Training runs without regression (healthy PPO metrics)

## Open Questions

1. ~~**Siphon quality sign bug**~~ **FIXED** — Removed double-negation in `RewardCalculator.swift:217`. Now `siphonSuboptimalPenalty * missedValue` = `-0.5 * missedValue` (correct negative penalty). Test added in `Tests/HackMatrixTests/RewardCalculatorTests.swift`.

2. ~~**Death penalty at stage 1**~~ **RESOLVED** — 0 is correct. No stages completed = no penalty. Spec and comments updated to match.
