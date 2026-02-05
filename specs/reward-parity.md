# Reward Parity Spec

**Status:** Draft
**Depends on:** [game-mechanics.md](./game-mechanics.md), [jax-implementation.md](./jax-implementation.md)

## Goal

Bring JAX `rewards.py` to full parity with Swift `RewardCalculator.swift`. The Swift implementation is the source of truth. Add parity tests that verify each reward component independently.

## Current State

### Rewards at parity (8 components)

| Component | JAX | Swift | Notes |
|-----------|-----|-------|-------|
| Step penalty | -0.01 | -0.01 | Identical |
| Stage completion | [1,2,4,8,16,32,64,100] | [1,2,4,8,16,32,64,100] | Identical |
| Score gain | delta * 0.5 | delta * 0.5 | Identical |
| Kill reward | 0.3 per kill | 0.3 per kill | Identical |
| Data siphon collection | 1.0 | 1.0 | Identical |
| Distance shaping coef | 0.05 one-directional | 0.05 one-directional | Coefficient matches, but distance calculation differs (see below) |
| Victory bonus | 500 + score * 100 | 500 + score * 100 | Identical |
| HP change | hp_delta * 1.0 | damage: -1.0/HP, recovery: +1.0/HP | Net effect identical (Swift separates for breakdown display) |

### Missing from JAX (5 components)

#### 1. Resource Gain Reward

**Swift:** `credits_delta * 0.05 + energy_delta * 0.05`

Rewards collecting credits and energy (primarily via siphon). Small but provides dense feedback for resource acquisition.

**JAX state available:** `prev_credits`, `prev_energy`, `player.credits`, `player.energy` — all already tracked.

**Difficulty:** Low — straightforward delta calculation.

#### 2. Resource Holding Bonus (stage completion only)

**Swift:** `credits * 0.01 + energy * 0.01` — only granted when `stageAdvanced == true`

Rewards carrying resources into the next stage. Gated on stage completion to prevent farming.

**JAX state available:** `player.credits`, `player.energy`, `stage_completed` flag already passed to `calculate_reward`.

**Difficulty:** Low — conditional addition.

#### 3. Siphon Quality Penalty

**Swift:** Checks if a strictly better siphon position exists. "Strictly better" means: same blocks siphoned, but >= credits AND >= energy with at least one strictly greater.

**Constants:**
- `siphonSuboptimalPenalty = -0.5`
- `missedValue = missedCredits * 0.05 + missedEnergy * 0.05`
- Penalty = `0.5 * missedValue` (note: double-negation in Swift code)

**Algorithm (`checkForBetterSiphonPosition`):**
1. Compute siphon yield at current position (cross pattern: center + 4 cardinal)
2. For every cell on the 6x6 grid, compute siphon yield
3. Filter candidates that siphon the *exact same blocks* (matching data values and programs)
4. Among those, check if any has strictly more resources (>= both, > at least one)
5. If found: `wasOptimal = false`, record the resource difference

**JAX implementation notes:**
- Siphon range is a cross pattern (center + 4 cardinal = 5 cells max)
- Grid is 6x6 = 36 positions to check
- Yield comparison requires sorting block values for set equality
- Must only run when action is siphon (action == 4)
- Need to pass `action` to `calculate_reward` or compute in env.py and pass as flag

**Difficulty:** High — most complex reward to port. Requires grid-wide search with set comparison.

#### 4. Program Waste Penalty (RESET at 2 HP)

**Swift:** `-0.3` when RESET program is used at 2 HP (wastes 1 HP of healing since max is 3).

**Detection:** `action == PROGRAM_RESET (action index 19) && prev_hp == 2`

**JAX state available:** `prev_hp` already tracked. Need to know which action was taken.

**Difficulty:** Low — need to pass `action` to `calculate_reward`.

#### 5. Siphon-Caused Death Penalty

**Swift:** `-10.0` extra penalty when player dies to an enemy with `spawnedFromSiphon == true` that is adjacent to the player.

**Detection after enemy turn:** If `player_died`, scan enemies adjacent to player. If any has `spawned_from_siphon` flag (enemy array column 6), apply penalty.

**JAX state available:** Enemy array column 6 tracks `spawned_from_siphon`. Player position known. Need adjacency check.

**Difficulty:** Medium — requires scanning enemy array for adjacency + flag check.

### Discrepancies to fix (2 components)

#### 6. Death Penalty Calculation

**JAX (current):** `-cumulative_reward * 0.5` — uses ALL accumulated reward (shaping, kills, etc.)

**Swift (canonical):** `-stage_cumulative * 0.5` — uses only stage completion rewards:
```
cumulative = sum(stageRewards[0..<(currentStage - 1)])
penalty = -cumulative * 0.5
```

| Stage at death | Swift penalty | JAX penalty (varies) |
|---------------|---------------|---------------------|
| 1 | -0.5 | depends on accumulated shaping |
| 3 | -3.5 | could be much larger |
| 8 | -113.5 | could be much larger |

**Decision:** Use Swift (stage-only) as canonical. More predictable signal — doesn't punish exploration/kills.

**Difficulty:** Low — replace `cumulative_reward` with stage-based calculation.

#### 7. Distance Calculation

**JAX (current):** Manhattan distance: `|row - exit_row| + |col - exit_col|`

**Swift (canonical):** A* pathfinding via `Pathfinding.findDistance()` — accounts for obstacles (blocks, enemies on blocks).

On a 6x6 grid, these can disagree when blocks create walls. Manhattan gives credit for moving "closer" even if the path goes through a wall.

**Decision:** Port A* pathfinding to JAX.

**JAX implementation notes:**
- Grid is small (6x6 = 36 cells) so bounded-iteration BFS is tractable
- Obstacles: unsiphoned blocks, possibly enemies on blocks
- Must be JIT-compatible (fixed iteration count with early-exit via masking)
- BFS with max 36 iterations is sufficient for 6x6 grid

**Difficulty:** Medium — BFS/A* in JAX with fixed iteration count.

## Implementation Plan

### Phase 1: Simple Reward Additions + Death Penalty Fix

Low-complexity changes that don't require new algorithms.

1. **Extend `calculate_reward` signature** to accept `action: jnp.int32`
2. **Resource gain reward** — delta calculation from existing state fields
3. **Resource holding bonus** — conditional on `stage_completed`
4. **Program waste penalty** — check `action == 19 && prev_hp == 2`
5. **Death penalty fix** — replace `cumulative_reward * 0.5` with stage-only calculation
6. **Update `env.py`** — pass `action` to `calculate_reward`
7. **Parity tests** for all Phase 1 components

### Phase 2: Siphon-Caused Death Penalty

Requires adjacency scanning of enemy array.

1. **Implement adjacency check** — scan enemies for `spawned_from_siphon` flag within Manhattan distance 1 of player
2. **Apply -10.0 penalty** when `player_died` and adjacent siphon enemy found
3. **Parity tests** for siphon death scenarios

### Phase 3: Distance Shaping (BFS)

Replace Manhattan distance with BFS pathfinding.

1. **Implement JIT-compatible BFS** on 6x6 grid with fixed iteration count
2. **Define obstacle mask** — unsiphoned blocks are obstacles
3. **Replace Manhattan distance** in reward calculation with BFS distance
4. **Handle edge case:** BFS returns -1 (no path) → fall back to Manhattan or large constant
5. **Parity tests** for distance with/without obstacles

### Phase 4: Siphon Quality Penalty

Most complex component — grid-wide search with yield comparison.

1. **Implement `compute_siphon_yield`** — compute (credits, energy, block_values, programs) for a given position
2. **Implement `check_siphon_optimality`** — compare yields across all 36 grid positions
3. **Set equality in JAX** — sort block values and compare (fixed-size arrays with padding)
4. **Apply penalty** when suboptimal siphon detected
5. **Only evaluate when action == siphon** — skip for all other actions
6. **Parity tests** for optimal vs suboptimal siphon scenarios

## Parity Tests

Tests go in `python/tests/test_reward_parity.py`. Each test constructs a specific `EnvState`, calls `calculate_reward` (or `step`), and asserts the exact reward value.

### Test Structure

```python
# Helper to create a minimal state with specific fields
def make_state(**overrides) -> EnvState:
    """Create a base EnvState, apply overrides for test scenario."""
    ...

# Each test is self-contained: set up state, call reward fn, assert value
def test_step_penalty_on_neutral_move():
    """Moving to empty cell with no other effects should give exactly -0.01."""
    ...
```

### Phase 1 Tests

#### Step penalty
- `test_step_penalty_on_neutral_move` — move to empty cell, no enemies, no exit = exactly -0.01

#### Stage completion
- `test_stage_completion_reward_stage_1` — complete stage 1 = 1.0 (+ step penalty + distance)
- `test_stage_completion_reward_stage_8` — complete stage 8 = 100.0

#### Score gain
- `test_score_gain_from_siphon` — siphon data block worth 5 points = 2.5

#### Kill reward
- `test_kill_single_enemy` — kill 1 enemy = 0.3
- `test_kill_multiple_enemies` — kill 3 enemies = 0.9

#### Data siphon collection
- `test_collect_data_siphon` — walk onto data siphon cell = 1.0

#### HP change
- `test_damage_penalty` — lose 1 HP from enemy attack = -1.0
- `test_hp_recovery_from_reset` — RESET from 1 HP to 3 HP = +2.0
- `test_hp_recovery_from_stage_completion` — stage complete heals 1 HP = +1.0

#### Victory bonus
- `test_victory_bonus` — win with score 10 = 500 + 1000 = 1500

#### Resource gain reward (NEW)
- `test_resource_gain_credits` — gain 10 credits via siphon = 0.5
- `test_resource_gain_energy` — gain 5 energy via siphon = 0.25
- `test_resource_gain_both` — gain 10 credits + 5 energy = 0.75
- `test_resource_loss_from_program` — spend 3 credits on CRASH = -0.15

#### Resource holding bonus (NEW)
- `test_resource_holding_on_stage_complete` — complete stage with 20 credits, 10 energy = 0.3
- `test_resource_holding_not_on_normal_step` — normal step with resources = 0.0 holding bonus

#### Program waste penalty (NEW)
- `test_reset_at_2hp_penalty` — RESET at 2 HP = -0.3
- `test_reset_at_1hp_no_penalty` — RESET at 1 HP = no waste penalty

#### Death penalty fix (FIX)
- `test_death_penalty_stage_1` — die at stage 1 = -0.5 (not cumulative-based)
- `test_death_penalty_stage_3` — die at stage 3 = -(1+2)*0.5 = -1.5
- `test_death_penalty_with_high_cumulative` — die at stage 1 with high cumulative reward still = -0.5

### Phase 2 Tests

#### Siphon-caused death penalty (NEW)
- `test_siphon_death_penalty` — die adjacent to siphon-spawned enemy = -10.0 extra
- `test_death_not_from_siphon_enemy` — die adjacent to non-siphon enemy = no extra penalty
- `test_death_from_siphon_enemy_not_adjacent` — siphon enemy exists but not adjacent = no extra penalty

### Phase 3 Tests

#### Distance shaping with BFS (FIX)
- `test_distance_closer_no_obstacles` — move toward exit on clear grid = +0.05
- `test_distance_with_block_wall` — block wall between player and exit, move around it: BFS says closer, Manhattan might not
- `test_distance_no_path` — completely blocked path = no distance reward (fallback)
- `test_distance_on_death` — death = 0 distance reward
- `test_distance_on_stage_complete` — reach exit = full old distance * 0.05

### Phase 4 Tests

#### Siphon quality (NEW)
- `test_optimal_siphon_no_penalty` — siphon at best position = 0 penalty
- `test_suboptimal_siphon_missed_credits` — better position has 10 more credits = penalty
- `test_suboptimal_siphon_different_blocks` — position with more resources but different blocks = no penalty (not comparable)
- `test_siphon_quality_only_on_siphon_action` — non-siphon action = no quality check

## Files

| File | Change |
|------|--------|
| `python/hackmatrix/jax_env/rewards.py` | Add 5 reward components, fix death penalty, accept action param |
| `python/hackmatrix/jax_env/env.py` | Pass `action` to `calculate_reward` |
| `python/hackmatrix/jax_env/pathfinding.py` | New: BFS distance on 6x6 grid (JIT-compatible) |
| `python/hackmatrix/jax_env/siphon_quality.py` | New: siphon optimality check |
| `python/tests/test_reward_parity.py` | New: all parity tests |

## Success Criteria

- [ ] All 5 missing reward components implemented in JAX
- [ ] Death penalty uses stage-only calculation (matches Swift)
- [ ] Distance shaping uses BFS pathfinding (matches Swift)
- [ ] All parity tests pass
- [ ] Existing `test_purejaxrl.py` tests still pass
- [ ] Training runs without regression (healthy PPO metrics)

## Appendix: Swift RewardCalculator Full Interface

For reference, the complete Swift `RewardCalculator.calculate()` parameters:

```swift
static func calculate(
    oldScore: Int, currentScore: Int, currentStage: Int,
    oldHP: Int, currentHP: Int,
    oldCredits: Int, currentCredits: Int,
    oldEnergy: Int, currentEnergy: Int,
    playerDied: Bool, gameWon: Bool, stageAdvanced: Bool,
    blocksSiphoned: Int, programsAcquired: Int,
    creditsGained: Int, energyGained: Int,
    totalKills: Int, dataSiphonCollected: Bool,
    distanceToExitDelta: Int,
    siphonWasUsed: Bool, siphonWasOptimal: Bool,
    siphonMissedCredits: Int, siphonMissedEnergy: Int,
    resetWasWasteful: Bool,
    diedToSiphonEnemy: Bool
) -> RewardBreakdown
```
