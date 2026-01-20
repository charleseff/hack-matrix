# Environment Parity Test Suite Implementation Plan

Based on analysis of `specs/env-parity-tests.md`, `everything_wrong_with_impl_plan.txt`, and codebase exploration.

## Current State Assessment

### What Exists (Verified)

| Component | Status | Location |
|-----------|--------|----------|
| Swift JSON protocol | **Complete** | `HackMatrix/GameCommandProtocol.swift` |
| `reset`, `step`, `getValidActions`, `getActionSpace`, `getObservationSpace` commands | **Complete** | GameCommandProtocol.swift lines 71-105 |
| `HackEnv` Gymnasium wrapper | **Complete** | `python/hackmatrix/gym_env.py` |
| JAX dummy environment | **Complete** | `python/hackmatrix/jax_env.py` |
| Basic parity tests (shapes/dtypes) | **Complete** | `python/scripts/test_env_parity.py` |
| `SwiftEnvAdapter` class | **Complete** | `python/scripts/test_env_parity.py` lines 44-64 |
| `JaxEnvAdapter` class | **Complete** | `python/scripts/test_env_parity.py` lines 66-106 |
| `EnvAdapter` Protocol (basic) | **Complete** | `python/scripts/test_env_parity.py` lines 27-36 |
| Observation utilities | **Complete** | `python/hackmatrix/observation_utils.py` |
| Reward system | **Complete** | `HackMatrix/RewardCalculator.swift` |
| Program definitions (23 programs) | **Complete** | `HackMatrix/Program.swift` |
| `isProgramApplicable` function | **Complete** | `HackMatrix/GameState.swift` lines 903-1003 |

### What's Missing (Verified)

| Component | Priority | Notes |
|-----------|----------|-------|
| `set_state`/`setState` JSON command | **P0** | Core blocker - not in Swift protocol (verified via search) |
| pytest dependency | **P0** | Not in `python/requirements.txt` |
| `python/tests/` directory | **P0** | Does not exist (verified via glob) |
| `EnvInterface` Protocol (full spec) | **P1** | Current `EnvAdapter` lacks `set_state()` method |
| Comprehensive test cases | **P2** | Movement, siphon, programs, enemies, turns, stages, rewards |
| Action mask verification tests | **P2** | Per spec requirements |
| Reward verification tests | **P2** | Per spec requirements |
| `specs/game-mechanics.md` | **P1** | Authoritative mechanics reference |

### Spec Discrepancies

1. **Interface location**: Spec specifies `python/tests/env_interface.py`, current adapters in `python/scripts/test_env_parity.py`
2. **Wrapper naming**: Spec uses `SwiftEnvWrapper`/`JaxEnvWrapper`, current code uses `SwiftEnvAdapter`/`JaxEnvAdapter`
3. **`set_state` missing**: Critical feature not yet in Swift JSON protocol (confirmed via code search)
4. **pytest missing**: Not in requirements.txt, needs to be added

## Game Mechanics Reference

### Grid & Coordinates
- **Grid size**: 6x6 (Constants.gridSize = 6)
- **Valid positions**: row 0-5, col 0-5
- **Direction offsets** (row, col):
  - Up (action 0): (+1, 0) - row increases
  - Down (action 1): (-1, 0) - row decreases
  - Left (action 2): (0, -1) - col decreases
  - Right (action 3): (0, +1) - col increases

### Enemy Types

| Type | HP | Move Speed | Special |
|------|-----|-----------|---------|
| Virus | 2 | 2 cells/turn | Fastest |
| Daemon | 3 | 1 cell/turn | Most HP |
| Glitch | 2 | 1 cell/turn | Can move on blocks |
| Cryptog | 2 | 1 cell/turn | Invisible unless in same row/col |

### Turn Structure
- **Move/Attack** → Ends turn, enemy turn executes
- **Siphon** → Ends turn, enemy turn executes
- **Program (except wait)** → Does NOT end turn, can chain
- **Wait program** → Ends turn, enemy turn executes

### Player Stats
- **HP**: 3 (max), 0 = dead
- **Attack damage**: 1 (default), 2 (with ATK+)
- **Data siphons**: Required for siphon action

### Resource Collection
- **Credits/Energy**: Collected ONLY via siphoning blocks - NOT by walking onto cells
- **Data Siphons**: Collected by walking onto cells containing data siphons (unique collection method)

### Line-of-Sight Attacks
- When player takes directional action (0-3) toward an enemy that is:
  - More than 1 cell away, AND
  - In the player's line-of-sight (same row or column with no blocking obstacles)
- The result is an ATTACK, not a movement
- This applies even if the enemy is on a block (e.g., Glitch)

### Data Block Invariant
- Data blocks always have `score == spawnCount` (transmissions spawned equals points awarded)

### Program Costs (from Program.swift)

| Program | Index | Credits | Energy | Applicability Condition |
|---------|-------|---------|--------|-------------------------|
| PUSH | 5 | 0 | 2 | Enemies exist |
| PULL | 6 | 0 | 2 | Enemies exist |
| CRASH | 7 | 3 | 2 | Blocks/enemies/transmissions in 8 surrounding cells |
| WARP | 8 | 2 | 2 | Enemies OR transmissions exist |
| POLY | 9 | 1 | 1 | Enemies exist |
| WAIT | 10 | 0 | 1 | Always applicable |
| DEBUG | 11 | 3 | 0 | Enemies on blocks exist |
| ROW | 12 | 3 | 1 | Enemies in player's row |
| COL | 13 | 3 | 1 | Enemies in player's column |
| UNDO | 14 | 1 | 0 | Game history not empty |
| STEP | 15 | 0 | 3 | Always applicable |
| SIPH+ | 16 | 5 | 0 | Always applicable |
| EXCH | 17 | 4 | 0 | Player has >= 4 credits (after cost check) |
| SHOW | 18 | 2 | 0 | `showActivated` is false |
| RESET | 19 | 0 | 4 | Player HP < 3 |
| CALM | 20 | 2 | 4 | `scheduledTasksDisabled` is false |
| D_BOM | 21 | 3 | 0 | Daemon enemy exists |
| DELAY | 22 | 1 | 2 | Transmissions exist |
| ANTI-V | 23 | 3 | 0 | Virus enemy exists |
| SCORE | 24 | 0 | 5 | Not on last stage (stage < 8) |
| REDUC | 25 | 2 | 1 | Unsiphoned blocks with spawnCount > 0 exist |
| ATK+ | 26 | 4 | 4 | Not used this stage AND attackDamage < 2 |
| HACK | 27 | 2 | 2 | Siphoned cells exist |

### Reward Components (from RewardCalculator.swift)

| Component | Multiplier | Notes |
|-----------|------------|-------|
| Stage completion | [1, 2, 4, 8, 16, 32, 64, 100] | Exponential by stage |
| Score gain | 0.5× | Per point |
| Kills | 0.3× | Per enemy killed |
| Data siphon collected | 1.0 | Flat |
| Distance shaping | 0.05× | Per cell closer to exit |
| Victory bonus | 500 + score×100 | On winning |
| Death penalty | -0.5× cumulative stage rewards | Scales with progress |
| Resource gain | 0.05× | Per credit/energy |
| Resource holding | 0.01× | Per credit/energy on stage completion |
| Damage penalty | -1.0× | Per HP lost |
| HP recovery | 1.0× | Per HP gained |
| Siphon quality | -0.5× missed value | Suboptimal siphon penalty |
| Program waste | -0.3 | RESET at 2 HP |
| Siphon-caused death | -10.0 | Extra penalty |

## Implementation Tasks

### Phase 0: Corrections & Documentation

- [x] **0.1** Create `specs/game-mechanics.md` with authoritative mechanics reference (already exists)
- [x] **0.2** Update `IMPLEMENTATION_PLAN.md` with all corrections from `everything_wrong_with_impl_plan.txt`

### Phase 1: Interface & Infrastructure

- [x] **1.1** Add `pytest>=7.0.0` to `python/requirements.txt`
- [x] **1.2** Create `python/tests/` directory structure with `__init__.py`
- [x] **1.3** Create `python/tests/conftest.py` with pytest fixtures
- [x] **1.4** Create `python/tests/env_interface.py` with `EnvInterface` Protocol and dataclasses
- [x] **1.5** Add `setState` command to Swift JSON protocol (`GameCommandProtocol.swift`)
- [x] **1.6** Implement `executeSetState()` in `HeadlessGame.swift` and `HeadlessGameCLI.swift`
- [x] **1.7** Create `python/tests/swift_env_wrapper.py` implementing `EnvInterface`
- [x] **1.8** Create `python/tests/jax_env_wrapper.py` skeleton implementing `EnvInterface`
- [x] **1.9** Create `python/tests/test_interface_smoke.py`
- [x] **1.10** Smoke tests pass (7/7 tests passing)

### Phase 2: Comprehensive Test Cases

#### Movement Tests (`test_movement.py`)

- [ ] **2.1** Move to empty cell (all 4 directions)

**Test: `test_move_up_to_empty_cell`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[],
      transmissions=[],
      blocks=[],
      resources=[],
      owned_programs=[],
      stage=1,
      turn=0,
      showActivated=False,
      scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up)
- **Expected Observation Changes**:
  - `player.row`: 3 → 4 (row increases for up)
  - `player.col`: unchanged (3)
  - `turn`: 0 → 1 (enemy turn executed)
- **Expected Reward**: 0.0 (no score change, no kills, no resources)
- **Expected Valid Actions**: [0, 1, 2, 3] (all directions valid from center)
- **Variants**: `test_move_down_to_empty_cell`, `test_move_left_to_empty_cell`, `test_move_right_to_empty_cell` with corresponding direction changes

- [ ] **2.2** Move blocked by grid edge

**Test: `test_move_blocked_by_top_edge`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=5, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[], blocks=[], transmissions=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up)
- **Expected**: Action should be masked (not in valid_actions)
- **Pre-step Valid Actions**: Should NOT include 0 (up blocked by edge)
- **Variants**:
  - `test_move_blocked_by_bottom_edge` (row=0, action=1)
  - `test_move_blocked_by_left_edge` (col=0, action=2)
  - `test_move_blocked_by_right_edge` (col=5, action=3)

- [ ] **2.3** Move blocked by block

**Test: `test_move_blocked_by_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
      enemies=[], transmissions=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up toward block)
- **Expected**: Action should be masked (block at row=4, col=3)
- **Pre-step Valid Actions**: Should NOT include 0


- [ ] **2.4** Move onto cell with data siphon

**Test: `test_move_collects_data_siphon`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      resources=[Resource(row=4, col=3, dataSiphon=True, credits=0, energy=0)],
      enemies=[], transmissions=[], blocks=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up)
- **Expected Observation Changes**:
  - `player.row`: 3 → 4
  - `player.dataSiphons`: 0 → 1 (collected by walking)
  - Resource at (4,3) removed from observation
  - `turn`: 0 → 1
- **Expected Reward**: 1.0 (data siphon collected reward)
- **Note**: This is the ONLY resource type collected by walking - credits/energy require siphoning

- [ ] **2.5** Line-of-sight attack on distant enemy

**Test: `test_line_of_sight_attack_distant_enemy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=5, col=3, hp=1, stunned=False)],  # 5 cells away, same column
      transmissions=[], blocks=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up - toward enemy in line of sight)
- **Expected Observation Changes**:
  - `player.row`: unchanged (0) - attack, not move
  - Enemy killed (was 1 HP, took 1 damage)
  - `turn`: 0 → 1
- **Expected Reward**: 0.3 (kill reward)
- **Key**: Directional action toward enemy in line-of-sight triggers ATTACK, not movement

- [ ] **2.6** Line-of-sight attack on enemy on block

**Test: `test_line_of_sight_attack_enemy_on_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="glitch", row=4, col=3, hp=1, stunned=False)],  # On block, same column
      blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
      transmissions=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up - toward enemy on block)
- **Expected Observation Changes**:
  - `player.row`: unchanged (0) - attack, not move
  - Enemy is damaged (glitch on block)
  - Block still exists
  - `turn`: 0 → 1
- **Expected Reward**: 0.3 (kill reward)
- **Key**: Player can attack enemies on blocks via line-of-sight (even though movement is blocked by the block)

- [ ] **2.7** Move into adjacent enemy (attack and kill)

**Test: `test_attack_kills_enemy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=4, col=3, hp=1, stunned=False)],  # 1 HP enemy, adjacent
      transmissions=[], blocks=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up into enemy)
- **Expected Observation Changes**:
  - `player.row`: unchanged (3) - attack doesn't move player
  - Enemy removed from observation
  - `turn`: 0 → 1
- **Expected Reward**: 0.3 (kill reward)

- [ ] **2.8** Move into adjacent enemy (attack, enemy survives)

**Test: `test_attack_damages_enemy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],  # 2 HP enemy
      transmissions=[], blocks=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up into enemy)
- **Expected Observation Changes**:
  - `player.row`: unchanged (3)
  - Enemy HP: 2 → 1
  - Enemy still in observation
  - `turn`: 0 → 1
- **Expected Reward**: 0.0 (no kill, no score)

- [ ] **2.9** Move into transmission

**Test: `test_attack_destroys_transmission`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      transmissions=[Transmission(row=4, col=3, turnsRemaining=3, enemyType="virus")],
      enemies=[], blocks=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up into transmission)
- **Expected Observation Changes**:
  - `player.row`: unchanged (3) - attack doesn't move
  - Transmission removed from observation
  - `turn`: 0 → 1
- **Expected Reward**: 0.0 (transmissions don't give kill reward)

#### Siphon Tests (`test_siphon.py`)

- [ ] **2.10** Siphon adjacent block (cross pattern)

**Test: `test_siphon_data_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=1, attackDamage=1, score=0),
      blocks=[Block(row=4, col=3, type="data", points=9, spawnCount=9, siphoned=False)],  # INVARIANT: points == spawnCount
      transmissions=[], enemies=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 4 (siphon)
- **Expected Observation Changes**:
  - `player.dataSiphons`: 1 → 0 (consumed)
  - `player.score`: 0 → 10
  - Block at (4,3) marked as siphoned=True
  - 10 transmissions spawned (from spawnCount, which equals points)
  - `turn`: 0 → 1
- **Expected Reward**: 10 × 0.5 = 5.0 (score gain)
- **INVARIANT CHECK**: `block.points == block.spawnCount` (data block invariant)

- [ ] **2.11** Siphon always valid with data siphons

**Test: `test_siphon_always_valid_with_siphons`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=1, attackDamage=1, score=0),
      blocks=[],  # NO blocks anywhere
      transmissions=[], enemies=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Pre-step Valid Actions**: SHOULD include 4 (siphon is valid when player has siphons)
- **Note**: Siphon validity only depends on having data siphons, not on block adjacency

- [ ] **2.12** Siphon invalid without data siphons

**Test: `test_siphon_invalid_without_siphons`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
      transmissions=[], enemies=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Pre-step Valid Actions**: Should NOT include 4 (no siphons available)
- **Note**: Even with adjacent block, siphon requires data siphons

- [ ] **2.13** Siphon spawns transmissions

**Test: `test_siphon_spawns_transmissions`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=1, attackDamage=1, score=0),
      blocks=[Block(row=4, col=3, type="data", points=3, spawnCount=3, siphoned=False)],
      transmissions=[],
      enemies=[], resources=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 4 (siphon)
- **Expected Observation Changes**:
  - 3 new transmissions appear in observation
  - Transmissions spawned at valid empty cells

- [ ] **2.14** Siphon does NOT reveal resources under block

**Test: `test_siphon_does_not_reveal_resources`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=1, attackDamage=1, score=0),
      blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=False)],
      resources=[Resource(row=4, col=3, credits=5, energy=3)],  # Hidden under block
      transmissions=[], enemies=[],
      owned_programs=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 4 (siphon)
- **Expected Observation Changes**:
  - Block at (4,3) marked as siphoned=True
  - Resources at (4,3) still NOT visible/collectible (block still exists, just siphoned)
- **Key**: Siphoning marks block as siphoned but does NOT destroy it or reveal resources

- [ ] **2.15** Siphon program block

**Test: `test_siphon_program_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, dataSiphons=1, attackDamage=1, score=0),
      blocks=[Block(row=4, col=3, type="program", programType="push", programActionIndex=5, spawnCount=2, siphoned=False)],
      owned_programs=[],
      transmissions=[], enemies=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 4 (siphon)
- **Expected Observation Changes**:
  - `owned_programs`: [] → [5] (push program acquired)
  - `programs[0]`: 0 → 1 (push at index 0 in programs array)

#### Program Tests (`test_programs.py`)

**IMPORTANT**: All program tests must verify:
1. Correct resource deduction (credits/energy per Program.swift costs)
2. Program applicability conditions (per `isProgramApplicable`)
3. Primary effect
4. Secondary effects (stuns, block destruction, etc.)

- [ ] **2.16** PUSH (index 5)

**Test: `test_push_enemies_away`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=2, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],
      owned_programs=[5],  # Has push
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 5 (push)
- **Expected Observation Changes**:
  - `player.energy`: 2 → 0 (cost: 0C, 2E)
  - Enemy position: (4,3) → (5,3) (pushed away from player)
  - `turn`: unchanged (programs don't end turn except wait)
- **Expected Reward**: 0.0
- **Post-step Valid Actions**: Should still include movement options (turn not ended)
- **Applicability**: Requires enemies to exist

- [ ] **2.17** PULL (index 6)

**Test: `test_pull_enemies_toward`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=2, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=5, col=3, hp=2, stunned=False)],
      owned_programs=[6],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 6 (pull)
- **Expected Observation Changes**:
  - `player.energy`: 2 → 0 (cost: 0C, 2E)
  - Enemy at (5,3) → (4,3) (pulled toward player)
- **Applicability**: Requires enemies to exist

- [ ] **2.18** CRASH (index 7)

**Test: `test_crash_clears_surrounding`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=2, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="virus", row=4, col=3, hp=1, stunned=False),
          Enemy(type="virus", row=3, col=4, hp=1, stunned=False),
      ],
      blocks=[
          Block(row=4, col=4, type="data", points=5, spawnCount=5, siphoned=False),  # Unsiphoned
          Block(row=2, col=3, type="data", points=3, spawnCount=3, siphoned=True),   # Siphoned
      ],
      resources=[Resource(row=4, col=4, credits=5, energy=0)],  # Under unsiphoned block
      transmissions=[], owned_programs=[7],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 7 (crash)
- **Expected Observation Changes**:
  - `player.credits`: 3 → 0 (cost: 3C, 2E)
  - `player.energy`: 2 → 0
  - Both enemies killed (in 8 surrounding cells)
  - BOTH blocks destroyed (siphoned AND unsiphoned)
  - Resources at (4,4) now visible/accessible
  - Cell at (4,4) and (2,3) now traversable
- **Expected Reward**: 2 × 0.3 = 0.6 (2 kills)
- **Applicability**: Requires blocks/enemies/transmissions in 8 surrounding cells
- CRASH destroys ALL blocks (siphoned or not), exposing resources underneath

- [ ] **2.19** WARP (index 8)

**Test: `test_warp_to_random_enemy_kills_it`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=2, energy=2, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="virus", row=3, col=3, hp=2, stunned=False),
          Enemy(type="virus", row=5, col=5, hp=2, stunned=False),
      ],
      owned_programs=[8],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 8 (warp)
- **Expected Observation Changes**:
  - `player.credits`: 2 → 0 (cost: 2C, 2E)
  - `player.energy`: 2 → 0
  - Player position becomes one of: (3,3) OR (5,5) (random target)
  - The enemy at that position is KILLED
  - One enemy removed, one remains
- **Expected Reward**: 0.3 (1 kill)
- **Applicability**: Requires enemies OR transmissions to exist
- Warp teleports TO the enemy position and KILLS it (not adjacent to enemy)
- **Test strategy**: With multiple enemies, verify player ends up at one of the enemy positions and that enemy is dead

- [ ] **2.20** POLY (index 9)

**Test: `test_poly_randomizes_enemy_types`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=1, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False)],  # Full HP (0 damage dealt)
      owned_programs=[9],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 9 (poly)
- **Expected Observation Changes**:
  - `player.credits`: 1 → 0 (cost: 1C, 1E)
  - `player.energy`: 1 → 0
  - Enemy type changes to a DIFFERENT type (not virus)
  - **Damage carries over**: 0 damage dealt → newHP = newType.maxHP - 0 = newType.maxHP
  - If daemon: HP = 3; if glitch/cryptog: HP = 2
- **Assert**: Enemy still exists, type is NOT virus (guaranteed different)
- **Applicability**: Requires enemies to exist

**Test: `test_poly_daemon_1hp_edge_case`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=1, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="daemon", row=5, col=5, hp=1, stunned=False)],  # Daemon at 1/3 HP (2 damage dealt)
      owned_programs=[9],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 9 (poly)
- **Expected Observation Changes**:
  - Enemy type changes to Cryptog, Virus, or Glitch (random, guaranteed different from daemon)
  - **Damage carries over**: 2 damage dealt to daemon carries to new type
  - New type maxHP = 2, so newHP = 2 - 2 = 0 → **Enemy dies**
  - Enemy removed from observation
- **Assert**: Enemy is killed (no enemies remain)
- **Expected Reward**: 0.3 (kill reward)
- **Code Reference**: `GameState.swift:1473-1498` - damage preservation logic
- **VERIFIED**: Fix in commit ac8d827 - POLY now preserves damage across type transformation

- [ ] **2.21** WAIT (index 10)

**Test: `test_wait_ends_turn`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="daemon", row=5, col=3, hp=3, stunned=False)],
      owned_programs=[10],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 10 (wait)
- **Expected Observation Changes**:
  - `player.energy`: 1 → 0 (cost: 0C, 1E)
  - `turn`: 0 → 1 (turn ends, enemy turn executes)
  - Enemy position: (5,3) → (4,3) (daemon moved 1 cell toward player)
- **Key**: This is the ONLY program that ends the turn
- **Applicability**: Always applicable

- [ ] **2.22** DEBUG (index 11)

**Test: `test_debug_damages_and_stuns_enemies_on_blocks`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="glitch", row=4, col=4, hp=1, stunned=False),  # On block, will die
          Enemy(type="glitch", row=2, col=2, hp=2, stunned=False),  # On block, will survive
          Enemy(type="virus", row=5, col=5, hp=2, stunned=False),   # NOT on block
      ],
      blocks=[
          Block(row=4, col=4, type="data", points=5, spawnCount=5, siphoned=False),
          Block(row=2, col=2, type="data", points=3, spawnCount=3, siphoned=False),
      ],
      owned_programs=[11],
      transmissions=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 11 (debug)
- **Expected Observation Changes**:
  - `player.credits`: 3 → 0 (cost: 3C, 0E)
  - Enemy at (4,4) killed (1 HP, took 1 damage)
  - Enemy at (2,2) damaged (2 → 1 HP) AND STUNNED
  - Enemy at (5,5) unaffected (not on block)
- **Expected Reward**: 0.3 (1 kill)
- **Applicability**: Requires enemies on blocks to exist
- DEBUG also STUNS surviving enemies on blocks

- [ ] **2.23** ROW (index 12)

**Test: `test_row_attacks_and_stuns_all_in_row`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="virus", row=3, col=0, hp=1, stunned=False),   # Same row, will die
          Enemy(type="daemon", row=3, col=5, hp=3, stunned=False),  # Same row, will survive
          Enemy(type="virus", row=4, col=3, hp=2, stunned=False),   # Different row - not hit
      ],
      owned_programs=[12],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 12 (row)
- **Expected Observation Changes**:
  - `player.credits`: 3 → 0 (cost: 3C, 1E)
  - `player.energy`: 1 → 0
  - Enemy at (3,0) killed
  - Enemy at (3,5) damaged (3 → 2 HP) AND STUNNED
  - Enemy at (4,3) untouched
- **Expected Reward**: 0.3 (1 kill)
- **Applicability**: Requires enemies in player's row
- ROW also STUNS surviving enemies

- [ ] **2.24** COL (index 13)

**Test: `test_col_attacks_and_stuns_all_in_column`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="virus", row=0, col=3, hp=1, stunned=False),   # Same col, will die
          Enemy(type="daemon", row=5, col=3, hp=3, stunned=False),  # Same col, will survive
          Enemy(type="virus", row=3, col=4, hp=2, stunned=False),   # Different col - not hit
      ],
      owned_programs=[13],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 13 (col)
- **Expected Observation Changes**:
  - `player.credits`: 3 → 0 (cost: 3C, 1E)
  - `player.energy`: 1 → 0
  - Enemy at (0,3) killed
  - Enemy at (5,3) damaged (3 → 2 HP) AND STUNNED
  - Enemy at (3,4) untouched
- **Expected Reward**: 0.3 (1 kill)
- **Applicability**: Requires enemies in player's column
- COL also STUNS surviving enemies

- [ ] **2.25** UNDO (index 14)

**Test: `test_undo_restores_player_and_enemy_positions`**
- **Preconditions**: (This requires 2 steps)
  1. Initial state with player at (3,3), enemy at (5,5)
  2. Move player to (4,3), enemy moves during enemy turn
  3. Execute undo
- **Setup**:
  ```python
  # After a move has been made (history exists):
  GameState(
      player=PlayerState(row=4, col=3, hp=3, credits=1, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="daemon", row=4, col=5, hp=3, stunned=False)],  # Moved from (5,5)
      owned_programs=[14],
      # Internal: gameHistory contains previous state
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=1, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 14 (undo)
- **Expected Observation Changes**:
  - `player.credits`: 1 → 0 (cost: 1C, 0E)
  - Player position reverts to (3,3)
  - Enemy position reverts to (5,5)
  - `turn` may revert
- **Applicability**: Requires game history (not empty)
- UNDO also reverses ENEMY positions (not just player)

- [ ] **2.26** STEP (index 15)

**Test: `test_step_prevents_enemy_movement`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=3, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=5, col=3, hp=2, stunned=False)],
      owned_programs=[15],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 15 (step), then 0 (move up to trigger turn end)
- **Expected Observation Changes**:
  - After STEP: `player.energy`: 3 → 0, `stepActive` = true
  - After move: Enemy position unchanged (5,3) - didn't move due to STEP
- **Applicability**: Always applicable

- [ ] **2.27** SIPH+ (index 16)

**Test: `test_siph_plus_gains_data_siphon`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=5, energy=0, dataSiphons=0, attackDamage=1, score=0),
      owned_programs=[16],
      enemies=[], transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 16 (siph+)
- **Expected Observation Changes**:
  - `player.credits`: 5 → 0 (cost: 5C, 0E)
  - `player.dataSiphons`: 0 → 1
- **Expected Reward**: 1.0 (data siphon collected reward)
- **Applicability**: Always applicable

- [ ] **2.28** EXCH (index 17)

**Test: `test_exch_converts_credits_to_energy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=8, energy=0, dataSiphons=0, attackDamage=1, score=0),
      owned_programs=[17],
      enemies=[], transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 17 (exch)
- **Expected Observation Changes**:
  - `player.credits`: 8 → 4 (cost: 4C, 0E)
  - `player.energy`: 0 → 4 (gained from exchange)
- **Applicability**: Requires player to have >= 4 credits (for cost + conversion)
- **Note**: Cost is 4C, then converts additional 4C to 4E - so need 8C total

**Note**: The cost is 4C and the exchange adds 4E. So with 4C you pay cost and gain 4E. Testing with 8C to be safe.

- [ ] **2.29** SHOW (index 18)

**Test: `test_show_reveals_cryptogs_and_transmissions`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="cryptog", row=5, col=5, hp=2, stunned=False)],  # Hidden (different row/col)
      transmissions=[Transmission(row=0, col=0, turnsRemaining=5, enemyType="virus")],
      showActivated=False,
      owned_programs=[18],
      blocks=[], resources=[],
      stage=1, turn=0, scheduledTasksDisabled=False
  )
  ```
- **Action**: 18 (show)
- **Expected Observation Changes**:
  - `player.credits`: 2 → 0 (cost: 2C, 0E)
  - `showActivated`: False → True
  - Cryptog at (5,5) now visible in observation (even though different row/col)
  - Transmissions now show their incoming enemy type
- **Applicability**: Requires `showActivated` to be false
- SHOW additionally makes transmissions show the incoming enemy type

- [ ] **2.30** RESET (index 19)

**Test: `test_reset_restores_hp`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=1, credits=0, energy=4, dataSiphons=0, attackDamage=1, score=0),
      owned_programs=[19],
      enemies=[], transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 19 (reset)
- **Expected Observation Changes**:
  - `player.energy`: 4 → 0 (cost: 0C, 4E)
  - `player.hp`: 1 → 3
- **Expected Reward**: HP recovery: 2 × 1.0 = 2.0
- **Applicability**: Requires player HP < 3

**Test: `test_reset_wasteful_at_2hp`**
- **Preconditions**: Same but with hp=2
- **Expected Reward**: HP recovery: 1 × 1.0 = 1.0, MINUS program waste: -0.3
- **Net Reward**: 0.7

- [ ] **2.31** CALM (index 20)

**Test: `test_calm_disables_scheduled_spawns`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=4, dataSiphons=0, attackDamage=1, score=0),
      scheduledTasksDisabled=False,
      owned_programs=[20],
      enemies=[], transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False
  )
  ```
- **Action**: 20 (calm)
- **Expected Observation Changes**:
  - `player.credits`: 2 → 0 (cost: 2C, 4E)
  - `player.energy`: 4 → 0
  - `scheduledTasksDisabled`: False → True
- **Applicability**: Requires `scheduledTasksDisabled` to be false

- [ ] **2.32** D_BOM (index 21)

**Test: `test_d_bom_destroys_daemon_and_damages_surrounding`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="daemon", row=5, col=5, hp=3, stunned=False),  # Target daemon
          Enemy(type="virus", row=5, col=4, hp=1, stunned=False),   # Adjacent to daemon, will die
          Enemy(type="virus", row=4, col=5, hp=2, stunned=False),   # Adjacent to daemon, will survive and stun
          Enemy(type="virus", row=0, col=0, hp=2, stunned=False),   # Far away, unaffected
      ],
      owned_programs=[21],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 21 (d_bom)
- **Expected Observation Changes**:
  - `player.credits`: 3 → 0 (cost: 3C, 0E)
  - Daemon at (5,5) destroyed
  - Enemy at (5,4) killed (took splash damage)
  - Enemy at (4,5) damaged (2 → 1 HP) AND STUNNED
  - Enemy at (0,0) unaffected
- **Expected Reward**: 2 × 0.3 = 0.6 (daemon + virus killed)
- **Applicability**: Requires daemon enemy to exist
- D_BOM does splash damage AND STUNS enemies in 8 surrounding cells of the daemon

- [ ] **2.33** DELAY (index 22)

**Test: `test_delay_extends_transmissions`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=1, energy=2, dataSiphons=0, attackDamage=1, score=0),
      transmissions=[Transmission(row=5, col=5, turnsRemaining=2, enemyType="virus")],
      owned_programs=[22],
      enemies=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 22 (delay)
- **Expected Observation Changes**:
  - `player.credits`: 1 → 0 (cost: 1C, 2E)
  - `player.energy`: 2 → 0
  - Transmission `turnsRemaining`: 2 → 5 (+3)
- **Applicability**: Requires transmissions to exist

- [ ] **2.34** ANTI-V (index 23)

**Test: `test_antiv_damages_and_stuns_all_viruses`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="virus", row=4, col=3, hp=1, stunned=False),   # Will die
          Enemy(type="virus", row=5, col=5, hp=2, stunned=False),   # Will survive and stun
          Enemy(type="daemon", row=0, col=0, hp=3, stunned=False),  # Not a virus, unaffected
      ],
      owned_programs=[23],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 23 (anti-v)
- **Expected Observation Changes**:
  - `player.credits`: 3 → 0 (cost: 3C, 0E)
  - Virus at (4,3) killed
  - Virus at (5,5) damaged (2 → 1 HP) AND STUNNED
  - Daemon at (0,0) untouched
- **Expected Reward**: 0.3 (1 kill)
- **Applicability**: Requires virus enemy to exist
- ANTI-V also STUNS all surviving viruses

- [ ] **2.35** SCORE (index 24)

**Test: `test_score_gains_points_by_stages_left`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=5, dataSiphons=0, attackDamage=1, score=0),
      stage=2,  # 8-2 = 6 stages left
      owned_programs=[24],
      enemies=[], transmissions=[], blocks=[], resources=[],
      turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 24 (score)
- **Expected Observation Changes**:
  - `player.energy`: 5 → 0 (cost: 0C, 5E)
  - `player.score`: 0 → 6 (stages left = 8 - 2)
- **Expected Reward**: Score gain: 6 × 0.5 = 3.0
- **Applicability**: Requires not being on last stage (stage < 8)

- [ ] **2.36** REDUC (index 25)

**Test: `test_reduc_reduces_block_spawn_counts`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=1, dataSiphons=0, attackDamage=1, score=0),
      blocks=[
          Block(row=4, col=3, type="data", points=3, spawnCount=3, siphoned=False),
          Block(row=4, col=4, type="data", points=2, spawnCount=2, siphoned=False),
          Block(row=2, col=2, type="data", points=1, spawnCount=1, siphoned=True),  # Siphoned, unaffected
      ],
      owned_programs=[25],
      enemies=[], transmissions=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 25 (reduc)
- **Expected Observation Changes**:
  - `player.credits`: 2 → 0 (cost: 2C, 1E)
  - `player.energy`: 1 → 0
  - Block at (4,3) spawnCount: 3 → 2
  - Block at (4,4) spawnCount: 2 → 1
  - Block at (2,2) unchanged (siphoned blocks not affected)
- **Applicability**: Requires unsiphoned blocks with spawnCount > 0

- [ ] **2.37** ATK+ (index 26)

**Test: `test_atkplus_increases_damage`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=4, energy=4, dataSiphons=0, attackDamage=1, score=0),
      owned_programs=[26],
      enemies=[], transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 26 (atk+)
- **Expected Observation Changes**:
  - `player.credits`: 4 → 0 (cost: 4C, 4E)
  - `player.energy`: 4 → 0
  - `player.attackDamage`: 1 → 2
- **Applicability**: Requires not used this stage AND attackDamage < 2
- **Subsequent attacks**: Should deal 2 damage instead of 1

- [ ] **2.38** HACK (index 27)

**Test: `test_hack_damages_enemies`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=2, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="virus", row=4, col=3, hp=1, stunned=False),  # On siphoned cell, will die
          Enemy(type="virus", row=5, col=5, hp=2, stunned=False),  # On regular cell, unaffected
      ],
      blocks=[
          Block(row=4, col=3, type="data", points=5, spawnCount=5, siphoned=True),   # Siphoned, will be destroyed
          Block(row=2, col=2, type="data", points=3, spawnCount=3, siphoned=False),  # Not siphoned, unaffected
      ],
      resources=[Resource(row=4, col=3, credits=5, energy=0)],  # Under siphoned block
      owned_programs=[27],
      transmissions=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 27 (hack)
- **Expected Observation Changes**:
  - `player.credits`: 2 → 0 (cost: 2C, 2E)
  - `player.energy`: 2 → 0
  - Enemy at (4,3) killed
  - Enemy at (5,5) unaffected
  - Block at (4,3) DESTROYED (cell becomes traversable)
  - Block at (2,2) unchanged (not siphoned)
  - Resources at (4,3) now visible/accessible
- **Expected Reward**: 0.3 (1 kill)
- **Applicability**: Requires siphoned cells to exist

#### Enemy Tests (`test_enemies.py`)

- [ ] **2.39** Enemy spawns from transmission

**Test: `test_enemy_spawns_from_transmission`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      transmissions=[Transmission(row=5, col=5, turnsRemaining=1, enemyType="virus")],
      enemies=[],
      owned_programs=[],
      blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up, triggers enemy turn)
- **Expected Observation Changes**:
  - Transmission removed
  - New virus enemy at (5,5)
  - `turn`: 0 → 1

- [ ] **2.40** Enemy movement toward player

**Test: `test_enemy_moves_toward_player`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=0, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="daemon", row=3, col=0, hp=3, stunned=False)],  # 3 cells away, same col
      owned_programs=[10],  # Wait program
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 10 (wait - ends turn)
- **Expected Observation Changes**:
  - Enemy moves from (3,0) to (2,0) (1 cell closer, daemon speed = 1)
  - `turn`: 0 → 1

- [ ] **2.41** Virus double-move

**Test: `test_virus_moves_twice`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=0, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=4, col=0, hp=2, stunned=False)],  # 4 cells away
      owned_programs=[10],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 10 (wait)
- **Expected Observation Changes**:
  - Virus moves from (4,0) to (2,0) (2 cells, moveSpeed=2)

- [ ] **2.42** Glitch can move on blocks

**Test: `test_glitch_moves_on_blocks`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=0, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="glitch", row=3, col=0, hp=2, stunned=False)],
      blocks=[Block(row=2, col=0, type="data", points=5, spawnCount=5, siphoned=False)],  # Block between glitch and player
      owned_programs=[10],
      transmissions=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 10 (wait)
- **Expected Observation Changes**:
  - Glitch moves through/onto block (only enemy type that can)
  - Position: (3,0) → (2,0)

- [ ] **2.43** Cryptog visibility

**Test: `test_cryptog_visible_in_same_row`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=0, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="cryptog", row=3, col=5, hp=2, stunned=False)],  # Same row
      showActivated=False,
      owned_programs=[],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, scheduledTasksDisabled=False
  )
  ```
- **Expected**: Cryptog visible in observation (same row)

**Test: `test_cryptog_hidden_different_row_col`**
- **Preconditions**: Cryptog at (5,5), player at (3,0)
- **Expected**: Cryptog NOT visible in observation (different row and col, showActivated=false)

- [ ] **2.44** Enemy attack when adjacent

**Test: `test_enemy_attacks_adjacent_player`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],  # Adjacent
      owned_programs=[10],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 10 (wait)
- **Expected Observation Changes**:
  - `player.hp`: 3 → 2 (took 1 damage)
- **Expected Reward**: Damage penalty: -1.0

- [ ] **2.45** Stunned enemy doesn't move

**Test: `test_stunned_enemy_no_movement`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=0, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=3, col=0, hp=2, stunned=True)],  # Stunned
      owned_programs=[10],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 10 (wait)
- **Expected Observation Changes**:
  - Enemy position unchanged (3,0)
  - Enemy `stunned` may reset to False after turn

- [ ] **2.46** Non-stunned enemies move after turn-ending actions

**Test: `test_enemies_move_after_turn_end`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      enemies=[
          Enemy(type="virus", row=4, col=0, hp=2, stunned=False),  # Not stunned, should move
          Enemy(type="daemon", row=5, col=5, hp=3, stunned=True),  # Stunned, should NOT move
      ],
      owned_programs=[],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up - ends turn)
- **Expected Observation Changes**:
  - Virus moves toward player (2 cells for virus)
  - Daemon position unchanged (stunned)
- **Key**: Any turn-ending action causes non-stunned enemies to move

#### Turn Tests (`test_turns.py`)

- [ ] **2.47** Move/attack/siphon ends player turn

**Test: `test_move_ends_turn`**
- **Preconditions**: Player at (3,3), turn=0
- **Action**: 0 (move up)
- **Expected**: `turn`: 0 → 1

- [ ] **2.48** Program execution does NOT end turn

**Test: `test_program_does_not_end_turn`**
- **Preconditions**: Player with push program, energy=2, turn=0, enemy present
- **Action**: 5 (push)
- **Expected**: `turn`: 0 (unchanged)
- **Post-step Valid Actions**: Should include all 4 movement directions

- [ ] **2.49** Wait program ends turn

**Test: `test_wait_ends_turn`**
- **Preconditions**: Player with wait program, energy=1, turn=0
- **Action**: 10 (wait)
- **Expected**: `turn`: 0 → 1

- [ ] **2.50** Turn counter increments on turn end

**Test: `test_turn_counter_increments`**
- **Preconditions**: turn=5
- **Action**: 0 (move up)
- **Expected**: `turn`: 5 → 6

- [ ] **2.51** Enemy turn executes after player turn ends

**Test: `test_enemy_turn_after_player`**
- **Preconditions**: Enemy at (5,0), player at (0,0), turn=0
- **Action**: 0 (move up)
- **Expected**: Enemy closer to player after step completes

- [ ] **2.52** Chain multiple programs before turn ends

**Test: `test_chain_programs`**
- **Preconditions**: Player with push (5) and pull (6), energy=4, turn=0, enemies present
- **Actions**: 5 (push), then 6 (pull)
- **Expected**: Both execute, turn=0 after both
- **Then Action**: 0 (move)
- **Expected**: turn=1

#### Stage Tests (`test_stages.py`)

- [ ] **2.53** Stage completion trigger

**Test: `test_stage_completes_on_exit_reached`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=4, col=5, hp=3, credits=0, energy=0, dataSiphons=0, attackDamage=1, score=0),
      stage=1,
      # Exit at (5,5)
      owned_programs=[],
      enemies=[], transmissions=[], blocks=[], resources=[],
      turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 0 (move up to exit)
- **Expected**: `stage`: 1 → 2
- **Expected Reward**: Stage completion: 1.0

- [ ] **2.54** New stage maintains data block invariant

**Test: `test_new_stage_data_block_invariant`**
- **Note**: After stage transition, verify all data blocks have `points == spawnCount`
- **Assert**: For each data block, `block.points == block.spawnCount`

- [ ] **2.55** Player state preserved on stage transition

**Test: `test_player_state_preserved_on_stage_transition`**
- **Preconditions**: Player with credits=5, energy=3, score=10, stage=1
- **Action**: Move to exit
- **Expected**: `credits`, `energy`, `score` preserved after stage transition

#### Action Mask Tests (`test_action_mask.py`)

- [ ] **2.56** Movement masked by walls/edges

**Test: `test_movement_masked_at_edges`**
- **Preconditions**: Player at (0,0)
- **Expected Valid Actions**: NOT include 1 (down - row=0), NOT include 2 (left - col=0)

- [ ] **2.57** Movement masked by blocks

**Test: `test_movement_masked_by_block`**
- **Preconditions**: Player at (3,3), block at (4,3)
- **Expected Valid Actions**: NOT include 0 (up toward block)

- [ ] **2.58** Siphon valid when player has data siphons

**Test: `test_siphon_valid_with_siphons`**
- **Preconditions**: Player with dataSiphons=1
- **Expected Valid Actions**: Include 4 (siphon)

**Test: `test_siphon_invalid_without_siphons`**
- **Preconditions**: Player with dataSiphons=0
- **Expected Valid Actions**: NOT include 4

- [ ] **2.59** Programs masked when not owned

**Test: `test_program_masked_when_not_owned`**
- **Preconditions**: Player with owned_programs=[], credits=10, energy=10
- **Expected Valid Actions**: NOT include any programs (5-27)

- [ ] **2.60** Programs masked when insufficient credits

**Test: `test_program_masked_insufficient_credits`**
- **Preconditions**: Player with owned_programs=[7], credits=0, energy=10 (crash needs 3C)
- **Expected Valid Actions**: NOT include 7

- [ ] **2.61** Programs masked when insufficient energy

**Test: `test_program_masked_insufficient_energy`**
- **Preconditions**: Player with owned_programs=[5], credits=10, energy=0 (push needs 2E)
- **Expected Valid Actions**: NOT include 5

- [ ] **2.62** Programs masked when applicability conditions not met

**Test: `test_program_masked_no_enemies`**
- **Preconditions**: Player with owned_programs=[5], energy=10, NO enemies
- **Expected Valid Actions**: NOT include 5 (push requires enemies)

**Test: `test_program_masked_no_daemon`**
- **Preconditions**: Player with owned_programs=[21], credits=10, enemies (but no daemon)
- **Expected Valid Actions**: NOT include 21 (d_bom requires daemon)

**Test: `test_program_masked_no_virus`**
- **Preconditions**: Player with owned_programs=[23], credits=10, enemies (but no virus)
- **Expected Valid Actions**: NOT include 23 (anti-v requires virus)

**Test: `test_program_masked_show_already_activated`**
- **Preconditions**: Player with owned_programs=[18], credits=10, showActivated=True
- **Expected Valid Actions**: NOT include 18 (show requires showActivated=false)

**Test: `test_program_masked_reset_at_full_hp`**
- **Preconditions**: Player with owned_programs=[19], hp=3, energy=10
- **Expected Valid Actions**: NOT include 19 (reset requires hp < 3)

**Test: `test_program_masked_undo_empty_history`**
- **Preconditions**: Player with owned_programs=[14], credits=10, fresh game (no history)
- **Expected Valid Actions**: NOT include 14 (undo requires history)

- [ ] **2.63** Mask updates correctly after state changes

**Test: `test_mask_updates_after_resource_gain`**
- **Preconditions**: Player with owned_programs=[5], energy=0, resource giving energy at adjacent cell
- **Action**: Move to collect resource (note: only data siphons collected by walking, not energy)
- **Note**: This test needs rethinking since energy isn't collected by walking

**Test: `test_mask_updates_after_siphon`**
- **Preconditions**: Player with owned_programs=[5], energy=0, dataSiphons=1, energy-giving block adjacent
- **Action**: 4 (siphon block that gives energy)
- **Post-step Valid Actions**: Now include 5 (push, has enough energy)

#### Edge Case Tests (`test_edge_cases.py`)

- [ ] **2.64** Player death

**Test: `test_player_death`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=1, credits=0, energy=1, dataSiphons=0, attackDamage=1, score=0),
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],
      owned_programs=[10],
      transmissions=[], blocks=[], resources=[],
      stage=1, turn=0, showActivated=False, scheduledTasksDisabled=False
  )
  ```
- **Action**: 10 (wait - triggers enemy attack)
- **Expected**:
  - `done`: True
  - `player.hp`: 0
  - Reward includes death penalty

- [ ] **2.65** Win condition

**Test: `test_win_condition`**
- **Preconditions**: Player at stage 8, adjacent to exit
- **Action**: Move to exit
- **Expected**:
  - `done`: True
  - `stage`: 9 (indicates victory)
  - Reward includes victory bonus: 500.0 + score × 100.0

#### Reward Tests (`test_rewards.py`)

- [ ] **2.66** Stage completion rewards

**Tests for each stage**:
| Stage Completed | Reward |
|-----------------|--------|
| 1 | 1.0 |
| 2 | 2.0 |
| 3 | 4.0 |
| 4 | 8.0 |
| 5 | 16.0 |
| 6 | 32.0 |
| 7 | 64.0 |
| 8 | 100.0 |

- [ ] **2.67** Score gain reward

**Test: `test_score_gain_reward`**
- **Preconditions**: score=0, siphon data block worth 10 points
- **Action**: 4 (siphon)
- **Expected Reward Component**: 10 × 0.5 = 5.0

- [ ] **2.68** Kill reward

**Test: `test_kill_reward`**
- **Preconditions**: 1 HP enemy adjacent
- **Action**: Attack
- **Expected Reward Component**: 1 × 0.3 = 0.3

- [ ] **2.69** Data siphon collection reward

**Test: `test_data_siphon_collection_reward`**
- **Preconditions**: Player executes SIPH+ program OR walks onto cell with data siphon
- **Expected Reward Component**: 1.0

- [ ] **2.70** Distance shaping reward

**Test: `test_distance_closer_reward`**
- **Preconditions**: Player moves closer to exit
- **Expected Reward Component**: delta × 0.05 (positive)

- [ ] **2.71** Victory bonus

**Test: `test_victory_bonus`**
- **Preconditions**: Complete stage 8 with score=100
- **Expected Reward Component**: 500.0 + 100 × 100.0 = 10500.0

- [ ] **2.72** Death penalty

**Test: `test_death_penalty`**
- **Preconditions**: Player dies at stage 3 (stages 1,2 completed)
- **Expected**: -(1 + 2) × 0.5 = -1.5

- [ ] **2.73** Resource gain reward

**Test: `test_resource_gain_reward`**
- **Preconditions**: Siphon block that gives 3 credits, 2 energy
- **Expected**: (3 + 2) × 0.05 = 0.25

- [ ] **2.74** Resource holding bonus on stage complete

**Test: `test_resource_holding_bonus_on_stage_complete`**
- **Preconditions**: Complete stage with 5 credits, 3 energy
- **Expected**: 5 × 0.01 + 3 × 0.01 = 0.08

- [ ] **2.75** Damage penalty

**Test: `test_damage_penalty`**
- **Preconditions**: Player takes 2 damage
- **Expected**: 2 × -1.0 = -2.0

- [ ] **2.76** HP recovery reward

**Test: `test_hp_recovery_reward`**
- **Preconditions**: Player at 1 HP uses RESET
- **Expected**: 2 × 1.0 = 2.0 (gained 2 HP)

- [ ] **2.77** Program waste penalty

**Test: `test_reset_at_2hp_penalty`**
- **Preconditions**: Player at 2 HP uses RESET
- **Expected**: -0.3 (program waste penalty)

- [ ] **2.78** Siphon-caused death penalty

**Test: `test_siphon_caused_death_penalty`**
- **Preconditions**: Player dies to enemy that spawned from siphon
- **Expected**: -10.0 (on top of regular death penalty)

## GameState Serialization Format

For `set_state`, use this JSON structure:

```json
{
  "action": "setState",
  "state": {
    "player": {
      "row": 3,
      "col": 3,
      "hp": 3,
      "credits": 5,
      "energy": 3,
      "dataSiphons": 0,
      "attackDamage": 1,
      "score": 0
    },
    "enemies": [
      {"type": "virus", "row": 1, "col": 1, "hp": 2, "stunned": false}
    ],
    "transmissions": [
      {"row": 2, "col": 2, "turnsRemaining": 3, "enemyType": "daemon"}
    ],
    "blocks": [
      {"row": 0, "col": 0, "type": "data", "points": 5, "spawnCount": 5, "siphoned": false}
    ],
    "resources": [
      {"row": 1, "col": 2, "credits": 2, "energy": 1, "dataSiphon": false}
    ],
    "ownedPrograms": [5, 10, 15],
    "stage": 1,
    "turn": 0,
    "showActivated": false,
    "scheduledTasksDisabled": false
  }
}
```

## File Structure

```
python/
├── hackmatrix/
│   ├── gym_env.py          # Existing - no changes needed
│   └── jax_env.py          # Existing - no changes needed
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # pytest fixtures
│   ├── env_interface.py    # EnvInterface protocol & dataclasses
│   ├── swift_env_wrapper.py
│   ├── jax_env_wrapper.py
│   ├── test_interface_smoke.py
│   ├── test_movement.py
│   ├── test_siphon.py
│   ├── test_programs.py
│   ├── test_enemies.py
│   ├── test_turns.py
│   ├── test_stages.py
│   ├── test_action_mask.py
│   ├── test_edge_cases.py
│   └── test_rewards.py
├── requirements.txt        # Add pytest>=7.0.0
└── scripts/
    └── test_env_parity.py  # Existing - keep for standalone parity checks

HackMatrix/
├── GameCommandProtocol.swift  # Add setState command handling
└── HeadlessGame.swift         # Add executeSetState() implementation

specs/
└── game-mechanics.md          # NEW - authoritative mechanics reference
```

## Success Criteria

- [ ] pytest added to requirements.txt and installed
- [ ] `specs/game-mechanics.md` created with authoritative reference
- [ ] `EnvInterface` Protocol defined with all methods (`reset`, `step`, `get_valid_actions`, `set_state`)
- [ ] `SwiftEnvWrapper` fully implements `EnvInterface` including `set_state`
- [ ] `JaxEnvWrapper` skeleton implements `EnvInterface` (stub returns for `set_state`)
- [ ] Interface smoke tests pass for both wrappers
- [ ] `setState` JSON command added to Swift protocol and HeadlessGame
- [ ] All comprehensive tests pass against Swift environment
- [ ] Test coverage includes all 23 programs, all action types, key edge cases
- [ ] Reward tests verify all reward components from RewardCalculator
- [ ] Tests runnable with: `cd python && source venv/bin/activate && pytest tests/ -v`

## Running Tests

```bash
# Run all tests
cd python && source venv/bin/activate && pytest tests/ -v

# Run specific test file
pytest tests/test_movement.py -v

# Run only Swift tests (skip JAX)
pytest tests/ -v -k "swift"

# Run smoke tests only
pytest tests/test_interface_smoke.py -v
```

## Notes

- Tests should handle non-determinism (enemy movement ties) by asserting one of valid outcomes
- Stage generation randomness: only test deterministic properties (enemy counts, player preserved)
- JAX wrapper tests will fail until `jax-implementation.md` is complete (expected)
- The existing `python/scripts/test_env_parity.py` will be kept for standalone parity checks (shape/dtype validation)
- New pytest-based tests in `python/tests/` are for comprehensive behavioral testing with `set_state`

## Dependencies

Before implementing:
1. Swift binary must be built: `swift build` (from project root)
2. Python venv must be set up: `cd python && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Execution Order

Recommended implementation order:
1. [ ] Create `specs/game-mechanics.md`
2. [ ] Add pytest to requirements.txt → install
3. [ ] Swift: Add `setState` command (Protocol + HeadlessGame)
4. [ ] Python: Create tests/ directory structure
5. [ ] Python: Create env_interface.py with dataclasses
6. [ ] Python: Create swift_env_wrapper.py with set_state
7. [ ] Python: Create jax_env_wrapper.py skeleton
8. [ ] Python: Create conftest.py with fixtures
9. [ ] Python: Create test_interface_smoke.py
10. [ ] Run smoke tests to verify infrastructure
11. [ ] Implement comprehensive tests file by file
