# Environment Parity Test Suite Implementation Plan

Based on analysis of `specs/env-parity-tests.md` and codebase exploration.

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

## Implementation Tasks

### Phase 1: Interface & Infrastructure

- [ ] **1.1** Add `pytest>=7.0.0` to `python/requirements.txt`
- [ ] **1.2** Create `python/tests/` directory structure with `__init__.py`
- [ ] **1.3** Create `python/tests/conftest.py` with pytest fixtures
- [ ] **1.4** Create `python/tests/env_interface.py` with `EnvInterface` Protocol and dataclasses
- [ ] **1.5** Add `setState` command to Swift JSON protocol (`GameCommandProtocol.swift`)
- [ ] **1.6** Implement `executeSetState()` in `HeadlessGame.swift`
- [ ] **1.7** Create `python/tests/swift_env_wrapper.py` implementing `EnvInterface`
- [ ] **1.8** Create `python/tests/jax_env_wrapper.py` skeleton implementing `EnvInterface`
- [ ] **1.9** Create `python/tests/test_interface_smoke.py`

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
      player=PlayerState(row=5, col=3, hp=3, ...),  # Top row
      enemies=[], blocks=[], ...
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
      player=PlayerState(row=3, col=3, hp=3, ...),
      blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=2, siphoned=False)],
      ...
  )
  ```
- **Action**: 0 (move up toward block)
- **Expected**: Action should be masked (block at row=4, col=3)
- **Pre-step Valid Actions**: Should NOT include 0

- [ ] **2.4** Move onto cell with credits

**Test: `test_move_collects_credits`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, ...),
      resources=[Resource(row=4, col=3, credits=3, energy=0)],
      ...
  )
  ```
- **Action**: 0 (move up)
- **Expected Observation Changes**:
  - `player.row`: 3 → 4
  - `player.credits`: 0 → 3
  - Resource at (4,3) removed from observation
- **Expected Reward**:
  - Resource gain: 3 * 0.05 = 0.15
  - Total: 0.15

- [ ] **2.5** Move onto cell with energy

**Test: `test_move_collects_energy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, ...),
      resources=[Resource(row=4, col=3, credits=0, energy=2)],
      ...
  )
  ```
- **Action**: 0 (move up)
- **Expected Observation Changes**:
  - `player.row`: 3 → 4
  - `player.energy`: 0 → 2
- **Expected Reward**: 2 * 0.05 = 0.10

- [ ] **2.6** Move onto cell with both resources

**Test: `test_move_collects_both_resources`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, ...),
      resources=[Resource(row=4, col=3, credits=2, energy=3)],
      ...
  )
  ```
- **Action**: 0 (move up)
- **Expected Observation Changes**:
  - `player.credits`: 0 → 2
  - `player.energy`: 0 → 3
- **Expected Reward**: (2 + 3) * 0.05 = 0.25

- [ ] **2.7** Move into enemy (attack and kill)

**Test: `test_attack_kills_enemy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=0, attackDamage=1, ...),
      enemies=[Enemy(type="virus", row=4, col=3, hp=1, stunned=False)],  # 1 HP enemy
      ...
  )
  ```
- **Action**: 0 (move up into enemy)
- **Expected Observation Changes**:
  - `player.row`: unchanged (3) - attack doesn't move player
  - Enemy removed from observation
- **Expected Reward**:
  - Kill: 1 * 0.3 = 0.3
  - Total: 0.3

- [ ] **2.8** Move into enemy (attack, enemy survives)

**Test: `test_attack_damages_enemy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, attackDamage=1, ...),
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],  # 2 HP enemy
      ...
  )
  ```
- **Action**: 0 (move up into enemy)
- **Expected Observation Changes**:
  - `player.row`: unchanged (3)
  - Enemy HP: 2 → 1
  - Enemy still in observation
- **Expected Reward**: 0.0 (no kill, no score)

- [ ] **2.9** Move into transmission

**Test: `test_attack_destroys_transmission`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, ...),
      transmissions=[Transmission(row=4, col=3, turnsRemaining=3, enemyType="virus")],
      ...
  )
  ```
- **Action**: 0 (move up into transmission)
- **Expected Observation Changes**:
  - `player.row`: unchanged (3) - attack doesn't move
  - Transmission removed from observation
- **Expected Reward**: 0.0 (transmissions don't give kill reward)

#### Siphon Tests (`test_siphon.py`)

- [ ] **2.10** Siphon adjacent block (cross pattern)

**Test: `test_siphon_data_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, dataSiphons=1, score=0, ...),
      blocks=[Block(row=4, col=3, type="data", points=10, spawnCount=2, siphoned=False)],
      ...
  )
  ```
- **Action**: 4 (siphon)
- **Expected Observation Changes**:
  - `player.dataSiphons`: 1 → 0 (consumed)
  - `player.score`: 0 → 10
  - Block at (4,3) marked as siphoned=True
  - 2 transmissions spawned (from spawnCount)
- **Expected Reward**:
  - Score gain: 10 * 0.5 = 5.0
  - Total: 5.0

- [ ] **2.11** Siphon with no adjacent block (invalid)

**Test: `test_siphon_invalid_no_adjacent_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, dataSiphons=1, ...),
      blocks=[],  # No blocks
      ...
  )
  ```
- **Pre-step Valid Actions**: Should NOT include 4 (siphon not valid without adjacent block)

- [ ] **2.12** Siphon already-siphoned block

**Test: `test_siphon_already_siphoned_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, dataSiphons=1, ...),
      blocks=[Block(row=4, col=3, type="data", points=10, spawnCount=2, siphoned=True)],  # Already siphoned
      ...
  )
  ```
- **Pre-step Valid Actions**: Should NOT include 4 if only adjacent block is siphoned

- [ ] **2.13** Siphon spawns transmissions

**Test: `test_siphon_spawns_transmissions`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, dataSiphons=1, ...),
      blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=3, siphoned=False)],
      transmissions=[],  # No existing transmissions
      ...
  )
  ```
- **Action**: 4 (siphon)
- **Expected Observation Changes**:
  - 3 new transmissions appear in observation
  - Transmissions spawned at valid empty cells

- [ ] **2.14** Siphon reveals resources

**Test: `test_siphon_reveals_hidden_resources`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, dataSiphons=1, ...),
      blocks=[Block(row=4, col=3, type="data", points=5, spawnCount=0, siphoned=False)],
      # Resources at block position become visible after siphon
      ...
  )
  ```
- **Action**: 4 (siphon)
- **Expected**: Resources beneath block become collectible

- [ ] **2.15** Siphon different block types

**Test: `test_siphon_program_block`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, dataSiphons=1, ...),
      blocks=[Block(row=4, col=3, type="program", programType="push", programActionIndex=5, spawnCount=2, siphoned=False)],
      owned_programs=[],
      ...
  )
  ```
- **Action**: 4 (siphon)
- **Expected Observation Changes**:
  - `owned_programs`: [] → [5] (push program acquired)
  - `programs[0]`: 0 → 1 (push at index 0)

#### Program Tests (`test_programs.py`)

- [ ] **2.16** PUSH (index 5)

**Test: `test_push_enemies_away`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=2, ...),  # Needs 2 energy
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],  # Adjacent enemy
      owned_programs=[5],  # Has push
      ...
  )
  ```
- **Action**: 5 (push)
- **Expected Observation Changes**:
  - `player.energy`: 2 → 0 (cost: 0C, 2E)
  - Enemy position: (4,3) → (5,3) (pushed away from player)
  - `turn`: unchanged (programs don't end turn except wait)
- **Expected Reward**: 0.0
- **Post-step Valid Actions**: Should still include movement options (turn not ended)

- [ ] **2.17** PULL (index 6)

**Test: `test_pull_enemies_toward`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=2, ...),
      enemies=[Enemy(type="virus", row=5, col=3, hp=2, stunned=False)],  # 2 cells away
      owned_programs=[6],
      ...
  )
  ```
- **Action**: 6 (pull)
- **Expected**: Enemy at (5,3) → (4,3) (pulled toward player)
- **Cost**: 0C, 2E

- [ ] **2.18** CRASH (index 7)

**Test: `test_crash_clears_surrounding`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=2, ...),  # Cost: 3C, 2E
      enemies=[
          Enemy(type="virus", row=4, col=3, hp=1, stunned=False),  # Adjacent
          Enemy(type="virus", row=3, col=4, hp=1, stunned=False),
      ],
      owned_programs=[7],
      ...
  )
  ```
- **Action**: 7 (crash)
- **Expected**: All enemies in 8 surrounding cells damaged/killed
- **Expected Reward**: 2 * 0.3 = 0.6 (2 kills)

- [ ] **2.19** WARP (index 8)

**Test: `test_warp_to_enemy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, credits=2, energy=2, ...),  # Cost: 2C, 2E
      enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False)],  # Far enemy
      owned_programs=[8],
      ...
  )
  ```
- **Action**: 8 (warp)
- **Expected**: Player position changes to adjacent to enemy
- **Note**: Exact position may vary (non-deterministic)

- [ ] **2.20** POLY (index 9)

**Test: `test_poly_randomizes_enemy_types`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=1, energy=1, ...),  # Cost: 1C, 1E
      enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False)],
      owned_programs=[9],
      ...
  )
  ```
- **Action**: 9 (poly)
- **Expected**: Enemy type may change (non-deterministic)
- **Assert**: Enemy still exists, type is valid

- [ ] **2.21** WAIT (index 10)

**Test: `test_wait_ends_turn`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=1, ...),  # Cost: 0C, 1E
      enemies=[Enemy(type="virus", row=5, col=5, hp=2, stunned=False)],
      owned_programs=[10],
      turn=0,
      ...
  )
  ```
- **Action**: 10 (wait)
- **Expected Observation Changes**:
  - `turn`: 0 → 1 (turn ends, enemy turn executes)
  - Enemy may have moved (toward player)
- **Key**: This is the ONLY program that ends the turn

- [ ] **2.22** DEBUG (index 11)

**Test: `test_debug_damages_enemies_on_blocks`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0, ...),  # Cost: 3C, 0E
      enemies=[Enemy(type="virus", row=4, col=4, hp=1, stunned=False)],  # On block
      blocks=[Block(row=4, col=4, type="data", ...)],
      owned_programs=[11],
      ...
  )
  ```
- **Action**: 11 (debug)
- **Expected**: Enemy on block is killed
- **Expected Reward**: 0.3 (1 kill)

- [ ] **2.23** ROW (index 12)

**Test: `test_row_attacks_all_in_row`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=1, ...),  # Cost: 3C, 1E
      enemies=[
          Enemy(type="virus", row=3, col=0, hp=1, stunned=False),  # Same row
          Enemy(type="virus", row=3, col=5, hp=1, stunned=False),  # Same row
          Enemy(type="virus", row=4, col=3, hp=2, stunned=False),  # Different row - not hit
      ],
      owned_programs=[12],
      ...
  )
  ```
- **Action**: 12 (row)
- **Expected**: 2 enemies in row 3 killed, enemy in row 4 untouched
- **Expected Reward**: 2 * 0.3 = 0.6

- [ ] **2.24** COL (index 13)

**Test: `test_col_attacks_all_in_column`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=1, ...),  # Cost: 3C, 1E
      enemies=[
          Enemy(type="virus", row=0, col=3, hp=1, stunned=False),  # Same col
          Enemy(type="virus", row=5, col=3, hp=1, stunned=False),  # Same col
          Enemy(type="virus", row=3, col=4, hp=2, stunned=False),  # Different col - not hit
      ],
      owned_programs=[13],
      ...
  )
  ```
- **Action**: 13 (col)
- **Expected**: 2 enemies in col 3 killed

- [ ] **2.25** UNDO (index 14)

**Test: `test_undo_restores_previous_state`**
- **Preconditions**: (This requires 2 steps)
  1. Initial state with player at (3,3)
  2. Move player to (4,3)
  3. Execute undo
- **Setup**:
  ```python
  GameState(
      player=PlayerState(row=4, col=3, hp=3, credits=1, energy=0, ...),  # Cost: 1C, 0E
      owned_programs=[14],
      # Internal: previousState stored from before move
      ...
  )
  ```
- **Action**: 14 (undo)
- **Expected**: Player position reverts to (3,3)

- [ ] **2.26** STEP (index 15)

**Test: `test_step_prevents_enemy_movement`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=3, ...),  # Cost: 0C, 3E
      enemies=[Enemy(type="virus", row=5, col=3, hp=2, stunned=False)],
      owned_programs=[15],
      ...
  )
  ```
- **Action**: 15 (step)
- **Expected**: Enemy position unchanged on next turn
- **Note**: Need to trigger turn end (e.g., move) and verify enemy didn't move

- [ ] **2.27** SIPH+ (index 16)

**Test: `test_siph_plus_gains_data_siphon`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=5, energy=0, dataSiphons=0, ...),  # Cost: 5C, 0E
      owned_programs=[16],
      ...
  )
  ```
- **Action**: 16 (siph+)
- **Expected**: `player.dataSiphons`: 0 → 1
- **Expected Reward**: 1.0 (data siphon collected reward)

- [ ] **2.28** EXCH (index 17)

**Test: `test_exch_converts_credits_to_energy`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=4, energy=0, ...),  # Cost: 4C, 0E + converts 4C to 4E
      owned_programs=[17],
      ...
  )
  ```
- **Action**: 17 (exch)
- **Expected**:
  - `player.credits`: 4 → 0 (spent on program)
  - `player.energy`: 0 → 4 (gained from exchange)
- **Note**: May need 8 credits total (4 for cost + 4 for exchange)

- [ ] **2.29** SHOW (index 18)

**Test: `test_show_reveals_cryptogs`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=0, showActivated=False, ...),  # Cost: 2C, 0E
      enemies=[Enemy(type="cryptog", row=5, col=5, hp=2, stunned=False)],  # Hidden cryptog
      owned_programs=[18],
      ...
  )
  ```
- **Action**: 18 (show)
- **Expected**: `player.showActivated`: False → True
- **Expected**: Cryptog now visible in observation

- [ ] **2.30** RESET (index 19)

**Test: `test_reset_restores_hp`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=1, credits=0, energy=4, ...),  # Cost: 0C, 4E
      owned_programs=[19],
      ...
  )
  ```
- **Action**: 19 (reset)
- **Expected**: `player.hp`: 1 → 3
- **Expected Reward**: HP recovery: 2 * 1.0 = 2.0

**Test: `test_reset_wasteful_at_2hp`**
- **Preconditions**: Same but with hp=2
- **Expected Reward**: -0.3 (program waste penalty for RESET at 2 HP)

- [ ] **2.31** CALM (index 20)

**Test: `test_calm_disables_scheduled_spawns`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=4, scheduledTasksDisabled=False, ...),  # Cost: 2C, 4E
      owned_programs=[20],
      ...
  )
  ```
- **Action**: 20 (calm)
- **Expected**: `scheduledTasksDisabled`: False → True

- [ ] **2.32** D_BOM (index 21)

**Test: `test_d_bom_destroys_nearest_daemon`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0, ...),  # Cost: 3C, 0E
      enemies=[
          Enemy(type="daemon", row=4, col=3, hp=3, stunned=False),  # Nearest daemon
          Enemy(type="virus", row=5, col=5, hp=2, stunned=False),    # Not a daemon
      ],
      owned_programs=[21],
      ...
  )
  ```
- **Action**: 21 (d_bom)
- **Expected**: Daemon killed, virus untouched
- **Expected Reward**: 0.3 (1 kill)

- [ ] **2.33** DELAY (index 22)

**Test: `test_delay_extends_transmissions`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=1, energy=2, ...),  # Cost: 1C, 2E
      transmissions=[Transmission(row=5, col=5, turnsRemaining=2, enemyType="virus")],
      owned_programs=[22],
      ...
  )
  ```
- **Action**: 22 (delay)
- **Expected**: Transmission `turnsRemaining`: 2 → 5 (+3)

- [ ] **2.34** ANTI-V (index 23)

**Test: `test_antiv_damages_all_viruses`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=3, energy=0, ...),  # Cost: 3C, 0E
      enemies=[
          Enemy(type="virus", row=4, col=3, hp=1, stunned=False),
          Enemy(type="virus", row=5, col=5, hp=1, stunned=False),
          Enemy(type="daemon", row=0, col=0, hp=3, stunned=False),  # Not a virus
      ],
      owned_programs=[23],
      ...
  )
  ```
- **Action**: 23 (anti-v)
- **Expected**: 2 viruses killed, daemon untouched
- **Expected Reward**: 2 * 0.3 = 0.6

- [ ] **2.35** SCORE (index 24)

**Test: `test_score_gains_points_by_stages_left`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=0, energy=5, score=0, ...),  # Cost: 0C, 5E
      stage=2,  # 8-2 = 6 stages left
      owned_programs=[24],
      ...
  )
  ```
- **Action**: 24 (score)
- **Expected**: `player.score`: 0 → 6 (stages left = 8 - 2)
- **Expected Reward**: Score gain: 6 * 0.5 = 3.0

- [ ] **2.36** REDUC (index 25)

**Test: `test_reduc_reduces_block_spawn_counts`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=1, ...),  # Cost: 2C, 1E
      blocks=[
          Block(row=4, col=3, type="data", spawnCount=3, siphoned=False),
          Block(row=4, col=4, type="data", spawnCount=2, siphoned=False),
      ],
      owned_programs=[25],
      ...
  )
  ```
- **Action**: 25 (reduc)
- **Expected**: All block spawnCounts reduced by 1 (or to 0 minimum)

- [ ] **2.37** ATK+ (index 26)

**Test: `test_atkplus_increases_damage`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=4, energy=4, attackDamage=1, ...),  # Cost: 4C, 4E
      owned_programs=[26],
      ...
  )
  ```
- **Action**: 26 (atk+)
- **Expected**: `player.attackDamage`: 1 → 2
- **Subsequent attack**: Should deal 2 damage instead of 1

- [ ] **2.38** HACK (index 27)

**Test: `test_hack_damages_enemies_on_siphoned_cells`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, credits=2, energy=2, ...),  # Cost: 2C, 2E
      enemies=[
          Enemy(type="virus", row=4, col=3, hp=1, stunned=False),  # On siphoned cell
          Enemy(type="virus", row=5, col=5, hp=2, stunned=False),  # On regular cell
      ],
      blocks=[Block(row=4, col=3, type="data", siphoned=True)],  # Siphoned block
      owned_programs=[27],
      ...
  )
  ```
- **Action**: 27 (hack)
- **Expected**: Enemy on siphoned cell killed, other untouched
- **Expected Reward**: 0.3

#### Enemy Tests (`test_enemies.py`)

- [ ] **2.39** Enemy spawns from transmission

**Test: `test_enemy_spawns_from_transmission`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, ...),
      transmissions=[Transmission(row=5, col=5, turnsRemaining=1, enemyType="virus")],
      enemies=[],
      turn=0,
      ...
  )
  ```
- **Action**: 0 (move up, triggers enemy turn)
- **Expected**:
  - Transmission removed
  - New virus enemy at (5,5)

- [ ] **2.40** Enemy movement toward player

**Test: `test_enemy_moves_toward_player`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, ...),
      enemies=[Enemy(type="daemon", row=3, col=0, hp=3, stunned=False)],  # 3 cells away, same col
      turn=0,
      ...
  )
  ```
- **Action**: 10 (wait - ends turn)
- **Expected**: Enemy moves from (3,0) to (2,0) (1 cell closer)

- [ ] **2.41** Virus double-move

**Test: `test_virus_moves_twice`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, ...),
      enemies=[Enemy(type="virus", row=4, col=0, hp=2, stunned=False)],  # 4 cells away
      turn=0,
      ...
  )
  ```
- **Action**: 10 (wait)
- **Expected**: Virus moves from (4,0) to (2,0) (2 cells, moveSpeed=2)

- [ ] **2.42** Glitch can move on blocks

**Test: `test_glitch_moves_on_blocks`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, ...),
      enemies=[Enemy(type="glitch", row=3, col=0, hp=2, stunned=False)],
      blocks=[Block(row=2, col=0, type="data", ...)],  # Block between glitch and player
      turn=0,
      ...
  )
  ```
- **Action**: 10 (wait)
- **Expected**: Glitch moves through block (only enemy type that can)

- [ ] **2.43** Cryptog visibility

**Test: `test_cryptog_visible_in_same_row`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=0, hp=3, showActivated=False, ...),
      enemies=[Enemy(type="cryptog", row=3, col=5, hp=2, stunned=False)],  # Same row
      ...
  )
  ```
- **Expected**: Cryptog visible in observation (same row)

**Test: `test_cryptog_hidden_different_row_col`**
- **Preconditions**: Cryptog at (5,5), player at (3,0)
- **Expected**: Cryptog NOT visible (different row and col)

- [ ] **2.44** Enemy attack when adjacent

**Test: `test_enemy_attacks_adjacent_player`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=3, ...),
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],  # Adjacent
      turn=0,
      ...
  )
  ```
- **Action**: 10 (wait)
- **Expected**:
  - `player.hp`: 3 → 2 (took 1 damage)
- **Expected Reward**: Damage penalty: -1.0

- [ ] **2.45** Stunned enemy doesn't move

**Test: `test_stunned_enemy_no_movement`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=0, col=0, hp=3, ...),
      enemies=[Enemy(type="virus", row=3, col=0, hp=2, stunned=True)],  # Stunned
      turn=0,
      ...
  )
  ```
- **Action**: 10 (wait)
- **Expected**: Enemy position unchanged (3,0)

- [ ] **2.46** Disabled enemy behavior

**Test: `test_disabled_enemy_behavior`**
- **Preconditions**: Enemy with disabled counter > 0
- **Expected**: Enemy doesn't move/attack

#### Turn Tests (`test_turns.py`)

- [ ] **2.47** Move/attack/siphon ends player turn

**Test: `test_move_ends_turn`**
- **Preconditions**: Player at (3,3), turn=0
- **Action**: 0 (move up)
- **Expected**: `turn`: 0 → 1

- [ ] **2.48** Program execution does NOT end turn

**Test: `test_program_does_not_end_turn`**
- **Preconditions**: Player with push program, energy=2, turn=0
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
- **Preconditions**: Player with push (5) and pull (6), energy=4, turn=0
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
      player=PlayerState(row=4, col=5, hp=3, ...),  # Adjacent to exit
      stage=1,
      # Exit at (5,5)
      ...
  )
  ```
- **Action**: 0 (move up to exit)
- **Expected**: `stage`: 1 → 2
- **Expected Reward**: Stage completion: 1.0

- [ ] **2.54** New stage enemy count

**Test: `test_new_stage_enemy_count`**
- **Note**: Difficult to test exactly due to randomness
- **Assert**: After stage transition, enemies exist (count varies by stage)

- [ ] **2.55** Enemies persist across stage transitions

**Test: `test_enemies_persist_after_stage_complete`**
- **Note**: This depends on game logic - verify behavior

- [ ] **2.56** Player state preserved on stage transition

**Test: `test_player_state_preserved_on_stage_transition`**
- **Preconditions**: Player with credits=5, energy=3, score=10, stage=1
- **Action**: Move to exit
- **Expected**: `credits`, `energy`, `score` preserved

#### Action Mask Tests (`test_action_mask.py`)

- [ ] **2.57** Movement masked by walls/edges

**Test: `test_movement_masked_at_edges`**
- **Preconditions**: Player at (0,0)
- **Expected Valid Actions**: NOT include 1 (down), NOT include 2 (left)

- [ ] **2.58** Movement masked by blocks

**Test: `test_movement_masked_by_block`**
- **Preconditions**: Player at (3,3), block at (4,3)
- **Expected Valid Actions**: NOT include 0 (up toward block)

- [ ] **2.59** Siphon only valid adjacent to unsiphoned block

**Test: `test_siphon_masked_without_block`**
- **Preconditions**: Player at (3,3), no blocks
- **Expected Valid Actions**: NOT include 4

**Test: `test_siphon_valid_with_adjacent_block`**
- **Preconditions**: Player at (3,3), unsiphoned block at (4,3), dataSiphons=1
- **Expected Valid Actions**: Include 4

- [ ] **2.60** Programs masked when not owned

**Test: `test_program_masked_when_not_owned`**
- **Preconditions**: Player with owned_programs=[], credits=10, energy=10
- **Expected Valid Actions**: NOT include any programs (5-27)

- [ ] **2.61** Programs masked when insufficient credits

**Test: `test_program_masked_insufficient_credits`**
- **Preconditions**: Player with owned_programs=[7], credits=0, energy=10 (crash needs 3C)
- **Expected Valid Actions**: NOT include 7

- [ ] **2.62** Programs masked when insufficient energy

**Test: `test_program_masked_insufficient_energy`**
- **Preconditions**: Player with owned_programs=[5], credits=10, energy=0 (push needs 2E)
- **Expected Valid Actions**: NOT include 5

- [ ] **2.63** Mask updates correctly after state changes

**Test: `test_mask_updates_after_resource_gain`**
- **Preconditions**: Player with owned_programs=[5], energy=0, resource at (4,3) with energy=2
- **Action**: 0 (move up, collect energy)
- **Post-step Valid Actions**: Now include 5 (push, has 2E)

#### Edge Case Tests (`test_edge_cases.py`)

- [ ] **2.64** Player death

**Test: `test_player_death`**
- **Preconditions**:
  ```python
  GameState(
      player=PlayerState(row=3, col=3, hp=1, ...),
      enemies=[Enemy(type="virus", row=4, col=3, hp=2, stunned=False)],
      turn=0,
      ...
  )
  ```
- **Action**: 10 (wait - triggers enemy attack)
- **Expected**:
  - `done`: True
  - Reward includes death penalty

- [ ] **2.65** Win condition

**Test: `test_win_condition`**
- **Preconditions**: Player at stage 8, adjacent to exit
- **Action**: Move to exit
- **Expected**:
  - `done`: True
  - `stage`: 9 (indicates victory)
  - Reward includes victory bonus: 500.0 + score * 100.0

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
- **Expected Reward Component**: 10 * 0.5 = 5.0

- [ ] **2.68** Kill reward

**Test: `test_kill_reward`**
- **Preconditions**: 1 HP enemy adjacent
- **Action**: Attack
- **Expected Reward Component**: 1 * 0.3 = 0.3

- [ ] **2.69** Data siphon collection reward

**Test: `test_data_siphon_collection_reward`**
- **Preconditions**: Player executes SIPH+ program
- **Expected Reward Component**: 1.0

- [ ] **2.70** Distance shaping reward

**Test: `test_distance_closer_reward`**
- **Preconditions**: Player moves closer to exit
- **Expected Reward Component**: delta * 0.05 (positive)

- [ ] **2.71** Victory bonus

**Test: `test_victory_bonus`**
- **Preconditions**: Complete stage 8 with score=100
- **Expected Reward Component**: 500.0 + 100 * 100.0 = 10500.0

- [ ] **2.72** Death penalty

**Test: `test_death_penalty`**
- **Preconditions**: Player dies at stage 3 (stages 1,2 completed)
- **Expected**: -(1 + 2) * 0.5 = -1.5

- [ ] **2.73** Resource gain reward

**Test: `test_resource_gain_reward`**
- **Preconditions**: Collect 3 credits, 2 energy
- **Expected**: (3 + 2) * 0.05 = 0.25

- [ ] **2.74** Resource holding bonus

**Test: `test_resource_holding_bonus_on_stage_complete`**
- **Preconditions**: Complete stage with 5 credits, 3 energy
- **Expected**: 5 * 0.01 + 3 * 0.01 = 0.08

- [ ] **2.75** Damage penalty

**Test: `test_damage_penalty`**
- **Preconditions**: Player takes 2 damage
- **Expected**: 2 * -1.0 = -2.0

- [ ] **2.76** HP recovery reward

**Test: `test_hp_recovery_reward`**
- **Preconditions**: Player at 1 HP uses RESET
- **Expected**: 2 * 1.0 = 2.0 (gained 2 HP)

- [ ] **2.77** Siphon quality penalty

**Test: `test_siphon_quality_penalty`**
- **Preconditions**: Multiple siphon options, player chooses suboptimal
- **Expected**: Penalty proportional to missed resources

- [ ] **2.78** Program waste penalty

**Test: `test_reset_at_2hp_penalty`**
- **Preconditions**: Player at 2 HP uses RESET
- **Expected**: -0.3

- [ ] **2.79** Siphon-caused death penalty

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
      {"row": 0, "col": 0, "type": "data", "points": 5, "spawnCount": 2, "siphoned": false}
    ],
    "resources": [
      {"row": 1, "col": 2, "credits": 2, "energy": 1}
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
```

## Success Criteria

- [ ] pytest added to requirements.txt and installed
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
- [ ] Add pytest to requirements.txt → install
- [ ] Swift: Add `setState` command (Protocol + HeadlessGame)
- [ ] Python: Create tests/ directory structure
- [ ] Python: Create env_interface.py with dataclasses
- [ ] Python: Create swift_env_wrapper.py with set_state
- [ ] Python: Create jax_env_wrapper.py skeleton
- [ ] Python: Create conftest.py with fixtures
- [ ] Python: Create test_interface_smoke.py
- [ ] Run smoke tests to verify infrastructure
- [ ] Implement comprehensive tests file by file
