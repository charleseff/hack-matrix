# HackMatrix Game Mechanics Reference

This document provides an authoritative reference for HackMatrix game mechanics, extracted from the Swift source code. Use this as the single source of truth for implementing environment parity tests.

## Grid System

- **Grid Size**: 6×6 cells (Constants.gridSize = 6)
- **Valid Positions**: row 0-5, col 0-5
- **Corners**: (0,0), (0,5), (5,0), (5,5)
- **Exit Position**: Random corner, different from player's corner (changes each stage)

### Direction Mapping

| Action | Direction | Row Change | Col Change |
|--------|-----------|------------|------------|
| 0 | Up | +1 | 0 |
| 1 | Down | -1 | 0 |
| 2 | Left | 0 | -1 |
| 3 | Right | 0 | +1 |

## Action Space (28 actions)

| Index | Action Type |
|-------|-------------|
| 0-3 | Movement directions (up, down, left, right) |
| 4 | Siphon |
| 5-27 | Programs (23 total) |

## Player Mechanics

### Stats
- **HP**: 3 max, death at 0
- **Attack Damage**: 1 default, 2 with ATK+ program
- **Credits**: Currency for programs
- **Energy**: Currency for programs
- **Data Siphons**: Required for siphon action
- **Score**: Points accumulated

### Movement
- Player can move to empty, unblocked cells
- Movement ends the player's turn

### Line-of-Sight Attacks
When taking a directional action (0-3):
1. If an enemy is **adjacent** (1 cell away) in that direction → Attack
2. If an enemy is **further** but in **line-of-sight** (same row/col, no obstacles) → Attack (not move)
3. If cell is **empty and unblocked** → Move
4. If cell is **blocked or edge** → Invalid action (masked)

**Key**: Player attacks enemies in line-of-sight even if they're multiple cells away.

### Siphon Action
- **Validity**: Valid ONLY when `player.dataSiphons > 0`
- **NOT dependent on adjacent blocks** - siphon is always valid when player has siphons
- Consumes 1 data siphon
- Siphons the best adjacent block (cross pattern: up/down/left/right)
- Siphon ends the player's turn

### Resource Collection

| Resource Type | Collection Method |
|--------------|-------------------|
| Credits | Siphoning blocks only |
| Energy | Siphoning blocks only |
| Data Siphons | Walking onto cells |

**Critical**: Credits and energy are NOT collected by walking. Only data siphons are collected by walking.

## Turn Structure

### Player Turn
1. Player selects action
2. Action executes:
   - **Move/Attack/Siphon**: Turn ends immediately
   - **Program (except WAIT)**: Turn does NOT end, can chain more programs
   - **WAIT program**: Turn ends

### Enemy Turn (after player turn ends)
1. Transmissions tick down, may spawn enemies
2. Non-stunned enemies move toward player
3. Adjacent enemies attack player
4. Scheduled tasks execute (if not disabled)
5. Status effects reset (stun, disable counters)

## Enemy Types

| Type | Max HP | Move Speed | Special Ability |
|------|--------|------------|-----------------|
| Virus | 2 | 2 cells/turn | Fastest movement |
| Daemon | 3 | 1 cell/turn | Most HP |
| Glitch | 2 | 1 cell/turn | Can move onto blocks |
| Cryptog | 2 | 1 cell/turn | Invisible unless in same row/col as player |

### Enemy Visibility (Cryptog)
- Visible if in same row OR column as player
- Visible if `showActivated == true` (SHOW program)
- Otherwise invisible (not in observation)

### Enemy Movement
- Enemies use pathfinding to move toward player
- Viruses move twice per turn (speed = 2)
- Glitches can path through blocks
- Stunned enemies don't move or attack
- Disabled enemies don't move or attack

## Block Types

### Data Block
- **Awards**: Points (score). can be from 1 to 9, randomly
- **Spawns**: Transmissions (equal to points - INVARIANT)
- **Invariant**: `points == spawnCount` always

### Program Block
- **Awards**: Program ownership
- **Spawns**: Transmissions

### Question Block
- Randomly resolves to data or program when siphoned

### Block States
- **Unsiphoned**: Can be siphoned for rewards
- **Siphoned**: Already siphoned, still blocks movement (except HACK destroys siphoned blocks)

## Transmissions

- Spawn from siphoning blocks
- Count down each turn
- At turnsRemaining == 0, spawn enemy of `enemyType`
- Can be destroyed by attacking them

## Programs (23 total)

### Program Costs and Applicability

| Program | Index | Credits | Energy | Applicability Condition |
|---------|-------|---------|--------|-------------------------|
| PUSH | 5 | 0 | 2 | Enemies exist |
| PULL | 6 | 0 | 2 | Enemies exist |
| CRASH | 7 | 3 | 2 | Blocks/enemies/transmissions in 8 surrounding cells |
| WARP | 8 | 2 | 2 | Enemies OR transmissions exist |
| POLY | 9 | 1 | 1 | Enemies exist |
| WAIT | 10 | 0 | 1 | Always |
| DEBUG | 11 | 3 | 0 | Enemies on blocks exist |
| ROW | 12 | 3 | 1 | Enemies in player's row |
| COL | 13 | 3 | 1 | Enemies in player's column |
| UNDO | 14 | 1 | 0 | Game history not empty |
| STEP | 15 | 0 | 3 | Always |
| SIPH+ | 16 | 5 | 0 | Always |
| EXCH | 17 | 4 | 0 | Player has >= 4 credits |
| SHOW | 18 | 2 | 0 | `showActivated == false` |
| RESET | 19 | 0 | 4 | Player HP < 3 |
| CALM | 20 | 2 | 4 | `scheduledTasksDisabled == false` |
| D_BOM | 21 | 3 | 0 | Daemon enemy exists |
| DELAY | 22 | 1 | 2 | Transmissions exist |
| ANTI-V | 23 | 3 | 0 | Virus enemy exists |
| SCORE | 24 | 0 | 5 | Stage < 8 (not last stage) |
| REDUC | 25 | 2 | 1 | Unsiphoned blocks with spawnCount > 0 |
| ATK+ | 26 | 4 | 4 | Not used this stage AND attackDamage < 2 |
| HACK | 27 | 2 | 2 | Siphoned cells exist |

### Program Effects

#### PUSH (5)
- Pushes all enemies 1 cell away from player
- Does NOT end turn

#### PULL (6)
- Pulls all enemies 1 cell toward player
- Does NOT end turn

#### CRASH (7)
- Destroys everything in 8 surrounding cells
- Kills enemies, destroys transmissions
- **Destroys ALL blocks** (siphoned AND unsiphoned)
- **Exposes resources** under destroyed blocks
- Does NOT end turn

#### WARP (8)
- Teleports player TO a random enemy or transmission position
- **Kills the target** (enemy or destroys transmission)
- **Triggers stage completion** if target is at exit position
- Does NOT end turn

#### POLY (9)
- Changes all enemies to a DIFFERENT type (guaranteed different)
- Enemy HP resets to new type's maxHP
- Does NOT end turn

#### WAIT (10)
- **ONLY program that ends turn**
- Triggers enemy turn

#### DEBUG (11)
- Damages enemies standing on blocks
- **Stuns surviving enemies**
- Does NOT end turn

#### ROW (12)
- Damages all enemies in player's row
- **Stuns surviving enemies**
- Does NOT end turn

#### COL (13)
- Damages all enemies in player's column
- **Stuns surviving enemies**
- Does NOT end turn

#### UNDO (14)
- Restores game state to before previous turn
- **Restores both player AND enemy positions**
- Does NOT end turn

#### STEP (15)
- Next turn, enemies don't move
- Does NOT end turn

#### SIPH+ (16)
- Grants 1 data siphon
- Does NOT end turn

#### EXCH (17)
- Costs 4 credits, grants 4 energy
- Does NOT end turn

#### SHOW (18)
- Reveals all Cryptogs (sets `showActivated = true`)
- **Makes transmissions show their incoming enemy type**
- Does NOT end turn

#### RESET (19)
- Restores player HP to 3
- Does NOT end turn

#### CALM (20)
- Disables scheduled spawns (sets `scheduledTasksDisabled = true`)
- Does NOT end turn

#### D_BOM (21)
- Destroys nearest Daemon
- **Damages AND stuns enemies** in 8 cells around the daemon
- Does NOT end turn

#### DELAY (22)
- Adds 3 turns to all transmissions
- Does NOT end turn

#### ANTI-V (23)
- Damages all Viruses
- **Stuns surviving viruses**
- Does NOT end turn

#### SCORE (24)
- Gains points equal to stages remaining (8 - current stage)
- Does NOT end turn

#### REDUC (25)
- Reduces spawnCount of all unsiphoned blocks by 1
- Does NOT end turn

#### ATK+ (26)
- Increases player attackDamage to 2
- Can only be used once per stage
- Does NOT end turn

#### HACK (27)
- Damages enemies on siphoned cells
- **Stuns surviving enemies on siphoned cells**
- **Destroys siphoned blocks**, making cells traversable
- **Exposes resources** under destroyed blocks
- Does NOT end turn

## Reward System

### Components

| Component | Formula | Notes |
|-----------|---------|-------|
| Stage completion | [1, 2, 4, 8, 16, 32, 64, 100] | By stage number |
| Score gain | scoreDelta × 0.5 | Per point gained |
| Kill | kills × 0.3 | Per enemy killed |
| Data siphon collected | 1.0 | Flat per siphon |
| Distance shaping | distDelta × 0.05 | Per cell closer to exit |
| Victory bonus | 500 + score × 100 | On game win |
| Death penalty | -cumulativeRewards × 0.5 | Based on stages completed |
| Resource gain | (credits + energy) × 0.05 | Per resource gained |
| Resource holding | (credits + energy) × 0.01 | On stage completion |
| Damage penalty | hpLost × -1.0 | Per HP lost |
| HP recovery | hpGained × 1.0 | Per HP recovered |
| Siphon quality | -0.5 × missedValue | Suboptimal siphon penalty |
| Program waste | -0.3 | RESET at 2 HP |
| Siphon-caused death | -10.0 | Extra death penalty |

### Stage Completion Rewards
| Stage | Reward |
|-------|--------|
| 1 | 1.0 |
| 2 | 2.0 |
| 3 | 4.0 |
| 4 | 8.0 |
| 5 | 16.0 |
| 6 | 32.0 |
| 7 | 64.0 |
| 8 | 100.0 |

## Observation Space

### Player State (10 values, normalized)
```
[row, col, hp, credits, energy, stage, dataSiphons, baseAttack, showActivated, scheduledTasksDisabled]
```

### Programs (23 values)
Binary int32 vector indicating owned programs (1 = owned, 0 = not owned).

### Grid (6×6×42)
Each cell has 42 features encoding:
- Cell type (empty, block, etc.)
- Enemy presence and type
- Transmission presence
- Resources
- Special markers (exit, siphoned, etc.)

## Action Masking

An action is valid (not masked) if:

### Movement (0-3)
- Target cell is within grid bounds
- Target cell is not blocked by a block
- OR target has enemy/transmission in line-of-sight (triggers attack instead)

### Siphon (4)
- Player has dataSiphons > 0

### Programs (5-27)
1. Player owns the program
2. Player has sufficient credits
3. Player has sufficient energy
4. Program-specific applicability condition is met (see table above)

## Stage Transitions

### Stage Completion Triggers
- Player **moves** to exit position, OR
- Player uses **WARP** to teleport to an enemy/transmission at exit position

### Stage Completion Effects
1. Current stage increments
2. **Player position preserved** (stays at exit position)
3. **HP gains +1** (up to max 3, NOT reset to max)
4. Player stats (credits, energy, score) preserved
5. New stage is generated with:
   - New exit at random corner (different from player's position)
   - Data siphons at remaining 2 corners
   - 5-11 blocks placed randomly
   - Resources on empty cells
   - Transmissions spawned based on stage number

### Victory Condition
- Complete stage 8
- Stage becomes 9 (indicates victory)
- Game ends (`done = true`)

### Data Block Invariant
After stage generation, all data blocks must satisfy:
```
block.points == block.spawnCount
```

## Stage Generation

When a new stage begins, the following content is generated:

### Corner Placement
- **Player**: Stays at current position (the old exit)
- **Exit**: Random corner different from player
- **Data Siphons**: Placed at remaining 2 corners

### Block Placement
- **Count**: 5-11 blocks randomly
- **Positions**: Non-corner cells only, avoiding enemies/transmissions
- **Types**: 50% data blocks, 50% program blocks
- **Data blocks**: Points 1-9 (random), spawnCount == points
- **Program blocks**: Random program type, spawnCount = 2

### Resource Placement
- Placed on empty cells (not blocks, not corners)
- **Amount distribution**: 45% → 1, 45% → 2, 10% → 3
- **Type**: 50% credits, 50% energy

### Transmission Spawning
| Stage | Transmissions |
|-------|---------------|
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |

- Spawned at empty cells, preferring positions out of player's line of fire
- Enemy type: 25% each (virus, daemon, glitch, cryptog)
- Turns remaining: 1 (spawns as enemy next turn)

## Non-Deterministic Elements

### Random Elements
- **Exit position**: Random corner different from player (on stage transition)
- **Enemy type selection** for POLY
- **Target selection** for WARP (random from enemies/transmissions)
- **Block placement**: Count (5-11), positions, types (data/program)
- **Block contents**: Data points (1-9), program types
- **Resource placement**: Amount (1-3), type (credits/energy)
- **Transmission spawning**: Positions, enemy types

### Handling in Tests
- Use deterministic assertions (verify one of valid outcomes)
- For WARP with multiple targets: verify player is at ONE of the target positions
- For stage generation: test deterministic properties:
  - Player position preserved
  - HP gains +1 (up to max)
  - Exit at a corner different from player
  - Data siphons at 2 corners
  - Block count between 5-11
  - Transmission count matches stage number
