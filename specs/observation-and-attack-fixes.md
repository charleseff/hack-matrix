# Observation and Attack Fixes Spec

**Status:** Complete

## Goal

Fix three issues before proceeding with test reorganization:

1. Add `siphonCenter` to observation space (currently hidden but should be visible)
2. Add `spawnedFromSiphon` to enemy observation space (player should track which enemies spawned from siphoning)
3. Allow ATK+ program to be used twice per stage (attack: 1→2→3)

## Background

These are small game logic fixes that should be done before the test reorganization spec, as they affect what tests need to verify.

## Phase 1: Add siphonCenter to Observation Space

### Current State

- `siphonCenter` is tracked per cell in `Grid.swift` (line 44)
- Set to `true` when player siphons at that cell (`GameState.swift:791`)
- Used visually to show a smiley emoji (`GameScene.swift:214`)
- **NOT** included in `CellObservation` - only `isSiphoned` is exposed

### Why It Should Be Visible

The `hack` program deals damage to enemies based on siphon positions. Players need to see where they've siphoned to plan hack usage effectively. Currently this information is hidden from the observation space.

### Implementation

1. **Add to `CellObservation`** in `Observation.swift`:
   ```swift
   struct CellObservation: Codable {
       // ... existing fields ...
       let siphonCenter: Bool  // True if player siphoned from this cell
   }
   ```

2. **Update `ObservationBuilder.swift`** to include the field:
   ```swift
   return CellObservation(
       // ... existing fields ...
       siphonCenter: cell.siphonCenter
   )
   ```

3. **Update Python `hack_env.py`** to parse the new field (if needed for grid encoding)

4. **Add parity test** to verify `siphonCenter` is visible after siphoning

## Phase 2: Add spawnedFromSiphon to Enemy Observation Space

### Current State

- `spawnedFromSiphon` is tracked per enemy in `Enemy.swift` (line 57)
- Set to `true` when enemy spawns from siphoning a block
- Causes significant negative reward if player dies from such an enemy
- **NOT** included in `EnemyObservation` - player cannot see which enemies are siphon-spawned

### Why It Should Be Visible

Players need to track which enemies spawned from their siphoning to avoid dying to them (huge negative reward). Currently this information is hidden, making it impossible for the agent to learn to avoid these enemies specifically.

### Implementation

1. **Add to `EnemyObservation`** in `Observation.swift`:
   ```swift
   struct EnemyObservation: Codable {
       let type: String
       let hp: Int
       let isStunned: Bool
       let spawnedFromSiphon: Bool  // NEW: true if spawned from siphoning
   }
   ```

2. **Update `ObservationBuilder.swift`** to include the field:
   ```swift
   enemyObs = EnemyObservation(
       type: enemyTypeToString(enemy.type),
       hp: enemy.hp,
       isStunned: enemy.isStunned,
       spawnedFromSiphon: enemy.spawnedFromSiphon  // NEW
   )
   ```

3. **Update Python `hack_env.py`** grid encoding to include the new field

4. **Add parity test** to verify `spawnedFromSiphon` is visible on enemies

## Phase 3: Allow ATK+ Program to be Used Twice Per Stage

### Current State

- `atkPlusUsedThisStage` is a boolean flag in `GameState`
- Once ATK+ is used, it cannot be used again that stage
- Attack damage stays at 2 for the rest of the stage

### Desired Behavior

- ATK+ can be used **twice** per stage maximum
- First use: attack damage 1→2
- Second use: attack damage 2→3
- Third+ use: blocked (program not valid)

### Implementation

1. **Change `atkPlusUsedThisStage` from `Bool` to `Int`** in `GameState.swift`:
   ```swift
   var atkPlusUsesThisStage: Int = 0  // Renamed, tracks count
   ```

2. **Update ATK+ execution logic** to allow up to 2 uses:
   ```swift
   // In program execution
   guard atkPlusUsesThisStage < 2 else { return false }
   player.attackDamage += 1
   atkPlusUsesThisStage += 1
   ```

3. **Update stage reset** to reset counter to 0

4. **Update action mask** to block ATK+ when `atkPlusUsesThisStage >= 2`

5. **Add test** to verify ATK+ can be used twice but not three times

## Phase 4: Extend set_state for Enemy Flags

For testing `spawnedFromSiphon` and `isFromScheduledTask` reward logic, we need to be able to set these flags via `set_state`.

### Implementation

1. **Update `SetStateEnemy`** in `GameCommandProtocol.swift`:
   ```swift
   struct SetStateEnemy: Codable {
       let type: String
       let row: Int
       let col: Int
       let hp: Int
       let stunned: Bool?
       let spawnedFromSiphon: Bool?      // NEW
       let isFromScheduledTask: Bool?    // NEW
   }
   ```

2. **Update `HeadlessGame.swift`** to use these flags when creating enemies:
   ```swift
   let enemy = Enemy(
       type: enemyType,
       row: enemyData.row,
       col: enemyData.col,
       isFromScheduledTask: enemyData.isFromScheduledTask ?? false,
       spawnedFromSiphon: enemyData.spawnedFromSiphon ?? false
   )
   ```

3. **Update Python test helpers** to support the new fields

## Success Criteria

1. [ ] `siphonCenter` visible in cell observation space after player siphons
2. [ ] `spawnedFromSiphon` visible in enemy observation space
3. [ ] ATK+ program can be used twice per stage (attack: 1→2→3)
4. [ ] ATK+ blocked on third attempt in same stage
5. [ ] `set_state` supports `spawnedFromSiphon` and `isFromScheduledTask` on enemies
6. [ ] All existing tests pass

## Dependencies

- None (this spec should be completed before `test-reorganization.md`)

## References

- [test-reorganization.md](./test-reorganization.md) - Depends on this spec
- [game-mechanics.md](./game-mechanics.md) - May need updates for ATK+ change
