import Foundation

// MARK: - Observation Space
// Data structures for representing game state observations (used for ML training)

struct GameObservation {
    let playerRow: Int
    let playerCol: Int
    let playerHP: Int
    let credits: Int
    let energy: Int
    let stage: Int
    let turn: Int
    let dataSiphons: Int
    let baseAttack: Int
    let score: Int

    let cells: [[CellObservation]]
    let cryptogHints: [(row: Int, col: Int)]  // Last known positions for purple borders

    let showActivated: Bool
    let scheduledTasksDisabled: Bool
    let ownedPrograms: [Int]  // Action indices of owned programs (5-30)
}

struct CellObservation {
    let row: Int
    let col: Int

    let enemy: EnemyObservation?
    let block: BlockObservation?
    let transmission: TransmissionObservation?

    // Resources (integer quantities)
    // Visible if: no block OR any block is siphoned
    let credits: Int?
    let energy: Int?

    // Special cell types
    let isDataSiphon: Bool
    let isExit: Bool
}

struct EnemyObservation {
    let type: String  // EnemyType as string
    let hp: Int
    let isStunned: Bool
}

struct BlockObservation {
    let blockType: String  // "data", "program", or "question"
    let isSiphoned: Bool

    // Visibility rules:
    // - Data blocks: always visible (regardless of isSiphoned)
    // - Program blocks: always visible (regardless of isSiphoned)
    // - Question blocks: only visible if isSiphoned
    let points: Int?
    let programType: String?  // Program name (for debugging)
    let programActionIndex: Int?  // Explicit action index (5-27) for ML training
    let transmissionSpawnCount: Int?
}

struct TransmissionObservation {
    let turnsUntilSpawn: Int
    let enemyType: String?  // Only if showActivated, EnemyType rawValue
}
