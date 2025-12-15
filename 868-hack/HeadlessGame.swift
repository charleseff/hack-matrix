import Foundation

// Simplified wrapper for ML training - no UI, instant turn processing
class HeadlessGame {
    private var gameState: GameState

    init() {
        self.gameState = GameState()
    }

    // Reset to new game
    func reset() -> GameObservation {
        gameState = GameState()
        return getObservation()
    }

    // Execute action and advance game state
    // Returns: (observation, reward, isDone, info)
    func step(action: GameAction) -> (GameObservation, Double, Bool, [String: Any]) {
        let oldScore = gameState.player.score
        let oldStage = gameState.currentStage

        var isDone = false
        var info: [String: Any] = [:]

        // Execute the action
        switch action {
        case .move(let direction):
            let result = gameState.tryMove(direction: direction)
            if result.exitReached {
                // Reached exit - advance to next stage
                let continues = gameState.completeStage()
                isDone = !continues  // Game ends on victory
                info["stage_complete"] = true
            }

        case .siphon:
            _ = gameState.performSiphon()

        case .program(let programType):
            if gameState.canExecuteProgram(programType).canExecute {
                _ = gameState.executeProgram(programType)
            } else {
                // Invalid action - penalize slightly
                info["invalid_action"] = true
            }
        }

        // Check if player died
        if gameState.player.health == .dead {
            isDone = true
            info["death"] = true
        }

        // Calculate reward
        // Philosophy: Only winning (stage 8 completion) matters, but provide
        // small score signals during play to help agent learn
        let scoreDelta = Double(gameState.player.score - oldScore)

        var reward = scoreDelta * 0.01  // Small reward for gaining points during play

        // Episode end rewards
        if isDone {
            if gameState.player.health == .dead {
                // Death: No reward (any points accumulated don't matter)
                reward = 0.0
            } else {
                // Victory (completed stage 8): BIG reward based on final score
                // If isDone && not dead, player must have won
                reward = Double(gameState.player.score) * 10.0
            }
        }

        let observation = getObservation()
        return (observation, reward, isDone, info)
    }

    // Get valid actions based on current state
    func getValidActions() -> [GameAction] {
        var actions: [GameAction] = []

        // Movement actions - only if not blocked by edge or obstacle
        let playerRow = gameState.player.row
        let playerCol = gameState.player.col

        // Check each direction
        if playerRow > 0 { actions.append(.move(.up)) }
        if playerRow < 5 { actions.append(.move(.down)) }
        if playerCol > 0 { actions.append(.move(.left)) }
        if playerCol < 5 { actions.append(.move(.right)) }

        // Siphon - only if player has data siphons available
        if gameState.player.dataSiphons > 0 {
            actions.append(.siphon)
        }

        // Programs - use canExecuteProgram which checks ownership, resources, and applicability
        for programType in ProgramType.allCases {
            if gameState.canExecuteProgram(programType).canExecute {
                actions.append(.program(programType))
            }
        }

        return actions
    }

    // Get observation respecting partial observability
    func getObservation() -> GameObservation {
        var cells: [[CellObservation]] = []
        var cryptogHints: [(Int, Int)] = []

        for row in 0..<6 {
            var rowCells: [CellObservation] = []
            for col in 0..<6 {
                let cell = observeCell(row: row, col: col, cryptogHints: &cryptogHints)
                rowCells.append(cell)
            }
            cells.append(rowCells)
        }

        return GameObservation(
            playerRow: gameState.player.row,
            playerCol: gameState.player.col,
            playerHP: gameState.player.health.rawValue,
            credits: gameState.player.credits,
            energy: gameState.player.energy,
            stage: gameState.currentStage,
            turn: gameState.turnCount,
            dataSiphons: gameState.player.dataSiphons,
            baseAttack: gameState.player.attackDamage,
            cells: cells,
            cryptogHints: cryptogHints,
            cryptogsRevealed: gameState.cryptogsRevealed,
            transmissionsRevealed: gameState.transmissionsRevealed
        )
    }

    private func observeCell(row: Int, col: Int, cryptogHints: inout [(Int, Int)]) -> CellObservation {
        let cell = gameState.grid.cells[row][col]

        // Check for enemy at this position
        var enemyObs: EnemyObservation? = nil
        if let enemy = gameState.enemies.first(where: { $0.row == row && $0.col == col && $0.hp > 0 }) {
            // Check visibility rules
            if enemy.type == .cryptog {
                // Cryptog only visible if in same row/col OR revealed
                if (row == gameState.player.row || col == gameState.player.col) || gameState.cryptogsRevealed {
                    enemyObs = EnemyObservation(
                        type: enemyTypeToString(enemy.type),
                        hp: enemy.hp,
                        isStunned: enemy.isStunned
                    )
                } else if let lastRow = enemy.lastKnownRow, let lastCol = enemy.lastKnownCol {
                    // Not visible, but add hint for purple border
                    cryptogHints.append((lastRow, lastCol))
                }
            } else {
                // Non-cryptog enemies are always visible
                enemyObs = EnemyObservation(
                    type: enemyTypeToString(enemy.type),
                    hp: enemy.hp,
                    isStunned: enemy.isStunned
                )
            }
        }

        // Check for block at this position
        var blockObs: BlockObservation? = nil
        if case .block(let blockType) = cell.content {
            switch blockType {
            case .data(let points, let transmissionSpawn):
                // Data blocks: properties ALWAYS visible (regardless of siphoning)
                blockObs = BlockObservation(
                    blockType: "data",
                    isSiphoned: cell.isSiphoned,
                    points: points,
                    programType: nil,
                    transmissionSpawnCount: transmissionSpawn
                )

            case .program(let program, let transmissionSpawn):
                // Program blocks: properties ALWAYS visible (regardless of siphoning)
                blockObs = BlockObservation(
                    blockType: "program",
                    isSiphoned: cell.isSiphoned,
                    points: 0,  // Programs don't give points
                    programType: program.type.rawValue,
                    transmissionSpawnCount: transmissionSpawn
                )

            case .question(_, let points, let program, let transmissionSpawn):
                // Question blocks: properties only visible if siphoned
                blockObs = BlockObservation(
                    blockType: "question",
                    isSiphoned: cell.isSiphoned,
                    points: cell.isSiphoned ? points : nil,
                    programType: cell.isSiphoned ? program?.type.rawValue : nil,
                    transmissionSpawnCount: cell.isSiphoned ? transmissionSpawn : nil
                )
            }
        }

        // Check for transmission at this position
        var transmissionObs: TransmissionObservation? = nil
        if let transmission = gameState.transmissions.first(where: { $0.row == row && $0.col == col }) {
            // Extract turnsRemaining from state enum
            var turnsRemaining = 0
            if case .spawning(let turns) = transmission.state {
                turnsRemaining = turns
            }

            transmissionObs = TransmissionObservation(
                turnsUntilSpawn: turnsRemaining,
                // Only reveal enemy type if transmissions revealed
                enemyType: gameState.transmissionsRevealed ? enemyTypeToString(transmission.enemyType) : nil
            )
        }

        // Resources: integer quantities visible if no block OR if any block is siphoned
        var credits: Int? = nil
        var energy: Int? = nil
        let canSeeResources = blockObs == nil || cell.isSiphoned

        if canSeeResources {
            switch cell.resources {
            case .credits(let amount):
                credits = amount
            case .energy(let amount):
                energy = amount
            case .none:
                break
            }
        }

        // Check for special cell content
        var isDataSiphon = false
        var isExit = false
        switch cell.content {
        case .dataSiphon:
            isDataSiphon = true
        case .exit:
            isExit = true
        default:
            break
        }

        return CellObservation(
            row: row,
            col: col,
            enemy: enemyObs,
            block: blockObs,
            transmission: transmissionObs,
            credits: credits,
            energy: energy,
            isDataSiphon: isDataSiphon,
            isExit: isExit
        )
    }

    private func enemyTypeToString(_ type: EnemyType) -> String {
        switch type {
        case .virus: return "virus"
        case .daemon: return "daemon"
        case .glitch: return "glitch"
        case .cryptog: return "cryptog"
        }
    }
}

// MARK: - Action Space

enum GameAction: Equatable, Hashable {
    case move(Direction)
    case siphon
    case program(ProgramType)

    // Convert to integer index for ML (0-30)
    func toIndex() -> Int {
        switch self {
        case .move(.up): return 0
        case .move(.down): return 1
        case .move(.left): return 2
        case .move(.right): return 3
        case .siphon: return 4
        case .program(let type):
            // Programs indexed 5-30 (26 programs)
            let programIndex = ProgramType.allCases.firstIndex(of: type) ?? 0
            return 5 + programIndex
        }
    }

    // Convert from integer index
    static func fromIndex(_ index: Int) -> GameAction? {
        switch index {
        case 0: return .move(.up)
        case 1: return .move(.down)
        case 2: return .move(.left)
        case 3: return .move(.right)
        case 4: return .siphon
        case 5...30:
            let programIndex = index - 5
            guard programIndex < ProgramType.allCases.count else { return nil }
            return .program(ProgramType.allCases[programIndex])
        default: return nil
        }
    }
}

// MARK: - Observation Space

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

    let cells: [[CellObservation]]
    let cryptogHints: [(row: Int, col: Int)]  // Last known positions for purple borders

    let cryptogsRevealed: Bool
    let transmissionsRevealed: Bool
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
    let programType: String?
    let transmissionSpawnCount: Int?
}

struct TransmissionObservation {
    let turnsUntilSpawn: Int
    let enemyType: String?  // Only if transmissionsRevealed, EnemyType rawValue
}
