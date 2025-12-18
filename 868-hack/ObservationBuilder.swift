import Foundation

// Builds GameObservation from GameState for ML training
class ObservationBuilder {
    static func build(from gameState: GameState) -> GameObservation {
        var cells: [[CellObservation]] = []
        var cryptogHints: [(Int, Int)] = []

        for row in 0..<6 {
            var rowCells: [CellObservation] = []
            for col in 0..<6 {
                let cell = observeCell(
                    row: row,
                    col: col,
                    gameState: gameState,
                    cryptogHints: &cryptogHints
                )
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
            showActivated: gameState.showActivated
        )
    }

    private static func observeCell(
        row: Int,
        col: Int,
        gameState: GameState,
        cryptogHints: inout [(Int, Int)]
    ) -> CellObservation {
        let cell = gameState.grid.cells[row][col]

        // Check for enemy at this position
        var enemyObs: EnemyObservation? = nil
        if let enemy = gameState.enemies.first(where: { $0.row == row && $0.col == col && $0.hp > 0 }) {
            // Check visibility rules
            if enemy.type == .cryptog {
                // Cryptog only visible if in same row/col OR revealed
                if (row == gameState.player.row || col == gameState.player.col) || gameState.showActivated {
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
                enemyType: gameState.showActivated ? enemyTypeToString(transmission.enemyType) : nil
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

    private static func enemyTypeToString(_ type: EnemyType) -> String {
        switch type {
        case .virus: return "virus"
        case .daemon: return "daemon"
        case .glitch: return "glitch"
        case .cryptog: return "cryptog"
        }
    }
}
