import Foundation

// Simplified wrapper for ML training - no UI, instant turn processing
class HeadlessGame {
    var gameState: GameState  // Internal for observation building

    init() {
        self.gameState = GameState()
    }

    // Reset to new game
    func reset() -> GameObservation {
        gameState = GameState()
        return ObservationBuilder.build(from: gameState)
    }

    /// Build info dict from ActionResult - used by both HeadlessGame and VisualGameController
    static func buildInfoDict(from result: GameState.ActionResult) -> [String: Any] {
        var info: [String: Any] = [:]

        if !result.success {
            info["invalid_action"] = true
        }
        if result.stageAdvanced {
            info["stage_complete"] = true
        }
        if result.playerDied {
            info["death"] = true
        }

        info["reward_breakdown"] = [
            "stage": result.rewardBreakdown.stageCompletion,
            "score": result.rewardBreakdown.scoreGain,
            "kills": result.rewardBreakdown.kills,
            "dataSiphon": result.rewardBreakdown.dataSiphonCollected,
            "distance": result.rewardBreakdown.distanceShaping,
            "victory": result.rewardBreakdown.victory,
            "death": result.rewardBreakdown.deathPenalty,
            "resourceGain": result.rewardBreakdown.resourceGain,
            "resourceHolding": result.rewardBreakdown.resourceHolding,
            "damagePenalty": result.rewardBreakdown.damagePenalty,
            "hpRecovery": result.rewardBreakdown.hpRecovery,
            "siphonQuality": result.rewardBreakdown.siphonQuality,
            "programWaste": result.rewardBreakdown.programWaste,
            "siphonDeathPenalty": result.rewardBreakdown.siphonDeathPenalty
        ]

        return info
    }

    // Execute action and advance game state (including enemy turn)
    // Returns: (observation, reward, isDone, info)
    func step(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let action = GameAction.fromIndex(actionIndex) else {
            fatalError("Invalid action index: \(actionIndex)")
        }

        // Process action (handles player action + enemy turn)
        let result: GameState.ActionResult = gameState.tryExecuteAction(action)

        // Build info dict using shared helper
        let info = HeadlessGame.buildInfoDict(from: result)

        // Determine if episode is done
        var isDone = false
        if !result.success {
            isDone = true
            infoLog("HeadlessGame", "❌ Invalid action \(action) attempted - terminating episode")
        }
        if result.stageAdvanced {
            isDone = result.gameWon
            if isDone {
                infoLog("Completed the game! With points: \(gameState.player.score)")
            }
        }
        if result.playerDied {
            isDone = true
        }

        let observation = ObservationBuilder.build(from: gameState)
        let reward = result.rewardBreakdown.total

        if result.stageAdvanced {
            debugLog("Advanced to stage \(observation.stage)")
        } else {
            debugLog("Step \(String(describing: action)) -> reward: \(String(format: "%.3f", reward)), done: \(isDone), stage: \(observation.stage), credits: \(gameState.player.credits), energy: \(gameState.player.energy)")
        }

        return (observation, reward, isDone, info)
    }

    // Get valid actions based on current state
    func getValidActions() -> [GameAction] {
        let actions = gameState.getValidActions()
        let indices = actions.map { $0.toIndex() }
        debugLog(
            "HeadlessGame",
            "Valid actions: \(actions.map { String(describing: $0) }) → indices: \(indices)")
        return actions
    }

    // MARK: - Set State (for testing)

    /// Set the game to a specific state for deterministic testing.
    /// This allows tests to set up precise preconditions before executing actions.
    func setState(stateData: SetStateData) -> GameObservation {
        // Create fresh game state
        gameState = GameState()

        // Clear randomly generated content
        gameState.grid = Grid()
        gameState.enemies = []
        gameState.transmissions = []
        gameState.gameHistory = []

        // Set player state
        gameState.player = Player(row: stateData.player.row, col: stateData.player.col)
        if let hp = stateData.player.hp {
            gameState.player.health = PlayerHealth(rawValue: hp) ?? .full
        }
        if let credits = stateData.player.credits {
            gameState.player.credits = credits
        }
        if let energy = stateData.player.energy {
            gameState.player.energy = energy
        }
        if let dataSiphons = stateData.player.dataSiphons {
            gameState.player.dataSiphons = dataSiphons
        }
        if let attackDamage = stateData.player.attackDamage {
            gameState.player.attackDamage = attackDamage
        }
        if let score = stateData.player.score {
            gameState.player.score = score
        }

        // Set game state flags
        if let stage = stateData.stage {
            gameState.currentStage = stage
        }
        if let turn = stateData.turn {
            gameState.turnCount = turn
        }
        if let showActivated = stateData.showActivated {
            gameState.showActivated = showActivated
        }
        if let scheduledTasksDisabled = stateData.scheduledTasksDisabled {
            gameState.scheduledTasksDisabled = scheduledTasksDisabled
        }

        // Set owned programs (convert action indices to ProgramTypes)
        if let programIndices = stateData.ownedPrograms {
            gameState.ownedPrograms = programIndices.compactMap { index in
                ProgramType.allCases.first { $0.actionIndex == index }
            }
        }

        // Set enemies
        if let enemies = stateData.enemies {
            for enemyData in enemies {
                guard let enemyType = parseEnemyType(enemyData.type) else { continue }
                let enemy = Enemy(type: enemyType, row: enemyData.row, col: enemyData.col)
                enemy.hp = enemyData.hp
                if let stunned = enemyData.stunned, stunned {
                    enemy.isStunned = true
                }
                gameState.enemies.append(enemy)
            }
        }

        // Set transmissions
        if let transmissions = stateData.transmissions {
            for transData in transmissions {
                guard let enemyType = parseEnemyType(transData.enemyType) else { continue }
                let transmission = Transmission(
                    row: transData.row,
                    col: transData.col,
                    turnsUntilSpawn: transData.turnsRemaining,
                    enemyType: enemyType
                )
                gameState.transmissions.append(transmission)
            }
        }

        // Set blocks
        if let blocks = stateData.blocks {
            for blockData in blocks {
                guard gameState.grid.isValidPosition(row: blockData.row, col: blockData.col) else { continue }
                let cell = gameState.grid.cells[blockData.row][blockData.col]

                if blockData.type == "data" {
                    let points = blockData.points ?? 5
                    let spawnCount = blockData.spawnCount ?? points  // Default: spawnCount == points (invariant)
                    cell.content = .block(.data(points: points, transmissionSpawn: spawnCount))
                } else if blockData.type == "program" {
                    if let programTypeName = blockData.programType,
                       let programType = parseProgramType(programTypeName) {
                        cell.content = .block(.program(Program(type: programType), transmissionSpawn: blockData.spawnCount ?? 2))
                    }
                }

                if let siphoned = blockData.siphoned, siphoned {
                    cell.isSiphoned = true
                }
            }
        }

        // Set resources
        if let resources = stateData.resources {
            for resourceData in resources {
                guard gameState.grid.isValidPosition(row: resourceData.row, col: resourceData.col) else { continue }
                let cell = gameState.grid.cells[resourceData.row][resourceData.col]

                if let dataSiphon = resourceData.dataSiphon, dataSiphon {
                    cell.content = .dataSiphon
                } else if let credits = resourceData.credits, credits > 0 {
                    cell.resources = .credits(credits)
                } else if let energy = resourceData.energy, energy > 0 {
                    cell.resources = .energy(energy)
                }
            }
        }

        // Set exit position (default to corner opposite player if not set)
        // Find a valid exit position
        var exitSet = false
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                if gameState.grid.cells[row][col].isExit {
                    gameState.exitPosition = (row: row, col: col)
                    exitSet = true
                    break
                }
            }
            if exitSet { break }
        }

        // If no exit was set, place one at default position (5,5)
        if !exitSet {
            gameState.grid.cells[5][5].content = .exit
            gameState.exitPosition = (row: 5, col: 5)
        }

        infoLog("HeadlessGame", "setState: player at (\(stateData.player.row), \(stateData.player.col)), \(gameState.enemies.count) enemies, \(gameState.transmissions.count) transmissions, stage \(gameState.currentStage)")

        return ObservationBuilder.build(from: gameState)
    }

    // MARK: - Helper Functions

    private func parseEnemyType(_ typeString: String) -> EnemyType? {
        switch typeString.lowercased() {
        case "virus": return .virus
        case "daemon": return .daemon
        case "glitch": return .glitch
        case "cryptog": return .cryptog
        default: return nil
        }
    }

    private func parseProgramType(_ typeString: String) -> ProgramType? {
        // Try direct rawValue match first
        if let type = ProgramType(rawValue: typeString.lowercased()) {
            return type
        }
        // Handle common variations
        switch typeString.lowercased() {
        case "push": return .push
        case "pull": return .pull
        case "crash": return .crash
        case "warp": return .warp
        case "poly": return .poly
        case "wait": return .wait
        case "debug": return .debug
        case "row": return .row
        case "col": return .col
        case "undo": return .undo
        case "step": return .step
        case "siph+", "siphplus", "siph_plus": return .siphPlus
        case "exch": return .exch
        case "show": return .show
        case "reset": return .reset
        case "calm": return .calm
        case "d_bom", "dbom", "d_bomb": return .dBomb
        case "delay": return .delay
        case "anti-v", "antiv", "anti_v": return .antiV
        case "score": return .score
        case "reduc": return .reduc
        case "atk+", "atkplus", "atk_plus": return .atkPlus
        case "hack": return .hack
        default: return nil
        }
    }
}
