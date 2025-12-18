import Foundation

// MARK: - Action Space

enum GameAction: Equatable, Hashable {
    case direction(Direction)
    case siphon
    case program(ProgramType)

    // Convert to integer index for ML (0-30)
    func toIndex() -> Int {
        switch self {
        case .direction(.up): return 0
        case .direction(.down): return 1
        case .direction(.left): return 2
        case .direction(.right): return 3
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
        case 0: return .direction(.up)
        case 1: return .direction(.down)
        case 2: return .direction(.left)
        case 3: return .direction(.right)
        case 4: return .siphon
        case 5...30:
            let programIndex = index - 5
            guard programIndex < ProgramType.allCases.count else { return nil }
            return .program(ProgramType.allCases[programIndex])
        default: return nil
        }
    }
}

class GameState {
    var player: Player
    var grid: Grid
    var enemies: [Enemy]
    var transmissions: [Transmission]
    var currentStage: Int
    var turnCount: Int
    var ownedPrograms: [ProgramType]  // Array to preserve acquisition order
    var showActivated: Bool
    var scheduledTasksDisabled: Bool
    var stepActive: Bool
    var gameHistory: [GameStateSnapshot]
    var pendingSiphonTransmissions: Int
    var atkPlusUsedThisStage: Bool

    init() {
        self.grid = Grid()
        self.currentStage = 1
        self.turnCount = 0
        self.enemies = []
        self.transmissions = []
        self.ownedPrograms = []
        self.showActivated = false
        self.scheduledTasksDisabled = false
        self.stepActive = false
        self.gameHistory = []
        self.pendingSiphonTransmissions = 0
        self.atkPlusUsedThisStage = false

        let corners = grid.getCornerPositions()
        let playerCorner = corners.randomElement()!
        self.player = Player(row: playerCorner.0, col: playerCorner.1)

        initializeStage()
    }

    /// Debug initializer to reproduce specific scenarios
    /// Set useDebugScenario to true to load the debug scenario
    static func createDebugScenario() -> GameState {
        let state = GameState()

        // Clear the randomly generated stage
        state.grid = Grid()
        state.enemies = []
        state.transmissions = []
        state.turnCount = 15
        state.currentStage = 2

        // Set player position
        state.player = Player(row: 3, col: 1)
        state.player.health = .full
        state.player.score = 5
        state.player.credits = 120
        state.player.energy = 160
        state.player.dataSiphons = 0

        // Add owned programs (example - adjust as needed)
        state.ownedPrograms = [.col, .row, .warp, .crash, .exch, .show, .reset, .dBomb, .antiV, .calm, .delay, .atkPlus, .debug, .reduc, .score, .hack]

        // Place enemies
        let virus = Enemy(type: .virus, row: 1, col: 1)
        virus.hp = 2
        state.enemies.append(virus)

        let daemon1 = Enemy(type: .daemon, row: 1, col: 4)
        daemon1.hp = 2
        state.enemies.append(daemon1)
        let daemon2 = Enemy(type: .daemon, row: 0, col: 4)
        daemon2.hp = 2
        state.enemies.append(daemon2)

        // Setup grid - place blocks and resources based on image
        // Row 5 (top row, counting from 0)
        state.grid.cells[5][0].content = .empty
        state.grid.cells[5][0].resources = .none
        state.grid.cells[5][1].content = .block(.data(points: 8, transmissionSpawn: 1))
        state.grid.cells[5][2].content = .block(.data(points: 6, transmissionSpawn: 2))
        state.grid.cells[5][2].resources = .energy(2)
        state.grid.cells[5][3].content = .empty
        state.grid.cells[5][3].resources = .energy(2)
        state.grid.cells[5][4].content = .empty
        state.grid.cells[5][4].resources = .credits(1)
        state.grid.cells[5][5].content = .dataSiphon

        // Row 4
        state.grid.cells[4][0].content = .empty
        state.grid.cells[4][0].resources = .energy(2)
        state.grid.cells[4][1].content = .empty
        state.grid.cells[4][1].resources = .energy(1)
        state.grid.cells[4][2].content = .empty
        state.grid.cells[4][2].resources = .credits(1)
        state.grid.cells[4][3].content = .block(.data(points: 6, transmissionSpawn: 2))
        state.grid.cells[4][3].resources = .credits(2)
        state.grid.cells[4][4].content = .empty
        state.grid.cells[4][4].resources = .credits(1)
        state.grid.cells[4][5].content = .block(.data(points: 6, transmissionSpawn: 1))

        // Row 3
        state.grid.cells[3][0].content = .empty
        state.grid.cells[3][0].resources = .credits(1)
        state.grid.cells[3][1].content = .empty // Virus is here
        state.grid.cells[3][1].resources = .credits(1)
        state.grid.cells[3][2].content = .empty
        state.grid.cells[3][2].resources = .energy(2)
        state.grid.cells[3][3].content = .empty
        state.grid.cells[3][3].resources = .credits(1)
        state.grid.cells[3][4].content = .empty
        state.grid.cells[3][4].resources = .energy(2)
        state.grid.cells[3][5].content = .empty
        state.grid.cells[3][5].resources = .energy(2)

        // Row 2
        state.grid.cells[2][0].content = .block(.data(points: 4, transmissionSpawn: 4))
        state.grid.cells[2][1].content = .block(.data(points: 1, transmissionSpawn: 2))
        state.grid.cells[2][2].content = .block(.question(isData: true, points: 2, program: nil, transmissionSpawn: 2))
        state.grid.cells[2][3].content = .block(.program(Program(type: .warp), transmissionSpawn: 5))
        state.grid.cells[2][4].content = .empty
        state.grid.cells[2][4].resources = .energy(2)
        state.grid.cells[2][5].content = .empty
        state.grid.cells[2][5].resources = .energy(2)

        // Row 1
        state.grid.cells[1][0].content = .empty
        state.grid.cells[1][0].resources = .credits(1)
        state.grid.cells[1][1].content = .empty // Player is here
        state.grid.cells[1][1].resources = .credits(1)
        state.grid.cells[1][1].siphonCenter = true // Player siphoned here
        state.grid.cells[1][2].content = .empty
        state.grid.cells[1][2].resources = .credits(1)
        state.grid.cells[1][3].content = .block(.data(points: 2, transmissionSpawn: 2))
        state.grid.cells[1][4].content = .empty
        state.grid.cells[1][4].resources = .credits(1)
        state.grid.cells[1][5].content = .block(.data(points: 2, transmissionSpawn: 2))

        // Row 0 (bottom row)
        state.grid.cells[0][0].content = .exit
        state.grid.cells[0][1].content = .block(.program(Program(type: .row), transmissionSpawn: 2))
        state.grid.cells[0][2].content = .empty
        state.grid.cells[0][2].resources = .energy(1)
        state.grid.cells[0][3].content = .empty
        state.grid.cells[0][3].resources = .credits(1)
        state.grid.cells[0][4].content = .empty
        state.grid.cells[0][4].resources = .credits(2)
        state.grid.cells[0][5].content = .dataSiphon

        return state
    }

    func initializeStage() {
        // Keep enemies and transmissions (they persist across stages)
        turnCount = 0
        showActivated = false
        scheduledTasksDisabled = false
        stepActive = false
        atkPlusUsedThisStage = false

        grid = Grid()

        let corners = grid.getCornerPositions()
        var availableCorners = corners

        // Player stays at current position, exclude it from available corners
        availableCorners.removeAll { $0.0 == player.row && $0.1 == player.col }

        let exitCorner = availableCorners.randomElement()!
        availableCorners.removeAll { $0.0 == exitCorner.0 && $0.1 == exitCorner.1 }
        grid.cells[exitCorner.0][exitCorner.1].content = .exit

        for corner in availableCorners {
            grid.cells[corner.0][corner.1].content = .dataSiphon
        }

        // Calculate available space for blocks
        let totalCells = Constants.gridSize * Constants.gridSize
        let usedCells = 4 // corners (player, exit, 2 siphons)
        let enemyCells = enemies.count
        let transmissionCells = transmissions.count
        let newTransmissionCount = Constants.startingEnemies[currentStage - 1]
        let availableForBlocks = totalCells - usedCells - enemyCells - transmissionCells - newTransmissionCount

        // Place blocks (reduce count if necessary to fit)
        placeBlocksWithValidation()
        placeResources()

        // Spawn new transmissions (guaranteed to have space reserved)
        spawnRandomTransmissions(count: newTransmissionCount)
    }

    func placeBlocksWithValidation() {
        let blockCount = Int.random(in: 5...11)
        var attempts = 0
        let maxAttempts = 100

        while attempts < maxAttempts {
            let tempGrid = Grid()

            let corners = grid.getCornerPositions()
            var availableCorners = corners
            let playerCorner = (player.row, player.col)
            availableCorners.removeAll { $0.0 == playerCorner.0 && $0.1 == playerCorner.1 }

            let exitCorner = grid.cells.enumerated().flatMap { row, cells in
                cells.enumerated().compactMap { col, cell in
                    cell.isExit ? (row, col) : nil
                }
            }.first!
            availableCorners.removeAll { $0.0 == exitCorner.0 && $0.1 == exitCorner.1 }

            tempGrid.cells[exitCorner.0][exitCorner.1].content = .exit
            for corner in availableCorners {
                tempGrid.cells[corner.0][corner.1].content = .dataSiphon
            }

            var placed = 0
            var usedPrograms = Set<ProgramType>()  // Track programs used in this stage

            while placed < blockCount {
                let row = Int.random(in: 0..<Constants.gridSize)
                let col = Int.random(in: 0..<Constants.gridSize)

                // Skip corners
                let isCorner = (row == 0 || row == Constants.gridSize - 1) &&
                               (col == 0 || col == Constants.gridSize - 1)
                if isCorner {
                    continue
                }

                // Skip positions with enemies or transmissions (they persist across stages)
                let hasEnemy = enemies.contains(where: { $0.row == row && $0.col == col })
                let hasTransmission = transmissions.contains(where: { $0.row == row && $0.col == col })
                if hasEnemy || hasTransmission {
                    continue
                }

                let cell = tempGrid.cells[row][col]

                if case .empty = cell.content {
                    let isData = Bool.random()  // 50% data, 50% program
                    let isQuestion = Double.random(in: 0..<1) < 0.1  // 10% are question blocks

                    if isQuestion {
                        // Question block - hide the contents
                        if isData {
                            let pointsAndSpawn = Int.random(in: 1...9)
                            cell.content = .block(.question(
                                isData: true,
                                points: pointsAndSpawn,
                                program: nil,
                                transmissionSpawn: pointsAndSpawn
                            ))
                        } else {
                            // Get available programs (not used yet)
                            let programPool = Constants.devModePrograms ?? ProgramType.allCases
                            let availablePrograms = programPool.filter { !usedPrograms.contains($0) }

                            // If no programs available, skip this block (make it data instead)
                            if availablePrograms.isEmpty {
                                let pointsAndSpawn = Int.random(in: 1...9)
                                cell.content = .block(.question(
                                    isData: true,
                                    points: pointsAndSpawn,
                                    program: nil,
                                    transmissionSpawn: pointsAndSpawn
                                ))
                            } else {
                                let programType = availablePrograms.randomElement()!
                                usedPrograms.insert(programType)
                                let program = Program(type: programType)
                                cell.content = .block(.question(
                                    isData: false,
                                    points: nil,
                                    program: program,
                                    transmissionSpawn: program.enemySpawnCost
                                ))
                            }
                        }
                    } else {
                        // Regular block - visible
                        if isData {
                            let pointsAndSpawn = Int.random(in: 1...9)
                            cell.content = .block(.data(points: pointsAndSpawn, transmissionSpawn: pointsAndSpawn))
                        } else {
                            // Get available programs (not used yet)
                            let programPool = Constants.devModePrograms ?? ProgramType.allCases
                            let availablePrograms = programPool.filter { !usedPrograms.contains($0) }

                            // If no programs available, skip this block (make it data instead)
                            if availablePrograms.isEmpty {
                                let pointsAndSpawn = Int.random(in: 1...9)
                                cell.content = .block(.data(points: pointsAndSpawn, transmissionSpawn: pointsAndSpawn))
                            } else {
                                let programType = availablePrograms.randomElement()!
                                usedPrograms.insert(programType)
                                let program = Program(type: programType)
                                cell.content = .block(.program(program, transmissionSpawn: program.enemySpawnCost))
                            }
                        }
                    }
                    placed += 1
                }
            }

            if GridValidator.isValidPlacement(grid: tempGrid) {
                for row in 0..<Constants.gridSize {
                    for col in 0..<Constants.gridSize {
                        if case .block = tempGrid.cells[row][col].content {
                            grid.cells[row][col].content = tempGrid.cells[row][col].content
                        }
                    }
                }
                return
            }

            attempts += 1
        }

        print("Warning: Could not find valid block placement after \(maxAttempts) attempts")
    }

    func placeResources() {
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                let cell = grid.cells[row][col]
                // Place resources on empty cells AND data siphon cells
                if case .empty = cell.content {
                    // Weighted random: 45% chance of 1, 45% chance of 2, 10% chance of 3
                    let roll = Double.random(in: 0..<1)
                    let amount: Int
                    if roll < 0.45 {
                        amount = 1
                    } else if roll < 0.9 {
                        amount = 2
                    } else {
                        amount = 3
                    }
                    cell.resources = Bool.random() ? .credits(amount) : .energy(amount)
                } else if case .dataSiphon = cell.content {
                    // Data siphon cells also get resources
                    let roll = Double.random(in: 0..<1)
                    let amount: Int
                    if roll < 0.45 {
                        amount = 1
                    } else if roll < 0.9 {
                        amount = 2
                    } else {
                        amount = 3
                    }
                    cell.resources = Bool.random() ? .credits(amount) : .energy(amount)
                }
            }
        }
    }

    func spawnRandomTransmissions(count: Int) {
        for _ in 0..<count {
            // First try to find a cell out of player's line of fire
            if let pos = findEmptyCellOutOfLineOfFire() {
                let transmission = Transmission(row: pos.0, col: pos.1)
                transmissions.append(transmission)
            } else if let pos = findEmptyCell() {
                // Fallback: spawn in any empty cell (even in line of fire)
                let transmission = Transmission(row: pos.0, col: pos.1)
                transmissions.append(transmission)
            }
        }
    }

    func findEmptyCellOutOfLineOfFire() -> (Int, Int)? {
        var attempts = 0
        while attempts < 100 {
            let row = Int.random(in: 0..<Constants.gridSize)
            let col = Int.random(in: 0..<Constants.gridSize)

            // Check if position is empty AND not in line of fire
            if isPositionEmpty(row: row, col: col) && !isPositionInLineOfFire(row: row, col: col) {
                return (row, col)
            }
            attempts += 1
        }
        return nil
    }

    func findEmptyCell() -> (Int, Int)? {
        var attempts = 0
        while attempts < 100 {
            let row = Int.random(in: 0..<Constants.gridSize)
            let col = Int.random(in: 0..<Constants.gridSize)

            if isPositionEmpty(row: row, col: col) {
                return (row, col)
            }
            attempts += 1
        }
        return nil
    }

    func isPositionEmpty(row: Int, col: Int) -> Bool {
        guard grid.isValidPosition(row: row, col: col) else { return false }

        if player.row == row && player.col == col { return false }
        if enemies.contains(where: { $0.row == row && $0.col == col }) { return false }
        if transmissions.contains(where: { $0.row == row && $0.col == col }) { return false }
        if grid.cells[row][col].hasBlock { return false }

        return true
    }

    func isPositionInLineOfFire(row: Int, col: Int) -> Bool {
        // Must be in same row or column
        guard row == player.row || col == player.col else {
            return false
        }

        // Determine direction offset
        let rowOffset = row == player.row ? 0 : (row > player.row ? 1 : -1)
        let colOffset = col == player.col ? 0 : (col > player.col ? 1 : -1)

        var currentRow = player.row
        var currentCol = player.col

        // Trace line of fire from player toward target
        while true {
            currentRow += rowOffset
            currentCol += colOffset

            // Reached the target position - it's in line of fire
            if currentRow == row && currentCol == col {
                return true
            }

            // Check if there's an enemy at this position
            let hasEnemy = enemies.contains(where: { $0.row == currentRow && $0.col == currentCol })

            // If there's an enemy before reaching target, line stops at enemy
            if hasEnemy {
                return false
            }

            // If there's a block (and no enemy on it), line of fire is blocked
            if grid.cells[currentRow][currentCol].hasBlock {
                return false
            }
        }
    }

    func maybeExecuteScheduledTask() {
        guard !scheduledTasksDisabled else { return }

        let interval = Constants.scheduledTaskIntervals[currentStage - 1]
        if turnCount > 0 && turnCount % interval == 0 {
            spawnRandomTransmissions(count: 1)
        }
    }

    // For animated enemy movement - start turn without processing enemies
    // Returns true if enemies should move this turn
    func beginAnimatedEnemyTurn() -> Bool {
        guard !stepActive else {
            // Step program: free action, no enemy turn, no turn counter advance
            stepActive = false
            return false
        }

        // Normal turn: advance counter, enemies move
        turnCount += 1
        processTransmissions()
        maybeExecuteScheduledTask()

        return true
    }

    // For animated enemy movement - finalize turn after animations complete
    func finalizeAnimatedEnemyTurn() {
        // Spawn any pending transmissions from siphoning
        if pendingSiphonTransmissions > 0 {
            spawnRandomTransmissions(count: pendingSiphonTransmissions)
            pendingSiphonTransmissions = 0
        }

        for enemy in enemies {
            enemy.decrementDisable()
            enemy.isStunned = false
        }

        // Save snapshot for undo (after enemy turn completes)
        saveSnapshot()
    }

    // For animated enemy movement - processes one step at a time
    func executeEnemyStep(step: Int, enemiesWhoAttacked: inout Set<UUID>) -> Bool {
        let maxSteps = enemies.map { $0.type.moveSpeed }.max() ?? 1

        guard step < maxSteps else { return false }

        // First check for attacks this step
        for enemy in enemies {
            guard !enemy.isDisabled && !enemy.isStunned else { continue }
            guard !enemiesWhoAttacked.contains(enemy.id) else { continue }
            guard step < enemy.type.moveSpeed else { continue }

            if isAdjacentToPlayer(enemy) {
                player.health.takeDamage()
                enemiesWhoAttacked.insert(enemy.id)
            }
        }

        // Then move enemies who didn't attack
        moveEnemiesSimultaneously(step: step, enemiesWhoAttacked: enemiesWhoAttacked)

        return step + 1 < maxSteps
    }

    func getMaxEnemySteps() -> Int {
        return enemies.map { $0.type.moveSpeed }.max() ?? 1
    }

    func processTransmissions() {
        var newEnemies: [Enemy] = []

        for transmission in transmissions {
            if let enemy = transmission.decrementTimer() {
                newEnemies.append(enemy)
            }
        }

        enemies.append(contentsOf: newEnemies)

        transmissions.removeAll { transmission in
            if case .spawned = transmission.state {
                return true
            }
            return false
        }
    }

    func isAdjacentToPlayer(_ enemy: Enemy) -> Bool {
        let rowDiff = abs(enemy.row - player.row)
        let colDiff = abs(enemy.col - player.col)
        return (rowDiff == 1 && colDiff == 0) || (rowDiff == 0 && colDiff == 1)
    }

    func getOccupiedPositions(for movingEnemy: Enemy?) -> Set<String> {
        var positions = Set<String>()

        // Add enemy positions
        for enemy in enemies {
            if let movingEnemy = movingEnemy, enemy.id == movingEnemy.id {
                continue
            }
            positions.insert("\(enemy.row),\(enemy.col)")
        }

        // Add transmission positions (enemies cannot move onto transmissions)
        for transmission in transmissions {
            positions.insert("\(transmission.row),\(transmission.col)")
        }

        return positions
    }

    func moveEnemiesSimultaneously(step: Int, enemiesWhoAttacked: Set<UUID>) {
        // Filter to enemies that should move this step
        let enemiesToMove = enemies.filter { enemy in
            !enemy.isDisabled &&
            !enemy.isStunned &&
            !enemiesWhoAttacked.contains(enemy.id) &&
            step < enemy.type.moveSpeed &&
            !isAdjacentToPlayer(enemy)
        }

        var desiredMoves: [(enemy: Enemy, target: (row: Int, col: Int))] = []
        let allEnemyPositions = getOccupiedPositions(for: nil)

        for enemy in enemiesToMove {
            let otherEnemyPositions = getOccupiedPositions(for: enemy)

            if let nextMove = Pathfinding.findNextMove(
                from: (enemy.row, enemy.col),
                to: (player.row, player.col),
                grid: grid,
                canMoveOnBlocks: enemy.type.canMoveOnBlocks,
                occupiedPositions: otherEnemyPositions
            ) {
                desiredMoves.append((enemy, nextMove))
            } else {
                // Pathfinding failed - try to move towards player anyway
                // Find the adjacent cell that's closest to the player
                var bestMove: (row: Int, col: Int)?
                var bestDistance = Int.max

                for direction in Direction.allCases {
                    let offset = direction.offset
                    let newRow = enemy.row + offset.row
                    let newCol = enemy.col + offset.col

                    guard grid.isValidPosition(row: newRow, col: newCol) else { continue }

                    let cell = grid.cells[newRow][newCol]
                    if cell.hasBlock && !enemy.type.canMoveOnBlocks { continue }

                    let posKey = "\(newRow),\(newCol)"
                    if otherEnemyPositions.contains(posKey) { continue }

                    let distance = abs(player.row - newRow) + abs(player.col - newCol)
                    if distance < bestDistance {
                        bestDistance = distance
                        bestMove = (newRow, newCol)
                    }
                }

                if let move = bestMove {
                    desiredMoves.append((enemy, move))
                }
            }
        }

        var targetOccupied = Set<String>()

        for (enemy, optimalTarget) in desiredMoves {
            var target = optimalTarget
            let targetKey = "\(target.row),\(target.col)"

            if targetOccupied.contains(targetKey) {
                if let alternative = findAlternativeMove(enemy: enemy, occupiedTargets: targetOccupied, allEnemyPositions: allEnemyPositions) {
                    target = alternative
                } else {
                    continue
                }
            }

            if target.row != enemy.row || target.col != enemy.col {
                enemy.row = target.row
                enemy.col = target.col
                targetOccupied.insert("\(target.row),\(target.col)")

                // Update last known position only when Cryptog is visible
                if enemy.type == .cryptog {
                    let isVisible = enemy.row == player.row || enemy.col == player.col
                    if isVisible {
                        enemy.lastKnownRow = enemy.row
                        enemy.lastKnownCol = enemy.col
                    }
                }
            }
        }
    }

    func performSiphon() -> Bool {
        // Check if player has data siphons
        guard player.dataSiphons > 0 else {
            return false
        }

        // Get cells in siphon range (current cell + cardinal directions)
        let siphonCells = grid.getSiphonCells(centerRow: player.row, centerCol: player.col)

        // Mark center cell (where player is standing)
        grid.cells[player.row][player.col].siphonCenter = true

        // Process each cell
        for cell in siphonCells {
            // Skip if already siphoned
            if cell.isSiphoned {
                continue
            }

            // Check if there's a block - if so, only process the block, NOT resources
            if case .block(let blockType) = cell.content {
                switch blockType {
                case .data(let points, let transmissionSpawn):
                    player.score += points
                    pendingSiphonTransmissions += transmissionSpawn
                    // Update block to show 0 transmissions since they're being spawned
                    cell.content = .block(.data(points: points, transmissionSpawn: 0))

                case .program(let program, let transmissionSpawn):
                    if !ownedPrograms.contains(program.type) {
                        ownedPrograms.append(program.type)
                    }
                    pendingSiphonTransmissions += transmissionSpawn
                    // Update block to show 0 transmissions since they're being spawned
                    cell.content = .block(.program(program, transmissionSpawn: 0))

                case .question(let isData, let points, let program, let transmissionSpawn):
                    if isData, let pts = points {
                        player.score += pts
                        // Reveal the question block as a data block (with 0 transmissions since they're being spawned)
                        cell.content = .block(.data(points: pts, transmissionSpawn: 0))
                    } else if let prog = program {
                        if !ownedPrograms.contains(prog.type) {
                            ownedPrograms.append(prog.type)
                        }
                        // Reveal the question block as a program block (with 0 transmissions since they're being spawned)
                        cell.content = .block(.program(prog, transmissionSpawn: 0))
                    }
                    pendingSiphonTransmissions += transmissionSpawn
                }

                // Don't destroy the block - it stays but is marked as siphoned
                // (resources remain hidden until block is destroyed by other means)
            } else {
                // No block - collect resources from empty cells
                if case .credits(let amount) = cell.resources {
                    player.credits += amount
                    cell.resources = .none
                } else if case .energy(let amount) = cell.resources {
                    player.energy += amount
                    cell.resources = .none
                }
            }

            // Mark cell as siphoned
            cell.isSiphoned = true
        }

        // Consume the data siphon
        player.dataSiphons -= 1

        // Don't advance turn here - let caller handle animated turn flow
        return true
    }

    // MARK: - Program Execution

    /// Check if a program can be executed
    func canExecuteProgram(_ type: ProgramType) -> (canExecute: Bool, reason: String?) {
        guard ownedPrograms.contains(type) else {
            return (false, "Not owned")
        }

        let program = Program(type: type)

        // Check resources
        if player.credits < program.cost.credits {
            return (false, "Need \(program.cost.credits)💰")
        }
        if player.energy < program.cost.energy {
            return (false, "Need \(program.cost.energy)🔋")
        }

        // Check if applicable to current board state
        if !isProgramApplicable(type) {
            return (false, "N/A")
        }

        return (true, nil)
    }

    /// Check if program is applicable to current board state
    func isProgramApplicable(_ type: ProgramType) -> Bool {
        switch type {
        case .push, .pull, .poly:
            // Need enemies on board
            return !enemies.isEmpty

        case .crash:
            // Check 8 surrounding cells for blocks, enemies, or transmissions
            for rowOffset in -1...1 {
                for colOffset in -1...1 {
                    if rowOffset == 0 && colOffset == 0 { continue }
                    let checkRow = player.row + rowOffset
                    let checkCol = player.col + colOffset
                    if grid.isValidPosition(row: checkRow, col: checkCol) {
                        let cell = grid.cells[checkRow][checkCol]
                        if cell.hasBlock ||
                           enemies.contains(where: { $0.row == checkRow && $0.col == checkCol }) ||
                           transmissions.contains(where: { $0.row == checkRow && $0.col == checkCol }) {
                            return true
                        }
                    }
                }
            }
            return false

        case .warp:
            // Can warp if there are enemies OR transmissions
            return !enemies.isEmpty || !transmissions.isEmpty

        case .exch:
            return player.credits >= 4

        case .show:
            return !showActivated

        case .reset:
            return player.health.rawValue < 3

        case .dBomb:
            return enemies.contains { $0.type == .daemon }

        case .antiV:
            return enemies.contains { $0.type == .virus }

        case .calm:
            return !scheduledTasksDisabled

        case .delay:
            return !transmissions.isEmpty

        case .atkPlus:
            return !atkPlusUsedThisStage && player.attackDamage < 2

        case .row, .col:
            if type == .row {
                return enemies.contains { $0.row == player.row }
            } else {
                return enemies.contains { $0.col == player.col }
            }

        case .debug:
            return enemies.contains { enemy in
                grid.cells[enemy.row][enemy.col].hasBlock
            }

        case .reduc:
            for row in 0..<Constants.gridSize {
                for col in 0..<Constants.gridSize {
                    let cell = grid.cells[row][col]
                    if case .block(let blockType) = cell.content, !cell.isSiphoned {
                        if blockType.transmissionSpawnCount > 0 {
                            return true
                        }
                    }
                }
            }
            return false

        case .score:
            // Only applicable if not on last stage
            return currentStage < Constants.totalStages

        case .hack:
            // Only applicable if there are siphoned cells
            for row in 0..<Constants.gridSize {
                for col in 0..<Constants.gridSize {
                    if grid.cells[row][col].isSiphoned {
                        return true
                    }
                }
            }
            return false

        case .undo:
            return !gameHistory.isEmpty

        case .siphPlus, .wait, .step:
            // Always applicable
            return true
        }
    }

    /// Result of executing a program
    struct ProgramExecutionResult {
        let success: Bool
        let affectedPositions: [(row: Int, col: Int)]
    }

    /// Result of a single enemy step (for animation)
    struct EnemyStepResult {
        let step: Int  // 0, 1, etc (virus moves twice per turn)
        let movements: [(enemyId: UUID, fromRow: Int, fromCol: Int, toRow: Int, toCol: Int)]
        let attacks: [(enemyId: UUID, damage: Int)]
    }

    /// Player action data (for animation)
    struct PlayerActionResult {
        let fromRow: Int
        let fromCol: Int
        let movedTo: (row: Int, col: Int)?      // set if player moved
        let attackedTarget: (row: Int, col: Int)?  // set if player attacked
    }

    /// Result of processing any game action
    struct ActionResult {
        let success: Bool
        let exitReached: Bool
        let stageAdvanced: Bool  // stage was completed
        let gameWon: Bool        // player beat final stage
        let playerDied: Bool
        let playerAction: PlayerActionResult?  // nil if action failed
        let affectedPositions: [(row: Int, col: Int)]  // for explosion animations
        let enemySteps: [EnemyStepResult]  // for enemy movement animations

        static let failed = ActionResult(
            success: false,
            exitReached: false,
            stageAdvanced: false,
            gameWon: false,
            playerDied: false,
            playerAction: nil,
            affectedPositions: [],
            enemySteps: []
        )
    }

    /// Execute a game action - single entry point for all action processing
    /// Runs player action AND enemy turn (if applicable), returns all data for animation
    func tryExecuteAction(_ action: GameAction) -> ActionResult {
        // Capture player position before action
        let fromRow = player.row
        let fromCol = player.col

        var success = true
        var exitReached = false
        var affectedPositions: [(row: Int, col: Int)] = []
        var shouldRunEnemyTurn = false
        var movedTo: (row: Int, col: Int)? = nil
        var attackedTarget: (row: Int, col: Int)? = nil

        // 1. Handle player action
        switch action {
        case .direction(let direction):
            let result = tryAttackOrMove(direction: direction)
            success = result.success
            exitReached = result.exitReached
            movedTo = result.movedTo
            attackedTarget = result.attackedTarget
            shouldRunEnemyTurn = success && !exitReached

        case .siphon:
            success = performSiphon()
            shouldRunEnemyTurn = success

        case .program(let programType):
            let execResult = executeProgram(programType)
            success = execResult.success
            affectedPositions = execResult.affectedPositions
            // Only wait program advances enemy turn
            shouldRunEnemyTurn = programType == .wait && success
        }

        if !success {
            return .failed
        }

        // Build player action result for animation
        let playerAction = PlayerActionResult(
            fromRow: fromRow,
            fromCol: fromCol,
            movedTo: movedTo,
            attackedTarget: attackedTarget
        )

        // 2. Handle stage completion if exit reached
        var stageAdvanced = false
        var gameWon = false
        if exitReached {
            let gameContinues = completeStage()
            stageAdvanced = true
            gameWon = !gameContinues
        }

        // 3. Run enemy turn if needed (not if exit reached)
        var enemySteps: [EnemyStepResult] = []
        if shouldRunEnemyTurn {
            enemySteps = runEnemyTurn()
        }

        // 4. Return everything
        return ActionResult(
            success: true,
            exitReached: exitReached,
            stageAdvanced: stageAdvanced,
            gameWon: gameWon,
            playerDied: player.health == .dead,
            playerAction: playerAction,
            affectedPositions: affectedPositions,
            enemySteps: enemySteps
        )
    }

    /// Run full enemy turn, capturing step data for animation
    private func runEnemyTurn() -> [EnemyStepResult] {
        guard !stepActive else {
            stepActive = false
            return []
        }

        turnCount += 1
        processTransmissions()

        // Run all enemy steps, capturing movement/attack data
        var results: [EnemyStepResult] = []
        var enemiesWhoAttacked = Set<UUID>()
        let maxSteps = getMaxEnemySteps()

        for step in 0..<maxSteps {
            let stepResult = executeEnemyStepWithCapture(step: step, enemiesWhoAttacked: &enemiesWhoAttacked)
            results.append(stepResult)
        }

        maybeExecuteScheduledTask()

        // Finalize turn
        if pendingSiphonTransmissions > 0 {
            spawnRandomTransmissions(count: pendingSiphonTransmissions)
            pendingSiphonTransmissions = 0
        }
        for enemy in enemies {
            enemy.decrementDisable()
            enemy.isStunned = false
        }
        saveSnapshot()

        return results
    }

    /// Execute one enemy step and capture movement/attack data
    /// enemiesWhoAttacked tracks enemies that attacked this turn (for virus: if it attacks on step 0, skip step 1)
    private func executeEnemyStepWithCapture(step: Int, enemiesWhoAttacked: inout Set<UUID>) -> EnemyStepResult {
        var movements: [(enemyId: UUID, fromRow: Int, fromCol: Int, toRow: Int, toCol: Int)] = []
        var attacks: [(enemyId: UUID, damage: Int)] = []

        let maxSteps = enemies.map { $0.type.moveSpeed }.max() ?? 1
        guard step < maxSteps else {
            return EnemyStepResult(step: step, movements: [], attacks: [])
        }

        // First check for attacks this step
        for enemy in enemies {
            guard !enemy.isDisabled && !enemy.isStunned else { continue }
            guard !enemiesWhoAttacked.contains(enemy.id) else { continue }
            guard step < enemy.type.moveSpeed else { continue }

            if isAdjacentToPlayer(enemy) {
                player.health.takeDamage()
                enemiesWhoAttacked.insert(enemy.id)
                attacks.append((enemyId: enemy.id, damage: 1))
            }
        }

        // Capture positions before movement
        var positionsBefore: [UUID: (row: Int, col: Int)] = [:]
        for enemy in enemies {
            positionsBefore[enemy.id] = (enemy.row, enemy.col)
        }

        // Move enemies who didn't attack (reusing existing simultaneous movement logic)
        moveEnemiesSimultaneously(step: step, enemiesWhoAttacked: enemiesWhoAttacked)

        // Capture movements by comparing before/after positions
        for enemy in enemies {
            if let before = positionsBefore[enemy.id] {
                if enemy.row != before.row || enemy.col != before.col {
                    movements.append((enemyId: enemy.id, fromRow: before.row, fromCol: before.col, toRow: enemy.row, toCol: enemy.col))
                }
            }
        }

        return EnemyStepResult(step: step, movements: movements, attacks: attacks)
    }

    /// Execute a program's effect
    /// Returns execution result with affected positions for animations
    func executeProgram(_ type: ProgramType) -> ProgramExecutionResult {
        let check = canExecuteProgram(type)
        guard check.canExecute else { return ProgramExecutionResult(success: false, affectedPositions: []) }

        let program = Program(type: type)

        // Deduct resources
        player.credits -= program.cost.credits
        player.energy -= program.cost.energy

        var affectedPositions: [(row: Int, col: Int)] = []

        // Execute program effect
        switch type {
        case .exch:
            // Convert 4C to 4E
            player.credits -= 4
            player.energy += 4

        case .show:
            // Reveal Cryptogs and transmissions
            showActivated = true

        case .siphPlus:
            // Gain 1 data siphon
            player.dataSiphons += 1

        case .reset:
            // Restore to 3HP
            player.health = .full

        case .calm:
            // Disable scheduled spawns
            scheduledTasksDisabled = true

        case .atkPlus:
            // Increase damage to 2HP
            player.attackDamage = 2
            atkPlusUsedThisStage = true

        case .dBomb:
            // Destroy nearest Daemon and damage/stun surrounding enemies
            if let nearestDaemon = findNearestEnemy(ofType: .daemon) {
                let daemonRow = nearestDaemon.row
                let daemonCol = nearestDaemon.col

                // Add daemon position for explosion animation
                affectedPositions.append((daemonRow, daemonCol))

                // Remove the daemon
                enemies.removeAll { $0.id == nearestDaemon.id }

                // Damage and stun enemies in 8 surrounding cells
                for rowOffset in -1...1 {
                    for colOffset in -1...1 {
                        if rowOffset == 0 && colOffset == 0 { continue }
                        let checkRow = daemonRow + rowOffset
                        let checkCol = daemonCol + colOffset

                        for enemy in enemies where enemy.row == checkRow && enemy.col == checkCol {
                            affectedPositions.append((checkRow, checkCol))
                            enemy.takeDamage(1)
                            if enemy.hp > 0 {
                                enemy.isStunned = true
                            }
                        }
                    }
                }

                // Remove dead enemies
                enemies.removeAll { $0.hp <= 0 }
            }

        case .antiV:
            // Damage all Viruses and stun survivors
            for enemy in enemies where enemy.type == .virus {
                affectedPositions.append((enemy.row, enemy.col))
                enemy.takeDamage(1)
                if enemy.hp > 0 {
                    enemy.isStunned = true
                }
            }
            enemies.removeAll { $0.hp <= 0 }

        case .delay:
            // Extend transmissions +3 turns
            for transmission in transmissions {
                if case .spawning(let turns) = transmission.state {
                    transmission.state = .spawning(turnsRemaining: turns + 3)
                }
            }

        case .poly:
            // Randomize enemy types (each enemy becomes a DIFFERENT type)
            for enemy in enemies {
                let otherTypes = EnemyType.allCases.filter { $0 != enemy.type }
                let newType = otherTypes.randomElement()!
                enemy.type = newType
                enemy.hp = newType.maxHP

                // If converted to Cryptog, initialize last known position
                if newType == .cryptog {
                    // Check if currently visible to player
                    let isVisible = enemy.row == player.row || enemy.col == player.col
                    if isVisible {
                        enemy.lastKnownRow = enemy.row
                        enemy.lastKnownCol = enemy.col
                    } else {
                        // Not visible - set last known to current position (where it was just converted)
                        enemy.lastKnownRow = enemy.row
                        enemy.lastKnownCol = enemy.col
                    }
                }
            }

        case .reduc:
            // Reduce block spawn counts by 1 (minimum 0)
            for row in 0..<Constants.gridSize {
                for col in 0..<Constants.gridSize {
                    let cell = grid.cells[row][col]
                    if case .block(let blockType) = cell.content, !cell.isSiphoned {
                        switch blockType {
                        case .data(let points, let transmissionSpawn):
                            let newSpawn = max(0, transmissionSpawn - 1)
                            cell.content = .block(.data(points: points, transmissionSpawn: newSpawn))
                        case .program(let prog, let transmissionSpawn):
                            let newSpawn = max(0, transmissionSpawn - 1)
                            cell.content = .block(.program(prog, transmissionSpawn: newSpawn))
                        case .question(let isData, let points, let prog, let transmissionSpawn):
                            let newSpawn = max(0, transmissionSpawn - 1)
                            cell.content = .block(.question(isData: isData, points: points, program: prog, transmissionSpawn: newSpawn))
                        }
                    }
                }
            }

        case .score:
            // Gain points = levels left
            let levelsLeft = Constants.totalStages - currentStage
            player.score += levelsLeft

        case .row:
            // Attack all enemies in player's row
            for enemy in enemies where enemy.row == player.row {
                affectedPositions.append((enemy.row, enemy.col))
                enemy.takeDamage(player.attackDamage)
                if enemy.hp > 0 {
                    enemy.isStunned = true
                }
            }
            enemies.removeAll { $0.hp <= 0 }

        case .col:
            // Attack all enemies in player's column
            for enemy in enemies where enemy.col == player.col {
                affectedPositions.append((enemy.row, enemy.col))
                enemy.takeDamage(player.attackDamage)
                if enemy.hp > 0 {
                    enemy.isStunned = true
                }
            }
            enemies.removeAll { $0.hp <= 0 }

        case .debug:
            // Damage enemies standing on blocks
            for enemy in enemies where grid.cells[enemy.row][enemy.col].hasBlock {
                affectedPositions.append((enemy.row, enemy.col))
                enemy.takeDamage(player.attackDamage)
                if enemy.hp > 0 {
                    enemy.isStunned = true
                }
            }
            enemies.removeAll { $0.hp <= 0 }

        case .hack:
            // Damage enemies on siphoned cells and show explosions on all siphoned cells
            // First, collect all siphoned cells for animation
            for row in 0..<Constants.gridSize {
                for col in 0..<Constants.gridSize {
                    if grid.cells[row][col].isSiphoned {
                        affectedPositions.append((row, col))
                    }
                }
            }

            // Then damage enemies on siphoned cells
            for enemy in enemies where grid.cells[enemy.row][enemy.col].isSiphoned {
                enemy.takeDamage(player.attackDamage)
                if enemy.hp > 0 {
                    enemy.isStunned = true
                }
            }
            enemies.removeAll { $0.hp <= 0 }

        case .push:
            // Push all enemies one cell away from player
            var newPositions: [UUID: (row: Int, col: Int)] = [:]

            // Calculate desired push positions for all enemies
            for enemy in enemies {
                let rowDiff = enemy.row - player.row
                let colDiff = enemy.col - player.col

                // Determine push direction components (away from player)
                let pushRowDir = rowDiff == 0 ? 0 : (rowDiff > 0 ? 1 : -1)
                let pushColDir = colDiff == 0 ? 0 : (colDiff > 0 ? 1 : -1)

                // Try primary direction (both row and col)
                var candidates: [(row: Int, col: Int, dist: Int)] = []

                let primaryRow = enemy.row + pushRowDir
                let primaryCol = enemy.col + pushColDir
                if grid.isValidPosition(row: primaryRow, col: primaryCol) {
                    let dist = abs(primaryRow - player.row) + abs(primaryCol - player.col)
                    candidates.append((primaryRow, primaryCol, dist))
                }

                // Try row-only push
                if pushRowDir != 0 {
                    let rowOnlyRow = enemy.row + pushRowDir
                    let rowOnlyCol = enemy.col
                    if grid.isValidPosition(row: rowOnlyRow, col: rowOnlyCol) {
                        let dist = abs(rowOnlyRow - player.row) + abs(rowOnlyCol - player.col)
                        candidates.append((rowOnlyRow, rowOnlyCol, dist))
                    }
                }

                // Try col-only push
                if pushColDir != 0 {
                    let colOnlyRow = enemy.row
                    let colOnlyCol = enemy.col + pushColDir
                    if grid.isValidPosition(row: colOnlyRow, col: colOnlyCol) {
                        let dist = abs(colOnlyRow - player.row) + abs(colOnlyCol - player.col)
                        candidates.append((colOnlyRow, colOnlyCol, dist))
                    }
                }

                // Pick the candidate that's furthest from player
                if let best = candidates.max(by: { $0.dist < $1.dist }) {
                    newPositions[enemy.id] = (best.row, best.col)
                }
            }

            // Apply pushes (check for collisions with other pushed enemies)
            var occupiedAfterPush = Set<String>()
            for enemy in enemies {
                if let newPos = newPositions[enemy.id] {
                    let posKey = "\(newPos.row),\(newPos.col)"

                    if !occupiedAfterPush.contains(posKey) {
                        enemy.row = newPos.row
                        enemy.col = newPos.col
                        occupiedAfterPush.insert(posKey)
                    }
                    // If collision with another pushed enemy, stay in place
                }
            }

        case .pull:
            // Pull all enemies one cell toward player
            var newPositions: [UUID: (row: Int, col: Int)] = [:]

            // Calculate desired pull positions for all enemies
            for enemy in enemies {
                let rowDiff = enemy.row - player.row
                let colDiff = enemy.col - player.col

                // Determine pull direction components (toward player)
                let pullRowDir = rowDiff == 0 ? 0 : (rowDiff > 0 ? -1 : 1)
                let pullColDir = colDiff == 0 ? 0 : (colDiff > 0 ? -1 : 1)

                // Try primary direction (both row and col)
                var candidates: [(row: Int, col: Int, dist: Int)] = []

                let primaryRow = enemy.row + pullRowDir
                let primaryCol = enemy.col + pullColDir
                if grid.isValidPosition(row: primaryRow, col: primaryCol) &&
                   !(primaryRow == player.row && primaryCol == player.col) {
                    let dist = abs(primaryRow - player.row) + abs(primaryCol - player.col)
                    candidates.append((primaryRow, primaryCol, dist))
                }

                // Try row-only pull
                if pullRowDir != 0 {
                    let rowOnlyRow = enemy.row + pullRowDir
                    let rowOnlyCol = enemy.col
                    if grid.isValidPosition(row: rowOnlyRow, col: rowOnlyCol) &&
                       !(rowOnlyRow == player.row && rowOnlyCol == player.col) {
                        let dist = abs(rowOnlyRow - player.row) + abs(rowOnlyCol - player.col)
                        candidates.append((rowOnlyRow, rowOnlyCol, dist))
                    }
                }

                // Try col-only pull
                if pullColDir != 0 {
                    let colOnlyRow = enemy.row
                    let colOnlyCol = enemy.col + pullColDir
                    if grid.isValidPosition(row: colOnlyRow, col: colOnlyCol) &&
                       !(colOnlyRow == player.row && colOnlyCol == player.col) {
                        let dist = abs(colOnlyRow - player.row) + abs(colOnlyCol - player.col)
                        candidates.append((colOnlyRow, colOnlyCol, dist))
                    }
                }

                // Pick the candidate that's closest to player
                if let best = candidates.min(by: { $0.dist < $1.dist }) {
                    newPositions[enemy.id] = (best.row, best.col)
                }
            }

            // Apply pulls (check for collisions)
            var occupiedAfterPull = Set<String>()
            for enemy in enemies {
                if let newPos = newPositions[enemy.id] {
                    let posKey = "\(newPos.row),\(newPos.col)"

                    if !occupiedAfterPull.contains(posKey) {
                        enemy.row = newPos.row
                        enemy.col = newPos.col
                        occupiedAfterPull.insert(posKey)
                    }
                    // If collision, enemy stays in place
                }
            }

        case .crash:
            // Destroy blocks, enemies, and transmissions in 8 surrounding cells
            for rowOffset in -1...1 {
                for colOffset in -1...1 {
                    if rowOffset == 0 && colOffset == 0 { continue }

                    let checkRow = player.row + rowOffset
                    let checkCol = player.col + colOffset

                    if grid.isValidPosition(row: checkRow, col: checkCol) {
                        affectedPositions.append((checkRow, checkCol))
                        let cell = grid.cells[checkRow][checkCol]

                        // Destroy blocks and reveal resources underneath
                        if case .block = cell.content {
                            cell.content = .empty
                            // Resources remain (they were already placed)
                        }

                        // Destroy enemies at this position
                        enemies.removeAll { $0.row == checkRow && $0.col == checkCol }

                        // Destroy transmissions at this position
                        transmissions.removeAll { $0.row == checkRow && $0.col == checkCol }
                    }
                }
            }

        case .warp:
            // Warp to random enemy or transmission and destroy it
            var targets: [(row: Int, col: Int, isEnemy: Bool)] = []

            for enemy in enemies {
                targets.append((enemy.row, enemy.col, true))
            }
            for transmission in transmissions {
                targets.append((transmission.row, transmission.col, false))
            }

            if let target = targets.randomElement() {
                affectedPositions.append((target.row, target.col))

                // Warp player to target position
                player.row = target.row
                player.col = target.col

                // Destroy the target
                if target.isEnemy {
                    enemies.removeAll { $0.row == target.row && $0.col == target.col }
                } else {
                    transmissions.removeAll { $0.row == target.row && $0.col == target.col }
                }
            }

        case .wait:
            // Skip turn, enemies move - this will be handled by advancing the turn
            // (No immediate effect here, but caller should advance turn)
            break

        case .step:
            // Next turn enemies don't move
            stepActive = true

        case .undo:
            // Restore previous game state
            if restoreSnapshot() {
                // Successfully restored - no affected positions for animation
            } else {
                return ProgramExecutionResult(success: false, affectedPositions: [])
            }
        }

        return ProgramExecutionResult(success: true, affectedPositions: affectedPositions)
    }

    /// Find nearest enemy of a specific type
    func findNearestEnemy(ofType type: EnemyType) -> Enemy? {
        return enemies
            .filter { $0.type == type }
            .min { enemy1, enemy2 in
                let dist1 = abs(enemy1.row - player.row) + abs(enemy1.col - player.col)
                let dist2 = abs(enemy2.row - player.row) + abs(enemy2.col - player.col)
                return dist1 < dist2
            }
    }

    // MARK: - Snapshot Methods

    func saveSnapshot() {
        let snapshot = GameStateSnapshot(
            playerRow: player.row,
            playerCol: player.col,
            playerHealth: player.health,
            playerCredits: player.credits,
            playerEnergy: player.energy,
            playerSiphons: player.dataSiphons,
            playerScore: player.score,
            playerAttackDamage: player.attackDamage,
            turnCount: turnCount,
            ownedPrograms: ownedPrograms,
            enemies: enemies.map { enemy in
                EnemySnapshot(
                    id: enemy.id,
                    type: enemy.type,
                    row: enemy.row,
                    col: enemy.col,
                    hp: enemy.hp,
                    disabledTurns: enemy.disabledTurns,
                    isStunned: enemy.isStunned,
                    lastKnownRow: enemy.lastKnownRow,
                    lastKnownCol: enemy.lastKnownCol
                )
            },
            transmissions: transmissions.map { transmission in
                let turnsRemaining: Int
                if case .spawning(let turns) = transmission.state {
                    turnsRemaining = turns
                } else {
                    turnsRemaining = 0
                }
                return TransmissionSnapshot(
                    id: transmission.id,
                    row: transmission.row,
                    col: transmission.col,
                    turnsRemaining: turnsRemaining,
                    enemyType: transmission.enemyType
                )
            },
            gridCells: grid.cells.map { row in
                row.map { cell in
                    CellSnapshot(
                        content: cell.content,
                        resources: cell.resources,
                        isSiphoned: cell.isSiphoned,
                        siphonCenter: cell.siphonCenter
                    )
                }
            },
            showActivated: showActivated,
            scheduledTasksDisabled: scheduledTasksDisabled,
            atkPlusUsedThisStage: atkPlusUsedThisStage,
        )
        gameHistory.append(snapshot)
    }

    func restoreSnapshot() -> Bool {
        guard let snapshot: GameStateSnapshot = gameHistory.popLast() else { return false }

        player.row = snapshot.playerRow
        player.col = snapshot.playerCol
        player.health = snapshot.playerHealth
        player.credits = snapshot.playerCredits
        player.energy = snapshot.playerEnergy
        player.dataSiphons = snapshot.playerSiphons
        player.score = snapshot.playerScore
        player.attackDamage = snapshot.playerAttackDamage
        turnCount = snapshot.turnCount
        ownedPrograms = snapshot.ownedPrograms

        // Restore enemies
        enemies = snapshot.enemies.map { enemySnap in
            let enemy = Enemy(type: enemySnap.type, row: enemySnap.row, col: enemySnap.col)
            enemy.hp = enemySnap.hp
            enemy.disabledTurns = enemySnap.disabledTurns
            enemy.isStunned = enemySnap.isStunned
            enemy.lastKnownRow = enemySnap.lastKnownRow
            enemy.lastKnownCol = enemySnap.lastKnownCol
            return enemy
        }

        // Restore transmissions
        transmissions = snapshot.transmissions.map { transSnap in
            let transmission = Transmission(row: transSnap.row, col: transSnap.col, turnsUntilSpawn: transSnap.turnsRemaining, enemyType: transSnap.enemyType)
            return transmission
        }

        // Restore grid
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                let cellSnap = snapshot.gridCells[row][col]
                let cell = grid.cells[row][col]
                cell.content = cellSnap.content
                cell.resources = cellSnap.resources
                cell.isSiphoned = cellSnap.isSiphoned
                cell.siphonCenter = cellSnap.siphonCenter
            }
        }

        showActivated = snapshot.showActivated
        scheduledTasksDisabled = snapshot.scheduledTasksDisabled
        atkPlusUsedThisStage = snapshot.atkPlusUsedThisStage

        return true
    }

    func findAlternativeMove(enemy: Enemy, occupiedTargets: Set<String>, allEnemyPositions: Set<String>) -> (Int, Int)? {
        var candidates: [(pos: (Int, Int), dist: Int)] = []

        for direction in Direction.allCases {
            let offset = direction.offset
            let newRow = enemy.row + offset.row
            let newCol = enemy.col + offset.col

            guard grid.isValidPosition(row: newRow, col: newCol) else { continue }

            let cell = grid.cells[newRow][newCol]
            if cell.hasBlock && !enemy.type.canMoveOnBlocks { continue }

            let posKey = "\(newRow),\(newCol)"
            if occupiedTargets.contains(posKey) { continue }
            if allEnemyPositions.contains(posKey) { continue }

            let dist = abs(player.row - newRow) + abs(player.col - newCol)
            candidates.append(((newRow, newCol), dist))
        }

        return candidates.min(by: { $0.dist < $1.dist })?.pos
    }

    // MARK: - Player Action Methods (Pure Game Logic)

    /// Find target (transmission or enemy) in line of fire
    /// Returns the closest target, preferring transmissions over enemies if at same distance
    func findTargetInLineOfFire(direction: Direction) -> (transmission: Transmission?, enemy: Enemy?) {
        let offset = direction.offset
        var currentRow = player.row
        var currentCol = player.col

        while true {
            currentRow += offset.row
            currentCol += offset.col

            // Check bounds
            guard currentRow >= 0 && currentRow < Constants.gridSize &&
                  currentCol >= 0 && currentCol < Constants.gridSize else {
                return (nil, nil)
            }

            // Check for transmission first (same priority as enemy - first one hit)
            if let transmission = transmissions.first(where: { $0.row == currentRow && $0.col == currentCol }) {
                return (transmission, nil)
            }

            // Check for enemy
            if let enemy = enemies.first(where: { $0.row == currentRow && $0.col == currentCol }) {
                return (nil, enemy)
            }

            // If no target, check for block (blocks line of fire)
            if grid.cells[currentRow][currentCol].hasBlock {
                return (nil, nil)
            }
        }
    }

    /// Try to move player in direction, or attack if target in line of fire
    func tryAttackOrMove(direction: Direction) -> (success: Bool, exitReached: Bool, movedTo: (row: Int, col: Int)?, attackedTarget: (row: Int, col: Int)?) {
        // Check for transmission in line of fire
        let targetResult = findTargetInLineOfFire(direction: direction)

        if let transmission = targetResult.transmission {
            // Destroy the transmission (1 HP)
            let targetPos = (transmission.row, transmission.col)
            transmissions.removeAll { $0.id == transmission.id }
            return (true, false, nil, targetPos)
        }

        if let enemy = targetResult.enemy {
            // Attack the enemy
            let targetPos = (enemy.row, enemy.col)
            enemy.takeDamage(player.attackDamage)

            // Stun the enemy if it survives
            if enemy.hp > 0 {
                enemy.isStunned = true
            } else {
                // Remove dead enemy
                enemies.removeAll { $0.id == enemy.id }
            }
            return (true, false, nil, targetPos)
        }

        // No target to attack, try to move
        let offset = direction.offset
        let newRow = player.row + offset.row
        let newCol = player.col + offset.col

        if player.canMoveTo(row: newRow, col: newCol, grid: grid) {
            player.row = newRow
            player.col = newCol

            // Collect resources/siphons
            let cell = grid.cells[newRow][newCol]
            if cell.hasDataSiphon {
                player.dataSiphons += 1
                cell.content = .empty
            }

            let movedTo = (newRow, newCol)

            // Check for exit
            if cell.isExit {
                return (true, true, movedTo, nil)
            }

            return (true, false, movedTo, nil)
        }

        return (false, false, nil, nil)
    }

    /// Complete current stage and advance to next (or mark victory)
    /// Returns true if game continues, false if victory achieved
    func completeStage() -> Bool {
        // Gain 1 HP (up to max of 3)
        if player.health.rawValue < 3 {
            player.health = PlayerHealth(rawValue: player.health.rawValue + 1) ?? .full
        }

        if currentStage < Constants.totalStages {
            currentStage += 1
            initializeStage()
            return true  // Game continues
        } else {
            return false  // Victory!
        }
    }

    /// Get valid actions based on current state
    func getValidActions() -> [GameAction] {
        var actions: [GameAction] = []

        // Helper to check if a direction is valid (not blocked by edge or empty block)
        func canMoveOrAttack(toRow: Int, toCol: Int) -> Bool {
            // Check grid bounds
            guard toRow >= 0 && toRow < 6 && toCol >= 0 && toCol < 6 else {
                return false
            }

            let targetCell = grid.cells[toRow][toCol]

            // Can move/attack if:
            // 1. Cell is empty, OR
            // 2. Cell has an enemy (becomes attack), OR
            // 3. Cell has a block WITH an enemy on it (attack enemy on block)
            // Cannot move if: Cell has a block with NO enemy
            if case .block = targetCell.content {
                // Block exists - can only move here if there's an enemy on it
                return enemies.contains(where: { $0.row == toRow && $0.col == toCol })
            }

            return true
        }

        // Movement actions - check bounds and blocks
        // Note: row 0 = bottom of screen, row 5 = top of screen
        if canMoveOrAttack(toRow: player.row + 1, toCol: player.col) {
            actions.append(.direction(.up))  // Up = towards top = increasing row
        }
        if canMoveOrAttack(toRow: player.row - 1, toCol: player.col) {
            actions.append(.direction(.down))  // Down = towards bottom = decreasing row
        }
        if canMoveOrAttack(toRow: player.row, toCol: player.col - 1) {
            actions.append(.direction(.left))
        }
        if canMoveOrAttack(toRow: player.row, toCol: player.col + 1) {
            actions.append(.direction(.right))
        }

        // Siphon - only if player has data siphons available
        if player.dataSiphons > 0 {
            actions.append(.siphon)
        }

        // Programs - use canExecuteProgram which checks ownership, resources, and applicability
        for programType in ProgramType.allCases {
            if canExecuteProgram(programType).canExecute {
                actions.append(.program(programType))
            }
        }

        return actions
    }
}

struct GameStateSnapshot {
    let playerRow: Int
    let playerCol: Int
    let playerHealth: PlayerHealth
    let playerCredits: Int
    let playerEnergy: Int
    let playerSiphons: Int
    let playerScore: Int
    let playerAttackDamage: Int
    let turnCount: Int
    let ownedPrograms: [ProgramType]
    let enemies: [EnemySnapshot]
    let transmissions: [TransmissionSnapshot]
    let gridCells: [[CellSnapshot]]
    let showActivated: Bool
    let scheduledTasksDisabled: Bool
    let atkPlusUsedThisStage: Bool
}

struct EnemySnapshot {
    let id: UUID
    let type: EnemyType
    let row: Int
    let col: Int
    let hp: Int
    let disabledTurns: Int
    let isStunned: Bool
    let lastKnownRow: Int?
    let lastKnownCol: Int?
}

struct TransmissionSnapshot {
    let id: UUID
    let row: Int
    let col: Int
    let turnsRemaining: Int
    let enemyType: EnemyType
}

struct CellSnapshot {
    let content: CellContent
    let resources: ResourceType
    let isSiphoned: Bool
    let siphonCenter: Bool
}
