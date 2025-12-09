import Foundation

class GameState {
    var player: Player
    var grid: Grid
    var enemies: [Enemy]
    var transmissions: [Transmission]
    var currentStage: Int
    var turnCount: Int
    var ownedPrograms: Set<ProgramType>
    var cryptogsRevealed: Bool
    var scheduledTasksDisabled: Bool
    var stepActive: Bool
    var gameHistory: [GameStateSnapshot]

    init() {
        self.grid = Grid()
        self.currentStage = 1
        self.turnCount = 0
        self.enemies = []
        self.transmissions = []
        self.ownedPrograms = []
        self.cryptogsRevealed = false
        self.scheduledTasksDisabled = false
        self.stepActive = false
        self.gameHistory = []

        let corners = grid.getCornerPositions()
        let playerCorner = corners.randomElement()!
        self.player = Player(row: playerCorner.0, col: playerCorner.1)

        initializeStage()
    }

    func initializeStage() {
        // Keep enemies and transmissions (they persist across stages)
        turnCount = 0
        cryptogsRevealed = false
        scheduledTasksDisabled = false
        stepActive = false

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
            while placed < blockCount {
                let row = Int.random(in: 0..<Constants.gridSize)
                let col = Int.random(in: 0..<Constants.gridSize)

                // Skip corners
                let isCorner = (row == 0 || row == Constants.gridSize - 1) &&
                               (col == 0 || col == Constants.gridSize - 1)
                if isCorner {
                    continue
                }

                // Skip positions with enemies (they persist across stages)
                let hasEnemy = enemies.contains(where: { $0.row == row && $0.col == col })
                if hasEnemy {
                    continue
                }

                let cell = tempGrid.cells[row][col]

                if case .empty = cell.content {
                    let isData = Bool.random()  // 50% data, 50% program
                    if isData {
                        // Data blocks: 1-9 points (equally likely), transmissionSpawn = points
                        let pointsAndSpawn = Int.random(in: 1...9)
                        cell.content = .block(.data(points: pointsAndSpawn, transmissionSpawn: pointsAndSpawn))
                    } else {
                        let programType = ProgramType.allCases.randomElement()!
                        let program = Program(type: programType)
                        cell.content = .block(.program(program, transmissionSpawn: program.enemySpawnCost))
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

    func processScheduledTask() {
        guard !scheduledTasksDisabled else { return }

        let interval = Constants.scheduledTaskIntervals[currentStage - 1]
        if turnCount > 0 && turnCount % interval == 0 {
            spawnRandomTransmissions(count: 1)
        }
    }

    func advanceTurn() {
        turnCount += 1

        if !stepActive {
            processTransmissions()
            processEnemyTurn()
        }

        stepActive = false
        processScheduledTask()

        for enemy in enemies {
            enemy.decrementDisable()
            enemy.isStunned = false
        }
    }

    // For animated enemy movement - start turn without processing enemies
    func beginAnimatedTurn() {
        turnCount += 1

        if !stepActive {
            processTransmissions()
            // Don't process enemy turn here - will be done step-by-step with animations
        }

        stepActive = false
        processScheduledTask()
    }

    // For animated enemy movement - finalize turn after animations complete
    func finalizeAnimatedTurn() {
        for enemy in enemies {
            enemy.decrementDisable()
            enemy.isStunned = false
        }
    }

    // For animated enemy movement - processes one step at a time
    func processEnemyStep(step: Int, enemiesWhoAttacked: inout Set<UUID>) -> Bool {
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

    func processEnemyTurn() {
        // Track which enemies have attacked (they get no more actions)
        var enemiesWhoAttacked = Set<UUID>()

        let maxSteps = enemies.map { $0.type.moveSpeed }.max() ?? 1

        for step in 0..<maxSteps {
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
        }
    }

    func getOccupiedPositions(for movingEnemy: Enemy?) -> Set<String> {
        var positions = Set<String>()
        for enemy in enemies {
            if let movingEnemy = movingEnemy, enemy.id == movingEnemy.id {
                continue
            }
            positions.insert("\(enemy.row),\(enemy.col)")
        }
        return positions
    }

    func moveEnemiesSimultaneously(step: Int, enemiesWhoAttacked: Set<UUID>) {
        var desiredMoves: [(enemy: Enemy, target: (row: Int, col: Int))] = []
        let allEnemyPositions = getOccupiedPositions(for: nil)

        for enemy in enemies {
            guard !enemy.isDisabled && !enemy.isStunned else { continue }
            guard !enemiesWhoAttacked.contains(enemy.id) else { continue }
            guard step < enemy.type.moveSpeed else { continue }
            guard !isAdjacentToPlayer(enemy) else { continue }

            let otherEnemyPositions = getOccupiedPositions(for: enemy)

            if let nextMove = Pathfinding.findNextMove(
                from: (enemy.row, enemy.col),
                to: (player.row, player.col),
                grid: grid,
                canMoveOnBlocks: enemy.type.canMoveOnBlocks,
                occupiedPositions: otherEnemyPositions
            ) {
                desiredMoves.append((enemy, nextMove))
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
                    spawnRandomTransmissions(count: transmissionSpawn)

                case .program(let program, let transmissionSpawn):
                    ownedPrograms.insert(program.type)
                    spawnRandomTransmissions(count: transmissionSpawn)

                case .question(let isData, let points, let program, let transmissionSpawn):
                    if isData, let pts = points {
                        player.score += pts
                    } else if let prog = program {
                        ownedPrograms.insert(prog.type)
                    }
                    spawnRandomTransmissions(count: transmissionSpawn)
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

        // Advance turn
        advanceTurn()

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
}

struct GameStateSnapshot {
    let playerRow: Int
    let playerCol: Int
    let playerHealth: PlayerHealth
    let playerCredits: Int
    let playerEnergy: Int
    let playerSiphons: Int
    let playerScore: Int
    let turnCount: Int
    let ownedPrograms: Set<ProgramType>
}
