import Foundation

class GameState {
    var player: Player
    var grid: Grid
    var enemies: [Enemy]
    var transmissions: [Transmission]
    var currentStage: Int
    var turnCount: Int
    var ownedPrograms: [ProgramType]  // Array to preserve acquisition order
    var cryptogsRevealed: Bool
    var scheduledTasksDisabled: Bool
    var stepActive: Bool
    var gameHistory: [GameStateSnapshot]
    var pendingSiphonTransmissions: Int
    var atkPlusUsedThisStage: Bool
    var transmissionsRevealed: Bool  // For show program

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
        self.pendingSiphonTransmissions = 0
        self.atkPlusUsedThisStage = false
        self.transmissionsRevealed = false

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
        atkPlusUsedThisStage = false
        transmissionsRevealed = false

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
                            let programType = ProgramType.allCases.randomElement()!
                            let program = Program(type: programType)
                            cell.content = .block(.question(
                                isData: false,
                                points: nil,
                                program: program,
                                transmissionSpawn: program.enemySpawnCost
                            ))
                        }
                    } else {
                        // Regular block - visible
                        if isData {
                            let pointsAndSpawn = Int.random(in: 1...9)
                            cell.content = .block(.data(points: pointsAndSpawn, transmissionSpawn: pointsAndSpawn))
                        } else {
                            let programType = ProgramType.allCases.randomElement()!
                            let program = Program(type: programType)
                            cell.content = .block(.program(program, transmissionSpawn: program.enemySpawnCost))
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
        // Spawn any pending transmissions from siphoning
        if pendingSiphonTransmissions > 0 {
            spawnRandomTransmissions(count: pendingSiphonTransmissions)
            pendingSiphonTransmissions = 0
        }

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
            return !cryptogsRevealed || !transmissionsRevealed

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

    /// Execute a program's effect
    /// Returns true if successful, false otherwise
    func executeProgram(_ type: ProgramType) -> Bool {
        let check = canExecuteProgram(type)
        guard check.canExecute else { return false }

        let program = Program(type: type)

        // Deduct resources
        player.credits -= program.cost.credits
        player.energy -= program.cost.energy

        // Execute program effect
        switch type {
        case .exch:
            // Convert 4C to 4E
            player.credits -= 4
            player.energy += 4

        case .show:
            // Reveal Cryptogs and transmissions
            cryptogsRevealed = true
            transmissionsRevealed = true

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

                // Remove the daemon
                enemies.removeAll { $0.id == nearestDaemon.id }

                // Damage and stun enemies in 8 surrounding cells
                for rowOffset in -1...1 {
                    for colOffset in -1...1 {
                        if rowOffset == 0 && colOffset == 0 { continue }
                        let checkRow = daemonRow + rowOffset
                        let checkCol = daemonCol + colOffset

                        for enemy in enemies where enemy.row == checkRow && enemy.col == checkCol {
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

        // TODO: Complex programs need more implementation details
        case .wait, .step, .push, .pull, .crash, .warp, .row, .col, .debug, .undo, .hack:
            // These need more complex implementation - skip for now
            return false
        }

        return true
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
