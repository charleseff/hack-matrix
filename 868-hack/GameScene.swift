import SpriteKit
import SwiftUI

class GameScene: SKScene {
    var gameState: GameState!
    var cellNodes: [[SKShapeNode]] = []
    var entityNodes: [UUID: SKNode] = [:]
    var cellContentNodes: [SKNode] = []
    var isGameOver: Bool = false
    var isAnimating: Bool = false
    var enemiesWhoAttacked: Set<UUID> = []
    var programButtons: [ProgramType: SKNode] = [:]

    override func didMove(to view: SKView) {
        print("GameScene didMove called!")
        print("Scene size: \(size)")
        backgroundColor = .init(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)  // Darker background
        gameState = GameState()
        print("GameState initialized")
        setupGrid()
        print("Grid setup complete")
        updateDisplay()
        print("Display updated")
    }

    func setupGrid() {
        // Create grid cells centered in the scene
        let gridWidth = CGFloat(Constants.gridSize) * Constants.cellSize
        let gridHeight = CGFloat(Constants.gridSize) * Constants.cellSize
        let startX = (size.width - gridWidth) / 2
        let startY = (size.height - gridHeight) / 2 - 50 // Offset for HUD at top

        for row in 0..<Constants.gridSize {
            var rowNodes: [SKShapeNode] = []
            for col in 0..<Constants.gridSize {
                let cellNode = SKShapeNode(rectOf: CGSize(
                    width: Constants.cellSize - 2,
                    height: Constants.cellSize - 2
                ))

                // Position from bottom-left, row 0 = bottom
                let x = startX + CGFloat(col) * Constants.cellSize + Constants.cellSize / 2
                let y = startY + CGFloat(row) * Constants.cellSize + Constants.cellSize / 2

                cellNode.position = CGPoint(x: x, y: y)
                cellNode.strokeColor = .white
                cellNode.lineWidth = 2
                cellNode.fillColor = .clear

                addChild(cellNode)
                rowNodes.append(cellNode)
            }
            cellNodes.append(rowNodes)
        }
    }

    func updateDisplay() {
        // Clear entity nodes
        for (_, node) in entityNodes {
            node.removeFromParent()
        }
        entityNodes.removeAll()

        // Clear cell content nodes (siphons, exits)
        for node in cellContentNodes {
            node.removeFromParent()
        }
        cellContentNodes.removeAll()

        // Reset all cell borders and clear children (resource icons, etc.)
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                cellNodes[row][col].strokeColor = .white
                cellNodes[row][col].lineWidth = 2
                cellNodes[row][col].removeAllChildren()
            }
        }

        // Update grid cells
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                let cell = gameState.grid.cells[row][col]
                let cellNode = cellNodes[row][col]

                // Update cell appearance based on content
                switch cell.content {
                case .empty:
                    // Subtle dark fill for empty cells
                    cellNode.fillColor = .init(red: 0.15, green: 0.15, blue: 0.15, alpha: 1.0)
                case .block:
                    if cell.isSiphoned {
                        // Siphoned blocks are yellow/gold
                        cellNode.fillColor = .init(red: 0.7, green: 0.6, blue: 0.1, alpha: 1.0)
                        cellNode.strokeColor = .init(red: 0.9, green: 0.8, blue: 0.3, alpha: 1.0)
                        cellNode.lineWidth = 3
                    } else {
                        // Unsiphoned blocks are bright teal/cyan
                        cellNode.fillColor = .init(red: 0.1, green: 0.6, blue: 0.6, alpha: 1.0)
                        cellNode.strokeColor = .init(red: 0.2, green: 0.8, blue: 0.8, alpha: 1.0)
                        cellNode.lineWidth = 3
                    }
                case .dataSiphon:
                    cellNode.fillColor = .init(red: 0.2, green: 0.4, blue: 0.6, alpha: 0.5)
                case .exit:
                    cellNode.fillColor = .init(red: 0.2, green: 0.7, blue: 0.2, alpha: 0.5)
                }

                // Add resource icons for non-block cells, or block info for blocks
                if case .block(let blockType) = cell.content {
                    // Display block information
                    switch blockType {
                    case .data(let points, let transmissionSpawn):
                        // Show transmission spawn count in top-left corner
                        let spawnCount = cell.isSiphoned ? 0 : transmissionSpawn
                        let spawnLabel = SKLabelNode(text: "\(spawnCount)")
                        spawnLabel.fontName = "Helvetica-Bold"
                        spawnLabel.fontSize = 16
                        spawnLabel.fontColor = .init(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0) // Orange
                        spawnLabel.verticalAlignmentMode = .top
                        spawnLabel.horizontalAlignmentMode = .left
                        spawnLabel.position = CGPoint(x: -Constants.cellSize / 2 + 3, y: Constants.cellSize / 2 - 3)
                        spawnLabel.zPosition = 2
                        cellNode.addChild(spawnLabel)

                        // Show points value in center (larger)
                        let pointsLabel = SKLabelNode(text: "\(points)")
                        pointsLabel.fontName = "Helvetica-Bold"
                        pointsLabel.fontSize = 28
                        pointsLabel.fontColor = .white
                        pointsLabel.verticalAlignmentMode = .center
                        pointsLabel.horizontalAlignmentMode = .center
                        pointsLabel.position = CGPoint(x: 0, y: 0)
                        pointsLabel.zPosition = 2
                        cellNode.addChild(pointsLabel)

                    case .program(let program, let transmissionSpawn):
                        // Show transmission spawn count in top-left corner
                        let spawnCount = cell.isSiphoned ? 0 : transmissionSpawn
                        let spawnLabel = SKLabelNode(text: "\(spawnCount)")
                        spawnLabel.fontName = "Helvetica-Bold"
                        spawnLabel.fontSize = 16
                        spawnLabel.fontColor = .init(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0) // Orange
                        spawnLabel.verticalAlignmentMode = .top
                        spawnLabel.horizontalAlignmentMode = .left
                        spawnLabel.position = CGPoint(x: -Constants.cellSize / 2 + 3, y: Constants.cellSize / 2 - 3)
                        spawnLabel.zPosition = 2
                        cellNode.addChild(spawnLabel)

                        // Show program name in center
                        let programLabel = SKLabelNode(text: program.type.displayName)
                        programLabel.fontName = "Menlo-Bold"
                        programLabel.fontSize = 14
                        programLabel.fontColor = .init(red: 0.5, green: 0.9, blue: 1.0, alpha: 1.0) // Cyan
                        programLabel.verticalAlignmentMode = .center
                        programLabel.horizontalAlignmentMode = .center
                        programLabel.position = CGPoint(x: 0, y: 0)
                        programLabel.zPosition = 2
                        cellNode.addChild(programLabel)

                    case .question(let isData, let points, let program, let transmissionSpawn):
                        // Show ? for transmission spawn count in top-left corner (hidden until siphoned)
                        let spawnText = "?"
                        let spawnLabel = SKLabelNode(text: spawnText)
                        spawnLabel.fontName = "Helvetica-Bold"
                        spawnLabel.fontSize = 16
                        spawnLabel.fontColor = .init(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0) // Orange
                        spawnLabel.verticalAlignmentMode = .top
                        spawnLabel.horizontalAlignmentMode = .left
                        spawnLabel.position = CGPoint(x: -Constants.cellSize / 2 + 3, y: Constants.cellSize / 2 - 3)
                        spawnLabel.zPosition = 2
                        cellNode.addChild(spawnLabel)

                        // Show question mark in center
                        let questionLabel = SKLabelNode(text: "?")
                        questionLabel.fontName = "Helvetica-Bold"
                        questionLabel.fontSize = 32
                        questionLabel.fontColor = .init(red: 1.0, green: 0.8, blue: 0.2, alpha: 1.0) // Yellow
                        questionLabel.verticalAlignmentMode = .center
                        questionLabel.horizontalAlignmentMode = .center
                        questionLabel.position = CGPoint(x: 0, y: 0)
                        questionLabel.zPosition = 2
                        cellNode.addChild(questionLabel)
                    }
                } else {
                    // Show smiley for siphon center, otherwise show resources
                    if cell.siphonCenter {
                        let smiley = SKLabelNode(text: "😊")
                        smiley.fontSize = 14
                        smiley.verticalAlignmentMode = .top
                        smiley.horizontalAlignmentMode = .left
                        smiley.position = CGPoint(x: -Constants.cellSize / 2 + 3, y: Constants.cellSize / 2 - 3)
                        smiley.zPosition = 2
                        cellNode.addChild(smiley)
                    } else {
                        // Show resource icons stacked vertically in top-left corner
                        let (icon, amount) = resourceIcon(for: cell.resources)
                        if !icon.isEmpty {
                            for i in 0..<amount {
                                let resourceIcon = SKLabelNode(text: icon)
                                resourceIcon.fontSize = 12  // Smaller (was 14)
                                resourceIcon.alpha = 0.6    // More subtle/translucent
                                resourceIcon.verticalAlignmentMode = .top
                                resourceIcon.horizontalAlignmentMode = .left
                                // Position in top-left corner, stack vertically
                                let xOffset: CGFloat = -Constants.cellSize / 2 + 3
                                let yOffset: CGFloat = Constants.cellSize / 2 - CGFloat(i * 12) - 3  // Adjust spacing for smaller size
                                resourceIcon.position = CGPoint(x: xOffset, y: yOffset)
                                resourceIcon.zPosition = 2
                                cellNode.addChild(resourceIcon)
                            }
                        }
                    }
                }

                // Add emoji for data siphons and exits
                if case .dataSiphon = cell.content {
                    let emoji = SKLabelNode(text: "💎")
                    emoji.fontSize = 36
                    emoji.verticalAlignmentMode = .center
                    emoji.horizontalAlignmentMode = .center
                    emoji.position = cellNode.position
                    emoji.zPosition = 1 // Behind entities
                    addChild(emoji)
                    cellContentNodes.append(emoji)
                } else if case .exit = cell.content {
                    let emoji = SKLabelNode(text: "🌪️")
                    emoji.fontSize = 36
                    emoji.verticalAlignmentMode = .center
                    emoji.horizontalAlignmentMode = .center
                    emoji.position = cellNode.position
                    emoji.zPosition = 1 // Behind entities
                    addChild(emoji)
                    cellContentNodes.append(emoji)
                }
            }
        }

        // Draw player
        drawEntity(emoji: gameState.player.health.emoji, row: gameState.player.row, col: gameState.player.col, id: UUID())

        // Draw enemies
        for enemy in gameState.enemies {
            let isVisible = enemy.isVisible(
                playerRow: gameState.player.row,
                playerCol: gameState.player.col,
                cryptogsRevealed: gameState.cryptogsRevealed
            )

            if isVisible {
                drawEnemy(enemy: enemy)
            } else if enemy.type == .cryptog, let lastRow = enemy.lastKnownRow, let lastCol = enemy.lastKnownCol {
                // Draw purple border for last known position
                let cellNode = cellNodes[lastRow][lastCol]
                cellNode.strokeColor = .purple
                cellNode.lineWidth = 3
            }
        }

        // Draw transmissions
        for transmission in gameState.transmissions {
            drawEntity(emoji: "🌀", row: transmission.row, col: transmission.col, id: transmission.id)
        }

        // Update HUD
        updateHUD()
        updateProgramButtons()
    }

    func drawEntity(emoji: String, row: Int, col: Int, id: UUID) {
        let label = SKLabelNode(text: emoji)
        label.fontSize = 40
        label.verticalAlignmentMode = .center
        label.horizontalAlignmentMode = .center
        label.zPosition = 10 // On top of cell content

        let cellNode = cellNodes[row][col]
        label.position = cellNode.position

        addChild(label)
        entityNodes[id] = label
    }

    func drawEnemy(enemy: Enemy) {
        let container = SKNode()
        let cellNode = cellNodes[enemy.row][enemy.col]
        container.position = cellNode.position
        container.zPosition = 10 // On top of cell content

        // Draw emoji
        let emojiLabel = SKLabelNode(text: enemy.type.emoji)
        emojiLabel.fontSize = 40
        emojiLabel.verticalAlignmentMode = .center
        emojiLabel.horizontalAlignmentMode = .center
        emojiLabel.position = CGPoint(x: 0, y: 0)

        // Fade if stunned
        if enemy.isStunned {
            emojiLabel.alpha = 0.4
        }

        container.addChild(emojiLabel)

        // Show HP indicator
        let hpLabel = SKLabelNode(text: "\(enemy.hp)/\(enemy.type.maxHP)")
        hpLabel.fontName = "Helvetica-Bold"
        hpLabel.fontSize = 13
        hpLabel.fontColor = enemy.hp < enemy.type.maxHP ? .init(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0) : .white
        hpLabel.verticalAlignmentMode = .center
        hpLabel.horizontalAlignmentMode = .center
        hpLabel.position = CGPoint(x: 0, y: -25)

        // Fade HP label too if stunned
        if enemy.isStunned {
            hpLabel.alpha = 0.4
        }

        container.addChild(hpLabel)

        addChild(container)
        entityNodes[enemy.id] = container
    }

    func resourceLabel(for resource: ResourceType) -> String {
        switch resource {
        case .credits(let amount):
            return "\(amount)C"
        case .energy(let amount):
            return "\(amount)E"
        case .none:
            return ""
        }
    }

    func resourceIcon(for resource: ResourceType) -> (icon: String, amount: Int) {
        switch resource {
        case .credits(let amount):
            return ("💰", amount)
        case .energy(let amount):
            return ("💠", amount)  // Blue diamond
        case .none:
            return ("", 0)
        }
    }

    func updateHUD() {
        // Remove old HUD
        childNode(withName: "hud")?.removeFromParent()

        let hud = SKNode()
        hud.name = "hud"

        // Format owned programs list (in acquisition order)
        let programsList: String
        if gameState.ownedPrograms.isEmpty {
            programsList = "None"
        } else {
            programsList = gameState.ownedPrograms
                .map { $0.displayName }
                .joined(separator: ", ")
        }

        let hudText = """
        Stage: \(gameState.currentStage)/\(Constants.totalStages)  Turn: \(gameState.turnCount)
        Health: \(gameState.player.health.emoji)  Score: \(gameState.player.score)
        💰\(gameState.player.credits)  💠\(gameState.player.energy)  💎\(gameState.player.dataSiphons)
        Programs: \(programsList)
        Controls: Arrows = Move/Attack  |  S = Siphon (+ pattern)
        """

        let label = SKLabelNode(text: hudText)
        label.fontName = "Menlo-Bold"
        label.fontSize = 13
        label.fontColor = .white
        label.numberOfLines = 5
        label.horizontalAlignmentMode = .left
        label.verticalAlignmentMode = .top
        label.position = CGPoint(x: 10, y: size.height - 10)

        hud.addChild(label)
        addChild(hud)
    }

    func updateProgramButtons() {
        // Remove old buttons
        childNode(withName: "programButtons")?.removeFromParent()
        programButtons.removeAll()

        let buttonsNode = SKNode()
        buttonsNode.name = "programButtons"

        // Use programs in acquisition order
        let buttonWidth: CGFloat = 80
        let buttonHeight: CGFloat = 35
        let buttonSpacing: CGFloat = 5
        let startX: CGFloat = 10
        let startY: CGFloat = size.height - 120 // Below HUD

        for (index, programType) in gameState.ownedPrograms.enumerated() {
            let program = Program(type: programType)
            let check = gameState.canExecuteProgram(programType)

            // Calculate position (wrap to new row every 10 buttons)
            let column = index % 10
            let row = index / 10
            let xPos = startX + CGFloat(column) * (buttonWidth + buttonSpacing)
            let yPos = startY - CGFloat(row) * (buttonHeight + buttonSpacing)

            // Create button background
            let buttonBg = SKShapeNode(rectOf: CGSize(width: buttonWidth, height: buttonHeight), cornerRadius: 5)
            buttonBg.position = CGPoint(x: xPos + buttonWidth / 2, y: yPos - buttonHeight / 2)

            if check.canExecute {
                // Enabled: cyan background
                buttonBg.fillColor = .init(red: 0.2, green: 0.6, blue: 0.8, alpha: 1.0)
                buttonBg.strokeColor = .init(red: 0.3, green: 0.8, blue: 1.0, alpha: 1.0)
            } else {
                // Disabled: grey background
                buttonBg.fillColor = .init(red: 0.3, green: 0.3, blue: 0.3, alpha: 1.0)
                buttonBg.strokeColor = .init(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
            }
            buttonBg.lineWidth = 2

            // Add program name
            let nameLabel = SKLabelNode(text: programType.displayName)
            nameLabel.fontName = "Menlo-Bold"
            nameLabel.fontSize = 11
            nameLabel.fontColor = check.canExecute ? .white : .init(white: 0.6, alpha: 1.0)
            nameLabel.verticalAlignmentMode = .center
            nameLabel.horizontalAlignmentMode = .center
            nameLabel.position = CGPoint(x: 0, y: 5)
            buttonBg.addChild(nameLabel)

            // Add cost
            let costText = "\(program.cost.credits)💰 \(program.cost.energy)💠"
            let costLabel = SKLabelNode(text: costText)
            costLabel.fontName = "Menlo"
            costLabel.fontSize = 9
            costLabel.fontColor = check.canExecute ? .init(white: 0.9, alpha: 1.0) : .init(white: 0.5, alpha: 1.0)
            costLabel.verticalAlignmentMode = .center
            costLabel.horizontalAlignmentMode = .center
            costLabel.position = CGPoint(x: 0, y: -10)
            buttonBg.addChild(costLabel)

            // Store button node for click detection
            buttonBg.name = "program_\(programType.rawValue)"
            programButtons[programType] = buttonBg

            buttonsNode.addChild(buttonBg)
        }

        addChild(buttonsNode)
    }

    // MARK: - Animation Functions

    func getCellPosition(row: Int, col: Int) -> CGPoint {
        return cellNodes[row][col].position
    }

    func animateEntity(node: SKNode, fromRow: Int, fromCol: Int, toRow: Int, toCol: Int, duration: TimeInterval, completion: @escaping () -> Void) {
        let fromPos = getCellPosition(row: fromRow, col: fromCol)
        let toPos = getCellPosition(row: toRow, col: toCol)

        node.position = fromPos
        let moveAction = SKAction.move(to: toPos, duration: duration)
        moveAction.timingMode = .easeInEaseOut

        node.run(moveAction) {
            completion()
        }
    }

    func animatePlayerMove(fromRow: Int, fromCol: Int, toRow: Int, toCol: Int, completion: @escaping () -> Void) {
        let playerNode = entityNodes.values.first { node in
            // Find player node (it doesn't have a specific ID we track)
            let pos = getCellPosition(row: gameState.player.row, col: gameState.player.col)
            return abs(node.position.x - pos.x) < 1 && abs(node.position.y - pos.y) < 1
        }

        guard let node = playerNode else {
            completion()
            return
        }

        animateEntity(node: node, fromRow: fromRow, fromCol: fromCol, toRow: toRow, toCol: toCol, duration: 0.15, completion: completion)
    }

    func animateAttack(fromRow: Int, fromCol: Int, toRow: Int, toCol: Int, isPlayerAttack: Bool = true, completion: @escaping () -> Void) {
        let fromPos = getCellPosition(row: fromRow, col: fromCol)
        let toPos = getCellPosition(row: toRow, col: toCol)

        // Create laser line
        let path = CGMutablePath()
        path.move(to: fromPos)
        path.addLine(to: toPos)

        let laser = SKShapeNode(path: path)
        // Red for player attacks, purple for enemy attacks
        laser.strokeColor = isPlayerAttack ?
            .init(red: 1.0, green: 0.2, blue: 0.2, alpha: 1.0) :
            .init(red: 0.8, green: 0.2, blue: 1.0, alpha: 1.0)
        laser.lineWidth = 3
        laser.glowWidth = 2
        laser.zPosition = 100 // On top of everything

        addChild(laser)

        // Animate: fade in quickly, then fade out
        let fadeIn = SKAction.fadeAlpha(to: 1.0, duration: 0.05)
        let wait = SKAction.wait(forDuration: 0.1)
        let fadeOut = SKAction.fadeOut(withDuration: 0.15)
        let remove = SKAction.removeFromParent()

        let sequence = SKAction.sequence([fadeIn, wait, fadeOut, remove])
        laser.run(sequence) {
            completion()
        }
    }

    override func keyDown(with event: NSEvent) {
        // Block input during animations
        guard !isAnimating else { return }

        // Handle restart
        if isGameOver && event.keyCode == 15 { // R key
            restartGame()
            return
        }

        guard gameState.player.health != .dead else { return }

        // Handle siphon action
        if event.keyCode == 1 { // S key
            if gameState.performSiphon() {
                // Begin animated turn (processes transmissions and scheduled tasks, but not enemy movement)
                isAnimating = true
                gameState.beginAnimatedTurn()
                updateDisplay()

                // Process enemy movement step-by-step with animations
                enemiesWhoAttacked = Set<UUID>()
                animateEnemySteps(currentStep: 0)
            }
            return
        }

        var direction: Direction?

        switch event.keyCode {
        case 126: direction = .up      // Arrow up
        case 125: direction = .down    // Arrow down
        case 123: direction = .left    // Arrow left
        case 124: direction = .right   // Arrow right
        default: break
        }

        if let dir = direction {
            handlePlayerMove(direction: dir)
        }
    }

    override func mouseDown(with event: NSEvent) {
        // Block input during animations or game over
        guard !isAnimating && !isGameOver else { return }

        let location = event.location(in: self)
        let clickedNodes = nodes(at: location)

        // Check if a program button was clicked
        for node in clickedNodes {
            if let nodeName = node.name, nodeName.hasPrefix("program_") {
                let programName = String(nodeName.dropFirst("program_".count))

                // Find the program type
                if let programType = ProgramType.allCases.first(where: { $0.rawValue == programName }) {
                    // Try to execute the program
                    if gameState.executeProgram(programType) {
                        // Program executed successfully, update display
                        updateDisplay()
                    }
                }
                return
            }
        }
    }

    func restartGame() {
        // Remove all nodes
        removeAllChildren()

        // Reset state
        cellNodes = []
        entityNodes = [:]
        cellContentNodes = []
        isGameOver = false

        // Reinitialize game
        gameState = GameState()
        setupGrid()
        updateDisplay()
    }

    func handlePlayerMove(direction: Direction) {
        // Check for target in line of fire (transmission or enemy, whichever is closer)
        let targetResult = findTargetInLineOfFire(direction: direction)

        if let transmission = targetResult.transmission {
            // Animate attack
            isAnimating = true
            animateAttack(fromRow: gameState.player.row, fromCol: gameState.player.col,
                         toRow: transmission.row, toCol: transmission.col) { [weak self] in
                guard let self = self else { return }

                // Destroy the transmission (1 HP)
                self.gameState.transmissions.removeAll { $0.id == transmission.id }

                // Begin animated turn (processes transmissions and scheduled tasks, but not enemy movement)
                self.gameState.beginAnimatedTurn()
                self.updateDisplay()

                // Process enemy movement step-by-step with animations
                self.enemiesWhoAttacked = Set<UUID>()
                self.animateEnemySteps(currentStep: 0)
            }
            return
        }

        if let target = targetResult.enemy {
            // Animate attack
            isAnimating = true
            animateAttack(fromRow: gameState.player.row, fromCol: gameState.player.col,
                         toRow: target.row, toCol: target.col) { [weak self] in
                guard let self = self else { return }

                // Attack the enemy
                target.takeDamage(self.gameState.player.attackDamage)

                // Stun the enemy if it survives
                if target.hp > 0 {
                    target.isStunned = true
                } else {
                    // Remove dead enemy
                    self.gameState.enemies.removeAll { $0.id == target.id }
                }

                // Begin animated turn (processes transmissions and scheduled tasks, but not enemy movement)
                self.gameState.beginAnimatedTurn()
                self.updateDisplay()

                // Process enemy movement step-by-step with animations
                self.enemiesWhoAttacked = Set<UUID>()
                self.animateEnemySteps(currentStep: 0)
            }
            return
        }

        // No enemy to attack, try to move
        let offset = direction.offset
        let newRow = gameState.player.row + offset.row
        let newCol = gameState.player.col + offset.col

        if gameState.player.canMoveTo(row: newRow, col: newCol, grid: gameState.grid) {
            let oldRow = gameState.player.row
            let oldCol = gameState.player.col

            gameState.player.row = newRow
            gameState.player.col = newCol

            // Collect resources/siphons
            let cell = gameState.grid.cells[newRow][newCol]
            if cell.hasDataSiphon {
                gameState.player.dataSiphons += 1
                cell.content = .empty
            }

            // Check for exit
            if cell.isExit {
                advanceToNextStage()
                return
            }

            // Animate player movement
            isAnimating = true
            updateDisplay() // Redraw everything at new positions

            // Find and animate player node
            if let playerNode = findPlayerNode() {
                let fromPos = getCellPosition(row: oldRow, col: oldCol)
                playerNode.position = fromPos

                let moveAction = SKAction.move(to: getCellPosition(row: newRow, col: newCol), duration: 0.15)
                moveAction.timingMode = .easeInEaseOut

                playerNode.run(moveAction) { [weak self] in
                    self?.handlePlayerMoveComplete()
                }
            } else {
                // No animation needed, proceed immediately
                handlePlayerMoveComplete()
            }
        }
    }

    func findPlayerNode() -> SKNode? {
        let playerPos = getCellPosition(row: gameState.player.row, col: gameState.player.col)
        return entityNodes.values.first { node in
            abs(node.position.x - playerPos.x) < 1 && abs(node.position.y - playerPos.y) < 1
        }
    }

    func handlePlayerMoveComplete() {
        // Begin turn (processes transmissions and scheduled tasks, but not enemy movement)
        gameState.beginAnimatedTurn()
        updateDisplay()

        // Process enemy movement step-by-step with animations
        enemiesWhoAttacked = Set<UUID>()
        animateEnemySteps(currentStep: 0)
    }

    func animateEnemySteps(currentStep: Int) {
        // Capture positions before this step
        var enemyOldPositions: [UUID: (row: Int, col: Int)] = [:]
        for enemy in gameState.enemies {
            enemyOldPositions[enemy.id] = (enemy.row, enemy.col)
        }

        // Track which enemies are adjacent to player (will attack)
        let attackingEnemies = gameState.enemies.filter { enemy in
            !enemy.isDisabled && !enemy.isStunned &&
            !enemiesWhoAttacked.contains(enemy.id) &&
            currentStep < enemy.type.moveSpeed &&
            gameState.isAdjacentToPlayer(enemy)
        }

        // Process this step (enemies move and attack)
        let hasMoreSteps = gameState.processEnemyStep(step: currentStep, enemiesWhoAttacked: &enemiesWhoAttacked)

        // Animate enemy attacks first, then movements
        animateEnemyAttacks(attackingEnemies: attackingEnemies) { [weak self] in
            guard let self = self else { return }

            // Animate enemies to their new positions (don't call updateDisplay yet - it would destroy nodes)
            self.animateEnemyMovements(oldPositions: enemyOldPositions) { [weak self] in
                guard let self = self else { return }

                // Now update display after animations are complete
                self.updateDisplay()

                if hasMoreSteps {
                    // Continue to next step
                    self.animateEnemySteps(currentStep: currentStep + 1)
                } else {
                    // All steps complete, finalize turn
                    self.gameState.finalizeAnimatedTurn()
                    self.updateDisplay()
                    self.isAnimating = false

                    // Check for game over
                    if self.gameState.player.health == .dead {
                        self.showGameOver()
                    }
                }
            }
        }
    }

    func animateEnemyAttacks(attackingEnemies: [Enemy], completion: @escaping () -> Void) {
        guard !attackingEnemies.isEmpty else {
            completion()
            return
        }

        var attacksRemaining = attackingEnemies.count

        for enemy in attackingEnemies {
            animateAttack(fromRow: enemy.row, fromCol: enemy.col,
                         toRow: gameState.player.row, toCol: gameState.player.col,
                         isPlayerAttack: false) {
                attacksRemaining -= 1
                if attacksRemaining == 0 {
                    completion()
                }
            }
        }
    }

    func animateEnemyMovements(oldPositions: [UUID: (row: Int, col: Int)], completion: @escaping () -> Void) {
        var animationsRunning = 0

        for enemy in gameState.enemies {
            guard let oldPos = oldPositions[enemy.id] else { continue }

            // Skip if enemy didn't move
            if oldPos.row == enemy.row && oldPos.col == enemy.col {
                continue
            }

            guard let enemyNode = entityNodes[enemy.id] else { continue }

            animationsRunning += 1

            let fromPos = getCellPosition(row: oldPos.row, col: oldPos.col)
            enemyNode.position = fromPos

            let moveAction = SKAction.move(to: getCellPosition(row: enemy.row, col: enemy.col), duration: 0.2)
            moveAction.timingMode = .easeInEaseOut

            enemyNode.run(moveAction) {
                animationsRunning -= 1
                if animationsRunning == 0 {
                    completion()
                }
            }
        }

        // If no enemies moved, complete immediately
        if animationsRunning == 0 {
            completion()
        }
    }

    func advanceToNextStage() {
        // Gain 1 HP (up to max of 3)
        if gameState.player.health.rawValue < 3 {
            gameState.player.health = PlayerHealth(rawValue: gameState.player.health.rawValue + 1) ?? .full
        }

        if gameState.currentStage < Constants.totalStages {
            gameState.currentStage += 1
            gameState.initializeStage()
            updateDisplay()
        } else {
            showVictory()
        }
    }

    func findTargetInLineOfFire(direction: Direction) -> (transmission: Transmission?, enemy: Enemy?) {
        let offset = direction.offset
        var currentRow = gameState.player.row
        var currentCol = gameState.player.col

        // Move in the direction until we hit something
        while true {
            currentRow += offset.row
            currentCol += offset.col

            // Check bounds
            guard currentRow >= 0 && currentRow < Constants.gridSize &&
                  currentCol >= 0 && currentCol < Constants.gridSize else {
                return (nil, nil)
            }

            // Check for transmission first (same priority as enemy - first one hit)
            if let transmission = gameState.transmissions.first(where: { $0.row == currentRow && $0.col == currentCol }) {
                return (transmission, nil)
            }

            // Check for enemy
            if let enemy = gameState.enemies.first(where: { $0.row == currentRow && $0.col == currentCol }) {
                return (nil, enemy)
            }

            // If no target, check for block (blocks line of fire)
            if gameState.grid.cells[currentRow][currentCol].hasBlock {
                return (nil, nil)
            }
        }
    }

    func findEnemyInLineOfFire(direction: Direction) -> Enemy? {
        let offset = direction.offset
        var currentRow = gameState.player.row
        var currentCol = gameState.player.col

        // Move in the direction until we hit something
        while true {
            currentRow += offset.row
            currentCol += offset.col

            // Check bounds
            guard currentRow >= 0 && currentRow < Constants.gridSize &&
                  currentCol >= 0 && currentCol < Constants.gridSize else {
                return nil
            }

            // Check for enemy first (even if on a block)
            if let enemy = gameState.enemies.first(where: { $0.row == currentRow && $0.col == currentCol }) {
                return enemy
            }

            // If no enemy, check for block (blocks line of fire)
            if gameState.grid.cells[currentRow][currentCol].hasBlock {
                return nil
            }
        }
    }

    func showGameOver() {
        isGameOver = true

        let gameOverLabel = SKLabelNode(text: "GAME OVER")
        gameOverLabel.fontSize = 48
        gameOverLabel.fontColor = .red
        gameOverLabel.position = CGPoint(x: size.width / 2, y: size.height / 2 + 20)
        addChild(gameOverLabel)

        let restartLabel = SKLabelNode(text: "Press R to Restart")
        restartLabel.fontSize = 24
        restartLabel.fontColor = .white
        restartLabel.position = CGPoint(x: size.width / 2, y: size.height / 2 - 30)
        addChild(restartLabel)
    }

    func showVictory() {
        let victoryLabel = SKLabelNode(text: "VICTORY! Final Score: \(gameState.player.score)")
        victoryLabel.fontSize = 48
        victoryLabel.fontColor = .green
        victoryLabel.position = CGPoint(x: 0, y: 0)
        addChild(victoryLabel)
    }
}
