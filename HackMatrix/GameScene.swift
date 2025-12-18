import SpriteKit
import SwiftUI

class GameScene: SKScene {
    var gameState: GameState!
    var cellNodes: [[SKShapeNode]] = []
    var entityNodes: [UUID: SKNode] = [:]
    var cellContentNodes: [SKNode] = []
    var isGameOver: Bool = false
    var isAnimating: Bool = false
    var programButtons: [ProgramType: SKNode] = [:]

    // Visual CLI mode controller
    private var visualController: VisualGameController?

    // Called when scene is first presented - initialize game state and UI
    override func didMove(to view: SKView) {
        print("GameScene didMove called!")
        print("Scene size: \(size)")
        backgroundColor = .init(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)  // Darker background

        // Check for --debug-scenario command-line argument
        let useDebugScenario = CommandLine.arguments.contains("--debug-scenario")

        if useDebugScenario {
            gameState = GameState.createDebugScenario()
            print("Loaded debug scenario")
        } else {
            gameState = GameState()
            print("GameState initialized")
        }

        setupGrid()
        print("Grid setup complete")
        updateDisplay()
        print("Display updated")

        // Check for --visual-cli mode (GUI driven by stdin/stdout)
        if CommandLine.arguments.contains("--visual-cli") {
            print("[Visual CLI] Initializing controller...")
            visualController = VisualGameController(gameScene: self)
            visualController?.start()
            print("[Visual CLI] Controller started, listening for stdin commands")
        }
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
                showActivated: gameState.showActivated
            )

            if isVisible {
                drawEnemy(enemy: enemy)
            } else if enemy.type == .cryptog, let lastRow = enemy.lastKnownRow, let lastCol = enemy.lastKnownCol {
                // Draw purple border at last known position (where it was last visible)
                let cellNode = cellNodes[lastRow][lastCol]
                cellNode.strokeColor = .purple
                cellNode.lineWidth = 3
            }
        }

        // Draw transmissions
        for transmission in gameState.transmissions {
            if gameState.showActivated {
                // Show revealed transmission with enemy type overlay
                let container = SKNode()
                let cellNode = cellNodes[transmission.row][transmission.col]
                container.position = cellNode.position
                container.zPosition = 10

                // Draw transmission emoji (background)
                let transmissionLabel = SKLabelNode(text: "🌀")
                transmissionLabel.fontSize = 40
                transmissionLabel.verticalAlignmentMode = .center
                transmissionLabel.horizontalAlignmentMode = .center
                transmissionLabel.position = CGPoint(x: 0, y: 0)
                transmissionLabel.alpha = 0.5  // Make it translucent so enemy shows through
                container.addChild(transmissionLabel)

                // Draw enemy sprite (foreground)
                let enemySprite = SKSpriteNode(imageNamed: transmission.enemyType.spriteName)
                enemySprite.size = CGSize(width: Constants.cellSize * 0.6, height: Constants.cellSize * 0.6)
                enemySprite.position = CGPoint(x: 0, y: 0)
                container.addChild(enemySprite)

                addChild(container)
                entityNodes[transmission.id] = container
            } else {
                // Draw normal transmission
                drawEntity(emoji: "🌀", row: transmission.row, col: transmission.col, id: transmission.id)
            }
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

        // Draw sprite
        let sprite = SKSpriteNode(imageNamed: enemy.type.spriteName)
        sprite.size = CGSize(width: Constants.cellSize * 0.8, height: Constants.cellSize * 0.8)
        sprite.position = CGPoint(x: 0, y: 0)

        // Fade if stunned
        if enemy.isStunned {
            sprite.alpha = 0.4
        }

        container.addChild(sprite)

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

        let hudText = """
        Stage: \(gameState.currentStage)/\(Constants.totalStages)  Turn: \(gameState.turnCount)
        Health: \(gameState.player.health.emoji)  Score: \(gameState.player.score)
        💰\(gameState.player.credits)  💠\(gameState.player.energy)  💎\(gameState.player.dataSiphons)
        Controls: Arrows = Move/Attack  |  S = Siphon
        1=push 2=pull  3=crash  4=warp  5=poly  6=wait  7=debug  8=row  9=col
        Q=undo  W=step  E=siph+  R=exch  T=show  Y=reset  U=calm  I=d_bom  O=delay  P=anti-v
        A=score  S=reduc  D=atk+  F=hack
        """

        let label = SKLabelNode(text: hudText)
        label.fontName = "Menlo-Bold"
        label.fontSize = 11
        label.fontColor = .white
        label.numberOfLines = 7
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
        let buttonWidth: CGFloat = 65
        let buttonHeight: CGFloat = 32
        let buttonSpacing: CGFloat = 3
        let buttonsPerRow = 8  // Fit within 600px window: 8 * 68 = 544px
        let startX: CGFloat = 10
        let startY: CGFloat = size.height - 120 // Below HUD

        for (index, programType) in gameState.ownedPrograms.enumerated() {
            let program = Program(type: programType)
            let check = gameState.canExecuteProgram(programType)

            // Calculate position (wrap to new row based on buttonsPerRow)
            let column = index % buttonsPerRow
            let row = index / buttonsPerRow
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

            // Add cost (only show non-zero resources)
            var costParts: [String] = []
            if program.cost.credits > 0 {
                costParts.append("\(program.cost.credits)💰")
            }
            if program.cost.energy > 0 {
                costParts.append("\(program.cost.energy)💠")
            }
            let costText = costParts.isEmpty ? "Free" : costParts.joined(separator: " ")
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

    func animateExplosion(at position: (row: Int, col: Int), completion: (() -> Void)? = nil) {
        let pos = getCellPosition(row: position.row, col: position.col)

        // Create explosion circle
        let explosion = SKShapeNode(circleOfRadius: Constants.cellSize / 2)
        explosion.position = pos
        explosion.fillColor = .init(red: 1.0, green: 0.6, blue: 0.0, alpha: 0.8)
        explosion.strokeColor = .init(red: 1.0, green: 0.3, blue: 0.0, alpha: 1.0)
        explosion.lineWidth = 2
        explosion.zPosition = 100

        addChild(explosion)

        // Animate: expand and fade out
        let expand = SKAction.scale(to: 1.5, duration: 0.2)
        let fadeOut = SKAction.fadeOut(withDuration: 0.2)
        let remove = SKAction.removeFromParent()

        let group = SKAction.group([expand, fadeOut])
        let sequence = SKAction.sequence([group, remove])

        explosion.run(sequence) {
            completion?()
        }
    }

    func animateExplosions(at positions: [(row: Int, col: Int)], completion: @escaping () -> Void) {
        guard !positions.isEmpty else {
            completion()
            return
        }

        var explosionsRemaining = positions.count

        for position in positions {
            animateExplosion(at: position) {
                explosionsRemaining -= 1
                if explosionsRemaining == 0 {
                    completion()
                }
            }
        }
    }

    // Map keyboard keys to programs for ML-friendly input
    func getProgramForKeyCode(_ keyCode: UInt16) -> ProgramType? {
        // Number keys 1-9 (key codes 18-26) -> first 9 programs
        // Letter keys for remaining programs
        switch keyCode {
        // Numbers 1-9
        case 18: return .push      // 1
        case 19: return .pull      // 2
        case 20: return .crash     // 3
        case 21: return .warp      // 4
        case 23: return .poly      // 5
        case 22: return .wait      // 6
        case 26: return .debug     // 7
        case 28: return .row       // 8
        case 25: return .col       // 9

        // Letters q-p (top row)
        case 12: return .undo      // q
        case 13: return .step      // w
        case 14: return .siphPlus  // e
        case 15: return .exch      // r
        case 17: return .show      // t
        case 16: return .reset     // y
        case 32: return .calm      // u
        case 34: return .dBomb     // i
        case 31: return .delay     // o
        case 35: return .antiV     // p

        // Letters a-l (home row)
        case 0: return .score      // a
        // case 1 is 's' for siphon - skip
        case 2: return .reduc      // d
        case 3: return .atkPlus    // f
        case 5: return .hack       // g

        default: return nil
        }
    }
override func keyDown(with event: NSEvent) {
        guard !isAnimating else { return }

        // Handle restart
        if isGameOver && event.keyCode == 15 { // R key
            restartGame()
            return
        }

        // Handle siphon action
        if event.keyCode == 1 { // S key
            tryExecuteActionAndAnimate(.siphon)
            return
        }

        // Check for program keyboard shortcuts
        if let programType = getProgramForKeyCode(event.keyCode) {
            tryExecuteActionAndAnimate(.program(programType))
            return
        }

        // Check for direction keys
        switch event.keyCode {
        case 126: tryExecuteActionAndAnimate(.direction(.up))
        case 125: tryExecuteActionAndAnimate(.direction(.down))
        case 123: tryExecuteActionAndAnimate(.direction(.left))
        case 124: tryExecuteActionAndAnimate(.direction(.right))
        default: break
        }
    }



override func mouseDown(with event: NSEvent) {
        guard !isAnimating && !isGameOver else { return }

        let location = event.location(in: self)
        let clickedNodes = nodes(at: location)

        // Check if a program button was clicked (check node and parent)
        for node in clickedNodes {
            var checkNode: SKNode? = node
            // Check up to 2 levels (node or parent could be the button)
            for _ in 0..<2 {
                if let nodeName = checkNode?.name, nodeName.hasPrefix("program_") {
                    let programName = String(nodeName.dropFirst("program_".count))
                    if let programType = ProgramType.allCases.first(where: { $0.rawValue == programName }) {
                        tryExecuteActionAndAnimate(.program(programType))
                    }
                    return
                }
                checkNode = checkNode?.parent
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

    func findPlayerNode() -> SKNode? {
        let playerPos = getCellPosition(row: gameState.player.row, col: gameState.player.col)
        return entityNodes.values.first { node in
            abs(node.position.x - playerPos.x) < 1 && abs(node.position.y - playerPos.y) < 1
        }
    }

    // MARK: - Unified Action Processing

    /// Execute an action and animate the result
    func tryExecuteActionAndAnimate(_ action: GameAction) {
        guard !isAnimating else { return }
        guard gameState.player.health != .dead else { return }

        let result = gameState.tryExecuteAction(action)
        guard result.success else { return }

        if result.gameWon {
            animateActionResult(result) { [weak self] in
                self?.showVictory()
                self?.visualController?.onAnimationComplete()
            }
            return
        }

        animateActionResult(result) { [weak self] in
            guard let self = self else { return }
            self.updateDisplay()
            if result.playerDied {
                self.showGameOver()
            }
            self.visualController?.onAnimationComplete()
        }
    }

    /// Animate the result of an action
    func animateActionResult(_ result: GameState.ActionResult, completion: @escaping () -> Void) {
        isAnimating = true

        animatePlayerAction(result.playerAction) { [weak self] in
            guard let self = self else { return }
            self.animateExplosions(at: result.affectedPositions) { [weak self] in
                guard let self = self else { return }
                self.animateEnemyStepsFromResult(result.enemySteps, stepIndex: 0) { [weak self] in
                    self?.isAnimating = false
                    completion()
                }
            }
        }
    }

    func animatePlayerAction(_ playerAction: GameState.PlayerActionResult?, completion: @escaping () -> Void) {
        guard let action = playerAction else {
            completion()
            return
        }

        if let target = action.attackedTarget {
            animateAttack(fromRow: action.fromRow, fromCol: action.fromCol,
                         toRow: target.row, toCol: target.col, isPlayerAttack: true, completion: completion)
        } else if let moveTo = action.movedTo {
            // Find player node at OLD position (before move)
            let fromPos = getCellPosition(row: action.fromRow, col: action.fromCol)
            if let playerNode = entityNodes.values.first(where: {
                abs($0.position.x - fromPos.x) < 1 && abs($0.position.y - fromPos.y) < 1
            }) {
                let toPos = getCellPosition(row: moveTo.row, col: moveTo.col)
                let moveAction = SKAction.move(to: toPos, duration: 0.15)
                moveAction.timingMode = .easeInEaseOut
                playerNode.run(moveAction, completion: completion)
            } else {
                completion()
            }
        } else {
            completion()
        }
    }

    func animateEnemyStepsFromResult(_ steps: [GameState.EnemyStepResult], stepIndex: Int, completion: @escaping () -> Void) {
        guard stepIndex < steps.count else {
            completion()
            return
        }

        let step = steps[stepIndex]
        animateEnemyAttacksFromResult(step.attacks) { [weak self] in
            guard let self = self else { return }
            self.animateEnemyMovementsFromResult(step.movements) { [weak self] in
                guard let self = self else { return }
                self.updateDisplay()
                self.animateEnemyStepsFromResult(steps, stepIndex: stepIndex + 1, completion: completion)
            }
        }
    }

    func animateEnemyAttacksFromResult(_ attacks: [(enemyId: UUID, damage: Int)], completion: @escaping () -> Void) {
        guard !attacks.isEmpty else {
            completion()
            return
        }

        var remaining = attacks.count
        for attack in attacks {
            if let enemy = gameState.enemies.first(where: { $0.id == attack.enemyId }) {
                animateAttack(fromRow: enemy.row, fromCol: enemy.col,
                             toRow: gameState.player.row, toCol: gameState.player.col,
                             isPlayerAttack: false) {
                    remaining -= 1
                    if remaining == 0 { completion() }
                }
            } else {
                remaining -= 1
                if remaining == 0 { completion() }
            }
        }
    }

    func animateEnemyMovementsFromResult(_ movements: [(enemyId: UUID, fromRow: Int, fromCol: Int, toRow: Int, toCol: Int)], completion: @escaping () -> Void) {
        guard !movements.isEmpty else {
            completion()
            return
        }

        var remaining = movements.count
        for movement in movements {
            guard let enemyNode = entityNodes[movement.enemyId] else {
                remaining -= 1
                if remaining == 0 { completion() }
                continue
            }

            let fromPos = getCellPosition(row: movement.fromRow, col: movement.fromCol)
            enemyNode.position = fromPos
            let toPos = getCellPosition(row: movement.toRow, col: movement.toCol)
            let moveAction = SKAction.move(to: toPos, duration: 0.2)
            moveAction.timingMode = .easeInEaseOut
            enemyNode.run(moveAction) {
                remaining -= 1
                if remaining == 0 { completion() }
            }
        }
    }

    func showGameOver() {
        isGameOver = true

        // Save high score (died on current stage)
        HighScoreManager.shared.addScore(
            score: gameState.player.score,
            completed: false,
            stage: gameState.currentStage
        )

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
        // Save high score (completed all stages)
        HighScoreManager.shared.addScore(
            score: gameState.player.score,
            completed: true,
            stage: gameState.currentStage
        )

        let victoryLabel = SKLabelNode(text: "VICTORY! Final Score: \(gameState.player.score)")
        victoryLabel.fontSize = 48
        victoryLabel.fontColor = .green
        victoryLabel.position = CGPoint(x: 0, y: 0)
        addChild(victoryLabel)
    }
}
