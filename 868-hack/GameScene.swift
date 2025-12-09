import SpriteKit
import SwiftUI

class GameScene: SKScene {
    var gameState: GameState!
    var cellNodes: [[SKShapeNode]] = []
    var entityNodes: [UUID: SKNode] = [:]

    override func didMove(to view: SKView) {
        print("GameScene didMove called!")
        print("Scene size: \(size)")
        backgroundColor = .darkGray
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

        // Reset all cell borders to white (clears old Cryptog purple borders)
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                cellNodes[row][col].strokeColor = .white
                cellNodes[row][col].lineWidth = 2
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
                    cellNode.fillColor = .clear
                case .block:
                    cellNode.fillColor = .init(white: 0.3, alpha: 1.0)
                case .dataSiphon:
                    cellNode.fillColor = .init(red: 0.2, green: 0.4, blue: 0.6, alpha: 1.0)
                case .exit:
                    cellNode.fillColor = .init(red: 0.2, green: 0.6, blue: 0.2, alpha: 1.0)
                }

                // Add resource indicator
                if case .empty = cell.content {
                    let label = SKLabelNode(text: resourceLabel(for: cell.resources))
                    label.fontSize = 10
                    label.fontColor = .gray
                    label.position = CGPoint(x: 0, y: -20)
                    cellNode.addChild(label)
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
                drawEntity(emoji: enemy.type.emoji, row: enemy.row, col: enemy.col, id: enemy.id)
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
    }

    func drawEntity(emoji: String, row: Int, col: Int, id: UUID) {
        let label = SKLabelNode(text: emoji)
        label.fontSize = 40
        label.verticalAlignmentMode = .center
        label.horizontalAlignmentMode = .center

        let cellNode = cellNodes[row][col]
        label.position = cellNode.position

        addChild(label)
        entityNodes[id] = label
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

    func updateHUD() {
        // Remove old HUD
        childNode(withName: "hud")?.removeFromParent()

        let hud = SKNode()
        hud.name = "hud"

        let hudText = """
        Stage: \(gameState.currentStage)/\(Constants.totalStages)  Turn: \(gameState.turnCount)
        Health: \(gameState.player.health.emoji)  Score: \(gameState.player.score)
        Credits: \(gameState.player.credits)  Energy: \(gameState.player.energy)  Siphons: \(gameState.player.dataSiphons)
        """

        let label = SKLabelNode(text: hudText)
        label.fontSize = 14
        label.fontColor = .white
        label.numberOfLines = 3
        label.horizontalAlignmentMode = .left
        label.verticalAlignmentMode = .top
        label.position = CGPoint(x: 10, y: size.height - 10)

        hud.addChild(label)
        addChild(hud)
    }

    override func keyDown(with event: NSEvent) {
        guard gameState.player.health != .dead else { return }

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

    func handlePlayerMove(direction: Direction) {
        // Check for target in line of fire (transmission or enemy, whichever is closer)
        let targetResult = findTargetInLineOfFire(direction: direction)

        if let transmission = targetResult.transmission {
            // Destroy the transmission (1 HP)
            gameState.transmissions.removeAll { $0.id == transmission.id }

            // Advance turn after attack
            gameState.advanceTurn()
            updateDisplay()

            // Check for game over
            if gameState.player.health == .dead {
                showGameOver()
            }
            return
        }

        if let target = targetResult.enemy {
            // Attack the enemy
            target.takeDamage(gameState.player.attackDamage)

            // Stun the enemy if it survives
            if target.hp > 0 {
                target.isStunned = true
            } else {
                // Remove dead enemy
                gameState.enemies.removeAll { $0.id == target.id }
            }

            // Advance turn after attack
            gameState.advanceTurn()
            updateDisplay()

            // Check for game over
            if gameState.player.health == .dead {
                showGameOver()
            }
            return
        }

        // No enemy to attack, try to move
        let offset = direction.offset
        let newRow = gameState.player.row + offset.row
        let newCol = gameState.player.col + offset.col

        if gameState.player.canMoveTo(row: newRow, col: newCol, grid: gameState.grid) {
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

            gameState.advanceTurn()
            updateDisplay()

            // Check for game over
            if gameState.player.health == .dead {
                showGameOver()
            }
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
        let gameOverLabel = SKLabelNode(text: "GAME OVER")
        gameOverLabel.fontSize = 48
        gameOverLabel.fontColor = .red
        gameOverLabel.position = CGPoint(x: 0, y: 0)
        addChild(gameOverLabel)
    }

    func showVictory() {
        let victoryLabel = SKLabelNode(text: "VICTORY! Final Score: \(gameState.player.score)")
        victoryLabel.fontSize = 48
        victoryLabel.fontColor = .green
        victoryLabel.position = CGPoint(x: 0, y: 0)
        addChild(victoryLabel)
    }
}
