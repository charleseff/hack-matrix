import SpriteKit

class MenuScene: SKScene {
    var startButton: SKShapeNode?

    override func didMove(to view: SKView) {
        backgroundColor = .init(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        setupMenu()
    }

    func setupMenu() {
        // Remove old content
        removeAllChildren()

        // Title
        let titleLabel = SKLabelNode(text: "HackMatrix")
        titleLabel.fontName = "Menlo-Bold"
        titleLabel.fontSize = 48
        titleLabel.fontColor = .init(red: 0.5, green: 0.9, blue: 1.0, alpha: 1.0)
        titleLabel.position = CGPoint(x: size.width / 2, y: size.height - 80)
        titleLabel.verticalAlignmentMode = .center
        titleLabel.horizontalAlignmentMode = .center
        addChild(titleLabel)

        // High Scores title
        let highScoresTitle = SKLabelNode(text: "HIGH SCORES")
        highScoresTitle.fontName = "Menlo-Bold"
        highScoresTitle.fontSize = 24
        highScoresTitle.fontColor = .white
        highScoresTitle.position = CGPoint(x: size.width / 2, y: size.height - 150)
        highScoresTitle.verticalAlignmentMode = .center
        highScoresTitle.horizontalAlignmentMode = .center
        addChild(highScoresTitle)

        // Display high scores
        let highScores = HighScoreManager.shared.getHighScores()
        let startY: CGFloat = size.height - 200

        if highScores.isEmpty {
            let noScoresLabel = SKLabelNode(text: "No scores yet. Start playing!")
            noScoresLabel.fontName = "Menlo"
            noScoresLabel.fontSize = 16
            noScoresLabel.fontColor = .init(white: 0.6, alpha: 1.0)
            noScoresLabel.position = CGPoint(x: size.width / 2, y: startY - 20)
            noScoresLabel.verticalAlignmentMode = .center
            noScoresLabel.horizontalAlignmentMode = .center
            addChild(noScoresLabel)
        } else {
            for (index, entry) in highScores.enumerated() {
                let yPos = startY - CGFloat(index) * 35

                // Rank number
                let rankLabel = SKLabelNode(text: "\(index + 1).")
                rankLabel.fontName = "Menlo-Bold"
                rankLabel.fontSize = 18
                rankLabel.fontColor = .white
                rankLabel.position = CGPoint(x: size.width / 2 - 180, y: yPos)
                rankLabel.verticalAlignmentMode = .center
                rankLabel.horizontalAlignmentMode = .left
                addChild(rankLabel)

                // Score
                let scoreLabel = SKLabelNode(text: "\(entry.score)")
                scoreLabel.fontName = "Menlo-Bold"
                scoreLabel.fontSize = 18
                scoreLabel.fontColor = entry.completed ?
                    .init(red: 0.3, green: 0.9, blue: 0.3, alpha: 1.0) :  // Green for completed
                    .init(red: 0.9, green: 0.6, blue: 0.3, alpha: 1.0)    // Orange for incomplete
                scoreLabel.position = CGPoint(x: size.width / 2 - 140, y: yPos)
                scoreLabel.verticalAlignmentMode = .center
                scoreLabel.horizontalAlignmentMode = .left
                addChild(scoreLabel)

                // Status
                let statusText = entry.completed ? "COMPLETED" : "Died Stage \(entry.stage)"
                let statusLabel = SKLabelNode(text: statusText)
                statusLabel.fontName = "Menlo"
                statusLabel.fontSize = 14
                statusLabel.fontColor = entry.completed ?
                    .init(red: 0.3, green: 0.9, blue: 0.3, alpha: 1.0) :
                    .init(red: 0.9, green: 0.6, blue: 0.3, alpha: 1.0)
                statusLabel.position = CGPoint(x: size.width / 2 - 50, y: yPos)
                statusLabel.verticalAlignmentMode = .center
                statusLabel.horizontalAlignmentMode = .left
                addChild(statusLabel)

                // Date
                let dateFormatter = DateFormatter()
                dateFormatter.dateFormat = "MM/dd/yy"
                let dateString = dateFormatter.string(from: entry.date)
                let dateLabel = SKLabelNode(text: dateString)
                dateLabel.fontName = "Menlo"
                dateLabel.fontSize = 14
                dateLabel.fontColor = .init(white: 0.6, alpha: 1.0)
                dateLabel.position = CGPoint(x: size.width / 2 + 80, y: yPos)
                dateLabel.verticalAlignmentMode = .center
                dateLabel.horizontalAlignmentMode = .left
                addChild(dateLabel)
            }
        }

        // Start New Game button
        let buttonWidth: CGFloat = 250
        let buttonHeight: CGFloat = 60
        let button = SKShapeNode(rectOf: CGSize(width: buttonWidth, height: buttonHeight), cornerRadius: 10)
        button.position = CGPoint(x: size.width / 2, y: 100)
        button.fillColor = .init(red: 0.2, green: 0.6, blue: 0.8, alpha: 1.0)
        button.strokeColor = .init(red: 0.3, green: 0.8, blue: 1.0, alpha: 1.0)
        button.lineWidth = 3
        button.name = "startButton"

        let buttonLabel = SKLabelNode(text: "START NEW GAME")
        buttonLabel.fontName = "Menlo-Bold"
        buttonLabel.fontSize = 20
        buttonLabel.fontColor = .white
        buttonLabel.verticalAlignmentMode = .center
        buttonLabel.horizontalAlignmentMode = .center
        buttonLabel.position = CGPoint(x: 0, y: 0)
        button.addChild(buttonLabel)

        addChild(button)
        startButton = button
    }

    override func mouseDown(with event: NSEvent) {
        let location = event.location(in: self)
        let clickedNodes = nodes(at: location)

        for node in clickedNodes {
            if node.name == "startButton" || node.parent?.name == "startButton" {
                startGame()
                return
            }
        }
    }

    func startGame() {
        // Create and transition to GameScene
        let gameScene = GameScene()
        gameScene.size = self.size
        gameScene.scaleMode = .aspectFit

        let transition = SKTransition.fade(withDuration: 0.5)
        view?.presentScene(gameScene, transition: transition)
    }
}