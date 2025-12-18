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

    // Execute action and advance game state (including enemy turn)
    // Returns: (observation, reward, isDone, info)
    func step(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let action = GameAction.fromIndex(actionIndex) else {
            fatalError("Invalid action index: \(actionIndex)")
        }
        let oldScore = gameState.player.score

        var isDone = false
        var info: [String: Any] = [:]

        // Process action (handles player action + enemy turn)
        let result = gameState.tryExecuteAction(action)

        if !result.success {
            info["invalid_action"] = true
        }

        if result.stageAdvanced {
            isDone = result.gameWon
            info["stage_complete"] = true
        }

        if result.playerDied {
            isDone = true
            info["death"] = true
        }

        // Calculate reward
        let scoreDelta = Double(gameState.player.score - oldScore)
        var reward = scoreDelta * 0.01

        if isDone {
            if result.playerDied {
                reward = 0.0
            } else {
                reward = Double(gameState.player.score) * 10.0
            }
        }

        let observation = ObservationBuilder.build(from: gameState)
        return (observation, reward, isDone, info)
    }

    // Get valid actions based on current state
    func getValidActions() -> [GameAction] {
        return gameState.getValidActions()
    }

}
