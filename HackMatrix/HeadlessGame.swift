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

        let observation = ObservationBuilder.build(from: gameState)
        return (observation, result.reward, isDone, info)
    }

    // Get valid actions based on current state
    func getValidActions() -> [GameAction] {
        return gameState.getValidActions()
    }

}
