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
        let result: GameState.ActionResult = gameState.tryExecuteAction(action)

        if !result.success {
            // Invalid action - terminate episode to prevent infinite loops
            info["invalid_action"] = true
            isDone = true
            infoLog("HeadlessGame", "❌ Invalid action \(action) attempted - terminating episode")
        }

        if result.stageAdvanced {
            isDone = result.gameWon
            if isDone {
                infoLog(
                    "Completed the game! With points: \(gameState.player.score)")
            }

            info["stage_complete"] = true
        }

        if result.playerDied {
            isDone = true
            info["death"] = true
        }

        // Add reward breakdown to info dict for Python-side tracking
        info["reward_breakdown"] = [
            "stage": result.rewardBreakdown.stageCompletion,
            "score": result.rewardBreakdown.scoreGain,
            "kills": result.rewardBreakdown.kills,
            "dataSiphon": result.rewardBreakdown.dataSiphonCollected,
            "distance": result.rewardBreakdown.distanceShaping,
            "victory": result.rewardBreakdown.victory,
            "death": result.rewardBreakdown.deathPenalty
        ]

        let observation = ObservationBuilder.build(from: gameState)
        let reward = result.rewardBreakdown.total

        if result.stageAdvanced {
            debugLog(
                "Advanced to stage \(observation.stage)"
            )
        } else {
            debugLog(
                "Step \(String(describing: action)) -> reward: \(String(format: "%.3f", reward)), done: \(isDone), stage: \(observation.stage), credits: \(gameState.player.credits), energy: \(gameState.player.energy)"
            )
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

}
