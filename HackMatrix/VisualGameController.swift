import Foundation

// Handles stdin/stdout protocol for visual CLI mode
// Keeps GameScene focused on rendering/animation
class VisualGameController: GameCommandExecutor {
    weak var gameScene: GameScene?
    private let commandReader = StdinCommandReader()

    // Callback invoked when animation completes - signals semaphore to unblock stdin thread
    private var animationCompleteForStdin: ((GameObservation, Double, Bool, [String: Any]) -> Void)?

    init(gameScene: GameScene) {
        self.gameScene = gameScene
        commandReader.executor = self
    }

    func start() {
        Thread {
            self.commandReader.start()
        }.start()
    }

    // MARK: - GameCommandExecutor

    func executeReset() -> GameObservation {
        print("[Visual CLI] executeReset called")
        guard let scene = gameScene else {
            fatalError("GameScene not available")
        }

        DispatchQueue.main.sync {
            print("[Visual CLI] Resetting game state on main thread")
            scene.gameState = GameState()
            scene.updateDisplay()
        }

        let obs = ObservationBuilder.build(from: scene.gameState)
        print("[Visual CLI] Observation built, returning")
        return obs
    }

    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let scene = gameScene,
              let action = GameAction.fromIndex(actionIndex) else {
            fatalError("Invalid action or scene not available")
        }

        var result: (GameObservation, Double, Bool, [String: Any])?
        let semaphore = DispatchSemaphore(value: 0)

        DispatchQueue.main.async { [weak self] in
            guard let self = self, !scene.isAnimating else {
                semaphore.signal()
                return
            }

            // Set callback to be invoked when animation completes
            self.animationCompleteForStdin = { obs, reward, done, info in
                result = (obs, reward, done, info)
                semaphore.signal()
            }

            // Trigger action with animation
            scene.tryExecuteActionAndAnimate(action)
        }

        semaphore.wait()
        return result!
    }

    func executeGetValidActions() -> [Int] {
        guard let scene = gameScene else { return [] }
        return scene.gameState.getValidActions().map { $0.toIndex() }
    }

    // Called by GameScene when animation completes
    func onAnimationComplete(actionResult: GameState.ActionResult) {
        guard let scene = gameScene, let callback = animationCompleteForStdin else { return }

        animationCompleteForStdin = nil

        let obs = ObservationBuilder.build(from: scene.gameState)
        // TODO: Add GameState.isGameOver() method instead of inline check
        let done = scene.gameState.player.health == .dead || scene.gameState.currentStage > Constants.totalStages
        callback(obs, actionResult.reward, done, [:])
    }
}
