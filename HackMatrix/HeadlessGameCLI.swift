import Foundation

// Command-line interface for HeadlessGame
// Communicates via JSON over stdin/stdout for Python wrapper
class HeadlessGameCLI: GameCommandExecutor {
    private var game: HeadlessGame?
    private let commandReader = StdinCommandReader()

    func run() {
        // Check for --headless-cli flag
        guard CommandLine.arguments.contains("--headless-cli") else {
            return
        }

        commandReader.executor = self
        commandReader.start()
    }

    // MARK: - GameCommandExecutor

    func executeReset() -> GameObservation {
        game = HeadlessGame()
        return game!.reset()
    }

    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let game = game else {
            fatalError("Game not initialized")
        }
        return game.step(actionIndex: actionIndex)
    }

    func executeGetValidActions() -> [Int] {
        guard let game = game else { return [] }
        return game.getValidActions().map { $0.toIndex() }
    }
}
