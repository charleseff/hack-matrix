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

        // Configure logging level based on flags
        if CommandLine.arguments.contains("--debug") {
            DebugConfig.logLevel = .debug
            debugLog("HeadlessGameCLI", "Debug logging enabled (verbose)")
        } else if CommandLine.arguments.contains("--info") {
            DebugConfig.logLevel = .info
            infoLog("HeadlessGameCLI", "Info logging enabled")
        }

        commandReader.executor = self
        commandReader.start()
    }

    // MARK: - GameCommandExecutor

    func executeReset() -> GameObservation {
        infoLog("Reset called - creating new game")
        game = HeadlessGame()
        let obs = game!.reset()
        infoLog("Reset complete - stage \(obs.stage)")
        return obs
    }

    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let game = game else {
            fatalError("Game not initialized")
        }
        let (obs, reward, done, info) = game.step(actionIndex: actionIndex)
        return (obs, reward, done, info)
    }

    func executeGetValidActions() -> [Int] {
        guard let game = game else { return [] }
        return game.getValidActions().map { $0.toIndex() }
    }
}
