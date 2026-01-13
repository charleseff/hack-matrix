import Foundation

// Command-line interface for HeadlessGame
// Communicates via JSON over stdin/stdout for Python wrapper
public class HeadlessGameCLI: GameCommandExecutor {
    private var game: HeadlessGame?
    private let commandReader = StdinCommandReader()

    public init() {}

    public func run() {
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
        // Autoreleasepool on macOS to ensure all temporary objects from old game are freed
        #if canImport(ObjectiveC)
        autoreleasepool {
            cleanupOldGame()
        }
        #else
        cleanupOldGame()
        #endif

        game = HeadlessGame()
        let obs = game!.reset()
        return obs
    }

    private func cleanupOldGame() {
        // Log stats from previous game before resetting
        if let oldGame = game {
            let gs = oldGame.gameState
            infoLog("Reset - Stage: \(gs.currentStage), Score: \(gs.player.score), Siphons Collected: \(gs.totalDataSiphonsCollected), Siphons Used: \(gs.totalSiphonUses), Enemies Killed: \(gs.totalEnemiesKilled), Programs: \(gs.ownedPrograms.count), Program Uses: \(gs.totalProgramUses)")

            // Explicitly clear gameHistory to help ARC free memory
            oldGame.gameState.gameHistory.removeAll()
        } else {
            infoLog("Reset")
        }

        // Setting game to nil first ensures old game is deallocated before new one
        game = nil
    }

    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let game = game else {
            fatalError("Game not initialized")
        }
        // Autoreleasepool on macOS to drain temporary objects from game logic
        #if canImport(ObjectiveC)
        return autoreleasepool {
            let (obs, reward, done, info) = game.step(actionIndex: actionIndex)
            return (obs, reward, done, info)
        }
        #else
        let (obs, reward, done, info) = game.step(actionIndex: actionIndex)
        return (obs, reward, done, info)
        #endif
    }

    func executeGetValidActions() -> [Int] {
        guard let game = game else { return [] }
        return game.getValidActions().map { $0.toIndex() }
    }
}
