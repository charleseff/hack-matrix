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
        return game!.getObservation()
    }

    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any]) {
        guard let game = game,
              let action = GameAction.fromIndex(actionIndex) else {
            // Return error state
            return (GameObservation(
                playerRow: 0, playerCol: 0, playerHP: 0,
                credits: 0, energy: 0, stage: 0, turn: 0,
                dataSiphons: 0, baseAttack: 0,
                cells: [], cryptogHints: [],
                showActivated: false
            ), 0, true, [:])
        }

        return game.step(action: action)
    }

    func executeGetValidActions() -> [Int] {
        guard let game = game else { return [] }
        return game.getValidActions().map { $0.toIndex() }
    }
}
