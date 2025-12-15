import Foundation

// Simple test to measure headless game performance
class HeadlessTest {
    static func runPerformanceTest() {
        print("Testing headless game performance...")

        let numGames = 100
        let startTime = Date()
        var totalSteps = 0

        for i in 0..<numGames {
            let game = HeadlessGame()
            var isDone = false
            var stepCount = 0

            // Play random actions until game ends
            while !isDone && stepCount < 1000 {  // Max 1000 steps per game
                let validActions = game.getValidActions()
                guard let randomAction = validActions.randomElement() else { break }

                let (_, _, done, _) = game.step(action: randomAction)
                isDone = done
                stepCount += 1
            }

            totalSteps += stepCount

            if (i + 1) % 10 == 0 {
                let elapsed = Date().timeIntervalSince(startTime)
                let gamesPerSec = Double(i + 1) / elapsed
                let stepsPerSec = Double(totalSteps) / elapsed
                print("  Completed \(i + 1)/\(numGames) games (\(String(format: "%.1f", gamesPerSec)) games/sec, \(String(format: "%.0f", stepsPerSec)) steps/sec)")
            }
        }

        let totalTime = Date().timeIntervalSince(startTime)
        let gamesPerSecond = Double(numGames) / totalTime
        let stepsPerSecond = Double(totalSteps) / totalTime
        let avgStepsPerGame = Double(totalSteps) / Double(numGames)

        print("\nPerformance Test Results:")
        print("  Total games: \(numGames)")
        print("  Total steps: \(totalSteps)")
        print("  Avg steps per game: \(String(format: "%.1f", avgStepsPerGame))")
        print("  Total time: \(String(format: "%.2f", totalTime)) seconds")
        print("  Games per second: \(String(format: "%.1f", gamesPerSecond))")
        print("  Steps per second: \(String(format: "%.0f", stepsPerSecond))")
        print("  Estimated steps per hour: \(String(format: "%.1f", stepsPerSecond * 3600 / 1_000_000)) million")

        print("\nFor RL training, steps/second is the key metric.")
        print("This performance should be sufficient for initial training experiments.")
    }
}
