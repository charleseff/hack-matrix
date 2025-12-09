import Foundation
import CoreGraphics

enum Constants {
    static let gridSize = 6
    static let cellSize: CGFloat = 80.0
    static let totalStages = 8

    // Starting enemies per stage (as transmissions)
    static let startingEnemies = [1, 2, 3, 4, 5, 6, 7, 8]

    // Scheduled task spawn intervals per stage (in turns)
    static let scheduledTaskIntervals = [12, 11, 10, 9, 8, 7, 6, 5]

    // Siphon range (cross pattern: center + 4 cardinal)
    static let siphonRange = 5
}

enum Direction: CaseIterable {
    case up, down, left, right

    var offset: (row: Int, col: Int) {
        switch self {
        case .up: return (1, 0)
        case .down: return (-1, 0)
        case .left: return (0, -1)
        case .right: return (0, 1)
        }
    }
}
