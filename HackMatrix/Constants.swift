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

    // MARK: - Developer Mode

    /// Check if developer mode is enabled via launch arguments
    static var isDevModeEnabled: Bool {
        ProcessInfo.processInfo.arguments.contains("--dev-mode")
    }

    /// Get filtered programs for developer mode, or nil if dev mode is disabled
    static var devModePrograms: [ProgramType]? {
        guard isDevModeEnabled else { return nil }

        let args = ProcessInfo.processInfo.arguments

        // Look for --programs argument
        if let programsIndex = args.firstIndex(where: { $0.hasPrefix("--programs=") }) {
            let programsArg = args[programsIndex]
            let programNames = programsArg
                .replacingOccurrences(of: "--programs=", with: "")
                .split(separator: ",")
                .map { String($0) }

            // Filter to valid program types
            let validPrograms = ProgramType.allCases.filter { programNames.contains($0.rawValue) }

            if !validPrograms.isEmpty {
                print("[DEV MODE] Limiting programs to: \(validPrograms.map { $0.rawValue }.joined(separator: ", "))")
                return validPrograms
            } else {
                print("[DEV MODE] Warning: No valid programs found in --programs argument")
            }
        }

        // Dev mode enabled but no program filter specified
        print("[DEV MODE] Enabled (all programs available)")
        return nil
    }
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
