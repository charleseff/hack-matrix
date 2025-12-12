import Foundation

struct HighScoreEntry: Codable {
    let score: Int
    let completed: Bool
    let stage: Int
    let date: Date
}

class HighScoreManager {
    static let shared = HighScoreManager()

    private let userDefaults = UserDefaults.standard
    private let highScoresKey = "highScores"
    private let maxHighScores = 10

    private init() {}

    func addScore(score: Int, completed: Bool, stage: Int) {
        var scores = getHighScores()

        let newEntry = HighScoreEntry(score: score, completed: completed, stage: stage, date: Date())
        scores.append(newEntry)

        // Sort: completed games first, then by score within each group
        scores.sort { (entry1, entry2) -> Bool in
            if entry1.completed != entry2.completed {
                return entry1.completed  // Completed games come first
            }
            return entry1.score > entry2.score  // Higher score is better
        }

        // Keep only top scores
        if scores.count > maxHighScores {
            scores = Array(scores.prefix(maxHighScores))
        }

        saveHighScores(scores)
    }

    func getHighScores() -> [HighScoreEntry] {
        guard let data = userDefaults.data(forKey: highScoresKey),
              let scores = try? JSONDecoder().decode([HighScoreEntry].self, from: data) else {
            return []
        }
        return scores
    }

    private func saveHighScores(_ scores: [HighScoreEntry]) {
        if let data = try? JSONEncoder().encode(scores) {
            userDefaults.set(data, forKey: highScoresKey)
        }
    }

    func clearHighScores() {
        userDefaults.removeObject(forKey: highScoresKey)
    }
}
