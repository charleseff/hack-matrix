import Foundation

enum PlayerHealth: Int {
    case full = 3      // 😊
    case damaged = 2   // 😐
    case critical = 1  // ☹️
    case dead = 0      // ☠️

    var emoji: String {
        switch self {
        case .full: return "😊"
        case .damaged: return "😐"
        case .critical: return "☹️"
        case .dead: return "☠️"
        }
    }

    mutating func takeDamage() {
        if rawValue > 0 {
            self = PlayerHealth(rawValue: rawValue - 1) ?? .dead
        }
    }

    mutating func heal() {
        self = .full
    }
}

class Player {
    var row: Int
    var col: Int
    var health: PlayerHealth
    var credits: Int
    var energy: Int
    var dataSiphons: Int
    var score: Int
    var attackDamage: Int // 1 by default, 2 with Atk+

    init(row: Int, col: Int) {
        self.row = row
        self.col = col
        self.health = .full
        self.credits = 0
        self.energy = 0
        self.dataSiphons = 0
        self.score = 0
        self.attackDamage = 1
    }

    func move(direction: Direction) {
        let offset = direction.offset
        row += offset.row
        col += offset.col
    }

    func canMoveTo(row: Int, col: Int, grid: Grid) -> Bool {
        guard row >= 0 && row < Constants.gridSize &&
              col >= 0 && col < Constants.gridSize else {
            return false
        }

        let cell = grid.cells[row][col]
        return !cell.hasBlock
    }
}
