import Foundation

enum EnemyType {
    case virus    // Moves 2 cells/turn
    case daemon   // 3 HP
    case glitch   // Can move on blocks
    case cryptog  // Invisible except in same row/col

    var maxHP: Int {
        switch self {
        case .daemon: return 3
        default: return 2
        }
    }

    var moveSpeed: Int {
        switch self {
        case .virus: return 2
        default: return 1
        }
    }

    var canMoveOnBlocks: Bool {
        return self == .glitch
    }

    var emoji: String {
        switch self {
        case .virus: return "🦠"
        case .daemon: return "👹"
        case .glitch: return "⚡️"
        case .cryptog: return "👻"
        }
    }

    var spriteName: String {
        switch self {
        case .virus: return "virus_sprite"
        case .daemon: return "daemon_sprite"
        case .glitch: return "glitch_sprite"
        case .cryptog: return "cryptog_sprite"
        }
    }
}

class Enemy {
    let id: UUID
    var type: EnemyType
    var row: Int
    var col: Int
    var hp: Int
    var disabledTurns: Int
    var isStunned: Bool
    var lastKnownRow: Int?  // For Cryptog tracking
    var lastKnownCol: Int?

    init(type: EnemyType, row: Int, col: Int) {
        self.id = UUID()
        self.type = type
        self.row = row
        self.col = col
        self.hp = type.maxHP
        self.disabledTurns = 0
        self.isStunned = false

        // Initialize last known position for Cryptogs (visible as transmission before spawning)
        if type == .cryptog {
            self.lastKnownRow = row
            self.lastKnownCol = col
        }
    }

    var isDisabled: Bool {
        return disabledTurns > 0
    }

    func takeDamage(_ amount: Int) {
        hp = max(0, hp - amount)
    }

    func decrementDisable() {
        if disabledTurns > 0 {
            disabledTurns -= 1
        }
    }

    func isVisible(playerRow: Int, playerCol: Int, showActivated: Bool) -> Bool {
        if type != .cryptog {
            return true
        }

        if showActivated {
            return true
        }

        // Visible if in same row or column
        return row == playerRow || col == playerCol
    }
}

enum TransmissionState {
    case spawning(turnsRemaining: Int)
    case spawned(Enemy)
}

class Transmission {
    let id: UUID
    var row: Int
    var col: Int
    var state: TransmissionState
    let enemyType: EnemyType  // Store enemy type from creation (for show program)

    init(row: Int, col: Int, turnsUntilSpawn: Int = 1, enemyType: EnemyType? = nil) {
        self.id = UUID()
        self.row = row
        self.col = col
        self.state = .spawning(turnsRemaining: turnsUntilSpawn)
        self.enemyType = enemyType ?? EnemyType.allCases.randomElement()!
    }

    func decrementTimer() -> Enemy? {
        if case .spawning(let turns) = state {
            if turns <= 1 {
                // Spawn enemy using the predetermined type
                let enemy = Enemy(type: enemyType, row: row, col: col)
                enemy.disabledTurns = 1  // Disable for spawn turn
                state = .spawned(enemy)
                return enemy
            } else {
                state = .spawning(turnsRemaining: turns - 1)
            }
        }
        return nil
    }

    var isSpawning: Bool {
        if case .spawning = state {
            return true
        }
        return false
    }
}

extension EnemyType: CaseIterable {}
