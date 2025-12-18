import Foundation

enum ProgramType: String, CaseIterable {
    case push, pull, crash, warp, poly, wait, debug, row, col, undo, step
    case siphPlus = "siph+"
    case exch, show, reset, calm
    case dBomb = "d_bom"
    case delay, antiV = "anti-v", score, reduc
    case atkPlus = "atk+"
    case hack

    var displayName: String {
        return rawValue.uppercased()
    }
}

struct ProgramCost {
    let credits: Int
    let energy: Int
}

struct Program {
    let type: ProgramType

    var cost: ProgramCost {
        switch type {
        case .push: return ProgramCost(credits: 0, energy: 2)
        case .pull: return ProgramCost(credits: 0, energy: 2)
        case .crash: return ProgramCost(credits: 3, energy: 2)
        case .warp: return ProgramCost(credits: 2, energy: 2)
        case .poly: return ProgramCost(credits: 1, energy: 1)
        case .wait: return ProgramCost(credits: 0, energy: 1)
        case .debug: return ProgramCost(credits: 3, energy: 0)
        case .row: return ProgramCost(credits: 3, energy: 1)
        case .col: return ProgramCost(credits: 3, energy: 1)
        case .undo: return ProgramCost(credits: 1, energy: 0)
        case .step: return ProgramCost(credits: 0, energy: 3)
        case .siphPlus: return ProgramCost(credits: 5, energy: 0)
        case .exch: return ProgramCost(credits: 4, energy: 0)
        case .show: return ProgramCost(credits: 2, energy: 0)
        case .reset: return ProgramCost(credits: 0, energy: 4)
        case .calm: return ProgramCost(credits: 2, energy: 4)
        case .dBomb: return ProgramCost(credits: 3, energy: 0)
        case .delay: return ProgramCost(credits: 1, energy: 2)
        case .antiV: return ProgramCost(credits: 3, energy: 0)
        case .score: return ProgramCost(credits: 0, energy: 5)
        case .reduc: return ProgramCost(credits: 2, energy: 1)
        case .atkPlus: return ProgramCost(credits: 4, energy: 4)
        case .hack: return ProgramCost(credits: 2, energy: 2)
        }
    }

    var enemySpawnCost: Int {
        switch type {
        case .push: return 4
        case .pull: return 4
        case .crash: return 6
        case .warp: return 5
        case .poly: return 3
        case .wait: return 1
        case .debug: return 4
        case .row: return 2
        case .col: return 2
        case .undo: return 3
        case .step: return 4
        case .siphPlus: return 7
        case .exch: return 3
        case .show: return 2
        case .reset: return 2
        case .calm: return 6
        case .dBomb: return 4
        case .delay: return 4
        case .antiV: return 4
        case .score: return 5
        case .reduc: return 3
        case .atkPlus: return 8
        case .hack: return 4
        }
    }

    var description: String {
        switch type {
        case .push: return "Push enemies away 1 cell"
        case .pull: return "Pull enemies toward 1 cell"
        case .crash: return "Clear 8 surrounding cells"
        case .warp: return "Warp to random enemy"
        case .poly: return "Randomize enemy types"
        case .wait: return "Skip turn, enemies move"
        case .debug: return "Damage enemies on blocks"
        case .row: return "Attack all in row"
        case .col: return "Attack all in column"
        case .undo: return "Undo last turn"
        case .step: return "Next turn enemies don't move"
        case .siphPlus: return "Gain 1 data siphon"
        case .exch: return "Convert 4C to 4E"
        case .show: return "Reveal Cryptogs"
        case .reset: return "Restore to 3HP"
        case .calm: return "Disable scheduled spawns"
        case .dBomb: return "Destroy nearest Daemon"
        case .delay: return "Extend transmissions +3 turns"
        case .antiV: return "Damage all Viruses"
        case .score: return "Gain points = levels left"
        case .reduc: return "Reduce block spawn counts"
        case .atkPlus: return "Increase damage to 2HP"
        case .hack: return "Hack nearby enemies"
        }
    }
}
