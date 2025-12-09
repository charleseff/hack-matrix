import Foundation

enum ResourceType {
    case credits(Int)
    case energy(Int)
    case none

    var amount: Int {
        switch self {
        case .credits(let amt), .energy(let amt):
            return amt
        case .none:
            return 0
        }
    }
}

enum BlockType {
    case data(points: Int, enemySpawn: Int)
    case program(Program, enemySpawn: Int)
    case question(isData: Bool, points: Int?, program: Program?, enemySpawn: Int)

    var enemySpawnCount: Int {
        switch self {
        case .data(_, let count), .program(_, let count), .question(_, _, _, let count):
            return count
        }
    }
}

enum CellContent {
    case empty
    case block(BlockType)
    case dataSiphon
    case exit
}

class Cell {
    var row: Int
    var col: Int
    var content: CellContent
    var resources: ResourceType
    var isSiphoned: Bool

    init(row: Int, col: Int) {
        self.row = row
        self.col = col
        self.content = .empty
        self.resources = .none
        self.isSiphoned = false
    }

    var hasBlock: Bool {
        if case .block = content {
            return true
        }
        return false
    }

    var hasDataSiphon: Bool {
        if case .dataSiphon = content {
            return true
        }
        return false
    }

    var isExit: Bool {
        if case .exit = content {
            return true
        }
        return false
    }
}

class Grid {
    var cells: [[Cell]]

    init() {
        cells = []
        for row in 0..<Constants.gridSize {
            var rowCells: [Cell] = []
            for col in 0..<Constants.gridSize {
                rowCells.append(Cell(row: row, col: col))
            }
            cells.append(rowCells)
        }
    }

    func isValidPosition(row: Int, col: Int) -> Bool {
        return row >= 0 && row < Constants.gridSize &&
               col >= 0 && col < Constants.gridSize
    }

    func cellAt(row: Int, col: Int) -> Cell? {
        guard isValidPosition(row: row, col: col) else { return nil }
        return cells[row][col]
    }

    func getCornerPositions() -> [(Int, Int)] {
        return [
            (0, 0),
            (0, Constants.gridSize - 1),
            (Constants.gridSize - 1, 0),
            (Constants.gridSize - 1, Constants.gridSize - 1)
        ]
    }

    func getSiphonCells(centerRow: Int, centerCol: Int) -> [Cell] {
        var result: [Cell] = []

        // Center cell
        if let cell = cellAt(row: centerRow, col: centerCol) {
            result.append(cell)
        }

        // Cardinal directions
        for direction in Direction.allCases {
            let offset = direction.offset
            let newRow = centerRow + offset.row
            let newCol = centerCol + offset.col

            if let cell = cellAt(row: newRow, col: newCol) {
                result.append(cell)
            }
        }

        return result
    }
}
