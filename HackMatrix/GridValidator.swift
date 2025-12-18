import Foundation

struct GridValidator {
    /// Check if all non-block, non-exit cells are connected (no isolated areas)
    /// Exit is treated as a wall because stepping on it ends the stage
    static func isPathConnected(grid: Grid) -> Bool {
        var walkableCells: [(Int, Int)] = []

        // Find all walkable cells (not blocks, not exit)
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                let cell = grid.cells[row][col]
                if !cell.hasBlock && !cell.isExit {
                    walkableCells.append((row, col))
                }
            }
        }

        guard !walkableCells.isEmpty else { return true }

        // BFS from first walkable cell
        let start = walkableCells[0]
        var visited = Set<String>()
        var queue = [start]
        visited.insert("\(start.0),\(start.1)")

        while !queue.isEmpty {
            let current = queue.removeFirst()

            // Check all 4 cardinal directions
            for direction in Direction.allCases {
                let offset = direction.offset
                let newRow = current.0 + offset.row
                let newCol = current.1 + offset.col

                guard grid.isValidPosition(row: newRow, col: newCol) else { continue }

                let key = "\(newRow),\(newCol)"
                guard !visited.contains(key) else { continue }

                let cell = grid.cells[newRow][newCol]
                guard !cell.hasBlock && !cell.isExit else { continue }

                visited.insert(key)
                queue.append((newRow, newCol))
            }
        }

        // All walkable cells should be visited
        return visited.count == walkableCells.count
    }

    /// Check if all blocks have at least one adjacent non-block cell (siphonable)
    static func areAllBlocksSiphonable(grid: Grid) -> Bool {
        for row in 0..<Constants.gridSize {
            for col in 0..<Constants.gridSize {
                let cell = grid.cells[row][col]

                if cell.hasBlock {
                    // Check if any adjacent cell is non-block
                    var hasAdjacentNonBlock = false

                    for direction in Direction.allCases {
                        let offset = direction.offset
                        let adjRow = row + offset.row
                        let adjCol = col + offset.col

                        if grid.isValidPosition(row: adjRow, col: adjCol) {
                            if !grid.cells[adjRow][adjCol].hasBlock {
                                hasAdjacentNonBlock = true
                                break
                            }
                        }
                    }

                    if !hasAdjacentNonBlock {
                        return false
                    }
                }
            }
        }

        return true
    }

    /// Validate both rules
    static func isValidPlacement(grid: Grid) -> Bool {
        return isPathConnected(grid: grid) && areAllBlocksSiphonable(grid: grid)
    }
}
