import Foundation

struct Pathfinding {
    /// Find the next move for an enemy to get closer to the player using BFS
    /// When multiple moves are equally good, randomly picks one
    static func findNextMove(
        from start: (row: Int, col: Int),
        to target: (row: Int, col: Int),
        grid: Grid,
        canMoveOnBlocks: Bool,
        occupiedPositions: Set<String>
    ) -> (row: Int, col: Int)? {

        // BFS to find shortest path(s)
        var queue: [(pos: (Int, Int), path: [(Int, Int)])] = [(start, [start])]
        var visited = Set<String>()
        visited.insert("\(start.0),\(start.1)")

        var shortestPathLength: Int?
        var candidateFirstMoves: Set<String> = []

        while !queue.isEmpty {
            let (currentPos, path) = queue.removeFirst()

            // If we've found paths and this one is longer, stop searching
            if let shortest = shortestPathLength, path.count > shortest {
                break
            }

            // Check if we reached the target
            if currentPos.0 == target.0 && currentPos.1 == target.1 {
                if path.count > 1 {
                    let firstMove = path[1]

                    if shortestPathLength == nil {
                        // First path found
                        shortestPathLength = path.count
                        candidateFirstMoves.insert("\(firstMove.0),\(firstMove.1)")
                    } else if path.count == shortestPathLength {
                        // Another path of same length - add its first move
                        candidateFirstMoves.insert("\(firstMove.0),\(firstMove.1)")
                    }
                }
                continue // Don't expand from target
            }

            // Try all 4 directions
            for direction in Direction.allCases {
                let offset = direction.offset
                let newRow = currentPos.0 + offset.row
                let newCol = currentPos.1 + offset.col

                guard grid.isValidPosition(row: newRow, col: newCol) else { continue }

                let posKey = "\(newRow),\(newCol)"
                guard !visited.contains(posKey) else { continue }

                // Check if position is blocked
                let cell = grid.cells[newRow][newCol]
                if cell.hasBlock && !canMoveOnBlocks {
                    continue
                }

                // Check if occupied by another enemy (but allow moving to player's position)
                if !(newRow == target.0 && newCol == target.1) && occupiedPositions.contains(posKey) {
                    continue
                }

                visited.insert(posKey)
                var newPath = path
                newPath.append((newRow, newCol))
                queue.append(((newRow, newCol), newPath))
            }
        }

        // Randomly pick from equally good first moves
        if !candidateFirstMoves.isEmpty {
            let chosenKey = candidateFirstMoves.randomElement()!
            let parts = chosenKey.split(separator: ",")
            return (Int(parts[0])!, Int(parts[1])!)
        }

        // No path found - stay in place
        return nil
    }

    /// Get all occupied positions (for collision detection)
    static func getOccupiedPositions(enemies: [Enemy], excludingId: UUID? = nil) -> Set<String> {
        var occupied = Set<String>()
        for enemy in enemies {
            if let excludeId = excludingId, enemy.id == excludeId {
                continue
            }
            occupied.insert("\(enemy.row),\(enemy.col)")
        }
        return occupied
    }
}
