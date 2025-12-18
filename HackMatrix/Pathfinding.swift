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

        // Try pathfinding with enemy collision avoidance first
        let candidateMoves = findPath(from: start, to: target, grid: grid, canMoveOnBlocks: canMoveOnBlocks, occupiedPositions: occupiedPositions)
        if !candidateMoves.isEmpty {
            // Randomly pick one of the equally good moves
            let chosenKey = candidateMoves.randomElement()!
            let parts = chosenKey.split(separator: ",")
            return (Int(parts[0])!, Int(parts[1])!)
        }

        // If no path found (likely blocked by enemies), try again ignoring enemy positions
        // This ensures enemies still try to move closer even when blocked
        let candidateMovesIgnoringEnemies = findPath(from: start, to: target, grid: grid, canMoveOnBlocks: canMoveOnBlocks, occupiedPositions: Set<String>())

        // Filter out moves that would actually place us on an enemy
        let validMoves = candidateMovesIgnoringEnemies.filter { moveKey in
            !occupiedPositions.contains(moveKey)
        }

        if !validMoves.isEmpty {
            // Randomly pick one of the valid moves
            let chosenKey = validMoves.randomElement()!
            let parts = chosenKey.split(separator: ",")
            return (Int(parts[0])!, Int(parts[1])!)
        }

        // No path found at all - stay in place
        return nil
    }

    /// Internal pathfinding implementation using BFS
    /// Returns all candidate first moves that lead to the shortest path
    private static func findPath(
        from start: (row: Int, col: Int),
        to target: (row: Int, col: Int),
        grid: Grid,
        canMoveOnBlocks: Bool,
        occupiedPositions: Set<String>
    ) -> Set<String> {

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

        // Return all candidate first moves
        return candidateFirstMoves
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
