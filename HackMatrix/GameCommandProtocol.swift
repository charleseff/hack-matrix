import Foundation

// MARK: - Game Command Protocol

// Protocol for game command execution
// Implemented by both HeadlessGameCLI and GameScene for stdin/stdout control
protocol GameCommandExecutor: AnyObject {
    func executeReset() -> GameObservation
    func executeStep(actionIndex: Int) -> (GameObservation, Double, Bool, [String: Any])
    func executeGetValidActions() -> [Int]
    func executeSetState(stateData: SetStateData) -> GameObservation
}

// MARK: - Command Structure

// Command structure for JSON decoding
struct Command: Codable {
    let action: String
    let actionIndex: Int?
    let state: SetStateData?
}

// MARK: - SetState Data Structures

/// Data structure for setState command - used to set up specific game states for testing
struct SetStateData: Codable {
    let player: SetStatePlayer
    let enemies: [SetStateEnemy]?
    let transmissions: [SetStateTransmission]?
    let blocks: [SetStateBlock]?
    let resources: [SetStateResource]?
    let ownedPrograms: [Int]?  // Action indices (5-27)
    let stage: Int?
    let turn: Int?
    let showActivated: Bool?
    let scheduledTasksDisabled: Bool?
}

struct SetStatePlayer: Codable {
    let row: Int
    let col: Int
    let hp: Int?
    let credits: Int?
    let energy: Int?
    let dataSiphons: Int?
    let attackDamage: Int?
    let score: Int?
}

struct SetStateEnemy: Codable {
    let type: String  // "virus", "daemon", "glitch", "cryptog"
    let row: Int
    let col: Int
    let hp: Int
    let stunned: Bool?
}

struct SetStateTransmission: Codable {
    let row: Int
    let col: Int
    let turnsRemaining: Int
    let enemyType: String
}

struct SetStateBlock: Codable {
    let row: Int
    let col: Int
    let type: String  // "data" or "program"
    // Data block fields
    let points: Int?
    let spawnCount: Int?
    let siphoned: Bool?
    // Program block fields
    let programType: String?
    let programActionIndex: Int?
}

struct SetStateResource: Codable {
    let row: Int
    let col: Int
    let credits: Int?
    let energy: Int?
    let dataSiphon: Bool?
}

// MARK: - Stdin Command Reader

// Shared stdin/stdout command reader
// Reads JSON commands from stdin, calls executor, writes JSON to stdout
class StdinCommandReader {
    weak var executor: GameCommandExecutor?
    private var outputFile: FileHandle?

    // MARK: Command Processing

    func start() {
        // Save original stdout for JSON output
        let originalStdout = dup(STDOUT_FILENO)

        // Redirect print statements to stderr so stdout is clean for JSON
        dup2(STDERR_FILENO, STDOUT_FILENO)

        outputFile = FileHandle(fileDescriptor: originalStdout, closeOnDealloc: true)

        // autoreleasepool is CRITICAL on macOS to prevent memory leak
        // Foundation's JSON/String operations create Objective-C bridged objects
        // that must be released each iteration to prevent accumulation
        while let line = readLine() {
            #if canImport(ObjectiveC)
            autoreleasepool {
                processLine(line)
            }
            #else
            processLine(line)
            #endif
        }

        close(originalStdout)
    }

    private func processLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let command = try? JSONDecoder().decode(Command.self, from: data) else {
            sendError("Invalid command")
            return
        }
        handleCommand(command)
    }

    private func handleCommand(_ command: Command) {
        guard let executor = executor else {
            sendError("No executor")
            return
        }

        switch command.action {
        case "reset":
            let obs = executor.executeReset()
            sendResponse(["observation": encodeObservation(obs)])

        case "step":
            guard let actionIndex = command.actionIndex else {
                sendError("Missing actionIndex")
                return
            }
            let (obs, reward, done, info) = executor.executeStep(actionIndex: actionIndex)
            sendResponse([
                "observation": encodeObservation(obs),
                "reward": reward,
                "done": done,
                "info": info
            ])

        case "getValidActions":
            let actions = executor.executeGetValidActions()
            sendResponse(["validActions": actions])

        case "getActionSpace":
            sendResponse(["actionSpaceSize": 28])  // 4 moves + 1 siphon + 23 programs

        case "getObservationSpace":
            sendResponse([
                "gridSize": 6,
                "playerFeatures": 9,  // row, col, hp, credits, energy, stage, turn, dataSiphons, baseAttack
                "cellFeatures": 20    // Approximate - varies by cell
            ])

        case "setState":
            guard let stateData = command.state else {
                sendError("Missing state data for setState command")
                return
            }
            let obs = executor.executeSetState(stateData: stateData)
            sendResponse(["observation": encodeObservation(obs)])

        default:
            sendError("Unknown command: \(command.action)")
        }
    }

    // MARK: Observation Encoding

    func encodeObservation(_ obs: GameObservation) -> [String: Any] {
        var result: [String: Any] = [
            "playerRow": obs.playerRow,
            "playerCol": obs.playerCol,
            "playerHP": obs.playerHP,
            "credits": obs.credits,
            "energy": obs.energy,
            "stage": obs.stage,
            "turn": obs.turn,
            "dataSiphons": obs.dataSiphons,
            "baseAttack": obs.baseAttack,
            "score": obs.score,
            "showActivated": obs.showActivated,
            "scheduledTasksDisabled": obs.scheduledTasksDisabled,
            "ownedPrograms": obs.ownedPrograms
        ]

        // Encode cells as nested arrays
        var cellsArray: [[[String: Any]]] = []
        for row in obs.cells {
            var rowArray: [[String: Any]] = []
            for cell in row {
                var cellDict: [String: Any] = [
                    "row": cell.row,
                    "col": cell.col,
                    "isDataSiphon": cell.isDataSiphon,
                    "isExit": cell.isExit
                ]

                if let enemy = cell.enemy {
                    cellDict["enemy"] = [
                        "type": enemy.type,
                        "hp": enemy.hp,
                        "isStunned": enemy.isStunned
                    ]
                }

                if let block = cell.block {
                    var blockDict: [String: Any] = [
                        "blockType": block.blockType,
                        "isSiphoned": block.isSiphoned
                    ]
                    if let points = block.points { blockDict["points"] = points }
                    if let program = block.programType { blockDict["programType"] = program }
                    if let actionIndex = block.programActionIndex { blockDict["programActionIndex"] = actionIndex }
                    if let spawn = block.transmissionSpawnCount { blockDict["transmissionSpawnCount"] = spawn }
                    cellDict["block"] = blockDict
                }

                if let transmission = cell.transmission {
                    var transDict: [String: Any] = [
                        "turnsUntilSpawn": transmission.turnsUntilSpawn
                    ]
                    if let enemyType = transmission.enemyType { transDict["enemyType"] = enemyType }
                    cellDict["transmission"] = transDict
                }

                if let credits = cell.credits { cellDict["credits"] = credits }
                if let energy = cell.energy { cellDict["energy"] = energy }

                rowArray.append(cellDict)
            }
            cellsArray.append(rowArray)
        }
        result["cells"] = cellsArray

        // Encode cryptog hints
        result["cryptogHints"] = obs.cryptogHints.map { ["row": $0.0, "col": $0.1] }

        return result
    }

    // MARK: JSON Response Handling

    func sendResponse(_ data: [String: Any]) {
        guard let output = outputFile else { return }
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: data)
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                output.write((jsonString + "\n").data(using: .utf8)!)
                try? output.synchronize()  // Flush output
            }
        } catch {
            sendError("Failed to encode response: \(error)")
        }
    }

    private func sendError(_ message: String) {
        guard let output = outputFile else { return }
        let errorData: [String: Any] = ["error": message]
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: errorData)
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                output.write((jsonString + "\n").data(using: .utf8)!)
                try? output.synchronize()  // Flush output
            }
        } catch {
            // Can't even send error, just exit
            exit(1)
        }
    }
}
