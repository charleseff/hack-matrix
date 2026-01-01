import Foundation

/// Lightweight test runner for game logic tests
/// Run with: HackMatrix --run-tests
class GameLogicTests {

    static var testsPassed = 0
    static var testsFailed = 0

    static func runAllTests() {
        print("Running game logic tests...\n")

        // Starting Bonus Tests
        testStartingBonusIsApplied()
        testStartingBonusRandomness()

        // Scheduled Task Tests
        testInitialScheduledTaskState()
        testSiphonAffectsScheduledTiming()
        testScheduledTaskIntervalMinimum()
        testScheduledTaskTriggersAtCorrectTurn()
        testStageChangeResetsInterval()

        print("\n=== Test Results ===")
        print("Passed: \(testsPassed)")
        print("Failed: \(testsFailed)")

        if testsFailed > 0 {
            print("\n❌ TESTS FAILED")
            exit(1)
        } else {
            print("\n✅ ALL TESTS PASSED")
        }
    }

    // MARK: - Test Helpers

    static func assert(_ condition: Bool, _ message: String, file: String = #file, line: Int = #line) {
        if condition {
            testsPassed += 1
            print("  ✓ \(message)")
        } else {
            testsFailed += 1
            print("  ✗ FAILED: \(message) (at \(file):\(line))")
        }
    }

    static func assertEqual<T: Equatable>(_ actual: T, _ expected: T, _ message: String, file: String = #file, line: Int = #line) {
        if actual == expected {
            testsPassed += 1
            print("  ✓ \(message)")
        } else {
            testsFailed += 1
            print("  ✗ FAILED: \(message) - expected \(expected), got \(actual) (at \(file):\(line))")
        }
    }

    static func assertGreaterThan<T: Comparable>(_ actual: T, _ threshold: T, _ message: String, file: String = #file, line: Int = #line) {
        if actual > threshold {
            testsPassed += 1
            print("  ✓ \(message)")
        } else {
            testsFailed += 1
            print("  ✗ FAILED: \(message) - expected > \(threshold), got \(actual) (at \(file):\(line))")
        }
    }

    // MARK: - Starting Bonus Tests

    static func testStartingBonusIsApplied() {
        print("\nTest: Starting bonus is applied")
        let state = GameState()
        let hasBonus = state.player.credits == 10 ||
                       state.player.energy == 11 ||
                       state.player.dataSiphons == 1
        assert(hasBonus, "Player should have exactly one starting bonus")
    }

    static func testStartingBonusRandomness() {
        print("\nTest: Starting bonus randomness")
        var bonusCounts = [0, 0, 0]  // credits, energy, siphon
        for _ in 0..<60 {
            let state = GameState()
            if state.player.credits == 10 { bonusCounts[0] += 1 }
            else if state.player.energy == 11 { bonusCounts[1] += 1 }
            else if state.player.dataSiphons == 1 { bonusCounts[2] += 1 }
        }
        // Each bonus should appear at least once in 60 trials (probability of missing one is ~(2/3)^60 ≈ 0)
        assertGreaterThan(bonusCounts[0], 0, "Credits bonus should appear at least once")
        assertGreaterThan(bonusCounts[1], 0, "Energy bonus should appear at least once")
        assertGreaterThan(bonusCounts[2], 0, "Data siphon bonus should appear at least once")
    }

    // MARK: - Scheduled Task Tests

    static func testInitialScheduledTaskState() {
        print("\nTest: Initial scheduled task state")
        let state = GameState()
        assertEqual(state.scheduledTaskInterval, 12, "Initial interval should be 12")
        assertEqual(state.nextScheduledTaskTurn, 12, "Next scheduled turn should be 12")
    }

    static func testSiphonAffectsScheduledTiming() {
        print("\nTest: Siphon affects scheduled timing")
        let state = GameState()
        // Give player a siphon to use
        state.player.dataSiphons = 1
        let initialInterval = state.scheduledTaskInterval
        let initialNextTurn = state.nextScheduledTaskTurn

        _ = state.performSiphon()

        assertEqual(state.scheduledTaskInterval, initialInterval - 1, "Interval should decrease by 1 after siphon")
        assertEqual(state.nextScheduledTaskTurn, initialNextTurn + 5, "Next turn should increase by 5 after siphon")
    }

    static func testScheduledTaskIntervalMinimum() {
        print("\nTest: Scheduled task interval minimum")
        let state = GameState()
        state.scheduledTaskInterval = 1
        state.player.dataSiphons = 1

        _ = state.performSiphon()

        assertEqual(state.scheduledTaskInterval, 1, "Interval should not go below 1")
    }

    static func testScheduledTaskTriggersAtCorrectTurn() {
        print("\nTest: Scheduled task triggers at correct turn")
        let state = GameState()
        state.scheduledTasksDisabled = false
        let initialTransmissionCount = state.transmissions.count

        // Simulate turns until we reach the scheduled turn
        state.turnCount = 11
        state.maybeExecuteScheduledTask()
        let countBefore = state.transmissions.count

        state.turnCount = 12
        state.maybeExecuteScheduledTask()
        let countAfter = state.transmissions.count

        assertEqual(countBefore, initialTransmissionCount, "Should not spawn before turn 12")
        assertGreaterThan(countAfter, countBefore, "Should spawn transmission at turn 12")
    }

    static func testStageChangeResetsInterval() {
        print("\nTest: Stage change resets interval to base")
        let state = GameState()

        // Simulate siphoning to reduce interval on stage 1
        state.player.dataSiphons = 3
        _ = state.performSiphon()  // interval now 11
        _ = state.performSiphon()  // interval now 10
        _ = state.performSiphon()  // interval now 9

        assertEqual(state.scheduledTaskInterval, 9, "Interval should be 9 after 3 siphons")

        // Simulate stage advance (mimicking the logic in advanceToNextStage)
        state.currentStage = 2
        state.turnCount = 20
        state.scheduledTaskInterval = 13 - state.currentStage  // Resets to 11
        state.nextScheduledTaskTurn = state.turnCount + state.scheduledTaskInterval

        assertEqual(state.scheduledTaskInterval, 11, "Interval should be 11 for stage 2")
        assertEqual(state.nextScheduledTaskTurn, 31, "Next turn should be turnCount + interval")
    }
}
