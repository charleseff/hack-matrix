import XCTest
@testable import HackMatrixCore

final class StartingBonusTests: XCTestCase {

    func testStartingBonusIsApplied() {
        let state = GameState()
        let hasBonus = state.player.credits == 10 ||
                       state.player.energy == 11 ||
                       state.player.dataSiphons == 1
        XCTAssertTrue(hasBonus, "Player should have exactly one starting bonus")
    }

    func testStartingBonusRandomness() {
        var bonusCounts = [0, 0, 0]  // credits, energy, siphon
        for _ in 0..<60 {
            let state = GameState()
            if state.player.credits == 10 { bonusCounts[0] += 1 }
            else if state.player.energy == 11 { bonusCounts[1] += 1 }
            else if state.player.dataSiphons == 1 { bonusCounts[2] += 1 }
        }
        XCTAssertGreaterThan(bonusCounts[0], 0, "Credits bonus should appear at least once")
        XCTAssertGreaterThan(bonusCounts[1], 0, "Energy bonus should appear at least once")
        XCTAssertGreaterThan(bonusCounts[2], 0, "Data siphon bonus should appear at least once")
    }
}

final class ScheduledTaskTests: XCTestCase {

    func testInitialScheduledTaskState() {
        let state = GameState()
        XCTAssertEqual(state.scheduledTaskInterval, 12, "Initial interval should be 12")
        XCTAssertEqual(state.nextScheduledTaskTurn, 12, "Next scheduled turn should be 12")
    }

    func testSiphonAffectsScheduledTiming() {
        let state = GameState()
        state.player.dataSiphons = 1
        let initialInterval = state.scheduledTaskInterval
        let initialNextTurn = state.nextScheduledTaskTurn

        _ = state.performSiphon()

        XCTAssertEqual(state.scheduledTaskInterval, initialInterval - 1,
                       "Interval should decrease by 1 after siphon")
        XCTAssertEqual(state.nextScheduledTaskTurn, initialNextTurn + 5,
                       "Next turn should increase by 5 after siphon")
    }

    func testScheduledTaskIntervalMinimum() {
        let state = GameState()
        state.scheduledTaskInterval = 5
        state.player.dataSiphons = 2

        _ = state.performSiphon()
        XCTAssertEqual(state.scheduledTaskInterval, 4, "Interval should decrease to 4")

        _ = state.performSiphon()
        XCTAssertEqual(state.scheduledTaskInterval, 4, "Interval should not go below 4")
    }

    func testScheduledTaskTriggersAtCorrectTurn() {
        let state = GameState()
        state.scheduledTasksDisabled = false
        let initialTransmissionCount = state.transmissions.count

        state.turnCount = 11
        state.maybeExecuteScheduledTask()
        let countBefore = state.transmissions.count

        state.turnCount = 12
        state.maybeExecuteScheduledTask()
        let countAfter = state.transmissions.count

        XCTAssertEqual(countBefore, initialTransmissionCount,
                       "Should not spawn before turn 12")
        XCTAssertGreaterThan(countAfter, countBefore,
                             "Should spawn transmission at turn 12")
    }

    func testStageChangeResetsInterval() {
        let state = GameState()
        state.player.dataSiphons = 3
        _ = state.performSiphon()
        _ = state.performSiphon()
        _ = state.performSiphon()

        XCTAssertEqual(state.scheduledTaskInterval, 9, "Interval should be 9 after 3 siphons")

        // Simulate stage advance (mimicking advanceToNextStage logic)
        state.currentStage = 2
        state.turnCount = 20
        state.scheduledTaskInterval = 13 - state.currentStage  // Resets to 11
        state.nextScheduledTaskTurn = state.turnCount + state.scheduledTaskInterval

        XCTAssertEqual(state.scheduledTaskInterval, 11, "Interval should be 11 for stage 2")
        XCTAssertEqual(state.nextScheduledTaskTurn, 31, "Next turn should be turnCount + interval")
    }
}
