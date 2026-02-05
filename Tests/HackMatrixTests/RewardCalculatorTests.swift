import XCTest
@testable import HackMatrixCore

final class RewardCalculatorTests: XCTestCase {

    // MARK: - Helpers

    /// Calls RewardCalculator.calculate with all-zero/false defaults, applying overrides.
    private func calculate(
        oldScore: Int = 0, currentScore: Int = 0, currentStage: Int = 1,
        oldHP: Int = 3, currentHP: Int = 3,
        oldCredits: Int = 0, currentCredits: Int = 0,
        oldEnergy: Int = 0, currentEnergy: Int = 0,
        playerDied: Bool = false, gameWon: Bool = false, stageAdvanced: Bool = false,
        blocksSiphoned: Int = 0, programsAcquired: Int = 0,
        creditsGained: Int = 0, energyGained: Int = 0,
        totalKills: Int = 0, dataSiphonCollected: Bool = false,
        distanceToExitDelta: Int = 0,
        siphonWasUsed: Bool = false, siphonWasOptimal: Bool = true,
        siphonMissedCredits: Int = 0, siphonMissedEnergy: Int = 0,
        resetWasWasteful: Bool = false, diedToSiphonEnemy: Bool = false
    ) -> RewardBreakdown {
        RewardCalculator.calculate(
            oldScore: oldScore, currentScore: currentScore, currentStage: currentStage,
            oldHP: oldHP, currentHP: currentHP,
            oldCredits: oldCredits, currentCredits: currentCredits,
            oldEnergy: oldEnergy, currentEnergy: currentEnergy,
            playerDied: playerDied, gameWon: gameWon, stageAdvanced: stageAdvanced,
            blocksSiphoned: blocksSiphoned, programsAcquired: programsAcquired,
            creditsGained: creditsGained, energyGained: energyGained,
            totalKills: totalKills, dataSiphonCollected: dataSiphonCollected,
            distanceToExitDelta: distanceToExitDelta,
            siphonWasUsed: siphonWasUsed, siphonWasOptimal: siphonWasOptimal,
            siphonMissedCredits: siphonMissedCredits, siphonMissedEnergy: siphonMissedEnergy,
            resetWasWasteful: resetWasWasteful, diedToSiphonEnemy: diedToSiphonEnemy
        )
    }

    // MARK: - Siphon Quality Sign Bug

    func testSuboptimalSiphonIsNegativePenalty() {
        // Suboptimal siphon that missed 10 credits should produce a NEGATIVE siphonQuality
        let breakdown = calculate(
            siphonWasUsed: true,
            siphonWasOptimal: false,
            siphonMissedCredits: 10,
            siphonMissedEnergy: 0
        )

        XCTAssertLessThan(breakdown.siphonQuality, 0,
            "Suboptimal siphon should produce negative penalty, got \(breakdown.siphonQuality)")

        // missedValue = 10 * 0.05 = 0.5, penalty = -0.5 * 0.5 = -0.25
        XCTAssertEqual(breakdown.siphonQuality, -0.25, accuracy: 1e-10,
            "Expected -0.25 penalty for missing 10 credits")
    }

    func testSuboptimalSiphonMissedBothResources() {
        let breakdown = calculate(
            siphonWasUsed: true,
            siphonWasOptimal: false,
            siphonMissedCredits: 10,
            siphonMissedEnergy: 10
        )

        // missedValue = (10 * 0.05) + (10 * 0.05) = 1.0, penalty = -0.5 * 1.0 = -0.5
        XCTAssertEqual(breakdown.siphonQuality, -0.5, accuracy: 1e-10,
            "Expected -0.5 penalty for missing 10 credits and 10 energy")
    }

    func testOptimalSiphonNoPenalty() {
        let breakdown = calculate(
            siphonWasUsed: true,
            siphonWasOptimal: true,
            siphonMissedCredits: 0,
            siphonMissedEnergy: 0
        )

        XCTAssertEqual(breakdown.siphonQuality, 0.0,
            "Optimal siphon should have zero penalty")
    }

    func testNoSiphonNoPenalty() {
        let breakdown = calculate(
            siphonWasUsed: false,
            siphonWasOptimal: false,
            siphonMissedCredits: 20,
            siphonMissedEnergy: 20
        )

        XCTAssertEqual(breakdown.siphonQuality, 0.0,
            "Non-siphon action should have zero siphon quality penalty")
    }

    func testSuboptimalSiphonAffectsTotal() {
        let baseline = calculate()
        let suboptimal = calculate(
            siphonWasUsed: true,
            siphonWasOptimal: false,
            siphonMissedCredits: 10,
            siphonMissedEnergy: 0
        )

        XCTAssertLessThan(suboptimal.total, baseline.total,
            "Suboptimal siphon should reduce total reward")
    }
}
