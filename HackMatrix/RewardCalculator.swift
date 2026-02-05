import Foundation

// MARK: - Reward Breakdown for RL Training

/// Detailed breakdown of reward components for debugging and analysis
struct RewardBreakdown {
    var stepPenalty: Double = 0
    var stageCompletion: Double = 0
    var scoreGain: Double = 0
    var kills: Double = 0
    var dataSiphonCollected: Double = 0
    var distanceShaping: Double = 0
    var victory: Double = 0
    var deathPenalty: Double = 0

    // NEW: Resource rewards
    var resourceGain: Double = 0        // Reward for acquiring credits/energy
    var resourceHolding: Double = 0     // Small bonus for maintaining reserves

    // NEW: HP penalties and recovery
    var damagePenalty: Double = 0       // Negative reward for taking damage
    var hpRecovery: Double = 0          // Positive reward for healing (offsets penalty)

    // NEW: Siphon optimization
    var siphonQuality: Double = 0       // Penalty if siphon was suboptimal

    // NEW: Program usage optimization
    var programWaste: Double = 0        // Penalty for wasteful program usage (e.g., RESET at 2 HP)

    // NEW: Siphon-caused death
    var siphonDeathPenalty: Double = 0  // Extra penalty for dying to siphon-spawned enemy

    var total: Double {
        stepPenalty + stageCompletion + scoreGain + kills + dataSiphonCollected +
        distanceShaping + victory + deathPenalty +
        resourceGain + resourceHolding +
        damagePenalty + hpRecovery +
        siphonQuality + programWaste +
        siphonDeathPenalty
    }
}

// MARK: - Reward Calculator for Reinforcement Learning

/// Pure reward calculation logic for RL training
/// Separated from GameState for easier testing and experimentation
struct RewardCalculator {

    // MARK: Resource Reward Constants
    private static let creditGainMultiplier: Double = 0.05      // Reward per credit gained
    private static let energyGainMultiplier: Double = 0.05      // Reward per energy gained
    private static let creditHoldingMultiplier: Double = 0.01   // Bonus per credit held at stage completion
    private static let energyHoldingMultiplier: Double = 0.01   // Bonus per energy held at stage completion

    // MARK: HP Penalty Constants
    private static let damagePenaltyPerHP: Double = -1.0        // Penalty per HP lost
    private static let hpRecoveryRewardPerHP: Double = 1.0      // Reward per HP recovered (full offset)

    // MARK: Siphon Optimization Constants
    private static let siphonSuboptimalPenalty: Double = -0.5   // Penalty if strictly better option existed

    // MARK: Program Waste Constants
    private static let resetAt2HPPenalty: Double = -0.3         // Penalty for RESET at 2 HP

    // MARK: Siphon Death Constants
    private static let siphonCausedDeathPenalty: Double = -10.0 // Extra penalty for dying to siphon-spawned enemy

    // MARK: Step Penalty Constants
    private static let stepPenalty: Double = -0.01 // Per-step cost to create time pressure

    /// Calculate reward for reinforcement learning based on action outcome
    /// - Parameters:
    ///   - oldScore: Player score before action
    ///   - currentScore: Player score after action
    ///   - currentStage: Current stage number (1-8, or 9 if won)
    ///   - oldHP: Player HP before action
    ///   - currentHP: Player HP after action
    ///   - oldCredits: Player credits before action
    ///   - currentCredits: Player credits after action
    ///   - oldEnergy: Player energy before action
    ///   - currentEnergy: Player energy after action
    ///   - playerDied: Whether player died
    ///   - gameWon: Whether player won the game
    ///   - stageAdvanced: Whether stage was completed
    ///   - blocksSiphoned: Number of blocks siphoned this step
    ///   - programsAcquired: Number of programs acquired this step
    ///   - creditsGained: Credits gained this step
    ///   - energyGained: Energy gained this step
    ///   - totalKills: Number of enemies killed (excludes scheduled task spawns)
    ///   - dataSiphonCollected: Whether a data siphon was collected this step
    ///   - distanceToExitDelta: Change in path distance to exit (oldDist - newDist, positive = closer)
    ///   - siphonWasUsed: Whether siphon action was taken
    ///   - siphonWasOptimal: Whether siphon position was optimal
    ///   - siphonMissedCredits: Credits missed by suboptimal siphon
    ///   - siphonMissedEnergy: Energy missed by suboptimal siphon
    ///   - resetWasWasteful: Whether RESET was used at 2 HP
    ///   - diedToSiphonEnemy: Whether death was caused by siphon-spawned enemy
    /// - Returns: RewardBreakdown with all reward components
    static func calculate(
        oldScore: Int,
        currentScore: Int,
        currentStage: Int,
        oldHP: Int,
        currentHP: Int,
        oldCredits: Int,
        currentCredits: Int,
        oldEnergy: Int,
        currentEnergy: Int,
        playerDied: Bool,
        gameWon: Bool,
        stageAdvanced: Bool,
        blocksSiphoned: Int,
        programsAcquired: Int,
        creditsGained: Int,
        energyGained: Int,
        totalKills: Int,
        dataSiphonCollected: Bool,
        distanceToExitDelta: Int,
        siphonWasUsed: Bool,
        siphonWasOptimal: Bool,
        siphonMissedCredits: Int,
        siphonMissedEnergy: Int,
        resetWasWasteful: Bool,
        diedToSiphonEnemy: Bool
    ) -> RewardBreakdown {

        var breakdown = RewardBreakdown()

        // 0. STEP PENALTY (creates time pressure, discourages oscillation)
        breakdown.stepPenalty = stepPenalty

        // 1. PROGRESSIVE STAGE COMPLETION (guides agent toward winning)
        // Exponential scaling creates strong gradient toward later stages
        // Early stages easy to reach, later stages increasingly valuable
        if stageAdvanced {
            let stageRewards: [Double] = [1, 2, 4, 8, 16, 32, 64, 100]
            let completedStage = currentStage - 1  // Stage just completed (1-8)
            if completedStage >= 1 && completedStage <= 8 {
                breakdown.stageCompletion = stageRewards[completedStage - 1]
            }
        }

        // 2. SCORE GAIN (encourages collecting points during gameplay)
        // Meaningful but not dominant - provides feedback for collecting valuable blocks
        let scoreDelta = Double(currentScore - oldScore)
        breakdown.scoreGain = scoreDelta * 0.5

        // 3. INTERMEDIATE REWARDS (dense feedback for good tactical behaviors)
        // These provide immediate feedback during gameplay, helping agent learn good strategies

        // Reward kills (excludes scheduled task spawns - already filtered in removeDeadEnemies)
        breakdown.kills = Double(totalKills) * 0.3

        // Reward collecting data siphons (important strategic resource)
        if dataSiphonCollected {
            breakdown.dataSiphonCollected = 1.0
        }

        // 4. DISTANCE SHAPING (guides agent toward exit)
        // One-directional: only reward for getting closer, no penalty for moving away
        // This prevents oscillation reward farming (left-right-left-right)
        breakdown.distanceShaping = Double(max(distanceToExitDelta, 0)) * 0.05

        // 5. VICTORY BONUSES (massive rewards for winning)
        if gameWon {
            // Base victory bonus for completing all 8 stages
            // Plus SCORE BONUS - high-scoring wins are worth more
            breakdown.victory = 500.0 + Double(currentScore) * 100.0
        }

        // 6. DEATH PENALTY (scales with progress but always less than cumulative rewards)
        // Death penalty increases with each stage, but completing stages is still net positive
        if playerDied {
            let stageRewards: [Double] = [1, 2, 4, 8, 16, 32, 64, 100]

            // Calculate total rewards earned from all completed stages
            var cumulativeReward = 0.0
            for i in 0..<(currentStage - 1) {  // currentStage is next stage (1-8), so -1 for completed
                if i < stageRewards.count {
                    cumulativeReward += stageRewards[i]
                }
            }

            // Death penalty = 50% of cumulative rewards
            // Stage 1 death: -0.5, Stage 4 death: -7.5, Stage 8 death: -113.5
            // Always ensures net positive for completing stages before death
            breakdown.deathPenalty = -cumulativeReward * 0.5
        }

        // 7. RESOURCE GAIN REWARD (encourages collecting resources)
        let creditsGainedDelta = currentCredits - oldCredits
        let energyGainedDelta = currentEnergy - oldEnergy
        breakdown.resourceGain = Double(creditsGainedDelta) * creditGainMultiplier +
                                 Double(energyGainedDelta) * energyGainMultiplier

        // 8. RESOURCE HOLDING REWARD (bonus for maintaining reserves ONLY on stage completion)
        // Only granted when completing a stage to prevent infinite reward farming by staying in same stage
        if stageAdvanced {
            breakdown.resourceHolding = Double(currentCredits) * creditHoldingMultiplier +
                                        Double(currentEnergy) * energyHoldingMultiplier
        }

        // 9. DAMAGE PENALTY (negative reward for taking damage)
        let hpLost = oldHP - currentHP
        breakdown.damagePenalty = hpLost > 0 ? Double(hpLost) * damagePenaltyPerHP : 0.0

        // 10. HP RECOVERY REWARD (positive reward for healing - offsets damage penalty)
        let hpGained = currentHP - oldHP
        breakdown.hpRecovery = hpGained > 0 ? Double(hpGained) * hpRecoveryRewardPerHP : 0.0

        // 11. SIPHON QUALITY (penalty for suboptimal siphon position)
        if siphonWasUsed {
            if !siphonWasOptimal {
                // Penalty proportional to missed resources
                let missedValue = Double(siphonMissedCredits) * creditGainMultiplier +
                                 Double(siphonMissedEnergy) * energyGainMultiplier
                breakdown.siphonQuality = -siphonSuboptimalPenalty * missedValue
            }
        }

        // 12. PROGRAM WASTE PENALTY (wasteful program usage)
        breakdown.programWaste = resetWasWasteful ? resetAt2HPPenalty : 0.0

        // 13. SIPHON-CAUSED DEATH PENALTY (extra penalty for dying to siphon-spawned enemy)
        breakdown.siphonDeathPenalty = diedToSiphonEnemy ? siphonCausedDeathPenalty : 0.0

        return breakdown
    }
}
