import Foundation

// MARK: - Reward Breakdown for RL Training

/// Detailed breakdown of reward components for debugging and analysis
struct RewardBreakdown {
    var stageCompletion: Double = 0
    var scoreGain: Double = 0
    var kills: Double = 0
    var dataSiphonCollected: Double = 0
    var distanceShaping: Double = 0
    var victory: Double = 0
    var deathPenalty: Double = 0

    var total: Double {
        stageCompletion + scoreGain + kills + dataSiphonCollected +
        distanceShaping + victory + deathPenalty
    }
}

// MARK: - Reward Calculator for Reinforcement Learning

/// Pure reward calculation logic for RL training
/// Separated from GameState for easier testing and experimentation
struct RewardCalculator {

    /// Calculate reward for reinforcement learning based on action outcome
    /// - Parameters:
    ///   - oldScore: Player score before action
    ///   - currentScore: Player score after action
    ///   - currentStage: Current stage number (1-8, or 9 if won)
    ///   - oldHP: Player HP before action (unused in current implementation)
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
    /// - Returns: RewardBreakdown with all reward components
    static func calculate(
        oldScore: Int,
        currentScore: Int,
        currentStage: Int,
        oldHP: Int,
        playerDied: Bool,
        gameWon: Bool,
        stageAdvanced: Bool,
        blocksSiphoned: Int,
        programsAcquired: Int,
        creditsGained: Int,
        energyGained: Int,
        totalKills: Int,
        dataSiphonCollected: Bool,
        distanceToExitDelta: Int
    ) -> RewardBreakdown {

        var breakdown = RewardBreakdown()

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
        // Small reward for getting closer, penalty for getting farther
        // Also rewards destroying blocks that create shorter paths
        breakdown.distanceShaping = Double(distanceToExitDelta) * 0.05

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

        return breakdown
    }
}
