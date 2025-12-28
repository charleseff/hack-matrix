import Foundation

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
    /// - Returns: Reward value for RL training
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
        dataSiphonCollected: Bool
    ) -> Double {

        /* ============================================================================
         * BACKUP: Original reward structure (pre-2024-12-23)
         * Kept for reference in case we want to revert or compare performance
         * ============================================================================
         *
         * let scoreDelta = Double(currentScore - oldScore)
         * var reward = scoreDelta * 0.1
         *
         * if dataSiphonCollected {
         *     reward += 1.0
         * }
         *
         * reward += Double(totalKills) * 0.3
         *
         * let hpChange = currentHP - oldHP
         * reward += Double(hpChange) * 1.0
         *
         * if playerDied {
         *     reward = 0.0
         * } else if gameWon {
         *     reward = Double(currentScore) * 10.0 + 10
         * } else if stageAdvanced {
         *     reward += 1
         *     let resourceBonus = Double(credits + energy) * 0.02
         *     reward += resourceBonus
         * }
         * ============================================================================
         */

        // CURRENT REWARD STRUCTURE (2024-12-26)
        // Objective: Maximize score while winning the game
        // Philosophy: Reward both outcomes (stages, victory) AND tactics (kills, siphons)
        // Dense intermediate rewards help agent learn good strategies

        var reward = 0.0

        // 1. PROGRESSIVE STAGE COMPLETION (guides agent toward winning)
        // Exponential scaling creates strong gradient toward later stages
        // Early stages easy to reach, later stages increasingly valuable
        if stageAdvanced {
            let stageRewards: [Double] = [1, 2, 4, 8, 16, 32, 64, 100]
            let completedStage = currentStage - 1  // Stage just completed (1-8)
            if completedStage >= 1 && completedStage <= 8 {
                reward += stageRewards[completedStage - 1]
            }
        }

        // 2. SCORE GAIN (encourages collecting points during gameplay)
        // Meaningful but not dominant - provides feedback for collecting valuable blocks
        let scoreDelta = Double(currentScore - oldScore)
        reward += scoreDelta * 0.5

        // 3. INTERMEDIATE REWARDS (dense feedback for good tactical behaviors)
        // These provide immediate feedback during gameplay, helping agent learn good strategies

        // Reward kills (excludes scheduled task spawns - already filtered in removeDeadEnemies)
        reward += Double(totalKills) * 0.3

        // Reward collecting data siphons (important strategic resource)
        if dataSiphonCollected {
            reward += 1.0
        }

        // 4. VICTORY BONUSES (massive rewards for winning)
        if gameWon {
            // Base victory bonus for completing all 8 stages
            reward += 500.0

            // SCORE BONUS - This is what the game is really about!
            // High-scoring wins are worth MUCH more than low-scoring wins
            reward += Double(currentScore) * 100.0
        }

        // 5. DEATH PENALTY (scales with progress but always less than cumulative rewards)
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
            let deathPenalty = cumulativeReward * 0.5
            reward += -deathPenalty
        }

        return reward
    }
}
