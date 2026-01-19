"""
Validate that all observation space data is correctly passed from Swift to Python.

Tests across different game scenarios to ensure comprehensive coverage.
"""

import sys
from collections import defaultdict

import numpy as np

from hackmatrix import HackEnv


class ObservationValidator:
    """Validates observation space coverage and correctness."""

    def __init__(self):
        self.observed_states = defaultdict(set)
        self.issues = []

    def validate_observation(self, obs, step_num, episode_num):
        """Validate a single observation and track what we've seen."""

        # Check top-level structure
        expected_keys = {'player', 'programs', 'grid'}
        if set(obs.keys()) != expected_keys:
            self.issues.append(f"Ep{episode_num} Step{step_num}: Missing keys! Expected {expected_keys}, got {set(obs.keys())}")
            return

        # Validate player array (10 values)
        # [row, col, hp, credits, energy, stage, siphons, attack, showActivated, scheduledTasksDisabled]
        player = obs['player']
        if len(player) != 10:
            self.issues.append(f"Ep{episode_num} Step{step_num}: Player array should be length 10, got {len(player)}")
        else:
            # Track observed values (denormalized for readability)
            # Note: Values are normalized in [0,1], need to denormalize to check
            row = player[0] * 5  # 0-5
            col = player[1] * 5  # 0-5
            hp = player[2] * 3   # 0-3

            self.observed_states['player_hp'].add(round(hp))
            self.observed_states['player_show_activated'].add(int(player[8] > 0.5))
            self.observed_states['player_scheduled_tasks_disabled'].add(int(player[9] > 0.5))

            # Validate normalized ranges [0, 1]
            for i, value in enumerate(player):
                if not (0.0 <= value <= 1.0):
                    self.issues.append(f"Ep{episode_num} Step{step_num}: Player[{i}] = {value} out of range [0,1]")

        # Validate programs array (23 programs)
        programs = obs['programs']
        if len(programs) != 23:
            self.issues.append(f"Ep{episode_num} Step{step_num}: Programs array should be length 23, got {len(programs)}")
        else:
            # Track which programs we've seen owned
            for i, owned in enumerate(programs):
                if owned == 1:
                    self.observed_states['owned_programs'].add(i)
                elif owned != 0:
                    self.issues.append(f"Ep{episode_num} Step{step_num}: Programs[{i}] should be 0 or 1, got {owned}")

        # Validate grid (6x6x40)
        # Enemy: one-hot types (4) + hp + stunned = 6
        # Block: one-hot types (3) + points + siphoned = 5
        # Program: one-hot (23) + transmission_spawncount + transmission_turns = 25
        # Resources: credits + energy = 2
        # Special: is_data_siphon + is_exit = 2
        # Total: 40 channels
        grid = obs['grid']
        if grid.shape != (6, 6, 40):
            self.issues.append(f"Ep{episode_num} Step{step_num}: Grid should be (6,6,40), got {grid.shape}")
        else:
            # Check for non-zero values in each channel to track coverage
            for channel in range(40):
                if np.any(grid[:, :, channel] != 0):
                    self.observed_states['active_grid_channels'].add(channel)

            # Validate normalized ranges [0, 1]
            if np.any(grid < 0) or np.any(grid > 1):
                self.issues.append(f"Ep{episode_num} Step{step_num}: Grid has values outside [0,1]")

    def validate_info(self, info, step_num, episode_num):
        """Validate info dict structure."""

        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']

            # Check all 14 reward components are present
            expected_components = [
                'stage', 'score', 'kills', 'dataSiphon', 'distance', 'victory', 'death',
                'resourceGain', 'resourceHolding', 'damagePenalty', 'hpRecovery',
                'siphonQuality', 'programWaste', 'siphonDeathPenalty'
            ]

            missing = [c for c in expected_components if c not in breakdown]
            if missing:
                self.issues.append(f"Ep{episode_num} Step{step_num}: Missing reward components: {missing}")

            # Track which components have been non-zero
            for component, value in breakdown.items():
                if value != 0:
                    self.observed_states['nonzero_reward_components'].add(component)

    def report(self):
        """Print comprehensive validation report."""

        print("\n" + "="*80)
        print("OBSERVATION SPACE VALIDATION REPORT")
        print("="*80)

        # Issues
        if self.issues:
            print(f"\n❌ FOUND {len(self.issues)} ISSUES:")
            for issue in self.issues[:20]:  # Limit to first 20
                print(f"  - {issue}")
            if len(self.issues) > 20:
                print(f"  ... and {len(self.issues) - 20} more")
        else:
            print("\n✅ NO ISSUES FOUND")

        # Coverage statistics
        print("\n" + "-"*80)
        print("COVERAGE STATISTICS")
        print("-"*80)

        print("\n📊 Player State Coverage:")
        print(f"  HP values seen: {sorted(self.observed_states['player_hp'])}")
        print(f"  Show activated: {sorted(self.observed_states['player_show_activated'])}")
        print(f"  Scheduled tasks disabled: {sorted(self.observed_states['player_scheduled_tasks_disabled'])}")

        print("\n📚 Programs Coverage:")
        owned_programs = sorted(self.observed_states['owned_programs'])
        print(f"  Programs owned: {len(owned_programs)}/23")
        if owned_programs:
            print(f"  Program indices: {owned_programs}")

        print("\n🗺️  Grid Coverage:")
        active_channels = sorted(self.observed_states['active_grid_channels'])
        print(f"  Active grid channels: {len(active_channels)}/40")
        print(f"  Channel indices: {active_channels}")

        channel_names = [
            # Enemy (6 channels)
            "0: Enemy virus (one-hot)", "1: Enemy daemon (one-hot)",
            "2: Enemy glitch (one-hot)", "3: Enemy cryptog (one-hot)",
            "4: Enemy HP (normalized)", "5: Enemy stunned (binary)",
            # Block (5 channels)
            "6: Block data (one-hot)", "7: Block program (one-hot)", "8: Block question (one-hot)",
            "9: Block points (normalized)", "10: Block siphoned (binary)",
            # Program (25 channels: 23 one-hot + 2 transmission)
            "11-33: Program types (23 one-hot)",
            "34: Transmission spawn count", "35: Transmission turns remaining",
            # Resources (2 channels)
            "36: Credits (normalized)", "37: Energy (normalized)",
            # Special (2 channels)
            "38: Is data siphon (binary)", "39: Is exit (binary)"
        ]

        if active_channels != list(range(40)):
            print("\n  Missing channels:")
            for i in range(40):
                if i not in active_channels:
                    if i >= 11 and i <= 33:
                        print(f"    - Channel {i}: Program type {i-11} (one-hot)")
                    elif i < len(channel_names):
                        print(f"    - {channel_names[i]}")
                    else:
                        print(f"    - Channel {i}")

        print("\n💰 Reward Components Coverage:")
        active_rewards = sorted(self.observed_states['nonzero_reward_components'])
        print(f"  Non-zero components: {len(active_rewards)}/14")
        for component in active_rewards:
            print(f"    ✓ {component}")

        all_rewards = [
            'stage', 'score', 'kills', 'dataSiphon', 'distance', 'victory', 'death',
            'resourceGain', 'resourceHolding', 'damagePenalty', 'hpRecovery',
            'siphonQuality', 'programWaste', 'siphonDeathPenalty'
        ]
        missing_rewards = [r for r in all_rewards if r not in active_rewards]
        if missing_rewards:
            print(f"\n  Never non-zero (may be expected):")
            for component in missing_rewards:
                print(f"    - {component}")

        print("\n" + "="*80)


def run_validation(num_episodes=20, max_steps=200):
    """Run validation across multiple episodes."""

    print(f"Running observation validation for {num_episodes} episodes...")
    print(f"This will test different game scenarios to maximize coverage.\n")

    validator = ObservationValidator()
    env = HackEnv(debug=False)

    for episode in range(num_episodes):
        obs, info = env.reset()
        validator.validate_observation(obs, 0, episode)
        validator.validate_info(info, 0, episode)

        for step in range(max_steps):
            # Take random valid action (simple approach)
            # In reality, the action doesn't matter much for observation validation
            action = step % 28  # Cycle through all actions

            obs, reward, done, truncated, info = env.step(action)
            validator.validate_observation(obs, step + 1, episode)
            validator.validate_info(info, step + 1, episode)

            if done or truncated:
                break

        if (episode + 1) % 5 == 0:
            print(f"  Completed episode {episode + 1}/{num_episodes}")

    env.close()
    validator.report()

    return len(validator.issues) == 0


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
