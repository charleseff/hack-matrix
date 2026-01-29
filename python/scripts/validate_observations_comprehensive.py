"""
Comprehensive observation space validator using --debug-scenario mode.

Tests specific scenarios to ensure 100% coverage of all observation and reward components.
"""

import os
import sys
from collections import defaultdict

# Add python directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hackmatrix import HackEnv


class ComprehensiveValidator:
    """Comprehensive validator for observation space coverage."""

    def __init__(self):
        self.observed_states = defaultdict(set)
        self.issues = []
        self.test_results = {}

    def validate_observation(self, obs, context=""):
        """Validate observation structure and track coverage."""

        # Check top-level structure
        expected_keys = {"player", "programs", "grid"}
        if set(obs.keys()) != expected_keys:
            self.issues.append(
                f"{context}: Missing keys! Expected {expected_keys}, got {set(obs.keys())}"
            )
            return False

        # Validate player array
        player = obs["player"]
        if len(player) != 10:
            self.issues.append(f"{context}: Player array should be length 10, got {len(player)}")
            return False

        # Track player states
        hp = round(player[2] * 3)
        self.observed_states["player_hp"].add(hp)
        self.observed_states["player_show_activated"].add(int(player[8] > 0.5))
        self.observed_states["player_scheduled_tasks_disabled"].add(int(player[9] > 0.5))

        # Validate ranges
        for i, value in enumerate(player):
            if not (0.0 <= value <= 1.0):
                self.issues.append(f"{context}: Player[{i}] = {value} out of range [0,1]")
                return False

        # Validate programs
        programs = obs["programs"]
        if len(programs) != 23:
            self.issues.append(f"{context}: Programs should be length 23, got {len(programs)}")
            return False

        for i, owned in enumerate(programs):
            if owned == 1:
                self.observed_states["owned_programs"].add(i)

        # Validate grid
        grid = obs["grid"]
        if grid.shape != (6, 6, 42):
            self.issues.append(f"{context}: Grid should be (6,6,42), got {grid.shape}")
            return False

        if np.any(grid < 0) or np.any(grid > 1):
            self.issues.append(f"{context}: Grid has values outside [0,1]")
            return False

        # Track active channels
        for channel in range(40):
            if np.any(grid[:, :, channel] != 0):
                self.observed_states["active_grid_channels"].add(channel)

        return True

    def validate_reward(self, info, context=""):
        """Validate reward breakdown structure."""

        if "reward_breakdown" not in info:
            self.issues.append(f"{context}: Missing reward_breakdown in info")
            return False

        breakdown = info["reward_breakdown"]
        expected_components = [
            "stage",
            "score",
            "kills",
            "dataSiphon",
            "distance",
            "victory",
            "death",
            "resourceGain",
            "resourceHolding",
            "damagePenalty",
            "hpRecovery",
            "siphonQuality",
            "programWaste",
            "siphonDeathPenalty",
        ]

        missing = [c for c in expected_components if c not in breakdown]
        if missing:
            self.issues.append(f"{context}: Missing reward components: {missing}")
            return False

        # Track non-zero components
        for component, value in breakdown.items():
            if value != 0:
                self.observed_states["nonzero_reward_components"].add(component)

        return True

    def test_basic_movement(self, env):
        """Test basic movement and observation validity."""
        print("\n🧪 Test 1: Basic Movement")

        obs, info = env.reset()
        if not self.validate_observation(obs, "Reset"):
            self.test_results["basic_movement"] = "❌ Failed"
            return

        # Take a few movement actions
        for direction in [0, 1, 2, 3]:  # up, down, left, right
            obs, reward, done, truncated, info = env.step(direction)
            if not self.validate_observation(obs, f"Move {direction}"):
                self.test_results["basic_movement"] = "❌ Failed"
                return
            self.validate_reward(info, f"Move {direction}")

            if done or truncated:
                break

        self.test_results["basic_movement"] = "✅ Passed"
        print("  ✓ Movement actions work correctly")

    def test_siphon_mechanics(self, env):
        """Test siphon action and block siphoning."""
        print("\n🧪 Test 2: Siphon Mechanics")

        obs, info = env.reset()

        # Look for data blocks or program blocks in observation
        grid = obs["grid"]
        has_blocks = (
            np.any(grid[:, :, 6] > 0)  # Data blocks
            or np.any(grid[:, :, 7] > 0)  # Program blocks
            or np.any(grid[:, :, 8] > 0)  # Question blocks
        )

        if not has_blocks:
            print("  ⚠️  No blocks found in initial state, creating new game...")
            obs, info = env.reset()

        # Try to siphon (action 4)
        max_attempts = 50
        siphoned = False

        for i in range(max_attempts):
            obs, reward, done, truncated, info = env.step(4)  # Siphon action

            grid = obs["grid"]
            if np.any(grid[:, :, 10] > 0):  # Block siphoned channel
                self.observed_states["active_grid_channels"].add(10)
                siphoned = True
                print(f"  ✓ Block siphoned successfully (step {i + 1})")
                break

            if done or truncated:
                obs, info = env.reset()

        if siphoned:
            self.test_results["siphon_mechanics"] = "✅ Passed"
        else:
            self.test_results["siphon_mechanics"] = "⚠️  Partial (no blocks siphoned)"
            print("  ⚠️  Could not trigger block siphoning")

    def test_program_acquisition(self, env):
        """Test acquiring and using programs."""
        print("\n🧪 Test 3: Program Acquisition")

        obs, info = env.reset()

        # Try to acquire programs by siphoning
        max_steps = 100
        programs_acquired = False

        for i in range(max_steps):
            # Alternate between siphon and movement
            action = 4 if i % 2 == 0 else (i % 4)
            obs, reward, done, truncated, info = env.step(action)

            programs = obs["programs"]
            if np.any(programs == 1):
                programs_acquired = True
                print(f"  ✓ Program acquired (step {i + 1})")

                # Try to use the program
                for prog_idx in range(23):
                    if programs[prog_idx] == 1:
                        program_action = 5 + prog_idx  # Programs start at action 5
                        obs, reward, done, truncated, info = env.step(program_action)
                        print(f"  ✓ Used program at index {prog_idx} (action {program_action})")
                        break
                break

            if done or truncated:
                obs, info = env.reset()

        if programs_acquired:
            self.test_results["program_acquisition"] = "✅ Passed"
        else:
            self.test_results["program_acquisition"] = "⚠️  No programs acquired"
            print("  ⚠️  Could not acquire programs")

    def test_enemy_interactions(self, env):
        """Test enemy presence and interactions."""
        print("\n🧪 Test 4: Enemy Interactions")

        obs, info = env.reset()

        # Check for enemies in grid
        grid = obs["grid"]
        enemy_channels = [0, 1, 2, 3]  # virus, daemon, glitch, cryptog

        has_enemies = any(np.any(grid[:, :, ch] > 0) for ch in enemy_channels)

        if has_enemies:
            print("  ✓ Enemies present in observation")

            # Check enemy HP channel
            if np.any(grid[:, :, 4] > 0):
                print("  ✓ Enemy HP data present")

            self.test_results["enemy_interactions"] = "✅ Passed"
        else:
            self.test_results["enemy_interactions"] = "⚠️  No enemies found"
            print("  ⚠️  No enemies in observation")

    def test_reward_components(self, env):
        """Test that various reward components can be triggered."""
        print("\n🧪 Test 5: Reward Components")

        obs, info = env.reset()

        # Play for many steps to trigger various rewards
        max_steps = 200

        for i in range(max_steps):
            action = np.random.choice([0, 1, 2, 3, 4])  # Random move or siphon
            obs, reward, done, truncated, info = env.step(action)

            self.validate_reward(info, f"Step {i}")

            if done or truncated:
                # Check if we got death penalty
                if info.get("reward_breakdown", {}).get("death", 0) != 0:
                    print("  ✓ Death penalty triggered")
                obs, info = env.reset()

        active_rewards = len(self.observed_states["nonzero_reward_components"])
        print(f"  ✓ Triggered {active_rewards}/14 reward components")

        if active_rewards >= 5:
            self.test_results["reward_components"] = "✅ Passed"
        else:
            self.test_results["reward_components"] = "⚠️  Limited coverage"

    def test_special_cells(self, env):
        """Test data siphon and exit cells."""
        print("\n🧪 Test 6: Special Cells")

        obs, info = env.reset()
        grid = obs["grid"]

        has_data_siphon = np.any(grid[:, :, 38] > 0)
        has_exit = np.any(grid[:, :, 39] > 0)

        if has_data_siphon:
            print("  ✓ Data siphon cell detected")
        else:
            print("  ⚠️  No data siphon in initial state")

        if has_exit:
            print("  ✓ Exit cell detected")
        else:
            print("  ⚠️  No exit in initial state")

        if has_data_siphon or has_exit:
            self.test_results["special_cells"] = "✅ Passed"
        else:
            self.test_results["special_cells"] = "⚠️  No special cells found"

    def test_transmissions(self, env):
        """Test transmission observations."""
        print("\n🧪 Test 7: Transmissions")

        obs, info = env.reset()

        # Play for a while to see transmissions spawn
        max_steps = 100
        found_transmission = False

        for i in range(max_steps):
            grid = obs["grid"]

            # Check transmission spawn count and turns channels
            if np.any(grid[:, :, 34] > 0) or np.any(grid[:, :, 35] > 0):
                found_transmission = True
                print(f"  ✓ Transmission detected (step {i})")
                break

            action = 4 if i % 3 == 0 else (i % 4)  # Siphon occasionally
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                obs, info = env.reset()

        if found_transmission:
            self.test_results["transmissions"] = "✅ Passed"
        else:
            self.test_results["transmissions"] = "⚠️  No transmissions observed"
            print("  ⚠️  No transmissions spawned")

    def test_resources(self, env):
        """Test credit and energy observations."""
        print("\n🧪 Test 8: Resources")

        obs, info = env.reset()
        grid = obs["grid"]

        has_credits = np.any(grid[:, :, 36] > 0)
        has_energy = np.any(grid[:, :, 37] > 0)

        if has_credits:
            print("  ✓ Credits detected in grid")
        if has_energy:
            print("  ✓ Energy detected in grid")

        if has_credits or has_energy:
            self.test_results["resources"] = "✅ Passed"
        else:
            self.test_results["resources"] = "⚠️  No resources found"
            print("  ⚠️  No resources in grid")

    def run_all_tests(self, use_debug_scenario=True):
        """Run all validation tests."""

        print("=" * 80)
        print("COMPREHENSIVE OBSERVATION SPACE VALIDATION")
        print("=" * 80)
        print(f"\nDebug Scenario Mode: {'ENABLED' if use_debug_scenario else 'DISABLED'}")
        print("This creates predictable game states for systematic testing.\n")

        # Create environment with debug scenario
        env = HackEnv(debug=False, debug_scenario=use_debug_scenario)

        try:
            # Run all tests
            self.test_basic_movement(env)
            self.test_siphon_mechanics(env)
            self.test_program_acquisition(env)
            self.test_enemy_interactions(env)
            self.test_reward_components(env)
            self.test_special_cells(env)
            self.test_transmissions(env)
            self.test_resources(env)

        finally:
            env.close()

        # Print comprehensive report
        self.print_report()

    def print_report(self):
        """Print comprehensive validation report."""

        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        # Test results summary
        print("\n📋 Test Results:")
        for test_name, result in self.test_results.items():
            print(f"  {test_name.replace('_', ' ').title()}: {result}")

        # Issues
        if self.issues:
            print(f"\n❌ FOUND {len(self.issues)} ISSUES:")
            for issue in self.issues[:20]:
                print(f"  - {issue}")
            if len(self.issues) > 20:
                print(f"  ... and {len(self.issues) - 20} more")
        else:
            print("\n✅ NO STRUCTURAL ISSUES FOUND")

        # Coverage statistics
        print("\n" + "-" * 80)
        print("COVERAGE STATISTICS")
        print("-" * 80)

        # Player state
        print("\n📊 Player State Coverage:")
        print(f"  HP values: {sorted(self.observed_states['player_hp'])}")
        print(f"  Show activated: {sorted(self.observed_states['player_show_activated'])}")
        print(
            f"  Scheduled tasks disabled: {sorted(self.observed_states['player_scheduled_tasks_disabled'])}"
        )

        # Programs
        print("\n📚 Programs:")
        owned = sorted(self.observed_states["owned_programs"])
        print(f"  Owned: {len(owned)}/23")
        if owned:
            print(f"  Indices: {owned}")

        # Grid channels
        print("\n🗺️  Grid Channels:")
        active = sorted(self.observed_states["active_grid_channels"])
        print(f"  Active: {len(active)}/40")

        channel_map = {
            0: "Enemy virus",
            1: "Enemy daemon",
            2: "Enemy glitch",
            3: "Enemy cryptog",
            4: "Enemy HP",
            5: "Enemy stunned",
            6: "Block data",
            7: "Block program",
            8: "Block question",
            9: "Block points",
            10: "Block siphoned",
            34: "Transmission spawn count",
            35: "Transmission turns",
            36: "Credits",
            37: "Energy",
            38: "Data siphon cell",
            39: "Exit cell",
        }

        missing = [i for i in range(40) if i not in active]
        if missing:
            print(f"\n  Missing Channels ({len(missing)}):")
            for i in missing:
                if i in channel_map:
                    print(f"    - Channel {i}: {channel_map[i]}")
                elif 11 <= i <= 33:
                    print(f"    - Channel {i}: Program type {i - 11}")
                else:
                    print(f"    - Channel {i}")

        # Rewards
        print("\n💰 Reward Components:")
        active_rewards = sorted(self.observed_states["nonzero_reward_components"])
        print(f"  Active: {len(active_rewards)}/14")
        for comp in active_rewards:
            print(f"    ✓ {comp}")

        all_rewards = [
            "stage",
            "score",
            "kills",
            "dataSiphon",
            "distance",
            "victory",
            "death",
            "resourceGain",
            "resourceHolding",
            "damagePenalty",
            "hpRecovery",
            "siphonQuality",
            "programWaste",
            "siphonDeathPenalty",
        ]
        missing_rewards = [r for r in all_rewards if r not in active_rewards]
        if missing_rewards:
            print("\n  Never Triggered:")
            for comp in missing_rewards:
                print(f"    - {comp}")

        print("\n" + "=" * 80)

        # Final verdict
        issues_count = len(self.issues)
        coverage = len(active) / 40 * 100
        reward_coverage = len(active_rewards) / 14 * 100

        print("\n📊 FINAL VERDICT:")
        print(f"  Issues: {issues_count}")
        print(f"  Grid Coverage: {coverage:.1f}%")
        print(f"  Reward Coverage: {reward_coverage:.1f}%")

        if issues_count == 0 and coverage >= 80:
            print("\n✅ VALIDATION PASSED")
            print("   Observation space is correctly implemented!")
        elif issues_count == 0:
            print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
            print("   No structural issues, but limited scenario coverage")
        else:
            print("\n❌ VALIDATION FAILED")
            print("   Fix issues before training!")

        print("=" * 80)


if __name__ == "__main__":
    validator = ComprehensiveValidator()
    validator.run_all_tests(use_debug_scenario=True)

    # Exit with appropriate code
    sys.exit(0 if len(validator.issues) == 0 else 1)
