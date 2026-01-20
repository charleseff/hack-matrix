#!/usr/bin/env python3
"""
Comprehensive test runner that doesn't require pytest.

This script imports test classes and methods from the test files and runs them
using a simple test framework. Assertions raise AssertionError on failure.

Usage:
    export HACKMATRIX_BINARY=/workspaces/868-hack-2/.build/debug/HackMatrix
    cd python && python3 tests/run_all_tests.py

For specific test files:
    python3 tests/run_all_tests.py --file test_movement.py
"""

import sys
import traceback
import argparse
import time
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, '.')

from tests.swift_env_wrapper import SwiftEnvWrapper


@dataclass
class TestResult:
    name: str
    passed: bool
    error: str | None = None
    duration: float = 0.0


class TestRunner:
    """Simple test runner that mimics pytest behavior."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.env = None

    def setup_env(self):
        """Create environment for tests."""
        if self.env is None:
            self.env = SwiftEnvWrapper()
            self.env.__enter__()

    def teardown_env(self):
        """Close environment."""
        if self.env is not None:
            self.env.__exit__(None, None, None)
            self.env = None

    def run_test_method(self, test_class, method_name: str):
        """Run a single test method."""
        full_name = f"{test_class.__name__}.{method_name}"

        if self.verbose:
            print(f"  {method_name}...", end=" ", flush=True)

        start_time = time.time()
        try:
            # Create instance and run
            instance = test_class()
            method = getattr(instance, method_name)

            # Pass the env to the method
            method(self.env)

            duration = time.time() - start_time
            self.results.append(TestResult(full_name, True, duration=duration))
            if self.verbose:
                print(f"PASSED ({duration:.2f}s)")
            return True

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {e}"
            self.results.append(TestResult(full_name, False, error_msg, duration))
            if self.verbose:
                print(f"FAILED ({duration:.2f}s)")
                if self.verbose:
                    traceback.print_exc()
            return False

    def run_test_class(self, test_class):
        """Run all test methods in a class."""
        methods = [m for m in dir(test_class) if m.startswith('test_')]

        if not methods:
            return

        print(f"\n{test_class.__name__}:")

        for method_name in methods:
            self.run_test_method(test_class, method_name)

    def run_test_module(self, module_name: str):
        """Run all test classes from a module."""
        print(f"\n{'='*60}")
        print(f"Running {module_name}")
        print('='*60)

        self.setup_env()

        try:
            # Import the module
            if module_name == 'test_movement':
                from tests.test_movement import (
                    TestMoveToEmptyCell,
                    TestMoveBlockedByEdge,
                    TestMoveBlockedByBlock,
                    TestMoveCollectsDataSiphon,
                    TestLineOfSightAttack,
                    TestLineOfSightAttackOnBlock,
                    TestAttackKillsEnemy,
                    TestAttackEnemySurvives,
                    TestAttackTransmission,
                )
                test_classes = [
                    TestMoveToEmptyCell,
                    TestMoveBlockedByEdge,
                    TestMoveBlockedByBlock,
                    TestMoveCollectsDataSiphon,
                    TestLineOfSightAttack,
                    TestLineOfSightAttackOnBlock,
                    TestAttackKillsEnemy,
                    TestAttackEnemySurvives,
                    TestAttackTransmission,
                ]
            elif module_name == 'test_siphon':
                from tests.test_siphon import (
                    TestSiphonDataBlock,
                    TestSiphonValidWithSiphons,
                    TestSiphonInvalidWithoutSiphons,
                    TestSiphonSpawnsTransmissions,
                    TestSiphonDoesNotRevealResources,
                    TestSiphonProgramBlock,
                )
                test_classes = [
                    TestSiphonDataBlock,
                    TestSiphonValidWithSiphons,
                    TestSiphonInvalidWithoutSiphons,
                    TestSiphonSpawnsTransmissions,
                    TestSiphonDoesNotRevealResources,
                    TestSiphonProgramBlock,
                ]
            elif module_name == 'test_programs':
                from tests.test_programs import (
                    TestPushProgram,
                    TestPullProgram,
                    TestCrashProgram,
                    TestWarpProgram,
                    TestPolyProgram,
                    TestWaitProgram,
                    TestDebugProgram,
                    TestRowProgram,
                    TestColProgram,
                    TestSiphPlusProgram,
                    TestExchProgram,
                    TestShowProgram,
                    TestResetProgram,
                    TestCalmProgram,
                    TestDBomProgram,
                    TestDelayProgram,
                    TestAntiVProgram,
                    TestScoreProgram,
                    TestReducProgram,
                    TestAtkPlusProgram,
                    TestHackProgram,
                    TestProgramChaining,
                )
                test_classes = [
                    TestPushProgram,
                    TestPullProgram,
                    TestCrashProgram,
                    TestWarpProgram,
                    TestPolyProgram,
                    TestWaitProgram,
                    TestDebugProgram,
                    TestRowProgram,
                    TestColProgram,
                    TestSiphPlusProgram,
                    TestExchProgram,
                    TestShowProgram,
                    TestResetProgram,
                    TestCalmProgram,
                    TestDBomProgram,
                    TestDelayProgram,
                    TestAntiVProgram,
                    TestScoreProgram,
                    TestReducProgram,
                    TestAtkPlusProgram,
                    TestHackProgram,
                    TestProgramChaining,
                ]
            elif module_name == 'test_enemies':
                from tests.test_enemies import (
                    TestEnemySpawnsFromTransmission,
                    TestEnemyMovesIntoPlayer,
                    TestVirusDoubleMoves,
                    TestGlitchMovesOnBlocks,
                    TestCryptogVisibility,
                    TestEnemyAttacksPlayer,
                    TestStunnedEnemyNoMovement,
                    TestEnemiesMoveAfterTurnEnd,
                )
                test_classes = [
                    TestEnemySpawnsFromTransmission,
                    TestEnemyMovesIntoPlayer,
                    TestVirusDoubleMoves,
                    TestGlitchMovesOnBlocks,
                    TestCryptogVisibility,
                    TestEnemyAttacksPlayer,
                    TestStunnedEnemyNoMovement,
                    TestEnemiesMoveAfterTurnEnd,
                ]
            elif module_name == 'test_turns':
                from tests.test_turns import (
                    TestMoveEndsTurn,
                    TestProgramsDoNotEndTurn,
                    TestWaitEndsTurn,
                    TestAttackEndsTurn,
                    TestSiphonEndsTurn,
                    TestProgramChaining,
                )
                test_classes = [
                    TestMoveEndsTurn,
                    TestProgramsDoNotEndTurn,
                    TestWaitEndsTurn,
                    TestAttackEndsTurn,
                    TestSiphonEndsTurn,
                    TestProgramChaining,
                ]
            else:
                print(f"Unknown module: {module_name}")
                return

            for test_class in test_classes:
                self.run_test_class(test_class)

        finally:
            self.teardown_env()

    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        total_time = sum(r.duration for r in self.results)

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  FAILED: {r.name}")
                    if r.error:
                        print(f"          {r.error}")

        print(f"\nResults: {passed} passed, {failed} failed out of {total} tests")
        print(f"Total time: {total_time:.2f}s")
        print("=" * 60)

        return failed == 0


def main():
    parser = argparse.ArgumentParser(description='Run environment parity tests')
    parser.add_argument('--file', type=str, help='Specific test file to run (e.g., test_movement.py)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    args = parser.parse_args()

    runner = TestRunner(verbose=not args.quiet)

    if args.file:
        # Run specific file
        module_name = args.file.replace('.py', '')
        runner.run_test_module(module_name)
    else:
        # Run all test modules
        modules = ['test_movement', 'test_siphon', 'test_programs', 'test_enemies', 'test_turns']
        for module in modules:
            runner.run_test_module(module)

    success = runner.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
