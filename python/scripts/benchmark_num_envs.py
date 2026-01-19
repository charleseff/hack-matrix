"""
Benchmark different --num-envs values to find optimal parallelism.
"""

import os
import sys
import time

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add parent to path for hackmatrix import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hackmatrix import HackEnv
from hackmatrix.training_config import FAST_MODEL_CONFIG
from hackmatrix.training_utils import make_env


def benchmark_num_envs(num_envs: int, timesteps: int = 10_000) -> float:
    """
    Benchmark training speed with given number of environments.
    Returns steps per second.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking num_envs={num_envs} for {timesteps:,} timesteps...")
    print(f"{'='*60}")

    # Create environments
    if num_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
    else:
        env = DummyVecEnv([make_env])

    # Create model with minimal logging
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=0,  # Quiet mode
        **FAST_MODEL_CONFIG  # Use shared fast config for benchmarking
    )

    # Warmup - let environments initialize
    print("Warming up...")
    model.learn(total_timesteps=1000, progress_bar=False)

    # Benchmark
    print("Benchmarking...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    elapsed = time.time() - start_time

    steps_per_sec = timesteps / elapsed

    # Cleanup
    env.close()

    print(f"Result: {steps_per_sec:.1f} steps/sec ({elapsed:.1f}s for {timesteps:,} steps)")
    return steps_per_sec


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark --num-envs values")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="Timesteps per benchmark (default: 10,000)")
    parser.add_argument("--env-values", type=str, default="1,2,4,6,8,10,12,14",
                        help="Comma-separated list of num_envs values to test")
    args = parser.parse_args()

    env_values = [int(x) for x in args.env_values.split(",")]

    print(f"Benchmarking num_envs values: {env_values}")
    print(f"Timesteps per test: {args.timesteps:,}")

    results = {}
    for num_envs in env_values:
        try:
            steps_per_sec = benchmark_num_envs(num_envs, args.timesteps)
            results[num_envs] = steps_per_sec
        except Exception as e:
            print(f"Error with num_envs={num_envs}: {e}")
            results[num_envs] = 0

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"{'num_envs':>10} | {'steps/sec':>12} | {'relative':>10}")
    print("-"*40)

    baseline = results.get(1, 1)
    best_envs = max(results, key=results.get)

    for num_envs in sorted(results.keys()):
        sps = results[num_envs]
        relative = sps / baseline if baseline > 0 else 0
        marker = " <-- BEST" if num_envs == best_envs else ""
        print(f"{num_envs:>10} | {sps:>12.1f} | {relative:>9.2f}x{marker}")

    print("="*60)
    print(f"\nOptimal: --num-envs {best_envs} ({results[best_envs]:.1f} steps/sec)")
    print(f"Speedup vs single env: {results[best_envs]/baseline:.2f}x")


if __name__ == "__main__":
    main()
