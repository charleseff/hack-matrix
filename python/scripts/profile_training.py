#!/usr/bin/env python3
"""
Profile training performance to identify bottlenecks.
Measures environment step time vs NN training time.
"""

import time
import numpy as np
from hackmatrix import HackEnv

def profile_environment(num_steps=1000):
    """Profile environment performance."""
    print("="*70)
    print("PROFILING ENVIRONMENT")
    print("="*70)

    env = HackEnv()
    env.reset()

    step_times = []
    valid_steps = 0

    for i in range(num_steps):
        # Get valid actions
        valid_actions = env.get_valid_actions()
        action = np.random.choice(valid_actions)

        # Time the step
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - start
        step_times.append(step_time)
        valid_steps += 1

        if terminated or truncated:
            env.reset()

    env.close()

    # Calculate statistics
    step_times = np.array(step_times)
    avg_step_time = np.mean(step_times) * 1000  # Convert to ms
    std_step_time = np.std(step_times) * 1000
    min_step_time = np.min(step_times) * 1000
    max_step_time = np.max(step_times) * 1000
    steps_per_sec = 1.0 / np.mean(step_times)

    print(f"\nEnvironment steps: {valid_steps}")
    print(f"Average step time: {avg_step_time:.2f} ms (±{std_step_time:.2f} ms)")
    print(f"Min step time: {min_step_time:.2f} ms")
    print(f"Max step time: {max_step_time:.2f} ms")
    print(f"Steps per second: {steps_per_sec:.1f}")
    print(f"\nProjected training time for 1M steps: {(1_000_000 / steps_per_sec / 3600):.1f} hours")

    return {
        'avg_step_time_ms': avg_step_time,
        'steps_per_sec': steps_per_sec
    }


def profile_training_loop(num_steps=10000):
    """Profile full training loop with NN updates."""
    print("\n" + "="*70)
    print("PROFILING TRAINING LOOP (with NN updates)")
    print("="*70)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    def mask_fn(env):
        return env._get_action_mask()

    def make_env():
        env = HackEnv()
        env = ActionMasker(env, mask_fn)
        return env

    env = DummyVecEnv([make_env])

    print("Creating MaskablePPO agent...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        n_steps=2048,
        batch_size=64,
    )

    print(f"Training for {num_steps} steps...\n")
    start = time.time()
    model.learn(total_timesteps=num_steps, progress_bar=True)
    total_time = time.time() - start

    steps_per_sec = num_steps / total_time

    print(f"\nTotal time: {total_time:.1f} seconds")
    print(f"Steps per second: {steps_per_sec:.1f}")
    print(f"Time per step: {(total_time / num_steps * 1000):.2f} ms")
    print(f"\nProjected training time for 1M steps: {(1_000_000 / steps_per_sec / 3600):.1f} hours")

    env.close()

    return {
        'steps_per_sec': steps_per_sec,
        'total_time': total_time
    }


if __name__ == "__main__":
    print("\n🔍 PERFORMANCE PROFILING\n")

    # Profile environment only
    env_stats = profile_environment(num_steps=1000)

    # Profile training with NN updates
    train_stats = profile_training_loop(num_steps=10000)

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    env_sps = env_stats['steps_per_sec']
    train_sps = train_stats['steps_per_sec']
    overhead = (env_sps - train_sps) / env_sps * 100

    print(f"\nEnvironment only: {env_sps:.1f} steps/sec")
    print(f"Training loop:    {train_sps:.1f} steps/sec")
    print(f"NN overhead:      {overhead:.1f}%")

    if overhead < 20:
        print("\n✓ Bottleneck: ENVIRONMENT (CPU-bound)")
        print("  → Swift subprocess and game simulation are the limiting factor")
        print("  → GPU will NOT help significantly")
        print("  → Consider: parallel environments (DummyVecEnv → SubprocVecEnv)")
    else:
        print("\n✓ Bottleneck: NEURAL NETWORK (could benefit from GPU)")
        print("  → GPU might help, but unlikely for this small network")

    print("\nRecommendations:")
    print("  • Use SubprocVecEnv with multiple parallel environments")
    print("  • Run on CPU instance with many cores (not GPU)")
    print("  • Lambda Labs: Choose CPU-optimized instance, not GPU")
