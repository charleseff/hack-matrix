"""
Memory profiling test for HackMatrix environment.
Runs multiple episodes and tracks memory usage to identify leaks.

Last run showed major memory increases:

⏺ Bash(source venv/bin/activate && python scripts/memory_test.py --episodes 500 --steps 100) timeout: 5m 0s
  ⎿  Error: Exit code 1
     Starting memory test: 500 episodes, 100 max steps each
     ================================================================================
     Initial Python memory: 411.6 KB
     Initial Swift memory:  1.5 MB (PID: 68676)

     Episode  10: Python=  421.8 KB (✓) | Swift=  18.4 MB (📈) | Swift Growth=+16.9 MB
     Episode  20: Python=  422.3 KB (✓) | Swift=  33.4 MB (📈) | Swift Growth=+31.8 MB
     Episode  30: Python=  423.1 KB (✓) | Swift=  49.2 MB (📈) | Swift Growth=+47.7 MB
     Episode  40: Python=  423.2 KB (✓) | Swift=  55.8 MB (📈) | Swift Growth=+54.3 MB
     Episode  50: Python=  420.0 KB (✓) | Swift=  68.2 MB (📈) | Swift Growth=+66.7 MB
     Episode  60: Python=  423.5 KB (✓) | Swift=  83.2 MB (📈) | Swift Growth=+81.6 MB
     Episode  70: Python=  423.8 KB (✓) | Swift=  95.3 MB (📈) | Swift Growth=+93.7 MB
     Episode  80: Python=  424.0 KB (✓) | Swift= 106.5 MB (📈) | Swift Growth=+104.9 MB
     Episode  90: Python=  421.2 KB (✓) | Swift= 120.5 MB (📈) | Swift Growth=+118.9 MB
     Episode 100: Python=  424.6 KB (✓) | Swift= 129.6 MB (📈) | Swift Growth=+128.1 MB
     Episode 110: Python=  424.9 KB (✓) | Swift= 142.0 MB (📈) | Swift Growth=+140.5 MB
     Episode 120: Python=  425.0 KB (✓) | Swift= 153.1 MB (📈) | Swift Growth=+151.5 MB
     Episode 130: Python=  425.3 KB (✓) | Swift= 162.4 MB (📈) | Swift Growth=+160.9 MB
     Episode 140: Python=  425.4 KB (✓) | Swift= 178.1 MB (📈) | Swift Growth=+176.6 MB
     Episode 150: Python=  426.3 KB (✓) | Swift= 188.4 MB (📈) | Swift Growth=+186.8 MB
     Episode 160: Python=  425.9 KB (✓) | Swift= 200.9 MB (📈) | Swift Growth=+199.4 MB
     Episode 170: Python=  426.8 KB (✓) | Swift= 211.0 MB (📈) | Swift Growth=+209.5 MB
     Episode 180: Python=  426.8 KB (✓) | Swift= 224.6 MB (📈) | Swift Growth=+223.0 MB
     Episode 190: Python=  423.8 KB (✓) | Swift= 238.1 MB (📈) | Swift Growth=+236.5 MB
     Episode 200: Python=  427.1 KB (✓) | Swift= 248.0 MB (📈) | Swift Growth=+246.5 MB
     Episode 210: Python=  424.3 KB (✓) | Swift= 261.0 MB (📈) | Swift Growth=+259.4 MB
     Episode 220: Python=  427.8 KB (✓) | Swift= 271.6 MB (📈) | Swift Growth=+270.0 MB
     Episode 230: Python=  428.1 KB (✓) | Swift= 284.8 MB (📈) | Swift Growth=+283.2 MB
     Episode 240: Python=  428.6 KB (✓) | Swift= 299.0 MB (📈) | Swift Growth=+297.5 MB
     Episode 250: Python=  428.7 KB (✓) | Swift= 310.1 MB (📈) | Swift Growth=+308.6 MB
     Episode 260: Python=  429.4 KB (✓) | Swift= 323.1 MB (📈) | Swift Growth=+321.5 MB
     Episode 270: Python=  429.3 KB (✓) | Swift= 332.9 MB (📈) | Swift Growth=+331.3 MB
     Episode 280: Python=  429.5 KB (✓) | Swift= 347.0 MB (📈) | Swift Growth=+345.5 MB
     Episode 290: Python=  430.1 KB (✓) | Swift= 360.9 MB (📈) | Swift Growth=+359.4 MB
     Episode 300: Python=  430.1 KB (✓) | Swift= 372.0 MB (📈) | Swift Growth=+370.5 MB
     Episode 310: Python=  430.2 KB (✓) | Swift= 384.3 MB (📈) | Swift Growth=+382.8 MB
     Episode 320: Python=  430.9 KB (✓) | Swift= 393.8 MB (📈) | Swift Growth=+392.2 MB
     Episode 330: Python=  430.9 KB (✓) | Swift= 406.4 MB (📈) | Swift Growth=+404.8 MB
     Episode 340: Python=  431.6 KB (✓) | Swift= 418.9 MB (📈) | Swift Growth=+417.3 MB
     Episode 350: Python=  432.0 KB (✓) | Swift= 429.9 MB (📈) | Swift Growth=+428.3 MB
     Episode 360: Python=  428.6 KB (✓) | Swift= 440.8 MB (📈) | Swift Growth=+439.2 MB
     Episode 370: Python=  431.9 KB (✓) | Swift= 454.9 MB (📈) | Swift Growth=+453.3 MB
     Episode 380: Python=  432.6 KB (✓) | Swift= 465.4 MB (📈) | Swift Growth=+463.8 MB
     Episode 390: Python=  429.3 KB (✓) | Swift= 476.6 MB (📈) | Swift Growth=+475.0 MB
     Episode 400: Python=  433.0 KB (✓) | Swift= 488.8 MB (📈) | Swift Growth=+487.2 MB
     Episode 410: Python=  433.2 KB (✓) | Swift= 501.8 MB (📈) | Swift Growth=+500.3 MB
     Episode 420: Python=  433.3 KB (✓) | Swift= 512.3 MB (📈) | Swift Growth=+510.7 MB
     Episode 430: Python=  433.6 KB (✓) | Swift= 521.4 MB (📈) | Swift Growth=+519.9 MB
     Episode 440: Python=  433.7 KB (✓) | Swift= 534.1 MB (📈) | Swift Growth=+532.5 MB
     Episode 450: Python=  434.4 KB (✓) | Swift= 547.5 MB (📈) | Swift Growth=+545.9 MB
     Episode 460: Python=  431.3 KB (✓) | Swift= 560.4 MB (📈) | Swift Growth=+558.8 MB
     Episode 470: Python=  435.3 KB (✓) | Swift= 573.2 MB (📈) | Swift Growth=+571.7 MB
     Episode 480: Python=  435.3 KB (✓) | Swift= 581.0 MB (📈) | Swift Growth=+579.5 MB
     Episode 490: Python=  435.6 KB (✓) | Swift= 593.2 MB (📈) | Swift Growth=+591.6 MB
     Episode 500: Python=  435.7 KB (✓) | Swift= 601.8 MB (📈) | Swift Growth=+600.2 MB

     ================================================================================
     FINAL RESULTS
     ================================================================================
     Python - Initial: 411.6 KB, Final: 48.3 KB, Peak: 749.3 KB
     Swift  - Initial: 1.5 MB, Final: 601.8 MB, Peak: 601.8 MB

     Swift memory trend: +550.2 MB (early avg: 31.7 MB, late avg: 581.9 MB)

     ⚠️  SWIFT MEMORY LEAK DETECTED: 550.2 MB growth over 500 episodes

"""

import os
import sys
import gc
import tracemalloc
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hackmatrix import HackEnv


def format_bytes(size):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def get_swift_memory_mb(pid):
    """Get memory usage of Swift process in MB."""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(pid)],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            # rss is in KB
            return int(result.stdout.strip()) / 1024
    except Exception:
        pass
    return 0


def run_memory_test(num_episodes: int = 100, steps_per_episode: int = 100):
    """
    Run episodes and track memory usage.

    Args:
        num_episodes: Number of episodes to run
        steps_per_episode: Max steps per episode before forced reset
    """
    print(f"Starting memory test: {num_episodes} episodes, {steps_per_episode} max steps each")
    print("=" * 80)

    # Start memory tracking
    tracemalloc.start()

    env = HackEnv()
    swift_pid = env.process.pid if env.process else None

    initial_snapshot = tracemalloc.take_snapshot()
    initial_current, initial_peak = tracemalloc.get_traced_memory()
    initial_swift_mb = get_swift_memory_mb(swift_pid) if swift_pid else 0

    print(f"Initial Python memory: {format_bytes(initial_current)}")
    print(f"Initial Swift memory:  {initial_swift_mb:.1f} MB (PID: {swift_pid})")
    print()

    swift_memory_history = [initial_swift_mb]

    for episode in range(num_episodes):
        obs, info = env.reset()

        for step in range(steps_per_episode):
            # Random action from valid actions
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = valid_actions[step % len(valid_actions)]

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Force garbage collection after each episode
        gc.collect()

        # Report memory every 10 episodes
        if (episode + 1) % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            py_growth = current - initial_current

            swift_mb = get_swift_memory_mb(swift_pid) if swift_pid else 0
            swift_growth = swift_mb - initial_swift_mb
            swift_memory_history.append(swift_mb)

            py_status = '📈' if py_growth > 1024*1024 else '✓'
            swift_status = '📈' if swift_growth > 10 else '✓'

            print(f"Episode {episode + 1:3d}: "
                  f"Python={format_bytes(current):>10s} ({py_status}) | "
                  f"Swift={swift_mb:>6.1f} MB ({swift_status}) | "
                  f"Swift Growth={swift_growth:+.1f} MB")

    env.close()

    # Final snapshot and comparison
    final_snapshot = tracemalloc.take_snapshot()
    final_current, final_peak = tracemalloc.get_traced_memory()

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Python - Initial: {format_bytes(initial_current)}, Final: {format_bytes(final_current)}, Peak: {format_bytes(final_peak)}")
    print(f"Swift  - Initial: {initial_swift_mb:.1f} MB, Final: {swift_memory_history[-1]:.1f} MB, Peak: {max(swift_memory_history):.1f} MB")
    print()

    # Analyze Swift memory trend
    if len(swift_memory_history) > 5:
        early_avg = sum(swift_memory_history[:5]) / 5
        late_avg = sum(swift_memory_history[-5:]) / 5
        trend = late_avg - early_avg
        print(f"Swift memory trend: {trend:+.1f} MB (early avg: {early_avg:.1f} MB, late avg: {late_avg:.1f} MB)")

        if trend > 10:
            print(f"\n⚠️  SWIFT MEMORY LEAK DETECTED: {trend:.1f} MB growth over {num_episodes} episodes")
            return False

    # Show top Python memory consumers
    print("\nTop 5 Python memory allocations:")
    top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
    for i, stat in enumerate(top_stats[:5], 1):
        print(f"  {i}. {stat}")

    tracemalloc.stop()

    print(f"\n✓ Memory usage appears stable")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory profiling test for HackMatrix")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Max steps per episode (default: 100)")

    args = parser.parse_args()

    success = run_memory_test(args.episodes, args.steps)
    sys.exit(0 if success else 1)
