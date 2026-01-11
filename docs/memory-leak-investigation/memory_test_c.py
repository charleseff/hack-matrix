"""
Memory test using pure C program to isolate Swift from subprocess.
"""

import os
import sys
import subprocess
import json
import gc

def format_bytes(size):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def get_memory_mb(pid):
    """Get memory usage of process in MB."""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(pid)],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip()) / 1024
    except Exception:
        pass
    return 0

def run_test(num_episodes=100, steps_per_episode=100):
    """Run memory test with C program."""
    print(f"Testing C program: {num_episodes} episodes, {steps_per_episode} steps each")
    print("=" * 80)

    # Path to test program (C or Swift)
    # c_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_memory_c")
    # c_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_memory_swift")
    # c_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_memory_foundation")
    # c_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_memory_swiftui")
    # c_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_memory_spritekit")
    c_path = os.path.join(os.path.dirname(__file__), "..", "..", ".build", "debug", "HackMatrix")

    # Start program
    cmd = [c_path]
    # Add --headless-cli if running HackMatrix
    if "HackMatrix" in c_path:
        cmd.append("--headless-cli")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    pid = process.pid
    initial_mem = get_memory_mb(pid)
    print(f"Initial C memory: {initial_mem:.1f} MB (PID: {pid})")
    print()

    mem_history = [initial_mem]

    for episode in range(num_episodes):
        # Send reset
        process.stdin.write('{"action":"reset"}\n')
        process.stdin.flush()
        process.stdout.readline()

        for step in range(steps_per_episode):
            # Get valid actions first (like the real env does)
            process.stdin.write('{"action":"getValidActions"}\n')
            process.stdin.flush()
            process.stdout.readline()

            # Send step
            process.stdin.write('{"action":"step","actionIndex":0}\n')
            process.stdin.flush()
            process.stdout.readline()

        gc.collect()

        if (episode + 1) % 10 == 0:
            mem = get_memory_mb(pid)
            growth = mem - initial_mem
            mem_history.append(mem)
            status = '📈' if growth > 1 else '✓'
            print(f"Episode {episode + 1:3d}: C={mem:>6.1f} MB ({status}) | Growth={growth:+.1f} MB")

    process.terminate()
    process.wait()

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"C - Initial: {initial_mem:.1f} MB, Final: {mem_history[-1]:.1f} MB, Peak: {max(mem_history):.1f} MB")

    if len(mem_history) > 5:
        early_avg = sum(mem_history[:5]) / 5
        late_avg = sum(mem_history[-5:]) / 5
        trend = late_avg - early_avg
        print(f"\nC memory trend: {trend:+.1f} MB")

        if trend > 1:
            print(f"\n⚠️  C MEMORY LEAK DETECTED: {trend:.1f} MB growth")
            return False

    print(f"\n✓ C memory stable")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    success = run_test(args.episodes, args.steps)
    sys.exit(0 if success else 1)
