#!/bin/bash
# Debug script for visual CLI mode
# Runs test and splits Python/Swift logs to separate files

# Clean up old logs
rm -f /tmp/python_visual.log /tmp/swift_visual.log

# Run test in background, redirecting stderr to Python log
source venv/bin/activate
python test_env.py --visual 2> /tmp/python_visual.log &
PID=$!

# Wait a moment for startup
sleep 0.5

# Open two new terminal windows to tail logs
echo "Starting log viewers..."
echo "  Python logs: /tmp/python_visual.log"
echo "  Swift logs: /tmp/swift_visual.log"
echo ""
echo "Watch logs with:"
echo "  tail -f /tmp/python_visual.log"
echo "  tail -f /tmp/swift_visual.log"
echo ""
echo "Python process PID: $PID"
echo "Press Ctrl+C to stop"

# Wait for the process
wait $PID
