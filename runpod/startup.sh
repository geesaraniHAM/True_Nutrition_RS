#!/bin/bash

# --- Start Redis server ---
echo "Starting Redis server..."
# Use --daemonize yes to run Redis in the background
# --port 6379 is the default, explicitly set for clarity
redis-server --daemonize yes --port 6379 --loglevel warning

# Give Redis a moment to initialize
sleep 2

# Verify Redis is running
if pgrep redis-server > /dev/null
then
    echo "Redis server started successfully on port 6379."
else
    echo "ERROR: Failed to start Redis server. Exiting."
    exit 1
fi

# --- Start your Python application ---
echo "Starting Python handler.py..."
# Execute your main application script
# The -u flag is for unbuffered output
python -u ./src/handler1.py