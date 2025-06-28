#!/bin/bash

echo "Starting Woolly server with cache disabled..."

# Set up environment
export PATH="$HOME/.cargo/bin:$PATH"
export RUST_LOG=woolly_server=debug,woolly_core=debug,woolly_gguf=debug
export RUST_BACKTRACE=full

# Kill any existing server
pkill -f woolly-server 2>/dev/null || true
sleep 2

# Start server
echo "Starting server..."
./target/release/woolly-server 2>&1 | tee server_baseline_debug.log