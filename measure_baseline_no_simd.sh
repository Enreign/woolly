#!/bin/bash

echo "================================================================"
echo "Baseline Performance Measurement Without SIMD"
echo "================================================================"

echo ""
echo "NOTE: You need to rebuild woolly-server with the updated code:"
echo "  cargo build --release --features server"
echo ""

# Set environment variable to disable SIMD
export WOOLLY_DISABLE_SIMD=1

echo "WOOLLY_DISABLE_SIMD=$WOOLLY_DISABLE_SIMD"
echo ""

# Kill any existing server
pkill -f "woolly-server" || true
sleep 1

# Start server
echo "Starting server without SIMD..."
./target/release/woolly-server --bind 127.0.0.1:8080 2>&1 | tee woolly_baseline_no_simd.log &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo ""

# Wait a bit to see initialization messages
sleep 5

# Check if we see SIMD disabled message
echo "Checking for SIMD status messages..."
grep -i "WOOLLY.*SIMD\|SIMD.*enabled" woolly_baseline_no_simd.log | head -10

# Try an inference request to trigger matrix operations
echo ""
echo "Triggering inference to test matrix operations..."

curl -X POST http://localhost:8080/api/v1/inference/complete \
    -H "Content-Type: application/json" \
    -d '{
        "model": "granite-3.3-8b-instruct-Q4_K_M",
        "prompt": "Test",
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": false
    }' \
    -s > /dev/null 2>&1

sleep 2

# Check logs again
echo ""
echo "Checking logs after inference attempt..."
grep -i "WOOLLY.*SIMD\|SIMD.*DISABLED\|matrix multiplication" woolly_baseline_no_simd.log | tail -20

# Clean up
echo ""
echo "Cleaning up..."
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "================================================================"
echo "Check woolly_baseline_no_simd.log for full output"
echo "================================================================"
echo ""
echo "Key findings:"
if grep -q "SIMD DISABLED" woolly_baseline_no_simd.log; then
    echo "✓ SIMD was successfully disabled"
    echo "✓ Non-SIMD matrix multiplication path was used"
else
    echo "✗ No SIMD disable messages found"
    echo "  This might mean:"
    echo "  - The server needs to be rebuilt with the updated code"
    echo "  - No matrix operations were triggered"
fi
echo ""