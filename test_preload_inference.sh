#!/bin/bash
# Test inference performance with preloaded weights

echo "🚀 WOOLLY WEIGHT PRELOADING INFERENCE TEST"
echo "=========================================="

# Kill any existing servers
pkill -f woolly-server
sleep 2

# Start server with SIMD disabled
echo "Starting server with SIMD disabled and weight preloading..."
WOOLLY_DISABLE_SIMD=1 RUST_LOG=info ./target/release/woolly-server &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 10

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start!"
    exit 1
fi

# Load model
echo ""
echo "Loading model (this will preload all weights)..."
LOAD_START=$(date +%s)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./models/granite-3.3-8b-instruct-Q4_K_M.gguf"}'

LOAD_END=$(date +%s)
LOAD_TIME=$((LOAD_END - LOAD_START))
echo ""
echo "Model loaded in $LOAD_TIME seconds (includes weight preloading)"

# Wait a bit more
sleep 5

echo ""
echo "=== RUNNING INFERENCE TESTS ==="
echo ""

# First inference
echo "Test 1: First inference after preload"
INFERENCE_START=$(date +%s.%N)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 1, "temperature": 0.0, "stream": false}'

INFERENCE_END=$(date +%s.%N)
FIRST_TIME=$(echo "$INFERENCE_END - $INFERENCE_START" | bc)
echo ""
echo "First inference time: $FIRST_TIME seconds"

# Second inference
echo ""
echo "Test 2: Second inference (should be similar speed)"
INFERENCE_START=$(date +%s.%N)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "World", "max_length": 1, "temperature": 0.0, "stream": false}'

INFERENCE_END=$(date +%s.%N)
SECOND_TIME=$(echo "$INFERENCE_END - $INFERENCE_START" | bc)
echo ""
echo "Second inference time: $SECOND_TIME seconds"

# Third inference
echo ""
echo "Test 3: Third inference"
INFERENCE_START=$(date +%s.%N)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_length": 1, "temperature": 0.0, "stream": false}'

INFERENCE_END=$(date +%s.%N)
THIRD_TIME=$(echo "$INFERENCE_END - $INFERENCE_START" | bc)
echo ""
echo "Third inference time: $THIRD_TIME seconds"

echo ""
echo "=== SUMMARY ==="
echo "Model load time (with preloading): $LOAD_TIME seconds"
echo "First inference: $FIRST_TIME seconds"
echo "Second inference: $SECOND_TIME seconds"
echo "Third inference: $THIRD_TIME seconds"
echo ""
echo "Expected: All inference times should be similar (no first-token penalty)"
echo "Previous results without preloading: First ~194s, subsequent ~179s"

# Clean up
kill $SERVER_PID
echo ""
echo "Test complete!"