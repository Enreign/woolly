#!/bin/bash

echo "=== Testing BLAS Attention Performance ==="
echo

# Start server
echo "Starting server with debug logging..."
cd crates/woolly-server
RUST_LOG=debug cargo run --release > /tmp/blas_test.log 2>&1 &
SERVER_PID=$!

# Wait for server
echo "Waiting for server to start..."
sleep 10

# List models
echo "Listing available models..."
curl -s http://localhost:8080/api/v1/models | jq .

# Load model
echo
echo "Loading model..."
curl -s -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load | jq .

# Wait for model to load
echo "Waiting for model to load..."
sleep 5

# Make inference request
echo
echo "Making inference request (testing BLAS attention)..."
START_TIME=$(date +%s.%N)
curl -s -X POST http://localhost:8080/api/v1/inference/complete \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.3-8b-instruct-Q4_K_M",
    "prompt": "Hi",
    "max_tokens": 1,
    "temperature": 0,
    "stream": false
  }' > /tmp/inference_result.json 2>&1
END_TIME=$(date +%s.%N)

# Calculate time
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
echo "Inference time: $ELAPSED seconds"
echo "Tokens/sec: $(echo "scale=4; 1 / $ELAPSED" | bc)"

# Show result
echo
echo "Response:"
cat /tmp/inference_result.json | jq .

# Check for BLAS usage
echo
echo "Checking for BLAS usage in logs..."
grep -E "(BLAS|🚀|Using BLAS|GQA attention|Accelerate)" /tmp/blas_test.log | tail -30

# Kill server
echo
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null

echo
echo "=== Test Complete ===" 