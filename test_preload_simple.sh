#!/bin/bash
# Simple test for weight preloading performance

echo "🚀 WOOLLY WEIGHT PRELOADING PERFORMANCE TEST"
echo "============================================"

# Kill any existing servers
pkill -f woolly-server
sleep 2

# Start server with SIMD disabled
echo "Starting server with SIMD disabled and weight preloading enabled..."
WOOLLY_DISABLE_SIMD=1 RUST_LOG=info ./target/release/woolly-server > preload_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 10

# Load model
echo "Loading model..."
curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./models/granite-3.3-8b-instruct-Q4_K_M.gguf"}' \
  > load_response.json 2>/dev/null

echo "Model load response:"
cat load_response.json | jq .

# Give model time to fully load and preload weights
echo "Waiting for weight preloading to complete..."
sleep 10

# Run inference test
echo ""
echo "Running inference test..."
START_TIME=$(date +%s)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 3, "temperature": 0.0, "stream": false}' \
  > inference_response.json 2>/dev/null

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Inference completed in $DURATION seconds"
echo "Response:"
cat inference_response.json | jq .

# Check server logs
echo ""
echo "=== SERVER LOGS (preloading related) ==="
grep -E "(Preload|preload|dequantiz|cache|weight)" preload_server.log | tail -50

# Clean up
kill $SERVER_PID
echo ""
echo "Test complete!"