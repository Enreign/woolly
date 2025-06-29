#!/bin/bash
# Test BLAS-optimized attention performance

echo "🚀 TESTING BLAS-OPTIMIZED ATTENTION PERFORMANCE"
echo "=============================================="

# Kill any existing servers
pkill -f woolly-server
sleep 2

# Start server with BLAS optimization and SIMD disabled
echo "Starting server with BLAS optimization and SIMD disabled..."
WOOLLY_DISABLE_SIMD=1 RUST_LOG=info ./target/release/woolly-server > blas_attention_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 10

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start!"
    exit 1
fi

echo ""
echo "Loading model (with weight preloading + BLAS attention)..."
LOAD_START=$(date +%s)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./models/granite-3.3-8b-instruct-Q4_K_M.gguf"}' \
  > blas_load_response.json 2>/dev/null

LOAD_END=$(date +%s)
LOAD_TIME=$((LOAD_END - LOAD_START))
echo ""
echo "Model loaded in $LOAD_TIME seconds"

# Wait a bit more
sleep 5

echo ""
echo "=== RUNNING BLAS ATTENTION PERFORMANCE TESTS ==="
echo ""

# First inference (should be fast with preloaded weights + BLAS attention)
echo "Test 1: First inference with BLAS attention"
INFERENCE_START=$(date +%s.%N)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 5, "temperature": 0.0, "stream": false}' \
  > blas_inference_response.json 2>/dev/null

INFERENCE_END=$(date +%s.%N)
FIRST_TIME=$(echo "$INFERENCE_END - $INFERENCE_START" | bc)
echo ""
echo "First inference time: $FIRST_TIME seconds"

# Second inference
echo ""
echo "Test 2: Second inference with BLAS attention"
INFERENCE_START=$(date +%s.%N)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "World", "max_length": 5, "temperature": 0.0, "stream": false}' \
  > blas_inference_response2.json 2>/dev/null

INFERENCE_END=$(date +%s.%N)
SECOND_TIME=$(echo "$INFERENCE_END - $INFERENCE_START" | bc)
echo ""
echo "Second inference time: $SECOND_TIME seconds"

# Third inference (longer sequence to test attention scaling)
echo ""
echo "Test 3: Longer sequence inference with BLAS attention"
INFERENCE_START=$(date +%s.%N)

curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The quick brown fox jumps", "max_length": 10, "temperature": 0.0, "stream": false}' \
  > blas_inference_response3.json 2>/dev/null

INFERENCE_END=$(date +%s.%N)
THIRD_TIME=$(echo "$INFERENCE_END - $INFERENCE_START" | bc)
echo ""
echo "Third inference time: $THIRD_TIME seconds"

echo ""
echo "=== PERFORMANCE SUMMARY ==="
echo "Model load time (with preloading): $LOAD_TIME seconds"
echo "First inference (5 tokens): $FIRST_TIME seconds"
echo "Second inference (5 tokens): $SECOND_TIME seconds"
echo "Third inference (10 tokens): $THIRD_TIME seconds"
echo ""

# Calculate tokens per second for each test
if command -v bc &> /dev/null; then
    FIRST_TOKENS_PER_SEC=$(echo "scale=2; 5 / $FIRST_TIME" | bc)
    SECOND_TOKENS_PER_SEC=$(echo "scale=2; 5 / $SECOND_TIME" | bc)
    THIRD_TOKENS_PER_SEC=$(echo "scale=2; 10 / $THIRD_TIME" | bc)
    
    echo "Performance metrics:"
    echo "  First test: $FIRST_TOKENS_PER_SEC tokens/sec"
    echo "  Second test: $SECOND_TOKENS_PER_SEC tokens/sec"
    echo "  Third test: $THIRD_TOKENS_PER_SEC tokens/sec"
    echo ""
    
    # Compare with target
    echo "Target: 15+ tokens/sec"
    echo "Previous baseline (with preloading only): ~67 tokens/sec"
    echo "Expected improvement: 10-20x faster attention (should reach 150-1000+ tokens/sec)"
fi

echo ""
echo "=== SERVER LOGS (BLAS ATTENTION RELATED) ==="
grep -E "(BLAS|blas|acceleration|projection|attention)" blas_attention_server.log | tail -30

echo ""
echo "=== RESPONSE SAMPLES ==="
echo "First response:"
cat blas_inference_response.json | jq -r '.choices[0].text' 2>/dev/null || cat blas_inference_response.json
echo ""
echo "Second response:"
cat blas_inference_response2.json | jq -r '.choices[0].text' 2>/dev/null || cat blas_inference_response2.json

# Clean up
kill $SERVER_PID
echo ""
echo "BLAS attention test complete!"