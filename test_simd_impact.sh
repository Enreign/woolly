#\!/bin/bash

echo "🔍 Testing SIMD Performance Impact"
echo "=================================="
echo

# Function to test inference
test_inference() {
    local name=$1
    local env_vars=$2
    
    echo "Test: $name"
    echo "--------------"
    
    # Kill any existing server
    pkill -f woolly-server 2>/dev/null
    sleep 2
    
    # Start server with environment variables
    eval "$env_vars ./target/release/woolly-server > ${name}_server.log 2>&1 &"
    SERVER_PID=$\!
    sleep 5
    
    # Check if server started
    if \! ps -p $SERVER_PID > /dev/null; then
        echo "❌ Server failed to start"
        return
    fi
    
    # Check SIMD status from logs
    if grep -q "WOOLLY INIT" ${name}_server.log; then
        grep "WOOLLY INIT" ${name}_server.log | head -1
    fi
    
    # Load model (if not already loaded)
    echo "Loading model..."
    curl -s -X POST http://localhost:8080/api/v1/models/load \
        -H "Content-Type: application/json" \
        -d '{"model_path": "./models/granite-3.3-8b-instruct-Q4_K_M.gguf"}' \
        > /dev/null 2>&1 &
    
    # Wait for model to load
    echo -n "Waiting for model"
    for i in {1..90}; do
        sleep 1
        echo -n "."
        # Check if model is loaded
        if curl -s http://localhost:8080/api/v1/models | jq -e '.[0].loaded' > /dev/null 2>&1; then
            echo " ✅ Model loaded"
            break
        fi
    done
    echo
    
    # Test inference
    echo "Testing inference..."
    START=$(date +%s.%N)
    
    RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/inference/complete \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Hello",
            "max_tokens": 1,
            "temperature": 0.1
        }' 2>&1)
    
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc -l)
    
    # Check response
    if echo "$RESPONSE" | jq -e '.choices[0].text' > /dev/null 2>&1; then
        echo "✅ Inference successful"
        echo "Time: ${ELAPSED}s"
        TPS=$(echo "scale=4; 1 / $ELAPSED" | bc -l)
        echo "Performance: ${TPS} tokens/sec"
        echo "$ELAPSED" > ${name}_time.txt
    else
        echo "❌ Inference failed"
        echo "Response: $RESPONSE" | head -100
    fi
    
    # Kill server
    kill $SERVER_PID 2>/dev/null
    echo
}

# Test 1: SIMD Enabled (default)
test_inference "simd_enabled" ""

# Test 2: SIMD Disabled
test_inference "simd_disabled" "WOOLLY_DISABLE_SIMD=1"

# Summary
echo "📊 Summary"
echo "=========="

if [ -f simd_enabled_time.txt ] && [ -f simd_disabled_time.txt ]; then
    SIMD_TIME=$(cat simd_enabled_time.txt)
    NO_SIMD_TIME=$(cat simd_disabled_time.txt)
    
    echo "SIMD Enabled:  ${SIMD_TIME}s"
    echo "SIMD Disabled: ${NO_SIMD_TIME}s"
    echo
    
    # Calculate speedup
    if (( $(echo "$NO_SIMD_TIME < $SIMD_TIME" | bc -l) )); then
        SPEEDUP=$(echo "scale=2; $SIMD_TIME / $NO_SIMD_TIME" | bc -l)
        echo "⚠️  SIMD is making performance WORSE\!"
        echo "Disabling SIMD gives ${SPEEDUP}x speedup"
        echo
        echo "Recommendation: Set WOOLLY_DISABLE_SIMD=1 for better performance"
    else
        SPEEDUP=$(echo "scale=2; $NO_SIMD_TIME / $SIMD_TIME" | bc -l)
        echo "✅ SIMD provides ${SPEEDUP}x speedup"
    fi
else
    echo "Tests incomplete - check logs for errors"
fi
