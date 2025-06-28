#\!/bin/bash

echo "🔍 Testing SIMD Performance Impact on Woolly"
echo "==========================================="
echo

# Kill any existing server
pkill -f woolly-server 2>/dev/null
sleep 2

# Test 1: With SIMD enabled (default)
echo "Test 1: SIMD Enabled (default)"
echo "------------------------------"
RUST_LOG=info target/release/woolly-server > simd_enabled_server.log 2>&1 &
SERVER_PID=$\!
sleep 5

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server started with SIMD enabled"
    
    # Load model
    echo "Loading model..."
    curl -s -X POST http://localhost:8080/api/v1/models/load \
      -H "Content-Type: application/json" \
      -d '{"model_path": "./models/granite-3.3-8b-instruct-Q4_K_M.gguf"}' &
    LOAD_PID=$\!
    
    # Wait for model to load (with timeout)
    echo -n "Waiting for model to load"
    for i in {1..120}; do
        if \! ps -p $LOAD_PID > /dev/null 2>&1; then
            break
        fi
        echo -n "."
        sleep 1
    done
    echo
    
    # Test inference
    echo "Testing inference..."
    START=$(date +%s.%N)
    response=$(curl -s -X POST http://localhost:8080/api/v1/inference/complete \
      -H "Content-Type: application/json" \
      -d '{
        "prompt": "Hello",
        "max_tokens": 1,
        "temperature": 0.1
      }')
    END=$(date +%s.%N)
    TIME_SIMD=$(echo "$END - $START" | bc -l)
    
    if echo "$response" | jq -e '.choices[0].text' > /dev/null 2>&1; then
        echo "✅ Inference successful"
        echo "Time with SIMD: ${TIME_SIMD}s"
    else
        echo "❌ Inference failed: $(echo "$response" | jq -r '.error.message // "Unknown"' 2>/dev/null || echo "$response")"
    fi
else
    echo "❌ Server failed to start with SIMD"
fi

# Kill server
kill $SERVER_PID 2>/dev/null
sleep 2

echo
echo "Test 2: SIMD Disabled"
echo "--------------------"
WOOLLY_DISABLE_SIMD=1 RUST_LOG=info target/release/woolly-server > simd_disabled_server.log 2>&1 &
SERVER_PID=$\!
sleep 5

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server started with SIMD disabled"
    
    # Load model
    echo "Loading model..."
    curl -s -X POST http://localhost:8080/api/v1/models/load \
      -H "Content-Type: application/json" \
      -d '{"model_path": "./models/granite-3.3-8b-instruct-Q4_K_M.gguf"}' &
    LOAD_PID=$\!
    
    # Wait for model to load (with timeout)
    echo -n "Waiting for model to load"
    for i in {1..120}; do
        if \! ps -p $LOAD_PID > /dev/null 2>&1; then
            break
        fi
        echo -n "."
        sleep 1
    done
    echo
    
    # Test inference
    echo "Testing inference..."
    START=$(date +%s.%N)
    response=$(curl -s -X POST http://localhost:8080/api/v1/inference/complete \
      -H "Content-Type: application/json" \
      -d '{
        "prompt": "Hello",
        "max_tokens": 1,
        "temperature": 0.1
      }')
    END=$(date +%s.%N)
    TIME_NO_SIMD=$(echo "$END - $START" | bc -l)
    
    if echo "$response" | jq -e '.choices[0].text' > /dev/null 2>&1; then
        echo "✅ Inference successful"
        echo "Time without SIMD: ${TIME_NO_SIMD}s"
    else
        echo "❌ Inference failed: $(echo "$response" | jq -r '.error.message // "Unknown"' 2>/dev/null || echo "$response")"
    fi
else
    echo "❌ Server failed to start without SIMD"
fi

# Kill server
kill $SERVER_PID 2>/dev/null

echo
echo "📊 Summary"
echo "=========="
echo "SIMD Enabled:  ${TIME_SIMD:-Failed}s"
echo "SIMD Disabled: ${TIME_NO_SIMD:-Failed}s"

if [[ -n "$TIME_SIMD" && -n "$TIME_NO_SIMD" ]]; then
    SPEEDUP=$(echo "scale=2; $TIME_SIMD / $TIME_NO_SIMD" | bc -l)
    echo "Speedup factor: ${SPEEDUP}x"
    
    if (( $(echo "$TIME_NO_SIMD < $TIME_SIMD" | bc -l) )); then
        echo "⚠️  SIMD is making performance WORSE\!"
        SLOWDOWN=$(echo "scale=2; $TIME_SIMD / $TIME_NO_SIMD" | bc -l)
        echo "SIMD is ${SLOWDOWN}x SLOWER than without SIMD"
    else
        echo "✅ SIMD is providing speedup"
    fi
fi

echo
echo "Check log files for details:"
echo "- simd_enabled_server.log"
echo "- simd_disabled_server.log"
