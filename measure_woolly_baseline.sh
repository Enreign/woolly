#!/bin/bash

echo "🔍 Measuring Woolly Baseline Performance (SIMD Disabled)"
echo "========================================================"
echo

# Test single token
echo "Test 1: Single token generation..."
START=$(date +%s.%N)

response=$(curl -s -X POST http://localhost:8080/api/v1/inference/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 1,
    "temperature": 0.1
  }')

END=$(date +%s.%N)
TIME=$(echo "$END - $START" | bc -l)

if echo "$response" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    text=$(echo "$response" | jq -r '.choices[0].text')
    tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 1')
    tps=$(echo "scale=2; $tokens / $TIME" | bc -l)
    
    echo "✅ Success!"
    echo "Generated: '$text'"
    echo "Time: ${TIME}s"
    echo "Tokens: $tokens"
    echo "Performance: ${tps} tokens/sec"
else
    echo "❌ Error: $(echo "$response" | jq -r '.error.message // "Unknown"' | head -1)"
fi

echo
echo "Test 2: Multi-token generation (5 tokens)..."
START=$(date +%s.%N)

response=$(curl -s -X POST http://localhost:8080/api/v1/inference/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 5,
    "temperature": 0.1
  }')

END=$(date +%s.%N)
TIME=$(echo "$END - $START" | bc -l)

if echo "$response" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    text=$(echo "$response" | jq -r '.choices[0].text')
    tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 5')
    tps=$(echo "scale=2; $tokens / $TIME" | bc -l)
    
    echo "✅ Success!"
    echo "Generated: '$text'"
    echo "Time: ${TIME}s"
    echo "Tokens: $tokens"
    echo "Performance: ${tps} tokens/sec"
else
    echo "❌ Error: $(echo "$response" | jq -r '.error.message // "Unknown"' | head -1)"
fi

echo
echo "📊 Summary:"
echo "- SIMD: Disabled"
echo "- Model: Granite 3.3B-8B Q4_K_M"
echo "- Optimizations: Memory Pool + Lazy Loading + GQA"
echo
echo "📈 Comparison:"
echo "- Ollama (llama.cpp): 0.6-1.2 tokens/sec"
echo "- Woolly Target: 5-10x improvement (3-12 tokens/sec)"