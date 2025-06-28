#!/bin/bash

echo "🔍 Benchmarking Ollama (llama.cpp) Performance"
echo "=============================================="
echo

# Test with the same Granite model
model="granite3.3:8b"

echo "Model: $model"
echo

echo "Test 1: Single short completion..."
START_TIME=$(date +%s.%N)

response=$(curl -s http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$model\", \"prompt\": \"Hello\", \"stream\": false}")

END_TIME=$(date +%s.%N)
TIME_TAKEN=$(echo "$END_TIME - $START_TIME" | bc -l)

generated_text=$(echo "$response" | jq -r '.response // empty')
token_count=$(echo "$generated_text" | wc -w)

echo "Generated: '$generated_text'"
echo "Time: ${TIME_TAKEN}s"
echo "Estimated tokens: $token_count"
if [ "$token_count" -gt 0 ]; then
    tokens_per_sec=$(echo "scale=2; $token_count / $TIME_TAKEN" | bc -l)
    echo "Performance: ${tokens_per_sec} tokens/sec"
fi

echo
echo "Test 2: Longer completion..."
START_TIME=$(date +%s.%N)

response=$(curl -s http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$model\", \"prompt\": \"The capital of France is\", \"stream\": false}")

END_TIME=$(date +%s.%N)
TIME_TAKEN=$(echo "$END_TIME - $START_TIME" | bc -l)

generated_text=$(echo "$response" | jq -r '.response // empty')
token_count=$(echo "$generated_text" | wc -w)

echo "Generated: '$generated_text'"
echo "Time: ${TIME_TAKEN}s"
echo "Estimated tokens: $token_count"
if [ "$token_count" -gt 0 ]; then
    tokens_per_sec=$(echo "scale=2; $token_count / $TIME_TAKEN" | bc -l)
    echo "Performance: ${tokens_per_sec} tokens/sec"
fi

echo
echo "Getting Ollama model info..."
ollama show $model | grep -E "(parameters|family|parameter_size)"