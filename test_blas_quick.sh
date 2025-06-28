#!/bin/bash

echo "Starting server with debug logging..."
cd crates/woolly-server
RUST_LOG=debug cargo run --release > /tmp/server_test.log 2>&1 &
SERVER_PID=$!

echo "Waiting for server to start..."
sleep 10

echo "Making test request..."
time curl -X POST http://localhost:8080/api/v1/inference/complete \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.3-8b-instruct-Q4_K_M",
    "prompt": "Hi",
    "max_tokens": 1,
    "temperature": 0,
    "stream": false
  }' > /tmp/response.json 2>&1

echo "Checking for BLAS usage..."
grep -E "(BLAS|🚀|Using BLAS|tokens/sec|seconds)" /tmp/server_test.log | tail -20

echo "Killing server..."
kill $SERVER_PID

echo "Response:"
cat /tmp/response.json