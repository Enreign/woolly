#!/usr/bin/env python3
"""Simple test for BLAS attention performance"""

import subprocess
import time
import json
import os

# Start server
print("Starting server...")
server_proc = subprocess.Popen(
    ["cargo", "run", "--release"],
    cwd="crates/woolly-server",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env={**os.environ, "RUST_LOG": "debug"}
)

# Collect output
output_lines = []
def collect_output():
    for line in server_proc.stdout:
        output_lines.append(line.strip())
        if "BLAS" in line or "🚀" in line:
            print(f">>> BLAS: {line.strip()}")

import threading
output_thread = threading.Thread(target=collect_output, daemon=True)
output_thread.start()

# Wait for server
time.sleep(10)

# Load model
print("\nLoading model...")
load_result = subprocess.run([
    "curl", "-X", "POST", 
    "http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load",
    "-H", "Content-Type: application/json",
    "-d", "{}"
], capture_output=True, text=True)
print(f"Load result: {load_result.stdout}")

# Wait for model to load
time.sleep(5)

# Make inference request
print("\nMaking inference request...")
start = time.time()
result = subprocess.run([
    "curl", "-X", "POST", "http://localhost:8080/api/v1/inference/complete",
    "-H", "Content-Type: application/json",
    "-d", json.dumps({
        "model": "granite-3.3-8b-instruct-Q4_K_M",
        "prompt": "Hi",
        "max_tokens": 1,
        "temperature": 0,
        "stream": False
    })
], capture_output=True, text=True)
elapsed = time.time() - start

print(f"\nTime: {elapsed:.3f} seconds")
print(f"Response: {result.stdout}")

# Wait a bit for logs
time.sleep(2)

# Check for BLAS in output
print("\nChecking for BLAS usage...")
blas_found = False
for line in output_lines:
    if any(word in line for word in ["BLAS", "🚀", "Using BLAS", "GQA attention"]):
        print(f"  {line}")
        blas_found = True

if not blas_found:
    print("  No BLAS usage detected in logs")

# Save full log
with open("/tmp/full_server_log.txt", "w") as f:
    f.write("\n".join(output_lines))
print("\nFull log saved to /tmp/full_server_log.txt")

# Kill server
server_proc.terminate()
server_proc.wait()