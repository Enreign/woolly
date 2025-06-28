#!/usr/bin/env python3
"""Test the BLAS attention fix"""

import subprocess
import time
import json
import os
import threading

print("=== Testing BLAS Attention Fix ===\n")

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

# Monitor for BLAS usage
blas_count = 0
output_lines = []

def monitor_output():
    global blas_count
    for line in server_proc.stdout:
        output_lines.append(line.strip())
        if "BLAS" in line and "🚀" in line:
            blas_count += 1
            print(f">>> BLAS DETECTED: {line.strip()}")

monitor_thread = threading.Thread(target=monitor_output, daemon=True)
monitor_thread.start()

# Wait for server
time.sleep(10)

# Load model
print("\nLoading model...")
subprocess.run([
    "curl", "-X", "POST", 
    "http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load",
    "-H", "Content-Type: application/json",
    "-d", "{}"
], capture_output=True)

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

print(f"\nResponse: {result.stdout}")
if result.stderr:
    print(f"Errors: {result.stderr}")

print(f"\n=== RESULTS ===")
print(f"Inference time: {elapsed:.2f} seconds")
print(f"Tokens/sec: {1/elapsed:.4f}")
print(f"BLAS calls detected: {blas_count}")

# Wait for any remaining logs
time.sleep(2)

# Count operations in logs
simd_blas_count = len([l for l in output_lines if "SIMD path: Using BLAS" in l])
dequant_count = len([l for l in output_lines if "dequantization" in l])

print(f"\nOperation counts:")
print(f"  SIMD BLAS attention calls: {simd_blas_count}")
print(f"  Dequantization calls: {dequant_count}")
print(f"  Total BLAS operations: {blas_count}")

# Compare to before
print(f"\nImprovement:")
print(f"  Before: 75+ seconds, 2 BLAS calls")
print(f"  After: {elapsed:.2f} seconds, {blas_count} BLAS calls")
print(f"  Speedup: {75/elapsed:.1f}x")

# Kill server
server_proc.terminate()
server_proc.wait()

# Save detailed log
with open("/tmp/blas_fix_test.log", "w") as f:
    f.write("\n".join(output_lines))
print(f"\nDetailed log saved to /tmp/blas_fix_test.log")