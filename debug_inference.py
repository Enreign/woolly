#!/usr/bin/env python3
"""Deep debugging of Woolly inference to find bottlenecks"""

import subprocess
import time
import json
import os
import re

# Start server with max debug logging
print("Starting server with debug logging...")
env = os.environ.copy()
env["RUST_LOG"] = "trace"
env["RUST_BACKTRACE"] = "1"

server_proc = subprocess.Popen(
    ["cargo", "run", "--release"],
    cwd="crates/woolly-server",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env
)

# Collect all output
output_lines = []
def collect_output():
    for line in server_proc.stdout:
        output_lines.append(line.strip())

import threading
output_thread = threading.Thread(target=collect_output, daemon=True)
output_thread.start()

# Wait for server
print("Waiting for server...")
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
subprocess.run([
    "curl", "-X", "POST", "http://localhost:8080/api/v1/inference/complete",
    "-H", "Content-Type: application/json",
    "-d", json.dumps({
        "model": "granite-3.3-8b-instruct-Q4_K_M",
        "prompt": "Hi",
        "max_tokens": 1,
        "temperature": 0,
        "stream": False
    })
], capture_output=True)
elapsed = time.time() - start

print(f"\nTotal inference time: {elapsed:.3f} seconds")

# Wait for all logs
time.sleep(3)

# Kill server
server_proc.terminate()
server_proc.wait()

# Analyze output
print("\n=== ANALYSIS ===\n")

# 1. Find all BLAS calls
print("1. BLAS Usage:")
blas_calls = [l for l in output_lines if "BLAS" in l and ("Using" in l or "🚀" in l)]
for call in blas_calls[-10:]:  # Last 10 BLAS calls
    print(f"  {call}")
print(f"  Total BLAS calls: {len(blas_calls)}")

# 2. Find attention computation
print("\n2. Attention Computation:")
attention_lines = [l for l in output_lines if any(x in l for x in ["attention", "GQA", "QKV", "attn"])]
for line in attention_lines[-10:]:
    print(f"  {line}")

# 3. Find fallbacks
print("\n3. Fallback Paths:")
fallback_lines = [l for l in output_lines if any(x in l for x in ["fallback", "Fallback", "not available", "slow path"])]
for line in fallback_lines:
    print(f"  {line}")

# 4. Find tensor operations
print("\n4. Tensor Operations Pattern:")
matmul_calls = [l for l in output_lines if "matmul" in l.lower()]
print(f"  Matrix multiplications: {len(matmul_calls)}")

dequant_calls = [l for l in output_lines if "dequantization" in l]
print(f"  Dequantization calls: {len(dequant_calls)}")

# 5. Memory operations
print("\n5. Memory Patterns:")
alloc_lines = [l for l in output_lines if any(x in l for x in ["allocat", "buffer", "memory pool"])]
print(f"  Memory allocations: {len(alloc_lines)}")

# 6. Performance markers
print("\n6. Performance Indicators:")
slow_ops = [l for l in output_lines if any(x in l for x in ["slow", "took", "elapsed", "ms", "seconds"])]
for line in slow_ops[-10:]:
    print(f"  {line}")

# Save full log
with open("/tmp/debug_inference_full.log", "w") as f:
    f.write("\n".join(output_lines))
print(f"\nFull log saved to /tmp/debug_inference_full.log ({len(output_lines)} lines)")

# Find the actual inference execution
print("\n7. Inference Execution Flow:")
inference_start = None
inference_end = None
for i, line in enumerate(output_lines):
    if "inference/complete" in line and "started processing" in line:
        inference_start = i
    elif inference_start and "finished processing" in line:
        inference_end = i
        break

if inference_start and inference_end:
    print(f"  Inference span: lines {inference_start} to {inference_end}")
    print(f"  Total inference lines: {inference_end - inference_start}")
    
    # Extract key operations during inference
    inference_lines = output_lines[inference_start:inference_end]
    
    # Count operations
    op_counts = {
        "NEON": len([l for l in inference_lines if "NEON" in l]),
        "BLAS": len([l for l in inference_lines if "BLAS" in l]),
        "dequant": len([l for l in inference_lines if "dequant" in l.lower()]),
        "layer": len([l for l in inference_lines if "layer" in l.lower()]),
        "attention": len([l for l in inference_lines if "attention" in l.lower()])
    }
    
    print("\n  Operation counts during inference:")
    for op, count in op_counts.items():
        print(f"    {op}: {count}")