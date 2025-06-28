#!/usr/bin/env python3
"""Test BLAS attention performance"""

import subprocess
import time
import json
import sys
import threading

def start_server():
    """Start the server in the background"""
    print("Starting server...")
    proc = subprocess.Popen(
        ["cargo", "run", "--release"],
        cwd="crates/woolly-server",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Monitor server output
    def monitor_output():
        for line in proc.stdout:
            print(f"[SERVER] {line.strip()}")
            if "BLAS" in line or "🚀" in line:
                print(f">>> BLAS DETECTED: {line.strip()}")
    
    monitor_thread = threading.Thread(target=monitor_output, daemon=True)
    monitor_thread.start()
    
    # Wait for server to start
    time.sleep(5)
    return proc

def test_inference():
    """Test a single inference request"""
    print("\n=== Testing inference with BLAS attention ===")
    
    cmd = [
        "curl", "-X", "POST", "http://localhost:8080/v1/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "granite-3.3-8b-instruct-Q4_K_M",
            "prompt": "Hi",
            "max_tokens": 1,
            "temperature": 0,
            "stream": False
        })
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"\nInference time: {elapsed:.2f} seconds")
    print(f"Tokens/sec: {1/elapsed:.4f}")
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            print(f"Response: {json.dumps(response, indent=2)}")
        except:
            print(f"Raw response: {result.stdout}")
    else:
        print(f"Error: {result.stderr}")

if __name__ == "__main__":
    # Start server
    server_proc = start_server()
    
    try:
        # Test inference
        test_inference()
    finally:
        # Clean up
        print("\nShutting down server...")
        server_proc.terminate()
        server_proc.wait()