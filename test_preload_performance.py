#!/usr/bin/env python3
"""Test performance with weight preloading enabled."""

import requests
import time
import json
import subprocess
import os
import signal
import sys
from datetime import datetime

# Configuration
WOOLLY_URL = "http://localhost:8080"
MODEL_NAME = "granite-3.0b-instruct-Q4_K_M"
MODEL_PATH = "./models/granite-3.0b-instruct-Q4_K_M.gguf"
TEST_PROMPT = "Hello"
NUM_TOKENS = 3

def kill_existing_servers():
    """Kill any existing woolly-server processes."""
    try:
        subprocess.run(['pkill', '-f', 'woolly-server'], capture_output=True)
        time.sleep(2)
    except:
        pass

def start_server(disable_simd=True):
    """Start the woolly server with appropriate settings."""
    env = os.environ.copy()
    if disable_simd:
        env['WOOLLY_DISABLE_SIMD'] = '1'
    
    # Enable debug logging
    env['RUST_LOG'] = 'info'
    
    # Start server in background
    process = subprocess.Popen(
        ['./target/release/woolly-server'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    for i in range(30):
        try:
            response = requests.get(f"{WOOLLY_URL}/api/v1/models")
            if response.status_code == 200:
                print("Server started successfully!")
                return process
        except:
            time.sleep(1)
    
    raise Exception("Server failed to start")

def load_model():
    """Load the model into memory."""
    print(f"\nLoading model: {MODEL_NAME}")
    response = requests.post(
        f"{WOOLLY_URL}/api/v1/models/{MODEL_NAME}/load",
        json={"model_path": MODEL_PATH}
    )
    
    if response.status_code != 200:
        print(f"Failed to load model: {response.status_code}")
        print(response.text)
        return False
    
    print("Model loaded successfully!")
    return True

def test_inference(test_name):
    """Test inference performance."""
    print(f"\n=== {test_name} ===")
    
    # Prepare request
    data = {
        "prompt": TEST_PROMPT,
        "max_length": NUM_TOKENS,
        "temperature": 0.0,
        "stream": False
    }
    
    # Time the inference
    start_time = time.time()
    
    response = requests.post(
        f"{WOOLLY_URL}/api/v1/models/{MODEL_NAME}/generate",
        json=data,
        timeout=600  # 10 minute timeout
    )
    
    end_time = time.time()
    
    if response.status_code != 200:
        print(f"Inference failed: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    tokens_generated = len(result.get('tokens', []))
    total_time = end_time - start_time
    time_per_token = total_time / tokens_generated if tokens_generated > 0 else 0
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Time per token: {time_per_token:.2f} seconds")
    print(f"Tokens/sec: {1/time_per_token:.2f}" if time_per_token > 0 else "N/A")
    
    return {
        "total_time": total_time,
        "tokens": tokens_generated,
        "time_per_token": time_per_token,
        "tokens_per_sec": 1/time_per_token if time_per_token > 0 else 0
    }

def main():
    """Run the performance test."""
    print("WOOLLY WEIGHT PRELOADING PERFORMANCE TEST")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {TEST_PROMPT}")
    print(f"Tokens to generate: {NUM_TOKENS}")
    print(f"SIMD: DISABLED (WOOLLY_DISABLE_SIMD=1)")
    print("Weight Preloading: ENABLED")
    
    # Kill any existing servers
    kill_existing_servers()
    
    # Start server
    server_process = start_server(disable_simd=True)
    
    try:
        # Load model
        if not load_model():
            return
        
        # Run multiple inference tests
        results = []
        for i in range(3):
            result = test_inference(f"Inference Test {i+1}")
            if result:
                results.append(result)
            time.sleep(2)  # Brief pause between tests
        
        # Print summary
        if results:
            print("\n=== SUMMARY ===")
            avg_time_per_token = sum(r["time_per_token"] for r in results) / len(results)
            avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in results) / len(results)
            
            print(f"Average time per token: {avg_time_per_token:.2f} seconds")
            print(f"Average tokens/sec: {avg_tokens_per_sec:.2f}")
            
            # Compare with previous results
            print("\n=== COMPARISON ===")
            print("Without preloading (from cache debug):")
            print("  First token: ~194 seconds")
            print("  Subsequent tokens: ~179-212 seconds")
            print("\nWith preloading:")
            print(f"  All tokens: ~{avg_time_per_token:.0f} seconds")
            
            improvement = 194 / avg_time_per_token if avg_time_per_token > 0 else 0
            print(f"\nImprovement: {improvement:.1f}x faster on first token!")
            
    finally:
        # Clean up
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)

if __name__ == "__main__":
    main()