#!/usr/bin/env python3
import requests
import time
import json
import sys

def test_single_token():
    """Test single token generation performance"""
    url = "http://localhost:3001/v1/completions"
    
    # First, try to load the model
    load_url = "http://localhost:3001/v1/models/load"
    load_payload = {
        "model": "granite-3.3-8b-instruct-Q4_K_M.gguf"
    }
    
    print("Loading model...")
    try:
        resp = requests.post(load_url, json=load_payload, timeout=30)
        print(f"Load response: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Load error: {resp.text}")
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    # Test inference
    payload = {
        "model": "models/granite-3.3-8b-instruct-Q4_K_M.gguf",
        "prompt": "The capital of France is",
        "max_tokens": 1,
        "temperature": 0
    }
    
    print("\nRunning inference test...")
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        end_time = time.time()
        
        duration = end_time - start_time
        
        print(f"Response status: {response.status_code}")
        print(f"Response time: {duration:.2f} seconds")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
                
                # Calculate tokens/sec
                if duration > 0:
                    tokens_per_sec = 1 / duration
                    print(f"\nPerformance: {tokens_per_sec:.4f} tokens/sec")
                    
                    # Check if we hit our target
                    if tokens_per_sec > 0.1:
                        print("✅ TARGET ACHIEVED! >0.1 tokens/sec")
                    else:
                        print(f"❌ Below target: need {(0.1/tokens_per_sec):.2f}x improvement")
            except:
                print(f"Response text: {response.text}")
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_token()