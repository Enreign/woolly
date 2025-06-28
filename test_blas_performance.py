#!/usr/bin/env python3
import time
import json
import urllib.request

# First, run a warmup inference
print("Running warmup inference...")
payload = {
    "model": "granite-3.3-8b-instruct-Q4_K_M",
    "prompt": "Hi",
    "max_tokens": 1,
    "temperature": 0
}

req = urllib.request.Request(
    "http://localhost:3001/api/v1/inference/complete",
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

start = time.time()
try:
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read())
        end = time.time()
        
        duration = end - start
        print(f"\nInference completed in {duration:.2f} seconds")
        print(f"Performance: {1.0/duration:.4f} tokens/sec")
        
        if 1.0/duration > 0.1:
            print("✅ TARGET ACHIEVED! >0.1 tokens/sec")
        else:
            print(f"❌ Below target: need {0.1/(1.0/duration):.2f}x improvement")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()