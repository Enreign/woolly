#!/usr/bin/env python3
import time
import json
import urllib.request

def measure_inference_time():
    """Measure single token inference time"""
    
    # Prepare request
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
    
    # Measure time
    start_time = time.time()
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read())
            end_time = time.time()
            
            duration = end_time - start_time
            tokens_per_sec = 1.0 / duration
            
            print(f"Inference completed in {duration:.2f} seconds")
            print(f"Performance: {tokens_per_sec:.4f} tokens/sec")
            
            if tokens_per_sec > 0.1:
                print("✅ TARGET ACHIEVED! >0.1 tokens/sec")
            else:
                improvement_needed = 0.1 / tokens_per_sec
                print(f"❌ Below target: need {improvement_needed:.2f}x improvement")
                
            return duration, tokens_per_sec
            
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    print("=== Woolly Performance Measurement ===")
    print("Testing single token inference speed...")
    
    # Run multiple tests
    times = []
    for i in range(3):
        print(f"\nTest {i+1}/3:")
        duration, tps = measure_inference_time()
        if duration:
            times.append(duration)
    
    if times:
        avg_time = sum(times) / len(times)
        avg_tps = 1.0 / avg_time
        print(f"\n=== Summary ===")
        print(f"Average time: {avg_time:.2f} seconds")
        print(f"Average performance: {avg_tps:.4f} tokens/sec")
        print(f"Best time: {min(times):.2f} seconds ({1.0/min(times):.4f} tokens/sec)")

if __name__ == "__main__":
    main()