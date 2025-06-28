#!/usr/bin/env python3
import urllib.request
import json
import time

def test_simd_performance():
    """Direct test of SIMD performance without external dependencies"""
    
    print("SIMD Performance Test")
    print("=" * 40)
    
    # First check if model is available
    try:
        req = urllib.request.Request(
            "http://localhost:3001/api/v1/models",
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            models = json.loads(response.read())
            print(f"Available models: {len(models)}")
            for model in models:
                print(f"  - {model['name']} (loaded: {model.get('loaded', False)})")
    except Exception as e:
        print(f"Failed to list models: {e}")
        return
    
    # Try to load model if not loaded
    model_name = "granite-3.3-8b-instruct-Q4_K_M"
    try:
        print(f"\nLoading model {model_name}...")
        req = urllib.request.Request(
            f"http://localhost:3001/api/v1/models/{model_name}/load",
            data=json.dumps({}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            print(f"Load result: {result}")
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    # Run inference test
    print("\nRunning inference test...")
    try:
        payload = {
            "model": model_name,
            "prompt": "The capital of France is",
            "max_tokens": 1,
            "temperature": 0
        }
        
        req = urllib.request.Request(
            "http://localhost:3001/api/v1/inference/complete",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        start_time = time.time()
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read())
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"\nInference completed in {duration:.2f} seconds")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if duration > 0:
                tokens_per_sec = 1 / duration
                print(f"\nPerformance: {tokens_per_sec:.4f} tokens/sec")
                
                if tokens_per_sec > 0.1:
                    print("✅ TARGET ACHIEVED! >0.1 tokens/sec")
                else:
                    improvement_needed = 0.1 / tokens_per_sec
                    print(f"❌ Below target: need {improvement_needed:.2f}x improvement")
                    
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simd_performance()