#\!/usr/bin/env python3
import subprocess
import time
import requests
import json
import os

def start_server(simd_enabled=True):
    """Start woolly server with or without SIMD"""
    env = os.environ.copy()
    if not simd_enabled:
        env['WOOLLY_DISABLE_SIMD'] = '1'
    
    # Kill any existing server
    subprocess.run(['pkill', '-f', 'woolly-server'], stderr=subprocess.DEVNULL)
    time.sleep(1)
    
    # Start server
    log_file = 'simd_enabled.log' if simd_enabled else 'simd_disabled.log'
    proc = subprocess.Popen(
        ['./target/release/woolly-server'],
        env=env,
        stdout=open(log_file, 'w'),
        stderr=subprocess.STDOUT
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if server is running
    try:
        resp = requests.get('http://localhost:8080/api/v1/health', timeout=2)
        if resp.status_code == 200:
            print(f"✅ Server started {'with' if simd_enabled else 'without'} SIMD")
            return proc
    except:
        pass
    
    print(f"❌ Failed to start server {'with' if simd_enabled else 'without'} SIMD")
    return None

def test_inference():
    """Test a simple inference"""
    # First check if model needs to be loaded
    models_resp = requests.get('http://localhost:8080/api/v1/models')
    if models_resp.status_code == 200:
        models = models_resp.json()
        if models and not models[0].get('loaded', False):
            print("Loading model...")
            load_start = time.time()
            load_resp = requests.post(
                'http://localhost:8080/api/v1/models/load',
                json={'model_path': './models/granite-3.3-8b-instruct-Q4_K_M.gguf'},
                timeout=300
            )
            load_time = time.time() - load_start
            print(f"Model loading took {load_time:.1f}s")
    
    # Test inference
    start = time.time()
    try:
        resp = requests.post(
            'http://localhost:8080/api/v1/inference/complete',
            json={
                'prompt': 'Hello',
                'max_tokens': 1,
                'temperature': 0.1
            },
            timeout=120
        )
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            if 'choices' in data:
                tokens_per_sec = 1 / elapsed if elapsed > 0 else 0
                return elapsed, tokens_per_sec
        else:
            print(f"Inference error: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        print(f"Inference failed: {e}")
    
    return None, None

def main():
    print("🔍 SIMD Performance Impact Test")
    print("==============================\n")
    
    results = {}
    
    # Test with SIMD enabled
    print("Test 1: SIMD Enabled")
    proc = start_server(simd_enabled=True)
    if proc:
        time1, tps1 = test_inference()
        if time1:
            results['simd_enabled'] = {'time': time1, 'tps': tps1}
            print(f"Time: {time1:.2f}s, Performance: {tps1:.4f} tokens/sec")
        proc.terminate()
        time.sleep(2)
    
    print("\nTest 2: SIMD Disabled")
    proc = start_server(simd_enabled=False)
    if proc:
        time2, tps2 = test_inference()
        if time2:
            results['simd_disabled'] = {'time': time2, 'tps': tps2}
            print(f"Time: {time2:.2f}s, Performance: {tps2:.4f} tokens/sec")
        proc.terminate()
    
    # Summary
    print("\n📊 Summary")
    print("==========")
    if 'simd_enabled' in results and 'simd_disabled' in results:
        simd_time = results['simd_enabled']['time']
        no_simd_time = results['simd_disabled']['time']
        
        print(f"SIMD Enabled:  {simd_time:.2f}s ({results['simd_enabled']['tps']:.4f} tokens/sec)")
        print(f"SIMD Disabled: {no_simd_time:.2f}s ({results['simd_disabled']['tps']:.4f} tokens/sec)")
        
        if no_simd_time < simd_time:
            speedup = simd_time / no_simd_time
            print(f"\n⚠️  SIMD is making performance WORSE by {speedup:.1f}x\!")
            print(f"Disabling SIMD would give you a {speedup:.1f}x speedup")
        else:
            speedup = no_simd_time / simd_time
            print(f"\n✅ SIMD provides {speedup:.1f}x speedup")
    else:
        print("Tests incomplete - check logs for errors")

if __name__ == '__main__':
    main()
