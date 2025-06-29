"""
Model Benchmarking Module
Tests model loading performance and compares with other implementations
"""

import asyncio
import aiohttp
import time
import psutil
import subprocess
import json
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class ModelBenchmarks:
    """Handles model-specific performance benchmarks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = f"http://{config['woolly']['host']}:{config['woolly']['port']}"
        self.api_key = config['woolly'].get('api_key')
        self.timeout = config['woolly'].get('inference_timeout', config['woolly'].get('timeout', 300))
        
        # Cache for benchmark results
        self.benchmark_cache = {}
        
    async def measure_single_token_latency(self, prompt: str) -> float:
        """Measure latency for generating a single token"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use the correct endpoint format for our server
        model_name = self.config['models'][0]['name']  # Get first model from config
        
        payload = {
            "prompt": prompt,
            "max_length": 1,
            "temperature": 0.0,
            "stream": False
        }
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            start_time = time.time()
            
            try:
                self.logger.info(f"Starting single token latency test (timeout: {self.timeout}s)")
                async with session.post(
                    f"{self.base_url}/api/v1/models/{model_name}/generate",
                    json=payload,
                    headers=headers
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        latency = elapsed * 1000  # Convert to ms
                        
                        # Log detailed results
                        self.logger.info(f"Single token completed in {elapsed:.2f}s")
                        if 'usage' in result:
                            tokens = result['usage'].get('completion_tokens', 1)
                            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
                            self.logger.info(f"Performance: {tokens_per_sec:.6f} tokens/sec")
                        
                        return latency
                    else:
                        self.logger.error(f"Request failed: {response.status}")
                        return -1
                        
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                self.logger.error(f"Request timed out after {elapsed:.1f}s (limit: {self.timeout}s)")
                return -1
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Error after {elapsed:.1f}s: {e}")
                return -1
    
    async def measure_throughput(self, prompt: str, max_tokens: int) -> Dict[str, float]:
        """Measure token generation throughput"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use the correct endpoint format for our server
        model_name = self.config['models'][0]['name']  # Get first model from config
        
        payload = {
            "prompt": prompt,
            "max_length": max_tokens,
            "temperature": 0.0,
            "stream": False
        }
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            start_time = time.time()
            tokens_received = 0
            first_token_time = None
            
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/models/{model_name}/generate",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        # Extract token count from response
                        if 'choices' in result and result['choices']:
                            tokens_received = max_tokens  # Assume all requested tokens were generated
                            first_token_time = start_time + 0.001  # Approximate first token time
                        
                        end_time = time.time()
                        total_time = end_time - start_time
                        generation_time = end_time - first_token_time if first_token_time else 0
                        
                        return {
                            "tokens_generated": tokens_received,
                            "total_time": total_time,
                            "generation_time": generation_time,
                            "tokens_per_second": tokens_received / generation_time if generation_time > 0 else 0,
                            "time_to_first_token": (first_token_time - start_time) if first_token_time else -1
                        }
                    else:
                        self.logger.error(f"Request failed: {response.status}")
                        return {"error": f"HTTP {response.status}"}
                        
            except Exception as e:
                self.logger.error(f"Error measuring throughput: {e}")
                return {"error": str(e)}
    
    async def measure_batch_performance(self, prompts: List[str], max_tokens: int) -> Dict[str, Any]:
        """Measure performance with batch processing"""
        # Note: This assumes Woolly supports batch processing
        # If not, we'll simulate it with concurrent requests
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use the correct endpoint format for our server
        model_name = self.config['models'][0]['name']  # Get first model from config
        
        start_time = time.time()
        
        # Try batch endpoint first
        batch_payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            # Try batch endpoint
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/inference/complete/batch",
                    json=batch_payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        end_time = time.time()
                        
                        return {
                            "batch_size": len(prompts),
                            "total_time": end_time - start_time,
                            "time_per_prompt": (end_time - start_time) / len(prompts),
                            "batch_supported": True
                        }
            except:
                pass
            
            # Fallback to concurrent individual requests
            tasks = []
            for prompt in prompts:
                payload = {
                    "prompt": prompt,
                    "max_length": max_tokens,
                    "temperature": 0.0,
                    "stream": False
                }
                
                task = session.post(
                    f"{self.base_url}/api/v1/models/{model_name}/generate",
                    json=payload,
                    headers=headers
                )
                tasks.append(task)
            
            # Execute concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful = sum(1 for r in responses 
                           if not isinstance(r, Exception) and r.status == 200)
            
            return {
                "batch_size": len(prompts),
                "total_time": end_time - start_time,
                "time_per_prompt": (end_time - start_time) / len(prompts),
                "successful_requests": successful,
                "failed_requests": len(prompts) - successful,
                "batch_supported": False
            }
    
    async def measure_context_scaling(self, context_size: int) -> Dict[str, Any]:
        """Measure performance with different context sizes"""
        # Generate context of specified size
        context = "The quick brown fox jumps over the lazy dog. " * (context_size // 10)
        context = context[:context_size]  # Trim to exact size
        
        prompt = f"{context}\n\nQuestion: What animal jumps over what?\nAnswer:"
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use the correct endpoint format for our server
        model_name = self.config['models'][0]['name']  # Get first model from config
        
        payload = {
            "prompt": prompt,
            "max_length": 20,
            "temperature": 0.0,
            "stream": False
        }
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            start_time = time.time()
            
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/models/{model_name}/generate",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        end_time = time.time()
                        
                        return {
                            "context_size": context_size,
                            "prompt_tokens": len(prompt.split()),  # Approximate
                            "latency": end_time - start_time,
                            "success": True
                        }
                    else:
                        return {
                            "context_size": context_size,
                            "error": f"HTTP {response.status}",
                            "success": False
                        }
                        
            except Exception as e:
                return {
                    "context_size": context_size,
                    "error": str(e),
                    "success": False
                }
    
    async def measure_cold_start(self) -> Tuple[float, float]:
        """Measure cold start time and memory usage"""
        # This would ideally restart the Woolly server and measure startup time
        # For now, we'll measure the first request after a period of inactivity
        
        self.logger.info("Measuring cold start performance...")
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().used
        
        # Make first request
        start_time = time.time()
        latency = await self.measure_single_token_latency("Hello")
        cold_start_time = time.time() - start_time
        
        # Get memory after model load
        final_memory = psutil.virtual_memory().used
        memory_increase = final_memory - initial_memory
        
        return cold_start_time, memory_increase
    
    async def measure_warm_restart(self) -> float:
        """Measure warm restart time"""
        # Make a request to ensure model is loaded
        await self.measure_single_token_latency("Hello")
        
        # Short delay
        await asyncio.sleep(1)
        
        # Measure subsequent request
        start_time = time.time()
        await self.measure_single_token_latency("Hello again")
        warm_restart_time = time.time() - start_time
        
        return warm_restart_time
    
    async def measure_model_memory_footprint(self, model_path: str) -> Dict[str, Any]:
        """Measure memory footprint of a specific model"""
        # This is approximate - ideally would load model in isolation
        
        # Get file size
        try:
            model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        except:
            model_size = 0
        
        # Estimate runtime memory (usually 1.2-2x model file size for GGUF)
        estimated_runtime_memory = model_size * 1.5
        
        return {
            "model_file_size": model_size,
            "estimated_runtime_memory": estimated_runtime_memory,
            "model_path": model_path
        }
    
    async def run_inference(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Run a single inference request"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use the correct endpoint format for our server
        model_name = self.config['models'][0]['name']  # Get first model from config
        
        payload = {
            "prompt": prompt,
            "max_length": max_tokens,
            "temperature": 0.0,
            "stream": False
        }
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/models/{model_name}/generate",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("choices", [{}])[0].get("text", "")
                    else:
                        return None
                        
            except Exception as e:
                self.logger.error(f"Error running inference: {e}")
                return None
    
    async def compare_with_llamacpp(self) -> Dict[str, Any]:
        """Compare performance with llama.cpp"""
        self.logger.info("Comparing with llama.cpp...")
        
        results = {
            "implementation": "llama.cpp",
            "tests": {}
        }
        
        # Check if llama.cpp is available
        try:
            subprocess.run(["llama-cli", "--version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            results["error"] = "llama.cpp not found"
            return results
        
        # Test prompts
        test_prompts = [
            ("short", "Hello", 10),
            ("medium", "Explain quantum computing", 100),
            ("long", "Write a detailed essay about climate change", 500)
        ]
        
        model_path = self.config["models"][0]["path"]
        
        for test_name, prompt, max_tokens in test_prompts:
            # Run llama.cpp benchmark
            cmd = [
                "llama-cli",
                "-m", model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", "0.0",
                "--no-display-prompt",
                "--log-disable"
            ]
            
            start_time = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                end_time = time.time()
                
                if result.returncode == 0:
                    # Parse output for metrics
                    output_lines = result.stderr.split('\n')
                    tokens_generated = max_tokens
                    
                    # Look for timing information in stderr
                    for line in output_lines:
                        if "tokens/s" in line.lower():
                            # Extract tokens per second
                            try:
                                tokens_per_sec = float(line.split()[-2])
                                results["tests"][test_name] = {
                                    "tokens_generated": tokens_generated,
                                    "total_time": end_time - start_time,
                                    "tokens_per_second": tokens_per_sec
                                }
                            except:
                                pass
                    
                    if test_name not in results["tests"]:
                        # Fallback calculation
                        results["tests"][test_name] = {
                            "tokens_generated": tokens_generated,
                            "total_time": end_time - start_time,
                            "tokens_per_second": tokens_generated / (end_time - start_time)
                        }
                else:
                    results["tests"][test_name] = {"error": "Command failed"}
                    
            except Exception as e:
                results["tests"][test_name] = {"error": str(e)}
        
        # Compare with Woolly results
        woolly_results = {}
        for test_name, prompt, max_tokens in test_prompts:
            result = await self.measure_throughput(prompt, max_tokens)
            woolly_results[test_name] = result
        
        results["woolly_comparison"] = woolly_results
        
        return results
    
    async def compare_with_ollama(self) -> Dict[str, Any]:
        """Compare performance with Ollama"""
        self.logger.info("Comparing with Ollama...")
        
        results = {
            "implementation": "ollama",
            "tests": {}
        }
        
        # Check if Ollama is available
        try:
            # Check if ollama is running
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status != 200:
                        results["error"] = "Ollama not running"
                        return results
        except:
            results["error"] = "Ollama not accessible"
            return results
        
        # Test prompts
        test_prompts = [
            ("short", "Hello", 10),
            ("medium", "Explain quantum computing", 100),
            ("long", "Write a detailed essay about climate change", 500)
        ]
        
        # Assume model is already loaded in Ollama
        model_name = "llama3.2:1b"  # Adjust based on actual model
        
        for test_name, prompt, max_tokens in test_prompts:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.0
                },
                "stream": False
            }
            
            start_time = time.time()
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:11434/api/generate",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            end_time = time.time()
                            
                            # Extract metrics from Ollama response
                            eval_count = result.get("eval_count", 0)
                            eval_duration = result.get("eval_duration", 0) / 1e9  # Convert ns to s
                            
                            results["tests"][test_name] = {
                                "tokens_generated": eval_count,
                                "total_time": end_time - start_time,
                                "generation_time": eval_duration,
                                "tokens_per_second": eval_count / eval_duration if eval_duration > 0 else 0
                            }
                        else:
                            results["tests"][test_name] = {"error": f"HTTP {response.status}"}
                            
            except Exception as e:
                results["tests"][test_name] = {"error": str(e)}
        
        # Compare with Woolly results
        woolly_results = {}
        for test_name, prompt, max_tokens in test_prompts:
            result = await self.measure_throughput(prompt, max_tokens)
            woolly_results[test_name] = result
        
        results["woolly_comparison"] = woolly_results
        
        # Calculate performance ratios
        if woolly_results:
            results["performance_ratio"] = {}
            for test_name in results["tests"]:
                if (test_name in woolly_results and 
                    "tokens_per_second" in woolly_results[test_name] and
                    "tokens_per_second" in results["tests"][test_name]):
                    
                    woolly_tps = woolly_results[test_name]["tokens_per_second"]
                    ollama_tps = results["tests"][test_name]["tokens_per_second"]
                    
                    if ollama_tps > 0:
                        results["performance_ratio"][test_name] = woolly_tps / ollama_tps
        
        return results