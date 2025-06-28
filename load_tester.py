"""
Load Testing Module
Simulates concurrent users and measures system performance under load
"""

import asyncio
import aiohttp
import time
import random
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class RequestResult:
    """Result of a single request"""
    user_id: int
    request_id: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    tokens_generated: Optional[int] = None
    time_to_first_token: Optional[float] = None
    inter_token_times: Optional[List[float]] = None


class LoadTester:
    """Handles load testing and concurrent user simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = f"http://{config['woolly']['host']}:{config['woolly']['port']}"
        self.api_key = config['woolly'].get('api_key')
        self.timeout = config['woolly'].get('inference_timeout', config['woolly'].get('timeout', 600))
        
        # Test prompts
        self.test_prompts = [
            "Explain the concept of recursion in programming.",
            "What are the benefits of using async/await in Python?",
            "Write a Python function to calculate factorial.",
            "Describe the differences between TCP and UDP protocols.",
            "How does garbage collection work in modern programming languages?",
            "Explain the CAP theorem in distributed systems.",
            "What is the difference between a process and a thread?",
            "Write a SQL query to find duplicate records in a table.",
            "Explain the concept of Big O notation with examples.",
            "What are design patterns and why are they important?"
        ]
        
        # Results storage
        self.results = []
        self.active_requests = 0
        self.total_requests = 0
        self.failed_requests = 0
        
    async def run_load_test(self, concurrent_users: int, duration: int, 
                           ramp_up_time: int = 30) -> Dict[str, Any]:
        """Run load test with specified number of concurrent users"""
        self.logger.info(f"Starting load test: {concurrent_users} users, {duration}s duration, {ramp_up_time}s ramp-up")
        
        self.results = []
        self.active_requests = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        start_time = time.time()
        
        # Create user tasks
        tasks = []
        for user_id in range(concurrent_users):
            # Stagger user start times during ramp-up
            delay = (user_id / concurrent_users) * ramp_up_time if ramp_up_time > 0 else 0
            task = asyncio.create_task(
                self._simulate_user(user_id, start_time + delay, start_time + duration)
            )
            tasks.append(task)
        
        # Wait for all users to complete
        await asyncio.gather(*tasks)
        
        # Process results
        return self._analyze_results(concurrent_users, duration)
    
    async def _simulate_user(self, user_id: int, start_time: float, end_time: float):
        """Simulate a single user making requests"""
        # Wait until user's start time
        wait_time = start_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            request_count = 0
            
            while time.time() < end_time:
                # Select random prompt
                prompt = random.choice(self.test_prompts)
                
                # Add some randomness to request timing (0.5-2.5 seconds between requests)
                think_time = random.uniform(0.5, 2.5)
                
                # Make request
                result = await self._make_request(session, user_id, request_count, prompt)
                self.results.append(result)
                
                request_count += 1
                
                # Think time between requests
                remaining_time = end_time - time.time()
                if remaining_time > think_time:
                    await asyncio.sleep(think_time)
                else:
                    break
    
    async def _make_request(self, session: aiohttp.ClientSession, user_id: int, 
                           request_id: int, prompt: str) -> RequestResult:
        """Make a single inference request and measure performance"""
        self.active_requests += 1
        self.total_requests += 1
        
        request_id_str = f"user_{user_id}_req_{request_id}"
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        result = RequestResult(
            user_id=user_id,
            request_id=request_id_str,
            start_time=start_time,
            end_time=start_time,
            duration=0,
            success=False
        )
        
        try:
            async with session.post(
                f"{self.base_url}/api/v1/inference/complete",
                json=payload,
                headers=headers
            ) as response:
                result.status_code = response.status_code
                
                if response.status == 200:
                    # Process streaming response
                    first_token_time = None
                    last_token_time = start_time
                    tokens_received = 0
                    inter_token_times = []
                    
                    async for line in response.content:
                        if line:
                            current_time = time.time()
                            
                            # Time to first token
                            if first_token_time is None:
                                first_token_time = current_time
                                result.time_to_first_token = first_token_time - start_time
                            else:
                                # Inter-token time
                                inter_token_times.append(current_time - last_token_time)
                            
                            last_token_time = current_time
                            tokens_received += 1
                    
                    result.end_time = time.time()
                    result.duration = result.end_time - result.start_time
                    result.success = True
                    result.tokens_generated = tokens_received
                    result.inter_token_times = inter_token_times
                    
                else:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text}"
                    result.end_time = time.time()
                    result.duration = result.end_time - result.start_time
                    self.failed_requests += 1
                    
        except asyncio.TimeoutError:
            result.error = "Request timeout"
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            self.failed_requests += 1
            
        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            self.failed_requests += 1
        
        finally:
            self.active_requests -= 1
        
        return result
    
    def _analyze_results(self, concurrent_users: int, duration: int) -> Dict[str, Any]:
        """Analyze load test results"""
        if not self.results:
            return {"error": "No results collected"}
        
        # Filter successful requests
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {
                "error": "No successful requests",
                "total_requests": len(self.results),
                "failed_requests": len(self.results)
            }
        
        # Calculate metrics
        durations = [r.duration for r in successful_results]
        ttft_values = [r.time_to_first_token for r in successful_results if r.time_to_first_token]
        
        # Calculate throughput
        time_range = max(r.end_time for r in self.results) - min(r.start_time for r in self.results)
        throughput = len(successful_results) / time_range if time_range > 0 else 0
        
        # Inter-token latency
        all_inter_token_times = []
        for r in successful_results:
            if r.inter_token_times:
                all_inter_token_times.extend(r.inter_token_times)
        
        analysis = {
            "concurrent_users": concurrent_users,
            "test_duration": duration,
            "total_requests": len(self.results),
            "successful_requests": len(successful_results),
            "failed_requests": self.failed_requests,
            "success_rate": len(successful_results) / len(self.results) * 100,
            "throughput": throughput,
            "requests_per_user": len(self.results) / concurrent_users,
            
            "latency": {
                "mean": np.mean(durations),
                "median": np.median(durations),
                "min": np.min(durations),
                "max": np.max(durations),
                "std": np.std(durations),
                "percentiles": {
                    "p50": np.percentile(durations, 50),
                    "p90": np.percentile(durations, 90),
                    "p95": np.percentile(durations, 95),
                    "p99": np.percentile(durations, 99)
                }
            }
        }
        
        # Time to first token metrics
        if ttft_values:
            analysis["time_to_first_token"] = {
                "mean": np.mean(ttft_values),
                "median": np.median(ttft_values),
                "min": np.min(ttft_values),
                "max": np.max(ttft_values),
                "std": np.std(ttft_values),
                "percentiles": {
                    "p50": np.percentile(ttft_values, 50),
                    "p90": np.percentile(ttft_values, 90),
                    "p95": np.percentile(ttft_values, 95),
                    "p99": np.percentile(ttft_values, 99)
                }
            }
        
        # Inter-token latency metrics
        if all_inter_token_times:
            analysis["inter_token_latency"] = {
                "mean": np.mean(all_inter_token_times),
                "median": np.median(all_inter_token_times),
                "min": np.min(all_inter_token_times),
                "max": np.max(all_inter_token_times),
                "std": np.std(all_inter_token_times),
                "percentiles": {
                    "p50": np.percentile(all_inter_token_times, 50),
                    "p90": np.percentile(all_inter_token_times, 90),
                    "p95": np.percentile(all_inter_token_times, 95),
                    "p99": np.percentile(all_inter_token_times, 99)
                }
            }
        
        # Error analysis
        if self.failed_requests > 0:
            error_types = defaultdict(int)
            for r in self.results:
                if not r.success and r.error:
                    error_types[r.error.split(':')[0]] += 1
            
            analysis["errors"] = dict(error_types)
        
        # Temporal analysis (requests over time)
        time_buckets = defaultdict(int)
        bucket_size = 10  # 10 second buckets
        
        start_time = min(r.start_time for r in self.results)
        for r in self.results:
            bucket = int((r.start_time - start_time) / bucket_size)
            time_buckets[bucket] += 1
        
        analysis["temporal_distribution"] = {
            f"{k*bucket_size}-{(k+1)*bucket_size}s": v 
            for k, v in sorted(time_buckets.items())
        }
        
        return analysis
    
    async def run_reliability_test(self, duration: int = 3600, 
                                 check_interval: int = 60,
                                 memory_leak_detection: bool = True) -> Dict[str, Any]:
        """Run extended reliability test"""
        self.logger.info(f"Starting reliability test for {duration/3600:.1f} hours")
        
        start_time = time.time()
        end_time = start_time + duration
        
        results = {
            "duration": duration,
            "check_interval": check_interval,
            "checks": [],
            "errors": [],
            "memory_leak_detection": memory_leak_detection
        }
        
        # Import performance monitor for memory leak detection
        if memory_leak_detection:
            from performance_monitor import PerformanceMonitor
            perf_monitor = PerformanceMonitor()
            perf_monitor.start()
        
        check_count = 0
        last_check_time = start_time
        
        # Continuous load with moderate concurrency
        concurrent_users = 5
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            
            # Create user tasks
            user_tasks = []
            for user_id in range(concurrent_users):
                task = asyncio.create_task(
                    self._reliability_user_simulation(session, user_id, end_time)
                )
                user_tasks.append(task)
            
            # Monitor loop
            while time.time() < end_time:
                current_time = time.time()
                
                if current_time - last_check_time >= check_interval:
                    # Perform health check
                    check_result = await self._perform_health_check(session)
                    check_result["timestamp"] = current_time - start_time
                    check_result["check_number"] = check_count
                    
                    # Check for memory leaks
                    if memory_leak_detection and perf_monitor:
                        leak_info = perf_monitor.detect_memory_leak()
                        if leak_info and leak_info["detected"]:
                            check_result["memory_leak_detected"] = True
                            check_result["memory_leak_info"] = leak_info
                    
                    results["checks"].append(check_result)
                    check_count += 1
                    last_check_time = current_time
                    
                    self.logger.info(f"Health check {check_count}: {check_result['status']}")
                
                # Short sleep to prevent tight loop
                await asyncio.sleep(1)
            
            # Cancel user tasks
            for task in user_tasks:
                task.cancel()
            
            await asyncio.gather(*user_tasks, return_exceptions=True)
        
        # Stop performance monitoring
        if memory_leak_detection and perf_monitor:
            perf_data = perf_monitor.stop()
            results["performance_summary"] = perf_data
        
        # Analyze reliability
        results["summary"] = self._analyze_reliability(results)
        
        return results
    
    async def _reliability_user_simulation(self, session: aiohttp.ClientSession, 
                                         user_id: int, end_time: float):
        """Simulate user for reliability testing"""
        request_count = 0
        
        while time.time() < end_time:
            try:
                prompt = random.choice(self.test_prompts)
                result = await self._make_request(session, user_id, request_count, prompt)
                request_count += 1
                
                # Slower pace for reliability testing
                await asyncio.sleep(random.uniform(2, 5))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in reliability user {user_id}: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_check(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Perform health check on the Woolly server"""
        check_result = {
            "status": "unknown",
            "response_time": None,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # Simple health check request
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with session.get(
                f"{self.base_url}/health",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                check_result["response_time"] = time.time() - start_time
                
                if response.status == 200:
                    check_result["status"] = "healthy"
                else:
                    check_result["status"] = "unhealthy"
                    check_result["error"] = f"HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            check_result["status"] = "timeout"
            check_result["error"] = "Health check timeout"
            
        except Exception as e:
            check_result["status"] = "error"
            check_result["error"] = str(e)
        
        return check_result
    
    def _analyze_reliability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reliability test results"""
        checks = results["checks"]
        
        if not checks:
            return {"error": "No health checks performed"}
        
        # Count check statuses
        status_counts = defaultdict(int)
        for check in checks:
            status_counts[check["status"]] += 1
        
        total_checks = len(checks)
        uptime_percentage = (status_counts["healthy"] / total_checks * 100) if total_checks > 0 else 0
        
        # Response time analysis for successful checks
        response_times = [c["response_time"] for c in checks 
                         if c["response_time"] is not None]
        
        summary = {
            "total_checks": total_checks,
            "healthy_checks": status_counts["healthy"],
            "unhealthy_checks": status_counts["unhealthy"],
            "timeout_checks": status_counts["timeout"],
            "error_checks": status_counts["error"],
            "uptime_percentage": uptime_percentage,
            "memory_leaks_detected": any(c.get("memory_leak_detected", False) for c in checks)
        }
        
        if response_times:
            summary["health_check_response_time"] = {
                "mean": np.mean(response_times),
                "median": np.median(response_times),
                "min": np.min(response_times),
                "max": np.max(response_times),
                "std": np.std(response_times)
            }
        
        # Find longest continuous uptime
        max_continuous_uptime = 0
        current_uptime = 0
        
        for check in checks:
            if check["status"] == "healthy":
                current_uptime += 1
                max_continuous_uptime = max(max_continuous_uptime, current_uptime)
            else:
                current_uptime = 0
        
        summary["max_continuous_uptime_checks"] = max_continuous_uptime
        summary["max_continuous_uptime_minutes"] = (max_continuous_uptime * 
                                                   results["check_interval"] / 60)
        
        return summary