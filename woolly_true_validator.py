#!/usr/bin/env python3
"""
Woolly True Performance Validator
Main orchestrator for comprehensive performance validation suite
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from performance_monitor import PerformanceMonitor
from load_tester import LoadTester
from model_benchmarks import ModelBenchmarks
from quality_validator import QualityValidator
from report_generator import ReportGenerator


class WoollyTrueValidator:
    """Main validation orchestrator for Woolly performance testing"""
    
    def __init__(self, config_path: str = "validation_config.json"):
        self.config = self._load_config(config_path)
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "config": self.config
            },
            "tests": {}
        }
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.load_tester = LoadTester(self.config)
        self.model_benchmarks = ModelBenchmarks(self.config)
        self.quality_validator = QualityValidator(self.config)
        self.report_generator = ReportGenerator()
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if no config file exists"""
        return {
            "woolly": {
                "host": "localhost",
                "port": 8080,
                "api_key": None,
                "timeout": 300
            },
            "models": [
                {
                    "name": "llama-3.2-1b",
                    "path": "models/llama-3.2-1b-instruct.gguf",
                    "quantization": "Q4_K_M"
                }
            ],
            "tests": {
                "inference_speed": {
                    "enabled": True,
                    "prompts": [
                        "Hello, how are you?",
                        "Explain quantum computing in simple terms.",
                        "Write a Python function to sort a list."
                    ],
                    "max_tokens": [10, 100, 500],
                    "batch_sizes": [1, 4, 8, 16]
                },
                "resource_utilization": {
                    "enabled": True,
                    "sample_interval": 0.1,
                    "duration": 60
                },
                "model_loading": {
                    "enabled": True,
                    "iterations": 5,
                    "measure_memory": True
                },
                "quality_validation": {
                    "enabled": True,
                    "consistency_runs": 10,
                    "temperature": 0.0
                },
                "load_testing": {
                    "enabled": True,
                    "concurrent_users": [1, 5, 10, 20, 50],
                    "duration": 300,
                    "ramp_up_time": 30
                },
                "comparative": {
                    "enabled": True,
                    "compare_with": ["llama.cpp", "ollama"],
                    "metrics": ["throughput", "latency", "memory"]
                },
                "reliability": {
                    "enabled": True,
                    "duration": 3600,
                    "check_interval": 60,
                    "memory_leak_detection": True
                }
            },
            "reporting": {
                "output_dir": "validation_results",
                "generate_html": True,
                "generate_pdf": True,
                "include_graphs": True
            }
        }
    
    def _setup_logging(self):
        """Configure logging for the validation suite"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Woolly True Validator initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all enabled tests in the validation suite"""
        self.logger.info("Starting comprehensive validation suite")
        
        # Start performance monitoring
        self.performance_monitor.start()
        
        try:
            # Run each test category
            if self.config["tests"]["inference_speed"]["enabled"]:
                await self._run_inference_speed_tests()
            
            if self.config["tests"]["resource_utilization"]["enabled"]:
                await self._run_resource_utilization_tests()
            
            if self.config["tests"]["model_loading"]["enabled"]:
                await self._run_model_loading_tests()
            
            if self.config["tests"]["quality_validation"]["enabled"]:
                await self._run_quality_validation_tests()
            
            if self.config["tests"]["load_testing"]["enabled"]:
                await self._run_load_tests()
            
            if self.config["tests"]["comparative"]["enabled"]:
                await self._run_comparative_tests()
            
            if self.config["tests"]["reliability"]["enabled"]:
                await self._run_reliability_tests()
            
        finally:
            # Stop performance monitoring
            self.performance_monitor.stop()
            
        # Generate reports
        self._generate_reports()
        
        return self.results
    
    async def _run_inference_speed_tests(self):
        """Test inference speed metrics with enhanced timeout handling"""
        self.logger.info("Running inference speed tests...")
        
        results = {
            "single_token_latency": {},
            "sustained_throughput": {},
            "timeout_analysis": {},
            "performance_summary": {},
            "detailed_measurements": {}
        }
        
        test_config = self.config["tests"]["inference_speed"]
        timeout_limit = self.config["woolly"]["inference_timeout"]
        
        # Enhanced single token latency tests
        successful_tests = []
        total_tests = 0
        
        for prompt in test_config["prompts"]:
            total_tests += 1
            self.logger.info(f"Testing single token latency for prompt: '{prompt[:30]}...'")
            
            start_time = time.time()
            latency = await self.model_benchmarks.measure_single_token_latency(prompt)
            elapsed = time.time() - start_time
            
            key = prompt[:50]
            
            if latency > 0:
                # Successful completion
                results["single_token_latency"][key] = latency
                tokens_per_sec = 1000 / latency if latency > 0 else 0  # latency in ms
                successful_tests.append(tokens_per_sec)
                
                # Store detailed measurement
                results["detailed_measurements"][key] = {
                    "status": "completed",
                    "latency_ms": latency,
                    "tokens_per_second": tokens_per_sec,
                    "elapsed_time": elapsed,
                    "performance_class": self._classify_performance(tokens_per_sec)
                }
                
                self.logger.info(f"✅ Completed: {elapsed:.2f}s, {tokens_per_sec:.6f} tokens/sec")
            else:
                # Timeout or failure
                results["single_token_latency"][key] = -1
                results["detailed_measurements"][key] = {
                    "status": "timeout_or_error",
                    "elapsed_time": elapsed,
                    "timeout_limit": timeout_limit,
                    "estimated_max_tps": 1.0 / elapsed if elapsed > 0 else 0
                }
                
                self.logger.error(f"❌ Failed/Timeout: {elapsed:.2f}s (limit: {timeout_limit}s)")
        
        # Performance analysis
        if successful_tests:
            avg_tps = sum(successful_tests) / len(successful_tests)
            max_tps = max(successful_tests)
            min_tps = min(successful_tests)
            
            results["performance_summary"] = {
                "average_tokens_per_second": avg_tps,
                "max_tokens_per_second": max_tps,
                "min_tokens_per_second": min_tps,
                "successful_tests": len(successful_tests),
                "total_tests": total_tests,
                "success_rate": len(successful_tests) / total_tests,
                "meets_target_15tps": avg_tps >= 15.0,
                "target_performance_ratio": avg_tps / 15.0,
                "speedup_needed": 15.0 / avg_tps if avg_tps > 0 else float('inf'),
                "usability_rating": self._get_usability_rating(avg_tps),
                "performance_classification": self._classify_performance(avg_tps)
            }
            
            self.logger.info(f"📊 Average performance: {avg_tps:.6f} tokens/sec")
            self.logger.info(f"🎯 Target ratio: {avg_tps/15.0:.1%}")
        else:
            results["performance_summary"] = {
                "status": "all_tests_failed_or_timeout",
                "successful_tests": 0,
                "total_tests": total_tests,
                "meets_target_15tps": False,
                "usability_rating": "unusable",
                "performance_classification": "failed"
            }
            
            self.logger.error("❌ All inference speed tests failed or timed out")
        
        # Enhanced throughput tests (only run if we have some successful single token tests)
        if successful_tests:
            self.logger.info("Running throughput tests with varying token counts...")
            
            for max_tokens in test_config["max_tokens"]:
                self.logger.info(f"Testing {max_tokens} tokens generation")
                
                start_time = time.time()
                throughput = await self.model_benchmarks.measure_throughput(
                    prompt=test_config["prompts"][0],
                    max_tokens=max_tokens
                )
                elapsed = time.time() - start_time
                
                results["sustained_throughput"][f"{max_tokens}_tokens"] = throughput
                
                # Analyze throughput scaling
                if isinstance(throughput, dict) and "tokens_per_second" in throughput:
                    tps = throughput["tokens_per_second"]
                    efficiency = (tps * max_tokens) / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(f"  {max_tokens} tokens: {tps:.6f} tps, efficiency: {efficiency:.2f}")
                else:
                    self.logger.error(f"  {max_tokens} tokens: failed or timeout after {elapsed:.2f}s")
        else:
            self.logger.info("Skipping throughput tests due to single token test failures")
            results["sustained_throughput"]["status"] = "skipped_due_to_single_token_failures"
        
        # Timeout analysis summary
        results["timeout_analysis"] = {
            "timeout_limit_seconds": timeout_limit,
            "tests_completed_within_timeout": len(successful_tests),
            "tests_that_timed_out": total_tests - len(successful_tests),
            "timeout_rate": (total_tests - len(successful_tests)) / total_tests,
            "average_completion_time": sum(results["detailed_measurements"][k].get("elapsed_time", 0) 
                                         for k in results["detailed_measurements"] 
                                         if results["detailed_measurements"][k]["status"] == "completed") / len(successful_tests) if successful_tests else 0
        }
        
        self.results["tests"]["inference_speed"] = results
        self.logger.info("Inference speed tests completed")
    
    def _classify_performance(self, tokens_per_sec: float) -> str:
        """Classify performance level"""
        if tokens_per_sec >= 15:
            return "excellent"
        elif tokens_per_sec >= 5:
            return "good"
        elif tokens_per_sec >= 1:
            return "acceptable"
        elif tokens_per_sec >= 0.1:
            return "slow"
        elif tokens_per_sec > 0:
            return "very_slow"
        else:
            return "failed"
    
    def _get_usability_rating(self, tokens_per_sec: float) -> str:
        """Get usability rating for Ole desktop app"""
        if tokens_per_sec >= 15:
            return "excellent_for_desktop"
        elif tokens_per_sec >= 5:
            return "good_for_desktop"
        elif tokens_per_sec >= 1:
            return "usable_with_delays"
        elif tokens_per_sec >= 0.1:
            return "barely_usable"
        else:
            return "unusable"
    
    async def _run_resource_utilization_tests(self):
        """Test resource utilization during inference"""
        self.logger.info("Running resource utilization tests...")
        
        test_config = self.config["tests"]["resource_utilization"]
        
        # Run inference workload while monitoring resources
        workload_func = await self._generate_resource_test_workload()
        results = await self.performance_monitor.monitor_during_workload(
            workload_func=workload_func(),
            duration=test_config["duration"],
            sample_interval=test_config["sample_interval"]
        )
        
        self.results["tests"]["resource_utilization"] = results
        self.logger.info("Resource utilization tests completed")
    
    async def _run_model_loading_tests(self):
        """Test model loading performance"""
        self.logger.info("Running model loading tests...")
        
        test_config = self.config["tests"]["model_loading"]
        results = {
            "cold_start": [],
            "warm_restart": [],
            "memory_footprint": {},
            "success_rate": 0.0
        }
        
        successful_loads = 0
        
        for i in range(test_config["iterations"]):
            # Cold start test
            cold_start_time, cold_memory = await self.model_benchmarks.measure_cold_start()
            results["cold_start"].append(cold_start_time)
            
            # Warm restart test
            warm_restart_time = await self.model_benchmarks.measure_warm_restart()
            results["warm_restart"].append(warm_restart_time)
            
            if cold_start_time > 0:
                successful_loads += 1
        
        # Calculate success rate
        results["success_rate"] = successful_loads / test_config["iterations"]
        
        # Measure memory footprint for each model
        for model in self.config["models"]:
            memory_usage = await self.model_benchmarks.measure_model_memory_footprint(
                model["path"]
            )
            results["memory_footprint"][model["name"]] = memory_usage
        
        self.results["tests"]["model_loading"] = results
        self.logger.info("Model loading tests completed")
    
    async def _run_quality_validation_tests(self):
        """Test output quality and consistency"""
        self.logger.info("Running quality validation tests...")
        
        test_config = self.config["tests"]["quality_validation"]
        
        results = await self.quality_validator.run_all_validations(
            consistency_runs=test_config["consistency_runs"],
            temperature=test_config["temperature"]
        )
        
        self.results["tests"]["quality_validation"] = results
        self.logger.info("Quality validation tests completed")
    
    async def _run_load_tests(self):
        """Run load testing with multiple concurrent users"""
        self.logger.info("Running load tests...")
        
        test_config = self.config["tests"]["load_testing"]
        results = {}
        
        for concurrent_users in test_config["concurrent_users"]:
            self.logger.info(f"Testing with {concurrent_users} concurrent users")
            
            load_results = await self.load_tester.run_load_test(
                concurrent_users=concurrent_users,
                duration=test_config["duration"],
                ramp_up_time=test_config["ramp_up_time"]
            )
            
            results[f"{concurrent_users}_users"] = load_results
        
        self.results["tests"]["load_testing"] = results
        self.logger.info("Load tests completed")
    
    async def _run_comparative_tests(self):
        """Run comparative benchmarks against other implementations"""
        self.logger.info("Running comparative tests...")
        
        test_config = self.config["tests"]["comparative"]
        results = {}
        
        for implementation in test_config["compare_with"]:
            if implementation == "llama.cpp":
                comp_results = await self.model_benchmarks.compare_with_llamacpp()
            elif implementation == "ollama":
                comp_results = await self.model_benchmarks.compare_with_ollama()
            else:
                self.logger.warning(f"Unknown implementation for comparison: {implementation}")
                continue
            
            results[implementation] = comp_results
        
        self.results["tests"]["comparative"] = results
        self.logger.info("Comparative tests completed")
    
    async def _run_reliability_tests(self):
        """Run extended reliability and stability tests"""
        self.logger.info("Running reliability tests...")
        
        test_config = self.config["tests"]["reliability"]
        
        results = await self.load_tester.run_reliability_test(
            duration=test_config["duration"],
            check_interval=test_config["check_interval"],
            memory_leak_detection=test_config["memory_leak_detection"]
        )
        
        self.results["tests"]["reliability"] = results
        self.logger.info("Reliability tests completed")
    
    async def _generate_resource_test_workload(self):
        """Generate workload for resource utilization testing"""
        async def workload():
            prompts = self.config["tests"]["inference_speed"]["prompts"]
            for _ in range(10):
                for prompt in prompts:
                    await self.model_benchmarks.run_inference(prompt, max_tokens=100)
                    await asyncio.sleep(0.1)
        
        return workload
    
    def _generate_reports(self):
        """Generate comprehensive reports from test results"""
        self.logger.info("Generating reports...")
        
        output_dir = Path(self.config["reporting"]["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        # Save raw results
        results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate HTML report
        if self.config["reporting"]["generate_html"]:
            html_file = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.report_generator.generate_html_report(self.results, html_file)
        
        # Generate PDF report
        if self.config["reporting"]["generate_pdf"]:
            pdf_file = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            self.report_generator.generate_pdf_report(self.results, pdf_file)
        
        self.logger.info(f"Reports generated in {output_dir}")


async def main():
    """Main entry point for the validation suite"""
    parser = argparse.ArgumentParser(description="Woolly True Performance Validator")
    parser.add_argument(
        "--config",
        type=str,
        default="validation_config.json",
        help="Path to validation configuration file"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=[
            "inference_speed", "resource_utilization", "model_loading",
            "quality_validation", "load_testing", "comparative", "reliability"
        ],
        help="Specific tests to run (default: all enabled in config)"
    )
    
    args = parser.parse_args()
    
    # Create validator instance
    validator = WoollyTrueValidator(args.config)
    
    # Override config if specific tests requested
    if args.tests:
        for test_name in validator.config["tests"]:
            validator.config["tests"][test_name]["enabled"] = test_name in args.tests
    
    # Run validation suite
    results = await validator.run_all_tests()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {validator.config['reporting']['output_dir']}")
    
    # Print summary
    if "inference_speed" in results["tests"]:
        avg_latency = sum(results["tests"]["inference_speed"]["single_token_latency"].values()) / len(results["tests"]["inference_speed"]["single_token_latency"])
        print(f"Average single token latency: {avg_latency:.2f}ms")
    
    if "load_testing" in results["tests"]:
        for users, data in results["tests"]["load_testing"].items():
            if "throughput" in data:
                print(f"Throughput with {users}: {data['throughput']:.2f} req/s")


if __name__ == "__main__":
    asyncio.run(main())