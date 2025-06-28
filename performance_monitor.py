"""
Performance Monitor Module
Monitors system resources during Woolly validation tests
"""

import asyncio
import psutil
import time
import threading
import queue
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
import logging
import platform
import subprocess

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class PerformanceMonitor:
    """Monitors system performance metrics during tests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        self.start_time = None
        self.metrics_history = []
        
        # Initialize process handle for Woolly
        self.woolly_process = None
        
        # System info
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information"""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total,
            "gpu_available": GPU_AVAILABLE
        }
        
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            info["gpus"] = [
                {
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "driver": gpu.driver
                }
                for gpu in gpus
            ]
        
        return info
    
    def start(self):
        """Start monitoring system resources"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.metrics_history = []
        
        # Find Woolly process
        self._find_woolly_process()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics"""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Process collected metrics
        results = self._process_metrics()
        
        self.logger.info("Performance monitoring stopped")
        return results
    
    def _find_woolly_process(self):
        """Find the Woolly process by name"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Look for woolly in process name or command line
                if 'woolly' in proc.info['name'].lower():
                    self.woolly_process = psutil.Process(proc.info['pid'])
                    self.logger.info(f"Found Woolly process: PID {proc.info['pid']}")
                    return
                elif proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    if 'woolly' in cmdline:
                        self.woolly_process = psutil.Process(proc.info['pid'])
                        self.logger.info(f"Found Woolly process: PID {proc.info['pid']}")
                        return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        self.logger.warning("Woolly process not found, monitoring system-wide metrics")
    
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread"""
        sample_interval = 0.1  # 100ms sampling
        
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_queue.put(metrics)
                self.metrics_history.append(metrics)
                time.sleep(sample_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            "timestamp": time.time() - self.start_time,
            "datetime": datetime.now().isoformat()
        }
        
        # CPU metrics
        metrics["cpu"] = {
            "percent": psutil.cpu_percent(interval=0),
            "percent_per_core": psutil.cpu_percent(percpu=True, interval=0),
            "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
        
        # Memory metrics
        vm = psutil.virtual_memory()
        metrics["memory"] = {
            "percent": vm.percent,
            "used": vm.used,
            "available": vm.available,
            "cached": getattr(vm, 'cached', 0),
            "buffers": getattr(vm, 'buffers', 0)
        }
        
        # Swap metrics
        swap = psutil.swap_memory()
        metrics["swap"] = {
            "percent": swap.percent,
            "used": swap.used,
            "free": swap.free
        }
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics["disk_io"] = {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count
            }
        
        # Network I/O metrics
        net_io = psutil.net_io_counters()
        metrics["network_io"] = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        # GPU metrics if available
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            metrics["gpu"] = [
                {
                    "id": gpu.id,
                    "load": gpu.load * 100,
                    "memory_percent": gpu.memoryUtil * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "temperature": gpu.temperature
                }
                for gpu in gpus
            ]
        
        # Woolly process specific metrics
        if self.woolly_process:
            try:
                with self.woolly_process.oneshot():
                    metrics["woolly_process"] = {
                        "cpu_percent": self.woolly_process.cpu_percent(interval=0),
                        "memory_percent": self.woolly_process.memory_percent(),
                        "memory_info": self.woolly_process.memory_info()._asdict(),
                        "num_threads": self.woolly_process.num_threads(),
                        "status": self.woolly_process.status()
                    }
                    
                    # Get open files and connections
                    try:
                        metrics["woolly_process"]["open_files"] = len(self.woolly_process.open_files())
                        metrics["woolly_process"]["connections"] = len(self.woolly_process.connections())
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass
                        
            except psutil.NoSuchProcess:
                self.woolly_process = None
                self._find_woolly_process()
        
        return metrics
    
    def _process_metrics(self) -> Dict[str, Any]:
        """Process collected metrics into summary statistics"""
        if not self.metrics_history:
            return {}
        
        # Convert to numpy arrays for easier processing
        cpu_percents = [m["cpu"]["percent"] for m in self.metrics_history]
        memory_percents = [m["memory"]["percent"] for m in self.metrics_history]
        memory_used = [m["memory"]["used"] for m in self.metrics_history]
        
        results = {
            "duration": time.time() - self.start_time,
            "samples": len(self.metrics_history),
            "system_info": self.system_info
        }
        
        # CPU statistics
        results["cpu"] = {
            "mean": np.mean(cpu_percents),
            "max": np.max(cpu_percents),
            "min": np.min(cpu_percents),
            "std": np.std(cpu_percents),
            "percentiles": {
                "p50": np.percentile(cpu_percents, 50),
                "p90": np.percentile(cpu_percents, 90),
                "p95": np.percentile(cpu_percents, 95),
                "p99": np.percentile(cpu_percents, 99)
            }
        }
        
        # Memory statistics
        results["memory"] = {
            "percent": {
                "mean": np.mean(memory_percents),
                "max": np.max(memory_percents),
                "min": np.min(memory_percents),
                "std": np.std(memory_percents)
            },
            "used_bytes": {
                "mean": np.mean(memory_used),
                "max": np.max(memory_used),
                "min": np.min(memory_used),
                "std": np.std(memory_used)
            }
        }
        
        # Disk I/O statistics (calculate rates)
        if len(self.metrics_history) > 1 and "disk_io" in self.metrics_history[0]:
            disk_read_rates = []
            disk_write_rates = []
            
            for i in range(1, len(self.metrics_history)):
                dt = self.metrics_history[i]["timestamp"] - self.metrics_history[i-1]["timestamp"]
                if dt > 0:
                    read_rate = (self.metrics_history[i]["disk_io"]["read_bytes"] - 
                               self.metrics_history[i-1]["disk_io"]["read_bytes"]) / dt
                    write_rate = (self.metrics_history[i]["disk_io"]["write_bytes"] - 
                                self.metrics_history[i-1]["disk_io"]["write_bytes"]) / dt
                    disk_read_rates.append(max(0, read_rate))
                    disk_write_rates.append(max(0, write_rate))
            
            if disk_read_rates:
                results["disk_io"] = {
                    "read_rate_bytes_per_sec": {
                        "mean": np.mean(disk_read_rates),
                        "max": np.max(disk_read_rates),
                        "p95": np.percentile(disk_read_rates, 95)
                    },
                    "write_rate_bytes_per_sec": {
                        "mean": np.mean(disk_write_rates),
                        "max": np.max(disk_write_rates),
                        "p95": np.percentile(disk_write_rates, 95)
                    }
                }
        
        # GPU statistics
        if GPU_AVAILABLE and "gpu" in self.metrics_history[0]:
            gpu_loads = [[gpu["load"] for gpu in m["gpu"]] for m in self.metrics_history]
            gpu_memory = [[gpu["memory_percent"] for gpu in m["gpu"]] for m in self.metrics_history]
            
            results["gpu"] = []
            for gpu_idx in range(len(gpu_loads[0])):
                gpu_load_series = [loads[gpu_idx] for loads in gpu_loads]
                gpu_mem_series = [mems[gpu_idx] for mems in gpu_memory]
                
                results["gpu"].append({
                    "id": gpu_idx,
                    "load": {
                        "mean": np.mean(gpu_load_series),
                        "max": np.max(gpu_load_series),
                        "min": np.min(gpu_load_series),
                        "std": np.std(gpu_load_series)
                    },
                    "memory": {
                        "mean": np.mean(gpu_mem_series),
                        "max": np.max(gpu_mem_series),
                        "min": np.min(gpu_mem_series),
                        "std": np.std(gpu_mem_series)
                    }
                })
        
        # Woolly process statistics
        if any("woolly_process" in m for m in self.metrics_history):
            woolly_cpu = [m["woolly_process"]["cpu_percent"] 
                         for m in self.metrics_history if "woolly_process" in m]
            woolly_mem = [m["woolly_process"]["memory_percent"] 
                         for m in self.metrics_history if "woolly_process" in m]
            
            if woolly_cpu:
                results["woolly_process"] = {
                    "cpu": {
                        "mean": np.mean(woolly_cpu),
                        "max": np.max(woolly_cpu),
                        "min": np.min(woolly_cpu),
                        "std": np.std(woolly_cpu)
                    },
                    "memory": {
                        "mean": np.mean(woolly_mem),
                        "max": np.max(woolly_mem),
                        "min": np.min(woolly_mem),
                        "std": np.std(woolly_mem)
                    }
                }
        
        # Add raw time series data for graphs
        results["time_series"] = {
            "timestamps": [m["timestamp"] for m in self.metrics_history],
            "cpu_percent": cpu_percents,
            "memory_percent": memory_percents,
            "memory_used": memory_used
        }
        
        return results
    
    async def monitor_during_workload(self, workload_func: Callable, 
                                    duration: float = 60, 
                                    sample_interval: float = 0.1) -> Dict[str, Any]:
        """Monitor resources while running a specific workload"""
        self.start()
        
        # Run workload
        start_time = time.time()
        workload_task = asyncio.create_task(workload_func)
        
        # Let workload run for specified duration
        try:
            await asyncio.wait_for(workload_task, timeout=duration)
        except asyncio.TimeoutError:
            self.logger.info(f"Workload ran for full duration of {duration}s")
        
        # Stop monitoring and return results
        results = self.stop()
        results["workload_duration"] = time.time() - start_time
        
        return results
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current snapshot of metrics"""
        return self._collect_metrics()
    
    def detect_memory_leak(self, threshold_mb: float = 100) -> Optional[Dict[str, Any]]:
        """Detect potential memory leaks based on memory growth"""
        if len(self.metrics_history) < 10:
            return None
        
        # Analyze memory usage trend
        memory_used = [m["memory"]["used"] for m in self.metrics_history]
        timestamps = [m["timestamp"] for m in self.metrics_history]
        
        # Simple linear regression to detect trend
        coeffs = np.polyfit(timestamps, memory_used, 1)
        slope = coeffs[0]  # bytes per second
        
        # Convert to MB per hour
        mb_per_hour = (slope * 3600) / (1024 * 1024)
        
        if mb_per_hour > threshold_mb:
            return {
                "detected": True,
                "growth_rate_mb_per_hour": mb_per_hour,
                "total_growth_mb": (memory_used[-1] - memory_used[0]) / (1024 * 1024),
                "duration_seconds": timestamps[-1] - timestamps[0]
            }
        
        return {
            "detected": False,
            "growth_rate_mb_per_hour": mb_per_hour
        }