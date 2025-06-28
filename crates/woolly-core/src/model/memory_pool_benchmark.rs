//! Benchmarking utilities for memory pool performance

use std::time::{Duration, Instant};
use crate::Result;

/// Track allocation metrics
#[derive(Debug, Default)]
pub struct AllocationMetrics {
    pub total_allocations: u64,
    pub total_bytes_allocated: u64,
    pub allocation_time: Duration,
    pub deallocation_time: Duration,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Memory pool benchmarker
pub struct MemoryPoolBenchmark {
    metrics: AllocationMetrics,
}

impl MemoryPoolBenchmark {
    pub fn new() -> Self {
        Self {
            metrics: AllocationMetrics::default(),
        }
    }
    
    /// Run a benchmark comparing pooled vs non-pooled allocations
    pub fn run_comparison(&mut self, iterations: usize) -> Result<BenchmarkResults> {
        // Test different allocation sizes
        let sizes = vec![
            1024,        // 1KB
            10240,       // 10KB  
            102400,      // 100KB
            1048576,     // 1MB
            10485760,    // 10MB
        ];
        
        let mut results = BenchmarkResults::default();
        
        for &size in &sizes {
            // Benchmark non-pooled allocations
            let start = Instant::now();
            for _ in 0..iterations {
                let _vec: Vec<f32> = vec![0.0; size / 4]; // f32 is 4 bytes
            }
            let non_pooled_time = start.elapsed();
            
            // Benchmark pooled allocations
            let mut pool = super::memory_pool::TensorMemoryPool::new();
            let start = Instant::now();
            for _ in 0..iterations {
                let buffer = pool.get_buffer(size / 4);
                pool.return_buffer(buffer);
            }
            let pooled_time = start.elapsed();
            
            // Calculate improvement
            let improvement = 100.0 * (1.0 - pooled_time.as_secs_f64() / non_pooled_time.as_secs_f64());
            
            results.add_result(size, non_pooled_time, pooled_time, improvement);
        }
        
        Ok(results)
    }
    
    /// Measure allocation overhead in a typical transformer workload
    pub fn measure_transformer_overhead(&mut self) -> Result<f64> {
        let hidden_size = 4096;
        let seq_len = 512;
        let intermediate_size = 11008;
        let num_layers = 32;
        let iterations = 100;
        
        // Simulate transformer forward pass allocations
        let start = Instant::now();
        let mut total_allocated = 0u64;
        
        for _ in 0..iterations {
            for _ in 0..num_layers {
                // Attention allocations
                let _q = vec![0.0f32; seq_len * hidden_size];
                let _k = vec![0.0f32; seq_len * hidden_size];
                let _v = vec![0.0f32; seq_len * hidden_size];
                let _scores = vec![0.0f32; seq_len * seq_len];
                let _attn_out = vec![0.0f32; seq_len * hidden_size];
                
                // FFN allocations
                let _gate = vec![0.0f32; seq_len * intermediate_size];
                let _up = vec![0.0f32; seq_len * intermediate_size];
                let _ffn_out = vec![0.0f32; seq_len * hidden_size];
                
                // Track allocations
                total_allocated += ((seq_len * hidden_size * 5 + seq_len * seq_len + seq_len * intermediate_size * 2) * 4) as u64; // f32 = 4 bytes
            }
        }
        
        let allocation_time = start.elapsed();
        let total_compute_time = Duration::from_secs(10); // Assume 10s total compute
        
        let overhead_percentage = 100.0 * allocation_time.as_secs_f64() / total_compute_time.as_secs_f64();
        
        self.metrics.total_allocations = (iterations * num_layers * 8) as u64;
        self.metrics.total_bytes_allocated = total_allocated;
        self.metrics.allocation_time = allocation_time;
        
        Ok(overhead_percentage)
    }
    
    /// Get current metrics
    pub fn metrics(&self) -> &AllocationMetrics {
        &self.metrics
    }
}

/// Benchmark results container
#[derive(Debug, Default)]
pub struct BenchmarkResults {
    pub results: Vec<SizeResult>,
}

#[derive(Debug)]
pub struct SizeResult {
    pub size: usize,
    pub non_pooled_time: Duration,
    pub pooled_time: Duration,
    pub improvement_percent: f64,
}

impl BenchmarkResults {
    fn add_result(&mut self, size: usize, non_pooled: Duration, pooled: Duration, improvement: f64) {
        self.results.push(SizeResult {
            size,
            non_pooled_time: non_pooled,
            pooled_time: pooled,
            improvement_percent: improvement,
        });
    }
    
    pub fn print_summary(&self) {
        println!("\n=== Memory Pool Performance Comparison ===");
        println!("{:<12} {:>15} {:>15} {:>12}", "Size", "Non-Pooled (ms)", "Pooled (ms)", "Improvement");
        println!("{:-<60}", "");
        
        for result in &self.results {
            println!("{:<12} {:>15.2} {:>15.2} {:>11.1}%",
                format_size(result.size),
                result.non_pooled_time.as_secs_f64() * 1000.0,
                result.pooled_time.as_secs_f64() * 1000.0,
                result.improvement_percent
            );
        }
        
        let avg_improvement = self.results.iter()
            .map(|r| r.improvement_percent)
            .sum::<f64>() / self.results.len() as f64;
            
        println!("{:-<60}", "");
        println!("Average Improvement: {:.1}%", avg_improvement);
    }
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_benchmark() {
        let mut benchmark = MemoryPoolBenchmark::new();
        let results = benchmark.run_comparison(100).unwrap();
        
        // Should show improvement for all sizes
        for result in &results.results {
            assert!(result.improvement_percent > 0.0, 
                "Pool should be faster for size {}", result.size);
        }
    }
    
    #[test]
    fn test_transformer_overhead_measurement() {
        let mut benchmark = MemoryPoolBenchmark::new();
        let overhead = benchmark.measure_transformer_overhead().unwrap();
        
        println!("Measured allocation overhead: {:.2}%", overhead);
        
        // Check metrics were recorded
        let metrics = benchmark.metrics();
        assert!(metrics.total_allocations > 0);
        assert!(metrics.total_bytes_allocated > 0);
        assert!(metrics.allocation_time.as_secs_f64() > 0.0);
    }
}