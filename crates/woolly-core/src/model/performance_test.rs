//! Performance testing and benchmarking for optimized transformer

use std::time::Instant;
use crate::{Result, CoreError};
use crate::model::{optimized_transformer::OptimizedTransformer, lazy_transformer::LazyTransformer};
use crate::model::transformer::TransformerConfig;
use crate::tensor_utils_optimized::{matmul_fast, rms_norm_fast};
use crate::tensor_utils::{matmul, tensor_from_slice};
use crate::model::memory_pool::TensorMemoryPool;
use woolly_tensor::Shape;

/// Performance test suite for optimized implementations
pub struct PerformanceTestSuite {
    pool: TensorMemoryPool,
    results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub optimized_time_ms: f64,
    pub baseline_time_ms: f64,
    pub speedup: f64,
    pub ops_per_second: f64,
}

impl PerformanceTestSuite {
    pub fn new() -> Self {
        Self {
            pool: TensorMemoryPool::new(),
            results: Vec::new(),
        }
    }
    
    /// Run comprehensive performance tests
    pub fn run_all_tests(&mut self) -> Result<()> {
        println!("🚀 Starting Woolly Performance Test Suite");
        println!("==========================================");
        
        // Matrix multiplication benchmarks
        self.benchmark_matmul_small()?;
        self.benchmark_matmul_medium()?;
        self.benchmark_matmul_large()?;
        
        // Normalization benchmarks
        self.benchmark_rms_norm()?;
        
        // Memory pool benchmarks
        self.benchmark_memory_pool()?;
        
        // End-to-end transformer benchmarks
        self.benchmark_transformer_layer()?;
        
        self.print_results();
        
        Ok(())
    }
    
    /// Benchmark small matrix multiplication (typical attention heads)
    fn benchmark_matmul_small(&mut self) -> Result<()> {
        let sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)];
        
        for (m, n, k) in sizes {
            let test_name = format!("MatMul_{}x{}x{}", m, n, k);
            
            // Generate test data
            let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.01).collect();
            let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.01).collect();
            
            let a = tensor_from_slice(&a_data, Shape::matrix(m, k))?;
            let b = tensor_from_slice(&b_data, Shape::matrix(k, n))?;
            
            // Benchmark optimized version
            let start = Instant::now();
            for _ in 0..100 {
                let _result = matmul_fast(&a, &b, &mut self.pool, true)?;
            }
            let optimized_time = start.elapsed().as_secs_f64() * 1000.0;
            
            // Benchmark baseline version
            let start = Instant::now();
            for _ in 0..100 {
                let _result = matmul(&a, &b)?;
            }
            let baseline_time = start.elapsed().as_secs_f64() * 1000.0;
            
            let speedup = baseline_time / optimized_time;
            let ops_per_second = (100.0 * 2.0 * m as f64 * n as f64 * k as f64) / (optimized_time / 1000.0);
            
            self.results.push(BenchmarkResult {
                test_name,
                optimized_time_ms: optimized_time,
                baseline_time_ms: baseline_time,
                speedup,
                ops_per_second,
            });
        }
        
        Ok(())
    }
    
    /// Benchmark medium matrix multiplication (FFN layers)
    fn benchmark_matmul_medium(&mut self) -> Result<()> {
        let sizes = [(512, 2048, 512), (1024, 4096, 1024), (2048, 8192, 2048)];
        
        for (m, n, k) in sizes {
            let test_name = format!("MatMul_FFN_{}x{}x{}", m, n, k);
            
            // Generate test data
            let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.001).collect();
            let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.001).collect();
            
            let a = tensor_from_slice(&a_data, Shape::matrix(m, k))?;
            let b = tensor_from_slice(&b_data, Shape::matrix(k, n))?;
            
            // Benchmark optimized version (fewer iterations for large matrices)
            let iterations = if m * n * k > 1_000_000 { 10 } else { 50 };
            
            let start = Instant::now();
            for _ in 0..iterations {
                let _result = matmul_fast(&a, &b, &mut self.pool, true)?;
            }
            let optimized_time = start.elapsed().as_secs_f64() * 1000.0;
            
            // Benchmark baseline version
            let start = Instant::now();
            for _ in 0..iterations {
                let _result = matmul(&a, &b)?;
            }
            let baseline_time = start.elapsed().as_secs_f64() * 1000.0;
            
            let speedup = baseline_time / optimized_time;
            let ops_per_second = (iterations as f64 * 2.0 * m as f64 * n as f64 * k as f64) / (optimized_time / 1000.0);
            
            self.results.push(BenchmarkResult {
                test_name,
                optimized_time_ms: optimized_time,
                baseline_time_ms: baseline_time,
                speedup,
                ops_per_second,
            });
        }
        
        Ok(())
    }
    
    /// Benchmark large matrix multiplication (LM head)
    fn benchmark_matmul_large(&mut self) -> Result<()> {
        let sizes = [(1, 32000, 4096), (4, 32000, 4096), (8, 32000, 4096)];
        
        for (m, n, k) in sizes {
            let test_name = format!("MatMul_LMHead_{}x{}x{}", m, n, k);
            
            // Generate test data
            let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.0001).collect();
            let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.0001).collect();
            
            let a = tensor_from_slice(&a_data, Shape::matrix(m, k))?;
            let b = tensor_from_slice(&b_data, Shape::matrix(k, n))?;
            
            // Benchmark optimized version (single iteration for very large matrices)
            let iterations = 5;
            
            let start = Instant::now();
            for _ in 0..iterations {
                let _result = matmul_fast(&a, &b, &mut self.pool, true)?;
            }
            let optimized_time = start.elapsed().as_secs_f64() * 1000.0;
            
            // Benchmark baseline version
            let start = Instant::now();
            for _ in 0..iterations {
                let _result = matmul(&a, &b)?;
            }
            let baseline_time = start.elapsed().as_secs_f64() * 1000.0;
            
            let speedup = baseline_time / optimized_time;
            let ops_per_second = (iterations as f64 * 2.0 * m as f64 * n as f64 * k as f64) / (optimized_time / 1000.0);
            
            self.results.push(BenchmarkResult {
                test_name,
                optimized_time_ms: optimized_time,
                baseline_time_ms: baseline_time,
                speedup,
                ops_per_second,
            });
        }
        
        Ok(())
    }
    
    /// Benchmark RMS normalization
    fn benchmark_rms_norm(&mut self) -> Result<()> {
        let configs = [(1, 4096), (8, 4096), (32, 4096), (128, 4096)];
        
        for (seq_len, hidden_size) in configs {
            let test_name = format!("RMSNorm_{}x{}", seq_len, hidden_size);
            
            // Generate test data
            let input_data: Vec<f32> = (0..seq_len*hidden_size).map(|i| (i as f32) * 0.01).collect();
            let weight_data: Vec<f32> = (0..hidden_size).map(|i| 1.0 + (i as f32) * 0.001).collect();
            
            let input = tensor_from_slice(&input_data, Shape::matrix(seq_len, hidden_size))?;
            let weight = tensor_from_slice(&weight_data, Shape::vector(hidden_size))?;
            
            let iterations = 1000;
            
            // Benchmark optimized version
            let start = Instant::now();
            for _ in 0..iterations {
                let _result = rms_norm_fast(&input, &weight, 1e-5, &mut self.pool)?;
            }
            let optimized_time = start.elapsed().as_secs_f64() * 1000.0;
            
            // Benchmark baseline version
            let start = Instant::now();
            for _ in 0..iterations {
                let _result = crate::tensor_utils::rms_norm(&input, &weight, 1e-5)?;
            }
            let baseline_time = start.elapsed().as_secs_f64() * 1000.0;
            
            let speedup = baseline_time / optimized_time;
            let ops_per_second = (iterations as f64 * seq_len as f64 * hidden_size as f64) / (optimized_time / 1000.0);
            
            self.results.push(BenchmarkResult {
                test_name,
                optimized_time_ms: optimized_time,
                baseline_time_ms: baseline_time,
                speedup,
                ops_per_second,
            });
        }
        
        Ok(())
    }
    
    /// Benchmark memory pool efficiency
    fn benchmark_memory_pool(&mut self) -> Result<()> {
        let test_name = "MemoryPool_Allocation".to_string();
        let sizes = [1024, 4096, 16384, 65536];
        let iterations = 10000;
        
        // Benchmark with memory pool
        let start = Instant::now();
        for _ in 0..iterations {
            for &size in &sizes {
                let buffer = self.pool.get_buffer(size);
                self.pool.return_buffer(buffer);
            }
        }
        let optimized_time = start.elapsed().as_secs_f64() * 1000.0;
        
        // Benchmark without memory pool (direct allocation)
        let start = Instant::now();
        for _ in 0..iterations {
            for &size in &sizes {
                let _buffer = vec![0.0f32; size];
            }
        }
        let baseline_time = start.elapsed().as_secs_f64() * 1000.0;
        
        let speedup = baseline_time / optimized_time;
        let ops_per_second = (iterations as f64 * sizes.len() as f64) / (optimized_time / 1000.0);
        
        self.results.push(BenchmarkResult {
            test_name,
            optimized_time_ms: optimized_time,
            baseline_time_ms: baseline_time,
            speedup,
            ops_per_second,
        });
        
        Ok(())
    }
    
    /// Benchmark transformer layer computation
    fn benchmark_transformer_layer(&mut self) -> Result<()> {
        let test_name = "TransformerLayer_Forward".to_string();
        
        // Simulate a typical transformer layer forward pass
        let seq_len = 8;
        let hidden_size = 4096;
        let intermediate_size = 11008;
        let num_operations = 100;
        
        // Attention projections: Q, K, V, O
        let q_data: Vec<f32> = (0..seq_len*hidden_size).map(|i| (i as f32) * 0.001).collect();
        let k_data: Vec<f32> = (0..hidden_size*hidden_size).map(|i| (i as f32) * 0.001).collect();
        
        let q_input = tensor_from_slice(&q_data, Shape::matrix(seq_len, hidden_size))?;
        let k_weight = tensor_from_slice(&k_data, Shape::matrix(hidden_size, hidden_size))?;
        
        // FFN projections
        let ffn_gate_data: Vec<f32> = (0..hidden_size*intermediate_size).map(|i| (i as f32) * 0.0001).collect();
        let ffn_up_data: Vec<f32> = (0..hidden_size*intermediate_size).map(|i| (i as f32) * 0.0001).collect();
        let ffn_down_data: Vec<f32> = (0..intermediate_size*hidden_size).map(|i| (i as f32) * 0.0001).collect();
        
        let ffn_gate = tensor_from_slice(&ffn_gate_data, Shape::matrix(hidden_size, intermediate_size))?;
        let ffn_up = tensor_from_slice(&ffn_up_data, Shape::matrix(hidden_size, intermediate_size))?;
        let ffn_down = tensor_from_slice(&ffn_down_data, Shape::matrix(intermediate_size, hidden_size))?;
        
        // Benchmark optimized layer
        let start = Instant::now();
        for _ in 0..num_operations {
            // Attention
            let _q_proj = matmul_fast(&q_input, &k_weight, &mut self.pool, true)?;
            let _k_proj = matmul_fast(&q_input, &k_weight, &mut self.pool, true)?;
            let _v_proj = matmul_fast(&q_input, &k_weight, &mut self.pool, true)?;
            let _attn_out = matmul_fast(&q_input, &k_weight, &mut self.pool, true)?;
            
            // FFN
            let gate_proj = matmul_fast(&q_input, &ffn_gate, &mut self.pool, true)?;
            let up_proj = matmul_fast(&q_input, &ffn_up, &mut self.pool, true)?;
            let _swiglu_out = crate::tensor_utils_optimized::swiglu_fast(&gate_proj, &up_proj, &mut self.pool)?;
            let _ffn_out = matmul_fast(&gate_proj, &ffn_down, &mut self.pool, true)?; // Reuse gate_proj buffer
        }
        let optimized_time = start.elapsed().as_secs_f64() * 1000.0;
        
        // Benchmark baseline layer
        let start = Instant::now();
        for _ in 0..num_operations {
            // Attention
            let _q_proj = matmul(&q_input, &k_weight)?;
            let _k_proj = matmul(&q_input, &k_weight)?;
            let _v_proj = matmul(&q_input, &k_weight)?;
            let _attn_out = matmul(&q_input, &k_weight)?;
            
            // FFN
            let gate_proj = matmul(&q_input, &ffn_gate)?;
            let up_proj = matmul(&q_input, &ffn_up)?;
            let _swiglu_out = crate::tensor_utils::swiglu(&gate_proj, &up_proj)?;
            let _ffn_out = matmul(&gate_proj, &ffn_down)?;
        }
        let baseline_time = start.elapsed().as_secs_f64() * 1000.0;
        
        let speedup = baseline_time / optimized_time;
        
        // Calculate total ops (rough estimate)
        let attn_ops = 4 * seq_len * hidden_size * hidden_size;
        let ffn_ops = seq_len * (hidden_size * intermediate_size * 2 + intermediate_size * hidden_size);
        let total_ops = (attn_ops + ffn_ops) as f64;
        let ops_per_second = (num_operations as f64 * total_ops) / (optimized_time / 1000.0);
        
        self.results.push(BenchmarkResult {
            test_name,
            optimized_time_ms: optimized_time,
            baseline_time_ms: baseline_time,
            speedup,
            ops_per_second,
        });
        
        Ok(())
    }
    
    /// Print comprehensive performance results
    fn print_results(&self) {
        println!("\n📊 Performance Test Results");
        println!("==========================");
        
        let mut total_speedup = 0.0;
        let mut count = 0;
        
        for result in &self.results {
            println!("\n🔹 {}", result.test_name);
            println!("   Optimized: {:.2}ms", result.optimized_time_ms);
            println!("   Baseline:  {:.2}ms", result.baseline_time_ms);
            println!("   Speedup:   {:.2}x", result.speedup);
            println!("   Ops/sec:   {:.2e}", result.ops_per_second);
            
            total_speedup += result.speedup;
            count += 1;
        }
        
        let avg_speedup = total_speedup / count as f64;
        
        println!("\n🏆 Summary");
        println!("=========");
        println!("Average Speedup: {:.2}x", avg_speedup);
        
        // Performance targets based on analysis
        if avg_speedup >= 3.0 {
            println!("✅ EXCELLENT: Achieved target performance for llama.cpp comparison");
        } else if avg_speedup >= 2.0 {
            println!("✅ GOOD: Significant performance improvement achieved");
        } else if avg_speedup >= 1.5 {
            println!("⚠️  MODERATE: Some improvement, but more optimization needed");
        } else {
            println!("❌ NEEDS WORK: Optimization not effective, requires investigation");
        }
        
        // Memory efficiency analysis
        let memory_pool_result = self.results.iter()
            .find(|r| r.test_name.contains("MemoryPool"));
            
        if let Some(pool_result) = memory_pool_result {
            if pool_result.speedup >= 5.0 {
                println!("✅ Memory allocation overhead significantly reduced");
            } else {
                println!("⚠️  Memory pool efficiency could be improved");
            }
        }
        
        println!("\n🎯 Next Steps:");
        if avg_speedup < 3.0 {
            println!("   • Implement more aggressive SIMD optimizations");
            println!("   • Add specialized kernels for common matrix sizes");
            println!("   • Optimize quantized weight dequantization");
            println!("   • Implement Flash Attention for memory efficiency");
        } else {
            println!("   • Ready for integration with Ole");
            println!("   • Consider implementing quantized KV cache");
            println!("   • Add proper GGUF tokenizer integration");
        }
    }
    
    /// Get the average speedup across all tests
    pub fn average_speedup(&self) -> f64 {
        if self.results.is_empty() {
            return 1.0;
        }
        
        let total_speedup: f64 = self.results.iter().map(|r| r.speedup).sum();
        total_speedup / self.results.len() as f64
    }
    
    /// Check if performance targets are met
    pub fn meets_performance_targets(&self) -> bool {
        let avg_speedup = self.average_speedup();
        
        // Target: 2-3x speedup for readiness with Ole integration
        avg_speedup >= 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_suite() -> Result<()> {
        let mut suite = PerformanceTestSuite::new();
        
        // Run a subset of tests for CI
        suite.benchmark_matmul_small()?;
        suite.benchmark_rms_norm()?;
        suite.benchmark_memory_pool()?;
        
        // Ensure we have some performance improvement
        assert!(suite.average_speedup() > 1.0, "Optimizations should provide some speedup");
        
        Ok(())
    }
}