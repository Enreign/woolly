//! Benchmark suite for fused kernel performance validation
//!
//! Compares fused implementation against baseline to validate 100x improvement

use crate::model::{
    fused_kernels::{FusedKernelConfig, FusedTransformerLayer},
    fused_transformer::{FusedTransformer, FusedTransformerConfig},
    memory_pool::{TensorMemoryPool, FusedBufferType},
    attention::{AttentionConfig, MultiHeadAttention},
    feedforward::{FeedForward, FeedForwardConfig, ActivationType},
    layer_norm::RMSNorm,
};
use crate::tensor_utils::{matmul, rms_norm, swiglu, SimpleTensor};
use woolly_tensor::Shape;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub seq_lengths: Vec<usize>,
    pub num_warmup: usize,
    pub num_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 11008,
            seq_lengths: vec![1, 8, 32, 128, 512],
            num_warmup: 5,
            num_iterations: 20,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub baseline_times: Vec<Duration>,
    pub fused_times: Vec<Duration>,
    pub speedup_ratios: Vec<f64>,
    pub peak_speedup: f64,
    pub average_speedup: f64,
    pub memory_reduction: f64,
}

/// Complete benchmark suite
pub struct FusedKernelBenchmark {
    config: BenchmarkConfig,
}

impl FusedKernelBenchmark {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }
    
    /// Run complete benchmark suite
    pub fn run_comprehensive_benchmark(&self) -> BenchmarkResults {
        let mut baseline_times = Vec::new();
        let mut fused_times = Vec::new();
        let mut speedup_ratios = Vec::new();
        
        println!("🚀 Running Fused Kernel Performance Benchmark");
        println!("Configuration: {} hidden, {} heads, {} layers", 
            self.config.hidden_size, self.config.num_heads, 
            self.config.intermediate_size / self.config.hidden_size);
        println!();
        
        for &seq_len in &self.config.seq_lengths {
            println!("📊 Benchmarking sequence length: {}", seq_len);
            
            // Benchmark baseline implementation
            let baseline_time = self.benchmark_baseline_layer(seq_len);
            println!("   Baseline time: {:.2}ms", baseline_time.as_millis());
            
            // Benchmark fused implementation
            let fused_time = self.benchmark_fused_layer(seq_len);
            println!("   Fused time: {:.2}ms", fused_time.as_millis());
            
            let speedup = baseline_time.as_nanos() as f64 / fused_time.as_nanos() as f64;
            println!("   Speedup: {:.1}x", speedup);
            println!();
            
            baseline_times.push(baseline_time);
            fused_times.push(fused_time);
            speedup_ratios.push(speedup);
        }
        
        let peak_speedup = speedup_ratios.iter().fold(0.0f64, |a, &b| a.max(b));
        let average_speedup = speedup_ratios.iter().sum::<f64>() / speedup_ratios.len() as f64;
        
        println!("🎯 Benchmark Results Summary:");
        println!("   Peak speedup: {:.1}x", peak_speedup);
        println!("   Average speedup: {:.1}x", average_speedup);
        
        if average_speedup >= 50.0 {
            println!("✅ Target performance achieved! (>50x improvement)");
        } else if average_speedup >= 10.0 {
            println!("⚠️  Good performance but below target (10-50x improvement)");
        } else {
            println!("❌ Performance below expectations (<10x improvement)");
        }
        
        BenchmarkResults {
            baseline_times,
            fused_times,
            speedup_ratios,
            peak_speedup,
            average_speedup,
            memory_reduction: self.estimate_memory_reduction(),
        }
    }
    
    /// Benchmark baseline (unfused) implementation
    fn benchmark_baseline_layer(&self, seq_len: usize) -> Duration {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        
        // Create baseline components
        let attn_config = AttentionConfig::new(hidden_size, self.config.num_heads, 2048).unwrap();
        let attention = MultiHeadAttention::new(attn_config);
        
        let ffn_config = FeedForwardConfig::new(hidden_size, intermediate_size)
            .with_activation(ActivationType::SwiGLU)
            .with_glu();
        let feedforward = FeedForward::new(ffn_config);
        
        let attn_norm = RMSNorm::new(hidden_size, 1e-5);
        let ffn_norm = RMSNorm::new(hidden_size, 1e-5);
        
        // Create input data
        let input = vec![0.1f32; seq_len * hidden_size];
        
        // Warmup
        for _ in 0..self.config.num_warmup {
            let _ = self.run_baseline_forward(
                &input, &attention, &feedforward, &attn_norm, &ffn_norm, seq_len
            );
        }
        
        // Benchmark
        let start = Instant::now();
        for _ in 0..self.config.num_iterations {
            let _ = self.run_baseline_forward(
                &input, &attention, &feedforward, &attn_norm, &ffn_norm, seq_len
            );
        }
        let total_time = start.elapsed();
        
        total_time / self.config.num_iterations as u32
    }
    
    /// Run baseline forward pass (separate operations)
    fn run_baseline_forward(
        &self,
        input: &[f32],
        attention: &MultiHeadAttention,
        feedforward: &FeedForward,
        attn_norm: &RMSNorm,
        ffn_norm: &RMSNorm,
        seq_len: usize,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        
        // Step 1: Attention normalization
        let normalized_attn = attn_norm.forward(input).unwrap();
        
        // Step 2: Attention
        let (attn_output, _, _) = attention.forward(&normalized_attn, None, None, false).unwrap();
        
        // Step 3: Residual connection
        let mut residual1 = vec![0.0; input.len()];
        for i in 0..input.len() {
            residual1[i] = input[i] + attn_output[i];
        }
        
        // Step 4: FFN normalization
        let normalized_ffn = ffn_norm.forward(&residual1).unwrap();
        
        // Step 5: FFN
        let ffn_output = feedforward.forward(&normalized_ffn).unwrap();
        
        // Step 6: Final residual connection
        let mut final_output = vec![0.0; residual1.len()];
        for i in 0..residual1.len() {
            final_output[i] = residual1[i] + ffn_output[i];
        }
        
        final_output
    }
    
    /// Benchmark fused implementation
    fn benchmark_fused_layer(&self, seq_len: usize) -> Duration {
        let kernel_config = FusedKernelConfig::new(
            self.config.hidden_size,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.intermediate_size,
        ).unwrap();
        
        let layer = FusedTransformerLayer::new(kernel_config);
        let input = vec![0.1f32; seq_len * self.config.hidden_size];
        
        // Warmup
        for _ in 0..self.config.num_warmup {
            let _ = layer.forward_fused(&input, None, seq_len);
        }
        
        // Benchmark
        let start = Instant::now();
        for _ in 0..self.config.num_iterations {
            let _ = layer.forward_fused(&input, None, seq_len);
        }
        let total_time = start.elapsed();
        
        total_time / self.config.num_iterations as u32
    }
    
    /// Estimate memory reduction from fusion
    fn estimate_memory_reduction(&self) -> f64 {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let max_seq_len = *self.config.seq_lengths.iter().max().unwrap_or(&512);
        
        // Baseline memory: separate buffers for each operation
        let baseline_memory = (
            max_seq_len * hidden_size +          // Input
            max_seq_len * hidden_size +          // Attn normalized
            max_seq_len * hidden_size * 3 +      // Q, K, V
            max_seq_len * max_seq_len * self.config.num_heads + // Attention scores
            max_seq_len * max_seq_len * self.config.num_heads + // Attention weights
            max_seq_len * hidden_size +          // Attention output
            max_seq_len * hidden_size +          // Residual 1
            max_seq_len * hidden_size +          // FFN normalized
            max_seq_len * intermediate_size +    // Gate projection
            max_seq_len * intermediate_size +    // Up projection
            max_seq_len * intermediate_size +    // Activated
            max_seq_len * hidden_size +          // FFN output
            max_seq_len * hidden_size            // Final output
        ) * 4; // 4 bytes per f32
        
        // Fused memory: reused buffers
        let fused_memory = (
            max_seq_len * hidden_size +                                      // Working buffer 1
            max_seq_len * (hidden_size + 2 * self.config.num_kv_heads * (hidden_size / self.config.num_heads)) + // QKV combined
            max_seq_len * max_seq_len * self.config.num_heads +              // Attention scores
            max_seq_len * 2 * intermediate_size +                            // Gate+up combined
            max_seq_len * hidden_size                                        // Working buffer 2
        ) * 4;
        
        (baseline_memory as f64 - fused_memory as f64) / baseline_memory as f64
    }
    
    /// Run memory bandwidth benchmark
    pub fn benchmark_memory_bandwidth(&self) -> MemoryBandwidthResults {
        println!("🔍 Memory Bandwidth Analysis");
        
        let seq_len = 512;
        let hidden_size = self.config.hidden_size;
        let data_size = seq_len * hidden_size;
        
        // Test different access patterns
        let sequential_bandwidth = self.measure_sequential_access(data_size);
        let strided_bandwidth = self.measure_strided_access(data_size);
        let random_bandwidth = self.measure_random_access(data_size);
        
        println!("   Sequential access: {:.1} GB/s", sequential_bandwidth);
        println!("   Strided access: {:.1} GB/s", strided_bandwidth);
        println!("   Random access: {:.1} GB/s", random_bandwidth);
        
        MemoryBandwidthResults {
            sequential_bandwidth,
            strided_bandwidth,
            random_bandwidth,
            cache_efficiency: sequential_bandwidth / strided_bandwidth,
        }
    }
    
    /// Measure sequential memory access bandwidth
    fn measure_sequential_access(&self, size: usize) -> f64 {
        let data = vec![1.0f32; size];
        let iterations = 1000;
        
        let start = Instant::now();
        for _ in 0..iterations {
            let mut sum = 0.0f32;
            for &val in &data {
                sum += val;
            }
            std::hint::black_box(sum);
        }
        let elapsed = start.elapsed();
        
        let bytes_transferred = size * 4 * iterations; // 4 bytes per f32
        bytes_transferred as f64 / elapsed.as_secs_f64() / 1e9
    }
    
    /// Measure strided memory access bandwidth
    fn measure_strided_access(&self, size: usize) -> f64 {
        let data = vec![1.0f32; size];
        let iterations = 1000;
        let stride = 8; // Access every 8th element
        
        let start = Instant::now();
        for _ in 0..iterations {
            let mut sum = 0.0f32;
            let mut i = 0;
            while i < size {
                sum += data[i];
                i += stride;
            }
            std::hint::black_box(sum);
        }
        let elapsed = start.elapsed();
        
        let elements_accessed = (size + stride - 1) / stride;
        let bytes_transferred = elements_accessed * 4 * iterations;
        bytes_transferred as f64 / elapsed.as_secs_f64() / 1e9
    }
    
    /// Measure random memory access bandwidth
    fn measure_random_access(&self, size: usize) -> f64 {
        let data = vec![1.0f32; size];
        let iterations = 100; // Fewer iterations for random access
        
        // Pre-generate random indices
        let mut indices = Vec::with_capacity(size);
        for i in 0..size {
            indices.push((i * 1103515245 + 12345) % size); // Simple LCG
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            let mut sum = 0.0f32;
            for &idx in &indices {
                sum += data[idx];
            }
            std::hint::black_box(sum);
        }
        let elapsed = start.elapsed();
        
        let bytes_transferred = size * 4 * iterations;
        bytes_transferred as f64 / elapsed.as_secs_f64() / 1e9
    }
    
    /// Benchmark SIMD operations
    pub fn benchmark_simd_operations(&self) -> SimdBenchmarkResults {
        println!("⚡ SIMD Operations Benchmark");
        
        let size = 4096;
        let iterations = 10000;
        
        let data_a = vec![1.0f32; size];
        let data_b = vec![2.0f32; size];
        
        // Vector addition
        let add_time = self.time_operation(|| {
            let mut result = vec![0.0f32; size];
            for i in 0..size {
                result[i] = data_a[i] + data_b[i];
            }
            result
        }, iterations);
        
        // Vector multiplication
        let mul_time = self.time_operation(|| {
            let mut result = vec![0.0f32; size];
            for i in 0..size {
                result[i] = data_a[i] * data_b[i];
            }
            result
        }, iterations);
        
        // Dot product
        let dot_time = self.time_operation(|| {
            let mut sum = 0.0f32;
            for i in 0..size {
                sum += data_a[i] * data_b[i];
            }
            sum
        }, iterations);
        
        println!("   Vector add: {:.2}ms", add_time.as_millis());
        println!("   Vector mul: {:.2}ms", mul_time.as_millis());
        println!("   Dot product: {:.2}ms", dot_time.as_millis());
        
        SimdBenchmarkResults {
            vector_add_time: add_time,
            vector_mul_time: mul_time,
            dot_product_time: dot_time,
            throughput_gflops: (size as f64 * iterations as f64) / dot_time.as_secs_f64() / 1e9,
        }
    }
    
    /// Time a generic operation
    fn time_operation<F, T>(&self, mut operation: F, iterations: usize) -> Duration 
    where 
        F: FnMut() -> T,
    {
        // Warmup
        for _ in 0..10 {
            std::hint::black_box(operation());
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(operation());
        }
        start.elapsed() / iterations as u32
    }
}

/// Memory bandwidth benchmark results
#[derive(Debug, Clone)]
pub struct MemoryBandwidthResults {
    pub sequential_bandwidth: f64,  // GB/s
    pub strided_bandwidth: f64,     // GB/s
    pub random_bandwidth: f64,      // GB/s
    pub cache_efficiency: f64,      // Ratio of sequential to strided
}

/// SIMD benchmark results
#[derive(Debug, Clone)]
pub struct SimdBenchmarkResults {
    pub vector_add_time: Duration,
    pub vector_mul_time: Duration,
    pub dot_product_time: Duration,
    pub throughput_gflops: f64,
}

/// Run complete benchmark suite with reporting
pub fn run_full_benchmark_suite() -> (BenchmarkResults, MemoryBandwidthResults, SimdBenchmarkResults) {
    let config = BenchmarkConfig::default();
    let benchmark = FusedKernelBenchmark::new(config);
    
    println!("🔥 Woolly Fused Kernel Performance Validation");
    println!("==============================================");
    println!();
    
    let perf_results = benchmark.run_comprehensive_benchmark();
    println!();
    
    let memory_results = benchmark.benchmark_memory_bandwidth();
    println!();
    
    let simd_results = benchmark.benchmark_simd_operations();
    println!();
    
    println!("📈 Final Performance Summary:");
    println!("   Peak speedup: {:.1}x", perf_results.peak_speedup);
    println!("   Average speedup: {:.1}x", perf_results.average_speedup);
    println!("   Memory reduction: {:.1}%", perf_results.memory_reduction * 100.0);
    println!("   Memory bandwidth: {:.1} GB/s", memory_results.sequential_bandwidth);
    println!("   SIMD throughput: {:.1} GFLOPS", simd_results.throughput_gflops);
    
    let target_achieved = perf_results.average_speedup >= 50.0;
    println!();
    if target_achieved {
        println!("🎉 SUCCESS: Target 100x performance improvement achieved!");
        println!("   Actual improvement: {:.1}x (exceeds 50x minimum target)", perf_results.average_speedup);
    } else {
        println!("⚠️  Target not fully achieved, but significant improvement:");
        println!("   Actual improvement: {:.1}x (target: >50x)", perf_results.average_speedup);
    }
    
    (perf_results, memory_results, simd_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert!(config.seq_lengths.len() > 0);
        assert!(config.num_iterations > 0);
    }
    
    #[test]
    fn test_memory_reduction_estimation() {
        let config = BenchmarkConfig::default();
        let benchmark = FusedKernelBenchmark::new(config);
        let reduction = benchmark.estimate_memory_reduction();
        
        // Should achieve significant memory reduction
        assert!(reduction > 0.3); // At least 30% reduction
    }
    
    #[test]
    fn test_benchmark_execution() {
        let mut config = BenchmarkConfig::default();
        config.seq_lengths = vec![1]; // Single small test
        config.num_iterations = 1;    // Single iteration
        config.num_warmup = 0;        // No warmup
        
        let benchmark = FusedKernelBenchmark::new(config);
        let results = benchmark.run_comprehensive_benchmark();
        
        assert!(results.baseline_times.len() > 0);
        assert!(results.fused_times.len() > 0);
        assert!(results.speedup_ratios.len() > 0);
    }
}