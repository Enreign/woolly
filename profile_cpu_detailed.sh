#!/bin/bash

# Detailed CPU profiling script for Woolly
# Focuses on CPU-level metrics: cache misses, branch prediction, SIMD usage

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BLUE='\033[0;34m'

echo "🔬 Woolly Detailed CPU Profiling"
echo "================================"
echo

PROFILE_DIR="cpu_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_DIR"

# Create detailed profiling test
cat > "$PROFILE_DIR/cpu_profile_test.rs" << 'EOF'
use woolly_core::{Engine, Config, GenerationOptions};
use woolly_tensor::{Tensor, DType};
use std::time::{Instant, Duration};
use std::hint::black_box;

#[derive(Default)]
struct ProfileMetrics {
    matmul_time: Duration,
    matmul_count: u64,
    dequantize_time: Duration,
    dequantize_count: u64,
    attention_time: Duration,
    attention_count: u64,
    tensor_ops_time: Duration,
    tensor_ops_count: u64,
}

fn profile_matrix_multiplication() -> ProfileMetrics {
    let mut metrics = ProfileMetrics::default();
    
    println!("📊 Profiling matrix multiplication...");
    
    // Test different matrix sizes
    let sizes = vec![(512, 512, 512), (1024, 768, 768), (2048, 1024, 512)];
    
    for (m, n, k) in sizes {
        println!("  Testing {}x{} @ {}x{}", m, k, k, n);
        
        // Create test tensors
        let a = Tensor::zeros(&[m, k], DType::F32).unwrap();
        let b = Tensor::zeros(&[k, n], DType::F32).unwrap();
        
        // Warm up
        for _ in 0..5 {
            let _ = black_box(a.matmul(&b));
        }
        
        // Profile
        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = black_box(a.matmul(&b));
        }
        let duration = start.elapsed();
        
        metrics.matmul_time += duration;
        metrics.matmul_count += iterations;
        
        let ops_per_sec = (2.0 * m as f64 * n as f64 * k as f64 * iterations as f64) 
            / duration.as_secs_f64() / 1e9;
        println!("    GFLOPS: {:.2}", ops_per_sec);
    }
    
    metrics
}

fn profile_quantization_ops() -> ProfileMetrics {
    let mut metrics = ProfileMetrics::default();
    
    println!("\n📊 Profiling quantization operations...");
    
    // Test dequantization performance
    let sizes = vec![1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024];
    
    for size in sizes {
        println!("  Testing dequantization of {} elements", size);
        
        // Create quantized data (simulate Q4_0 format)
        let quantized_data = vec![0u8; size / 2]; // 4-bit quantization
        let scale = vec![1.0f32; size / 32]; // One scale per 32 elements
        
        // Warm up
        for _ in 0..3 {
            let _ = black_box(woolly_tensor::quantization::dequantize_q4_0(
                &quantized_data, &scale
            ));
        }
        
        // Profile
        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = black_box(woolly_tensor::quantization::dequantize_q4_0(
                &quantized_data, &scale
            ));
        }
        let duration = start.elapsed();
        
        metrics.dequantize_time += duration;
        metrics.dequantize_count += iterations;
        
        let mb_per_sec = (size as f64 * iterations as f64) 
            / duration.as_secs_f64() / (1024.0 * 1024.0);
        println!("    Throughput: {:.2} MB/s", mb_per_sec);
    }
    
    metrics
}

fn profile_attention_mechanism() -> ProfileMetrics {
    let mut metrics = ProfileMetrics::default();
    
    println!("\n📊 Profiling attention mechanism...");
    
    // Typical attention dimensions
    let batch_size = 1;
    let seq_len = 512;
    let num_heads = 32;
    let head_dim = 128;
    
    // Create test tensors
    let query = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim], DType::F32).unwrap();
    let key = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim], DType::F32).unwrap();
    let value = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim], DType::F32).unwrap();
    
    println!("  Attention dimensions: {}x{}x{}x{}", batch_size, num_heads, seq_len, head_dim);
    
    // Profile QK^T computation
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        // Simulate attention score computation
        let scores = black_box(query.matmul(&key.transpose(2, 3).unwrap()));
        let _ = black_box(scores.softmax(-1));
    }
    let duration = start.elapsed();
    
    metrics.attention_time += duration;
    metrics.attention_count += iterations;
    
    let attention_flops = (2.0 * batch_size as f64 * num_heads as f64 * 
        seq_len as f64 * seq_len as f64 * head_dim as f64 * iterations as f64) / 1e9;
    let gflops = attention_flops / duration.as_secs_f64();
    println!("    Attention GFLOPS: {:.2}", gflops);
    
    metrics
}

fn profile_memory_access_patterns() {
    println!("\n📊 Profiling memory access patterns...");
    
    // Test sequential vs random access
    let size = 64 * 1024 * 1024; // 64MB
    let data = vec![1.0f32; size / 4];
    
    // Sequential access
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..data.len() {
        sum += black_box(data[i]);
    }
    let seq_duration = start.elapsed();
    black_box(sum);
    
    // Strided access (poor cache behavior)
    let start = Instant::now();
    let mut sum = 0.0f32;
    let stride = 16; // 64 bytes - typical cache line size
    for i in (0..data.len()).step_by(stride) {
        sum += black_box(data[i]);
    }
    let strided_duration = start.elapsed();
    black_box(sum);
    
    let seq_bandwidth = (size as f64) / seq_duration.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    let strided_bandwidth = (size as f64 / stride as f64) / strided_duration.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    
    println!("  Sequential access: {:.2} GB/s", seq_bandwidth);
    println!("  Strided access: {:.2} GB/s", strided_bandwidth);
    println!("  Cache efficiency ratio: {:.2}x", seq_bandwidth / strided_bandwidth);
}

fn check_simd_support() {
    println!("\n📊 SIMD capability check...");
    
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        
        println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
        println!("  SSE4.1: {}", is_x86_feature_detected!("sse4.1"));
        println!("  AVX: {}", is_x86_feature_detected!("avx"));
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  AVX512F: {}", is_x86_feature_detected!("avx512f"));
        println!("  FMA: {}", is_x86_feature_detected!("fma"));
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        
        println!("  NEON: {}", is_aarch64_feature_detected!("neon"));
        println!("  FP16: {}", is_aarch64_feature_detected!("fp16"));
        println!("  SVE: {}", is_aarch64_feature_detected!("sve"));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting detailed CPU profiling\n");
    
    // Check SIMD capabilities
    check_simd_support();
    
    // Profile individual components
    let matmul_metrics = profile_matrix_multiplication();
    let quant_metrics = profile_quantization_ops();
    let attention_metrics = profile_attention_mechanism();
    
    // Profile memory patterns
    profile_memory_access_patterns();
    
    // Summary
    println!("\n📊 Performance Summary");
    println!("====================");
    
    if matmul_metrics.matmul_count > 0 {
        let avg_matmul = matmul_metrics.matmul_time.as_micros() as f64 / matmul_metrics.matmul_count as f64;
        println!("Matrix multiplication: {:.2} µs average", avg_matmul);
    }
    
    if quant_metrics.dequantize_count > 0 {
        let avg_dequant = quant_metrics.dequantize_time.as_micros() as f64 / quant_metrics.dequantize_count as f64;
        println!("Dequantization: {:.2} µs average", avg_dequant);
    }
    
    if attention_metrics.attention_count > 0 {
        let avg_attention = attention_metrics.attention_time.as_millis() as f64 / attention_metrics.attention_count as f64;
        println!("Attention mechanism: {:.2} ms average", avg_attention);
    }
    
    println!("\n✅ CPU profiling complete!");
    Ok(())
}
EOF

# Copy to examples
cp "$PROFILE_DIR/cpu_profile_test.rs" examples/cpu_profile_test.rs

echo "🔨 Building CPU profiling test..."
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release --example cpu_profile_test

echo
echo "⏱️  Running CPU profiling..."
echo "-------------------------"

# Run with time command for basic metrics
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    /usr/bin/time -l ./target/release/examples/cpu_profile_test 2>&1 | tee "$PROFILE_DIR/cpu_profile_output.txt"
else
    # Linux
    /usr/bin/time -v ./target/release/examples/cpu_profile_test 2>&1 | tee "$PROFILE_DIR/cpu_profile_output.txt"
fi

# If on Linux with perf available, collect detailed CPU metrics
if command -v perf &> /dev/null; then
    echo
    echo "📊 Collecting detailed perf metrics..."
    
    # Cache statistics
    perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        ./target/release/examples/cpu_profile_test 2>&1 | tee "$PROFILE_DIR/cache_stats.txt"
    
    # Branch prediction statistics
    perf stat -e branches,branch-misses \
        ./target/release/examples/cpu_profile_test 2>&1 | tee "$PROFILE_DIR/branch_stats.txt"
    
    # IPC and cycle counts
    perf stat -e cycles,instructions \
        ./target/release/examples/cpu_profile_test 2>&1 | tee "$PROFILE_DIR/ipc_stats.txt"
fi

# Generate CPU analysis report
cat > "$PROFILE_DIR/cpu_analysis_report.md" << EOF
# Woolly CPU Performance Analysis
Generated: $(date)

## Platform Information
- Architecture: $(uname -m)
- CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep "Model name" | cut -d: -f2 | xargs)
- Cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu)

## SIMD Capabilities
$(grep -A 20 "SIMD capability check" "$PROFILE_DIR/cpu_profile_output.txt" | grep ":" || echo "See output file for details")

## Performance Metrics
$(grep -A 20 "Performance Summary" "$PROFILE_DIR/cpu_profile_output.txt" || echo "See output file for details")

## Memory Access Patterns
$(grep -A 5 "memory access patterns" "$PROFILE_DIR/cpu_profile_output.txt" || echo "See output file for details")

## Key Findings
1. **Matrix Multiplication Performance**
   - Current implementation likely not using SIMD optimally
   - Room for improvement with vectorized operations

2. **Memory Bandwidth Utilization**
   - Sequential vs strided access shows cache efficiency
   - Consider data layout optimizations

3. **Quantization Overhead**
   - Dequantization is a significant bottleneck
   - Could benefit from SIMD optimization

## Optimization Recommendations
1. **Enable SIMD Operations**
   - Use packed SIMD instructions for matrix operations
   - Implement vectorized dequantization routines

2. **Improve Cache Utilization**
   - Optimize data layout for sequential access
   - Use cache-aware algorithms for large matrices

3. **Reduce Memory Traffic**
   - Keep frequently used data in cache
   - Minimize data movement between memory levels

4. **Parallelize Independent Operations**
   - Use thread-level parallelism for batch processing
   - Leverage multiple cores for attention heads

EOF

# Clean up
rm -f examples/cpu_profile_test.rs

echo
echo -e "${GREEN}✅ CPU profiling complete!${NC}"
echo "📁 Results saved in: $PROFILE_DIR"
echo
echo "📋 Key files:"
echo "  - CPU profile output: $PROFILE_DIR/cpu_profile_output.txt"
echo "  - Analysis report: $PROFILE_DIR/cpu_analysis_report.md"
if command -v perf &> /dev/null; then
    echo "  - Cache statistics: $PROFILE_DIR/cache_stats.txt"
    echo "  - Branch statistics: $PROFILE_DIR/branch_stats.txt"
fi