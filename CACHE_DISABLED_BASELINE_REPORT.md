# Dequantization Cache Disabled - Baseline Performance Report

## Summary

I have successfully disabled the dequantization cache in Woolly to establish a baseline performance measurement. The cache was consuming significant memory (512MB by default) and may have been causing memory pressure with the large Granite 3.3B-8B model.

## Changes Made

### 1. Cache Effectively Disabled in `lazy_loader.rs`

```rust
// Line 169-176: Cache configuration minimized
let cache_config = DequantizationCacheConfig {
    max_memory_bytes: 1, // Minimal size to effectively disable
    prefetch_ahead: 0,
    use_frequency_priority: false,
    frequency_window: Duration::from_secs(1),
    enable_async_prefetch: false,
};
```

### 2. Direct Dequantization Without Cache

```rust
// Line 190-209: Bypassed cache lookup, direct dequantization
pub fn get_tensor(&mut self, name: &str) -> Result<&[f32]> {
    // CACHE DISABLED FOR PERFORMANCE TESTING
    // Direct dequantization without cache
    let tensor = self.tensors.get_mut(name)?;
    tensor.data()  // Will dequantize on demand
}
```

## Expected Performance Impact

### Without Cache (Current State)
- **Pros:**
  - 512MB less memory usage
  - No cache management overhead
  - Simpler code path
  - More predictable performance

- **Cons:**
  - Repeated dequantization of the same weights
  - Higher CPU usage for quantized operations
  - Potentially slower for models with weight reuse

### Performance Expectations

Based on research and llama.cpp benchmarks:

1. **Reference Performance (llama.cpp)**
   - **Apple Silicon (M1/M2)**: 40-60 tokens/sec for Q4_K_M models
   - **CPU Only**: 10-20 tokens/sec for Q4_K_M models
   - **Minimum Usable**: ~2.5-3 tokens/sec (400ms per token)

2. **Q4_K_M Quantization Characteristics**
   - Best balance between size and quality
   - Uses 4-bit quantization with special handling for attention weights
   - Typical performance penalty: 20-30% vs FP16

3. **Expected Woolly Performance Without Cache**
   - Should be lower than llama.cpp initially due to:
     - Repeated dequantization overhead
     - No SIMD optimizations for dequantization
     - Memory access patterns not optimized
   - Target: 5-15 tokens/sec on modern CPUs

## Next Steps

### 1. Measure Baseline Performance
```bash
# Build the server
cargo build --release --bin woolly-server

# Run performance tests
./test_baseline_perf.sh
```

### 2. Profile Bottlenecks
- Use profiling tools to identify hot paths
- Focus on dequantization functions
- Measure memory bandwidth usage

### 3. Optimization Opportunities
- **SIMD Dequantization**: Implement vectorized dequantization
- **Block-wise Caching**: Cache only frequently accessed blocks
- **Memory Pool**: Reuse dequantization buffers
- **Prefetching**: Predictive weight loading

### 4. Compare with llama.cpp
- Run llama.cpp with same model and settings
- Compare tokens/sec metrics
- Analyze implementation differences

## Technical Details

### Dequantization Process
The Q4_K_M format stores weights in 4-bit blocks with:
- Block size: 32 elements
- Scale factors per block
- Min values per block
- Special 6-bit quantization for some attention weights

Each dequantization involves:
1. Reading quantized data (4 bits per weight)
2. Extracting scale and min values
3. Computing: `weight = scale * quantized_value + min`
4. Storing as FP32

### Memory Access Pattern
Without cache:
- Random access to weight tensors
- Poor cache locality for large models
- Repeated computation overhead

## Conclusion

Disabling the dequantization cache provides a clean baseline for performance measurement. While this will show lower performance initially, it:

1. Eliminates memory pressure issues
2. Provides clear optimization targets
3. Simplifies debugging and profiling
4. Establishes a foundation for targeted optimizations

The next phase should focus on measuring actual performance and implementing targeted optimizations based on profiling data.