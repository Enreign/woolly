# SIMD Performance Analysis for Woolly

## Problem Summary
SIMD optimizations are making Woolly 5.3x SLOWER than without SIMD:
- **No SIMD**: 0.10 tokens/sec (96 seconds per token)
- **With SIMD**: 0.019 tokens/sec (50 seconds per token)
- **Ollama**: 0.6-1.2 tokens/sec (expected performance)

## Key Findings

### 1. Memory Allocation Issues
- The SIMD implementation allocates new buffers in hot paths
- `tensor_utils_simd.rs` creates new vectors for results in every operation
- No buffer reuse between operations

### 2. Excessive Copying
- Data is copied multiple times:
  - From tensor to SIMD buffer
  - From SIMD result back to tensor
  - Between different tensor representations

### 3. Architecture Detection Overhead
- Runtime CPU feature detection happens on EVERY operation
- `is_x86_feature_detected!()` is called in hot loops
- Should be done once at initialization

### 4. Small Operation Overhead
- SIMD setup overhead dominates for small tensors
- No check for minimum size before using SIMD
- Context switching between scalar and SIMD code

### 5. Missing Optimizations
- No loop unrolling in scalar fallbacks
- No prefetching for large operations
- Cache-aware blocking only kicks in at 512+ elements

## Critical Issues Found

### Issue 1: Memory Allocation in Hot Path
```rust
// In simd_matvec (tensor_utils_simd.rs:53)
let mut output = vec![0.0f32; rows];  // NEW ALLOCATION EVERY CALL!
```

### Issue 2: Feature Detection in Inner Loop
```rust
// In SimdF32::add (simd.rs:104)
if is_x86_feature_detected!("avx2") {  // CHECKED EVERY TIME!
    unsafe { Self::add_avx2(a, b, out) }
}
```

### Issue 3: Unnecessary Tensor Conversions
```rust
// Multiple conversions happening:
SimpleTensor -> &[f32] -> SIMD operations -> Vec<f32> -> SimpleTensor
```

### Issue 4: No Buffer Pooling
The memory pool exists but isn't used effectively for SIMD operations.

## Recommendations

### 1. Immediate Fixes
- Use pre-allocated buffers from memory pool
- Cache CPU feature detection at startup
- Add minimum size thresholds for SIMD (e.g., > 1024 elements)

### 2. Code Restructuring
- Move SIMD dispatch to compile-time where possible
- Use workspace buffers instead of allocating
- Implement zero-copy operations

### 3. Performance Optimizations
- Align memory for SIMD operations
- Use streaming stores for large writes
- Implement cache-aware tiling for all sizes

### 4. Testing Approach
- Profile with `perf` to identify bottlenecks
- Measure overhead of SIMD setup vs computation
- Compare with native BLAS libraries

## Next Steps

1. **Disable SIMD temporarily**: Set `WOOLLY_DISABLE_SIMD=1`
2. **Profile the code**: Use `cargo flamegraph` to identify hot spots
3. **Fix allocations**: Implement buffer reuse
4. **Benchmark incrementally**: Test each optimization

## Expected Performance
With proper SIMD implementation, we should see:
- 2-4x speedup for large matrix operations
- Minimal overhead for small operations
- Performance approaching 0.5-1.0 tokens/sec