# Fix for SIMD Disable Flag

## Problem
The SIMD disable flag wasn't working because the compute path in `lazy_transformer.rs` was always calling the SIMD functions (`compute_simd_gqa_attention` and `compute_simd_swiglu_ffn`), even when the `WOOLLY_DISABLE_SIMD` environment variable was set.

## Root Cause
The errors showed dimension mismatches:
1. K projection: Vector length 4096 doesn't match matrix columns 1024
2. Q projection: Vector length 20480 doesn't match matrix columns 4096

This happened because GQA (Grouped Query Attention) uses different dimensions for K/V (1024) vs Q (4096), and the SIMD path wasn't being bypassed when disabled.

## Solution
Modified `lazy_transformer.rs` to check the `WOOLLY_DISABLE_SIMD` environment variable at runtime and choose the appropriate computation path:

1. In `process_layer` method, added runtime check for SIMD:
   ```rust
   let use_simd = std::env::var("WOOLLY_DISABLE_SIMD")
       .map(|v| v != "1" && v.to_lowercase() != "true")
       .unwrap_or(true);
   ```

2. For attention computation:
   - If SIMD enabled: calls `compute_simd_gqa_attention`
   - If SIMD disabled: calls `compute_optimized_gqa_attention`

3. For FFN computation:
   - If SIMD enabled: calls `compute_simd_swiglu_ffn`
   - If SIMD disabled: calls `compute_swiglu_ffn`

## Changes Made
- Modified `/Users/ssh/Documents/Code/ai-inference/woolly/crates/woolly-core/src/model/lazy_transformer.rs`
- Added runtime SIMD detection in two places:
  1. Line ~114: Before computing attention
  2. Line ~144: Before computing FFN

## Testing
To test the fix:
```bash
# Set environment variable to disable SIMD
export WOOLLY_DISABLE_SIMD=1

# Run the test
cargo run --bin test_simple_inference_no_simd
```

The non-SIMD path (`compute_optimized_gqa_attention` and `compute_swiglu_ffn`) properly handles GQA dimensions and should work without dimension mismatch errors.

## Performance Impact
When SIMD is disabled, the inference will be slower but should work correctly. This allows:
- Testing on systems without SIMD support
- Debugging dimension issues
- Measuring baseline performance without SIMD optimizations

The goal is to get a working inference path to measure tokens/sec, even if it's slower without SIMD.