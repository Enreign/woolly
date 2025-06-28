# CRITICAL PERFORMANCE BUG FOUND

## The Problem

The Woolly inference engine has a **critical performance bug** that makes it ~50x slower than it should be:

1. **SIMD attention path uses manual nested loops** instead of BLAS matrix operations
2. **Only 2 BLAS calls** for entire inference (should be 80+ for 40 layers)
3. **2,812 dequantization operations** for a single token

## Impact

- Current: **75 seconds per token** (0.013 tokens/sec)
- Expected with fix: **<5 seconds per token** (>0.2 tokens/sec)

## Root Cause

In `lazy_transformer.rs`, the function `compute_simd_gqa_attention` (lines 841-1047) contains:

```rust
// Manual nested loops for attention (SLOW!)
for h in 0..num_heads {
    for i in 0..seq_len {
        for j in 0..seq_len {
            // Scalar operations
            score += queries[q_start + d] * keys[k_start + d];
        }
    }
}
```

Instead of:
```rust
// BLAS matrix multiplication (FAST!)
grouped_query_attention_blas(queries, keys, values, ...)
```

## The Fix

Replace the manual attention loops in `compute_simd_gqa_attention` with a call to `grouped_query_attention_blas`.

The BLAS implementation already exists and works! It's just not being used in the default code path.

## Why This Happened

1. Two attention implementations exist:
   - `compute_optimized_gqa_attention` - has BLAS ✓
   - `compute_simd_gqa_attention` - manual loops ✗

2. By default, SIMD is enabled, so the slow path is used

3. The BLAS attention implementation was added but not integrated into the SIMD path

## Immediate Action Required

The syntax errors in my attempted fix need to be resolved, but the solution is clear:
Make `compute_simd_gqa_attention` call `grouped_query_attention_blas` instead of using manual loops.

This single fix should improve performance by 10-50x.