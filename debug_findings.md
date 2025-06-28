# Critical Performance Issues Found in Woolly

## 1. SIMD Attention Path Doesn't Use BLAS (CRITICAL)

**Issue**: The `compute_simd_gqa_attention` function uses manual nested loops instead of BLAS
- Location: `lazy_transformer.rs` lines ~940-1040
- Impact: **This is why inference takes 75+ seconds!**
- The manual loops compute attention with O(n²) scalar operations
- BLAS is only used for final output projection (vocabulary logits)

**Evidence from logs**:
```
Total BLAS calls: 2  (should be ~80 for 40 layers!)
Total dequantization calls: 2,812
```

## 2. Wrong Attention Path is Used

The code chooses between:
- `compute_simd_gqa_attention` (when SIMD enabled - DEFAULT)
- `compute_optimized_gqa_attention` (when SIMD disabled)

Only the "optimized" version has BLAS support! The SIMD version has manual loops.

## 3. Excessive Dequantization

2,812 dequantization operations for a single token inference indicates:
- Weights are being dequantized multiple times
- Cache might not be working effectively
- Each layer dequantizes Q, K, V, O, Gate, Up, Down weights

## 4. Simple Tokenizer Fallback

Using "simple tokenization fallback" instead of proper tokenizer.

## Fix Priority

1. **IMMEDIATE**: Make SIMD attention use BLAS (est. 10-50x speedup for attention)
2. **HIGH**: Reduce redundant dequantizations 
3. **MEDIUM**: Implement proper tokenizer
4. **FUTURE**: Add multi-threading, kernel fusion

## Expected Performance After Fix

With BLAS attention properly implemented:
- Current: 75 seconds/token
- Expected: 5-15 seconds/token (5-15x improvement)
- Still short of <10s target but much closer