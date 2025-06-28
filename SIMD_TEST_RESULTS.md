# SIMD Performance Test Results

## Summary

We successfully loaded the model and tested SIMD performance impact on Woolly.

## Model Loading

**Correct API endpoint**: `/api/v1/models/{model_name}/load`

Example:
```bash
curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Performance Results

### Single Token Generation Time

| Configuration | Time | Tokens/sec |
|--------------|------|------------|
| SIMD Enabled | 68.90s | 0.0145 |
| SIMD Disabled | 36.76s | 0.0272 |

### Performance Impact

- **SIMD makes performance 1.87x WORSE**
- Disabling SIMD provides a **1.87x speedup**

## Recommendations

1. **Immediate Action**: Set `WOOLLY_DISABLE_SIMD=1` for better performance
2. **Root Cause**: SIMD implementation has overhead from:
   - Memory allocations in hot paths
   - Runtime CPU feature detection
   - Excessive data copying

## How to Run Woolly with Best Performance

```bash
# Start server with SIMD disabled
WOOLLY_DISABLE_SIMD=1 ./target/release/woolly-server

# Load model
curl -X POST http://localhost:8080/api/v1/models/granite-3.3-8b-instruct-Q4_K_M/load -d '{}'

# Run inference
curl -X POST http://localhost:8080/api/v1/inference/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

## Next Steps

While disabling SIMD helps, performance is still far from the target:
- Current: 0.027 tokens/sec (with SIMD disabled)
- Target: >15 tokens/sec
- Gap: Still 550x slower than target

The main bottlenecks remain:
1. GGUF dequantization (90+ seconds during model loading)
2. Manual attention loops instead of BLAS
3. Memory management issues

See `WOOLLY_PERFORMANCE_ANALYSIS_OVERVIEW.md` for the complete optimization roadmap.