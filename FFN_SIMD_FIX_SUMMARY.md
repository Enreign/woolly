# FFN SIMD Dimension Mismatch Fix

## Problem Identified

The SIMD FFN computation in `tensor_utils_simd.rs` had a dimension mismatch error:
- Error: "Vector length 4096 doesn't match matrix columns 12800"
- Root cause: The `simd_ffn_forward` function was using `SimdMatVec::compute` (matrix-vector multiplication) when it should be using matrix-matrix multiplication for 2D tensors.

## The Issue

In transformers, the FFN computation involves:
1. **Gate projection**: hidden_states [seq_len, hidden_size] × gate_weight [hidden_size, intermediate_size] → [seq_len, intermediate_size]
2. **Up projection**: hidden_states [seq_len, hidden_size] × up_weight [hidden_size, intermediate_size] → [seq_len, intermediate_size]
3. **SwiGLU activation**: gate_output * silu(up_output) → [seq_len, intermediate_size]
4. **Down projection**: activated [seq_len, intermediate_size] × down_weight [intermediate_size, hidden_size] → [seq_len, hidden_size]

For this model:
- hidden_size = 4096
- intermediate_size = 12800 (which is 4096 * 3.125)

## Fix Applied

Modified `simd_ffn_forward` in `/Users/ssh/Documents/Code/ai-inference/woolly/crates/woolly-core/src/tensor_utils_simd.rs`:

1. **Added dimension checking**: Check if hidden_states is 2D (matrix) or 1D (vector)
2. **For 2D tensors**: Use `simd_matmul_general` for matrix-matrix multiplication
3. **For 1D tensors**: Keep existing `SimdMatVec::compute` for matrix-vector multiplication

### Key Changes:

```rust
// Gate and Up projections
if hidden_states.shape.ndim() == 2 {
    // Use general matrix multiplication for 2D tensors
    let gate_proj = simd_matmul_general(
        hidden_states,
        gate_weight,
        false,  // Don't transpose gate_weight
        1.0,    // alpha
        0.0,    // beta
    )?;
    gate_buffer.copy_from_slice(&gate_proj.data);
    
    let up_proj = simd_matmul_general(
        hidden_states,
        up_weight,
        false,  // Don't transpose up_weight
        1.0,    // alpha
        0.0,    // beta
    )?;
    up_buffer.copy_from_slice(&up_proj.data);
} else {
    // For 1D hidden states, use matrix-vector multiplication
    // ... existing SimdMatVec::compute code ...
}

// Down projection
if hidden_states.shape.ndim() == 2 {
    // Create tensor from swiglu_buffer with proper shape
    let swiglu_tensor = SimpleTensor::new(
        swiglu_buffer.clone(),
        Shape::matrix(seq_len, intermediate_size)
    )?;
    
    // Use general matrix multiplication for 2D tensors
    let down_proj = simd_matmul_general(
        &swiglu_tensor,
        down_weight,
        false,  // Don't transpose down_weight
        1.0,    // alpha
        0.0,    // beta
    )?;
    output_buffer.copy_from_slice(&down_proj.data);
} else {
    // For 1D case, use matrix-vector multiplication
    // ... existing SimdMatVec::compute code ...
}
```

## Testing Required

To verify the fix works:

1. **Rebuild the project**: 
   ```bash
   cargo build --release
   ```

2. **Run SIMD performance test**:
   ```bash
   ./test_simd_simple.sh
   ```

3. **Check for errors**: The "Vector length doesn't match matrix columns" error should be resolved

4. **Verify performance**: SIMD should now provide speedup for FFN operations

## Expected Outcome

With this fix:
- FFN SIMD computation will handle proper matrix dimensions
- No more dimension mismatch errors
- SIMD optimizations will work correctly for transformer FFN layers
- Performance measurements can proceed without errors

## Next Steps

1. Rebuild the project with the fix
2. Run comprehensive SIMD tests
3. Measure performance improvements
4. Compare SIMD vs non-SIMD performance for FFN operations