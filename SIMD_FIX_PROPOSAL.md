# SIMD Performance Fix Proposal

## Critical Fix #1: Remove Allocations from Hot Path

### Current Problem
```rust
// tensor_utils_simd.rs:53
pub fn simd_matvec(...) -> Result<SimpleTensor> {
    let mut output = vec![0.0f32; rows];  // ALLOCATION IN HOT PATH!
    // ...
    SimpleTensor::new(output, Shape::vector(rows))  // ANOTHER ALLOCATION!
}
```

### Proposed Fix
```rust
pub fn simd_matvec_inplace(
    matrix: &SimpleTensor,
    vector: &SimpleTensor,
    output: &mut SimpleTensor,  // Pre-allocated output
    transpose: bool,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    // Work directly with output.data - no allocations!
    SimdMatVec::compute(
        &matrix.data,
        &vector.data,
        &mut output.data,  // Reuse existing buffer
        &matrix.shape,
        &config,
    )?;
    Ok(())
}
```

## Critical Fix #2: Cache Feature Detection

### Current Problem
```rust
// Called on EVERY operation!
if is_x86_feature_detected!("avx2") {
    // ...
}
```

### Proposed Fix
```rust
// Detect once at startup
lazy_static! {
    static ref CPU_FEATURES: CpuFeatures = CpuFeatures::detect();
}

struct CpuFeatures {
    has_avx2: bool,
    has_fma: bool,
    has_sse2: bool,
}

impl SimdF32 {
    #[inline(always)]
    pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
        if CPU_FEATURES.has_avx2 {  // No runtime check!
            unsafe { Self::add_avx2(a, b, out) }
        } else {
            Self::add_scalar(a, b, out);
        }
    }
}
```

## Critical Fix #3: Minimum Size Threshold

### Current Problem
SIMD overhead dominates for small tensors.

### Proposed Fix
```rust
const SIMD_THRESHOLD: usize = 1024;

pub fn simd_matvec(...) -> Result<()> {
    let total_ops = rows * cols;
    
    // Use scalar for small operations
    if total_ops < SIMD_THRESHOLD {
        return scalar_matvec(...);
    }
    
    // Use SIMD for large operations
    // ...
}
```

## Quick Test

To verify these fixes work:

1. Set `WOOLLY_DISABLE_SIMD=1` and measure baseline performance
2. Apply fixes one by one
3. Measure performance after each fix
4. Target: Match or exceed no-SIMD performance

## Expected Results

With these fixes:
- Small operations: Same speed as scalar
- Large operations: 2-4x faster than scalar
- No memory allocation overhead
- No feature detection overhead
- Overall: 5-10x faster than current SIMD implementation