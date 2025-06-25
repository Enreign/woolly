//! Matrix multiplication operations

use crate::backend::{Result, TensorError};
use crate::shape::Shape;

/// Matrix multiplication configurations
#[derive(Debug, Clone, Copy)]
pub struct MatMulConfig {
    /// Whether to transpose the left matrix
    pub transpose_a: bool,
    /// Whether to transpose the right matrix
    pub transpose_b: bool,
    /// Alpha scaling factor (result = alpha * A @ B + beta * C)
    pub alpha: f32,
    /// Beta scaling factor for accumulation
    pub beta: f32,
}

impl Default for MatMulConfig {
    fn default() -> Self {
        Self {
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

/// General matrix multiplication (GEMM)
pub struct Gemm;

impl Gemm {
    /// Validates shapes for matrix multiplication
    pub fn validate_shapes(
        a_shape: &Shape,
        b_shape: &Shape,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Shape> {
        if a_shape.ndim() != 2 || b_shape.ndim() != 2 {
            return Err(TensorError::invalid_shape(
                "MATMUL_NON_2D_TENSORS",
                "Matrix multiplication requires 2D tensors",
                format!("A: {:?}, B: {:?}", a_shape, b_shape),
                "matrix multiplication validation",
                "Non-2D tensor provided",
                "Ensure both tensors are 2-dimensional"
            ));
        }
        
        let (m, k1) = if transpose_a {
            (a_shape.as_slice()[1], a_shape.as_slice()[0])
        } else {
            (a_shape.as_slice()[0], a_shape.as_slice()[1])
        };
        
        let (k2, n) = if transpose_b {
            (b_shape.as_slice()[1], b_shape.as_slice()[0])
        } else {
            (b_shape.as_slice()[0], b_shape.as_slice()[1])
        };
        
        if k1 != k2 {
            return Err(TensorError::incompatible_shapes(
                "MATMUL_INNER_DIM_MISMATCH",
                format!("Matrix dimensions don't match for multiplication: ({}, {}) @ ({}, {})", 
                    m, k1, k2, n),
                "matrix multiplication",
                format!("{:?}", a_shape),
                format!("{:?}", b_shape),
                "Ensure inner dimensions match for matrix multiplication"
            ));
        }
        
        Ok(Shape::matrix(m, n))
    }
    
    /// Performs matrix multiplication: C = alpha * A @ B + beta * C
    pub fn compute(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        a_shape: &Shape,
        b_shape: &Shape,
        config: &MatMulConfig,
    ) -> Result<()> {
        let c_shape = Self::validate_shapes(a_shape, b_shape, config.transpose_a, config.transpose_b)?;
        
        let m = c_shape.as_slice()[0];
        let n = c_shape.as_slice()[1];
        let k = if config.transpose_a { a_shape.as_slice()[0] } else { a_shape.as_slice()[1] };
        
        // Optimized implementation with SIMD where possible
        Self::compute_optimized(a, b, c, a_shape, b_shape, m, n, k, config)
    }
    
    fn compute_optimized(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        a_shape: &Shape,
        b_shape: &Shape,
        m: usize,
        n: usize,
        k: usize,
        config: &MatMulConfig,
    ) -> Result<()> {
        // For small matrices, use naive implementation
        if k < 32 || m < 8 || n < 8 {
            return Self::compute_naive(a, b, c, a_shape, b_shape, m, n, k, config);
        }
        
        // Use blocked matrix multiplication for better cache efficiency
        Self::compute_blocked(a, b, c, a_shape, b_shape, m, n, k, config)
    }
    
    /// High-performance blocked matrix multiplication with SIMD optimization
    fn compute_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        a_shape: &Shape,
        b_shape: &Shape,
        m: usize,
        n: usize,
        k: usize,
        config: &MatMulConfig,
    ) -> Result<()> {
        // Block sizes optimized for typical cache sizes
        const MC: usize = 384;  // Rows of A to keep in L2 cache
        const KC: usize = 384;  // Columns of A / Rows of B to keep in L1 cache  
        const NC: usize = 4096; // Columns of B to keep in L3 cache
        
        // Handle transposition by choosing the right algorithm
        if !config.transpose_a && !config.transpose_b {
            Self::gemm_nn_blocked(a, b, c, m, n, k, config, a_shape, b_shape, MC, KC, NC)
        } else if config.transpose_a && !config.transpose_b {
            Self::gemm_tn_blocked(a, b, c, m, n, k, config, a_shape, b_shape, MC, KC, NC)
        } else if !config.transpose_a && config.transpose_b {
            Self::gemm_nt_blocked(a, b, c, m, n, k, config, a_shape, b_shape, MC, KC, NC)
        } else {
            Self::gemm_tt_blocked(a, b, c, m, n, k, config, a_shape, b_shape, MC, KC, NC)
        }
    }
    
    /// GEMM for A normal, B normal (NN)
    fn gemm_nn_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        config: &MatMulConfig,
        a_shape: &Shape,
        b_shape: &Shape,
        mc: usize,
        kc: usize,
        nc: usize,
    ) -> Result<()> {
        let lda = a_shape.as_slice()[1];
        let ldb = b_shape.as_slice()[1];
        let ldc = n;
        
        // Apply beta scaling to C if needed
        if config.beta != 0.0 && config.beta != 1.0 {
            for c_val in c.iter_mut() {
                *c_val *= config.beta;
            }
        } else if config.beta == 0.0 {
            for c_val in c.iter_mut() {
                *c_val = 0.0;
            }
        }
        
        // Main blocking loops
        for jc in (0..n).step_by(nc) {
            let nc_cur = std::cmp::min(nc, n - jc);
            
            for pc in (0..k).step_by(kc) {
                let kc_cur = std::cmp::min(kc, k - pc);
                
                for ic in (0..m).step_by(mc) {
                    let mc_cur = std::cmp::min(mc, m - ic);
                    
                    // Inner kernel for this block
                    Self::gemm_micro_kernel_nn(
                        &a[ic * lda + pc..],
                        &b[pc * ldb + jc..],
                        &mut c[ic * ldc + jc..],
                        mc_cur,
                        nc_cur,
                        kc_cur,
                        lda,
                        ldb,
                        ldc,
                        config.alpha,
                    );
                }
            }
        }
        
        Ok(())
    }
    
    /// High-performance micro kernel for small blocks using SIMD
    fn gemm_micro_kernel_nn(
        a: &[f32],
        b: &[f32], 
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
    ) {
        // Try to use optimized kernels when possible
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") && m >= 8 && n >= 8 {
                return unsafe {
                    Self::gemm_micro_kernel_8x8_avx2(a, b, c, k, lda, ldb, ldc, alpha)
                };
            }
        }
        
        // Fallback to optimized scalar kernel with better memory access
        Self::gemm_micro_kernel_scalar_optimized(a, b, c, m, n, k, lda, ldb, ldc, alpha);
    }
    
    /// AVX2 8x8 micro kernel for maximum performance
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn gemm_micro_kernel_8x8_avx2(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
    ) {
        use crate::ops::x86_64::avx2::Avx2MatMul;
        
        // Process in 8x8 blocks
        let m_blocks = 8;
        let n_blocks = 8;
        
        for i in (0..m_blocks).step_by(8) {
            for j in (0..n_blocks).step_by(8) {
                let a_block = &a[i * lda..];
                let b_block = &b[j..];
                let c_block = &mut c[i * ldc + j..];
                
                // Allocate temporary storage for the result
                let mut temp_c = vec![0.0f32; 64]; // 8x8 = 64
                
                // Load existing C values if alpha != 1.0
                if alpha != 1.0 {
                    for ii in 0..8 {
                        for jj in 0..8 {
                            temp_c[ii * 8 + jj] = c_block[ii * ldc + jj];
                        }
                    }
                }
                
                // Call the optimized kernel
                Avx2MatMul::kernel_8x8_f32(a_block, b_block, &mut temp_c, k, lda, ldb, 8);
                
                // Store results back with alpha scaling
                for ii in 0..8 {
                    for jj in 0..8 {
                        c_block[ii * ldc + jj] = alpha * temp_c[ii * 8 + jj] + c_block[ii * ldc + jj];
                    }
                }
            }
        }
    }
    
    /// Optimized scalar micro kernel with better memory access patterns
    fn gemm_micro_kernel_scalar_optimized(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
    ) {
        // Process in 4x4 blocks for better register usage
        const BLOCK_SIZE: usize = 4;
        
        for i in (0..m).step_by(BLOCK_SIZE) {
            let i_end = std::cmp::min(i + BLOCK_SIZE, m);
            
            for j in (0..n).step_by(BLOCK_SIZE) {
                let j_end = std::cmp::min(j + BLOCK_SIZE, n);
                
                // 4x4 block computation with manual loop unrolling
                for ii in i..i_end {
                    for jj in j..j_end {
                        let mut sum = 0.0f32;
                        
                        // Unroll the inner loop for better performance
                        let mut kk = 0;
                        while kk + 4 <= k {
                            sum += a[ii * lda + kk] * b[kk * ldb + jj]
                                + a[ii * lda + kk + 1] * b[(kk + 1) * ldb + jj]
                                + a[ii * lda + kk + 2] * b[(kk + 2) * ldb + jj]
                                + a[ii * lda + kk + 3] * b[(kk + 3) * ldb + jj];
                            kk += 4;
                        }
                        
                        // Handle remaining elements
                        while kk < k {
                            sum += a[ii * lda + kk] * b[kk * ldb + jj];
                            kk += 1;
                        }
                        
                        c[ii * ldc + jj] += alpha * sum;
                    }
                }
            }
        }
    }
    
    /// GEMM for A transposed, B normal (TN) - common in neural networks
    fn gemm_tn_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        config: &MatMulConfig,
        a_shape: &Shape,
        b_shape: &Shape,
        _mc: usize,
        _kc: usize,
        _nc: usize,
    ) -> Result<()> {
        // This is commonly used for neural network inference where weights are transposed
        let lda = a_shape.as_slice()[1]; // A is k x m when transposed
        let ldb = b_shape.as_slice()[1]; // B is k x n
        let ldc = n;
        
        // Apply beta scaling to C
        if config.beta != 1.0 {
            for c_val in c.iter_mut() {
                *c_val *= config.beta;
            }
        }
        
        // Optimized implementation for TN case
        for i in 0..m {
            for j in 0..n {
                // Use SIMD for the dot product when possible
                let sum;
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if is_x86_feature_detected!("avx2") && k >= 8 {
                        sum = unsafe {
                            Self::dot_product_tn_avx2(a, b, i, j, k, lda, ldb)
                        };
                    } else {
                        sum = Self::dot_product_tn_scalar(a, b, i, j, k, lda, ldb);
                    }
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
                {
                    sum = Self::dot_product_tn_scalar(a, b, i, j, k, lda, ldb);
                }
                
                c[i * ldc + j] += config.alpha * sum;
            }
        }
        
        Ok(())
    }
    
    /// SIMD dot product for TN case
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_tn_avx2(
        a: &[f32],
        b: &[f32],
        i: usize,
        j: usize,
        k: usize,
        lda: usize,
        ldb: usize,
    ) -> f32 {
        use crate::ops::x86_64::avx2::dot_product_avx2_f32;
        
        // Create slices for the row of A^T and column of B
        let a_row: Vec<f32> = (0..k).map(|l| a[l * lda + i]).collect();
        let b_col: Vec<f32> = (0..k).map(|l| b[l * ldb + j]).collect();
        
        dot_product_avx2_f32(&a_row, &b_col)
    }
    
    /// Scalar dot product for TN case
    fn dot_product_tn_scalar(
        a: &[f32],
        b: &[f32],
        i: usize,
        j: usize,
        k: usize,
        lda: usize,
        ldb: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        for l in 0..k {
            sum += a[l * lda + i] * b[l * ldb + j];
        }
        sum
    }
    
    /// GEMM for A normal, B transposed (NT)
    fn gemm_nt_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        config: &MatMulConfig,
        a_shape: &Shape,
        b_shape: &Shape,
        _mc: usize,
        _kc: usize,
        _nc: usize,
    ) -> Result<()> {
        let lda = a_shape.as_slice()[1];
        let ldb = b_shape.as_slice()[1]; // B is n x k when transposed
        let ldc = n;
        
        // Apply beta scaling
        if config.beta != 1.0 {
            for c_val in c.iter_mut() {
                *c_val *= config.beta;
            }
        }
        
        // NT is efficient because both A and B are accessed row-wise
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                
                // Direct row-row dot product
                for l in 0..k {
                    sum += a[i * lda + l] * b[j * ldb + l];
                }
                
                c[i * ldc + j] += config.alpha * sum;
            }
        }
        
        Ok(())
    }
    
    /// GEMM for A transposed, B transposed (TT)
    fn gemm_tt_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        config: &MatMulConfig,
        a_shape: &Shape,
        b_shape: &Shape,
        _mc: usize,
        _kc: usize,
        _nc: usize,
    ) -> Result<()> {
        let lda = a_shape.as_slice()[1];
        let ldb = b_shape.as_slice()[1];
        let ldc = n;
        
        // Apply beta scaling
        if config.beta != 1.0 {
            for c_val in c.iter_mut() {
                *c_val *= config.beta;
            }
        }
        
        // TT case - both matrices need column access
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                
                // Column-column dot product
                for l in 0..k {
                    sum += a[l * lda + i] * b[j * ldb + l];
                }
                
                c[i * ldc + j] += config.alpha * sum;
            }
        }
        
        Ok(())
    }
    
    fn compute_naive(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        a_shape: &Shape,
        b_shape: &Shape,
        m: usize,
        n: usize,
        k: usize,
        config: &MatMulConfig,
    ) -> Result<()> {
        // Simple naive implementation for small matrices
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                
                for l in 0..k {
                    let a_idx = if config.transpose_a {
                        l * a_shape.as_slice()[1] + i
                    } else {
                        i * a_shape.as_slice()[1] + l
                    };
                    
                    let b_idx = if config.transpose_b {
                        j * b_shape.as_slice()[1] + l
                    } else {
                        l * b_shape.as_slice()[1] + j
                    };
                    
                    sum += a[a_idx] * b[b_idx];
                }
                
                let c_idx = i * n + j;
                c[c_idx] = config.alpha * sum + config.beta * c[c_idx];
            }
        }
        
        Ok(())
    }
}

/// Batched matrix multiplication
pub struct BatchedMatMul;

impl BatchedMatMul {
    /// Validates shapes for batched matrix multiplication
    pub fn validate_shapes(
        a_shape: &Shape,
        b_shape: &Shape,
    ) -> Result<Shape> {
        if a_shape.ndim() < 2 || b_shape.ndim() < 2 {
            return Err(TensorError::invalid_shape(
                "BATCHED_MATMUL_NON_2D_TENSORS",
                "Batched matrix multiplication requires at least 2D tensors",
                format!("A: {:?}, B: {:?}", a_shape, b_shape),
                "batched matrix multiplication validation",
                "Non-2D tensor provided",
                "Ensure both tensors have at least 2 dimensions"
            ));
        }
        
        // Check batch dimensions match
        let a_batch_dims = &a_shape.as_slice()[..a_shape.ndim() - 2];
        let b_batch_dims = &b_shape.as_slice()[..b_shape.ndim() - 2];
        
        if a_batch_dims != b_batch_dims {
            return Err(TensorError::incompatible_shapes(
                "BATCHED_MATMUL_BATCH_DIM_MISMATCH",
                format!("Batch dimensions don't match: {:?} vs {:?}", a_batch_dims, b_batch_dims),
                "batched matrix multiplication",
                format!("{:?}", a_shape),
                format!("{:?}", b_shape),
                "Ensure batch dimensions are identical"
            ));
        }
        
        // Check matrix dimensions
        let k1 = a_shape.as_slice()[a_shape.ndim() - 1];
        let k2 = b_shape.as_slice()[b_shape.ndim() - 2];
        
        if k1 != k2 {
            return Err(TensorError::incompatible_shapes(
                "BATCHED_MATMUL_INNER_DIM_MISMATCH",
                format!("Matrix dimensions don't match: {} != {}", k1, k2),
                "batched matrix multiplication",
                format!("{:?}", a_shape),
                format!("{:?}", b_shape),
                "Ensure inner dimensions match for matrix multiplication"
            ));
        }
        
        // Construct output shape
        let mut output_shape = a_batch_dims.to_vec();
        output_shape.push(a_shape.as_slice()[a_shape.ndim() - 2]);
        output_shape.push(b_shape.as_slice()[b_shape.ndim() - 1]);
        
        Ok(Shape::from_slice(&output_shape))
    }
    
    /// Performs batched matrix multiplication
    pub fn compute(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        a_shape: &Shape,
        b_shape: &Shape,
    ) -> Result<()> {
        let _c_shape = Self::validate_shapes(a_shape, b_shape)?;
        
        // Calculate batch size
        let batch_size = a_shape.as_slice()[..a_shape.ndim() - 2]
            .iter()
            .product::<usize>();
        
        // Matrix dimensions
        let m = a_shape.as_slice()[a_shape.ndim() - 2];
        let k = a_shape.as_slice()[a_shape.ndim() - 1];
        let n = b_shape.as_slice()[b_shape.ndim() - 1];
        
        // Size of each matrix in the batch
        let a_matrix_size = m * k;
        let b_matrix_size = k * n;
        let c_matrix_size = m * n;
        
        // Perform matrix multiplication for each batch
        for batch_idx in 0..batch_size {
            let a_offset = batch_idx * a_matrix_size;
            let b_offset = batch_idx * b_matrix_size;
            let c_offset = batch_idx * c_matrix_size;
            
            // Extract slices for current batch
            let a_batch = &a[a_offset..a_offset + a_matrix_size];
            let b_batch = &b[b_offset..b_offset + b_matrix_size];
            let c_batch = &mut c[c_offset..c_offset + c_matrix_size];
            
            // Perform matrix multiplication for this batch
            Gemm::compute(
                a_batch,
                b_batch,
                c_batch,
                &Shape::matrix(m, k),
                &Shape::matrix(k, n),
                &MatMulConfig::default(),
            )?;
        }
        
        Ok(())
    }
}

/// Matrix-vector multiplication
pub struct MatVec;

impl MatVec {
    /// Computes matrix-vector multiplication: y = A @ x
    pub fn compute(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        matrix_shape: &Shape,
    ) -> Result<()> {
        if matrix_shape.ndim() != 2 {
            return Err(TensorError::invalid_shape(
                "MATVEC_NON_2D_MATRIX",
                "Matrix must be 2D",
                format!("{:?}", matrix_shape),
                "matrix-vector multiplication",
                "Non-2D matrix provided",
                "Ensure matrix is 2-dimensional"
            ));
        }
        
        let m = matrix_shape.as_slice()[0];
        let n = matrix_shape.as_slice()[1];
        
        if vector.len() != n {
            return Err(TensorError::incompatible_shapes(
                "MATVEC_DIM_MISMATCH",
                format!("Matrix columns {} doesn't match vector length {}", n, vector.len()),
                "matrix-vector multiplication",
                format!("{:?}", matrix_shape),
                format!("[{}]", vector.len()),
                "Ensure matrix columns match vector length"
            ));
        }
        
        if output.len() != m {
            return Err(TensorError::invalid_shape(
                "MATVEC_OUTPUT_SIZE_MISMATCH",
                format!("Output length {} doesn't match matrix rows {}", output.len(), m),
                format!("[{}]", output.len()),
                "matrix-vector multiplication",
                "Output size mismatch",
                "Ensure output length matches matrix rows"
            ));
        }
        
        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += matrix[i * n + j] * vector[j];
            }
            output[i] = sum;
        }
        
        Ok(())
    }
}

/// Outer product
pub struct Outer;

impl Outer {
    /// Computes outer product: A = x ⊗ y
    pub fn compute(
        x: &[f32],
        y: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let m = x.len();
        let n = y.len();
        
        if output.len() != m * n {
            return Err(TensorError::invalid_shape(
                "OUTER_PRODUCT_OUTPUT_SIZE_MISMATCH",
                format!("Output size {} doesn't match expected {}", output.len(), m * n),
                format!("[{}]", output.len()),
                "outer product",
                "Output size mismatch",
                "Ensure output size matches x.len() * y.len()"
            ));
        }
        
        for i in 0..m {
            for j in 0..n {
                output[i * n + j] = x[i] * y[j];
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gemm_shapes() {
        let a_shape = Shape::matrix(3, 4);
        let b_shape = Shape::matrix(4, 5);
        
        let c_shape = Gemm::validate_shapes(&a_shape, &b_shape, false, false).unwrap();
        assert_eq!(c_shape.as_slice(), &[3, 5]);
        
        // Test transposed
        let c_shape = Gemm::validate_shapes(&a_shape, &a_shape, false, true).unwrap();
        assert_eq!(c_shape.as_slice(), &[3, 3]);
    }
    
    #[test]
    fn test_simple_matmul() {
        // 2x3 matrix
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        
        // 3x2 matrix
        let b = vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];
        
        let mut c = vec![0.0; 4];
        
        Gemm::compute(
            &a, &b, &mut c,
            &Shape::matrix(2, 3),
            &Shape::matrix(3, 2),
            &MatMulConfig::default(),
        ).unwrap();
        
        // Expected: [[58, 64], [139, 154]]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }
    
    #[test]
    fn test_matvec() {
        let matrix = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let vector = vec![7.0, 8.0, 9.0];
        let mut output = vec![0.0; 2];
        
        MatVec::compute(&matrix, &vector, &mut output, &Shape::matrix(2, 3)).unwrap();
        
        // Expected: [50, 122]
        assert_eq!(output, vec![50.0, 122.0]);
    }
}