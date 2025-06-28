//! BLAS-accelerated matrix multiplication for ARM64 Mac
//! Uses Accelerate framework for massive performance improvements

use crate::{Result, CoreError};
use crate::tensor_utils::SimpleTensor;
use woolly_tensor::Shape;

// Link to Accelerate framework on macOS
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    // Single-precision general matrix multiplication
    // C = alpha * A * B + beta * C
    fn cblas_sgemm(
        layout: i32,        // Row-major (101) or column-major (102)
        trans_a: i32,       // No transpose (111) or transpose (112)
        trans_b: i32,       // No transpose (111) or transpose (112)
        m: i32,             // Rows of A
        n: i32,             // Columns of B
        k: i32,             // Columns of A / Rows of B
        alpha: f32,         // Scalar multiplier for A*B
        a: *const f32,      // Matrix A
        lda: i32,           // Leading dimension of A
        b: *const f32,      // Matrix B
        ldb: i32,           // Leading dimension of B
        beta: f32,          // Scalar multiplier for C
        c: *mut f32,        // Matrix C (output)
        ldc: i32,           // Leading dimension of C
    );
    
    // Single-precision matrix-vector multiplication
    // y = alpha * A * x + beta * y
    fn cblas_sgemv(
        layout: i32,        // Row-major (101) or column-major (102)
        trans: i32,         // No transpose (111) or transpose (112)
        m: i32,             // Rows of A
        n: i32,             // Columns of A
        alpha: f32,         // Scalar multiplier
        a: *const f32,      // Matrix A
        lda: i32,           // Leading dimension of A
        x: *const f32,      // Vector x
        incx: i32,          // Stride of x
        beta: f32,          // Scalar multiplier for y
        y: *mut f32,        // Vector y (output)
        incy: i32,          // Stride of y
    );
}

// CBLAS constants
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// Check if BLAS acceleration is available
pub fn is_blas_available() -> bool {
    let available = cfg!(target_os = "macos");
    eprintln!("🔍 BLAS availability check: {}", available);
    available
}

/// Perform matrix multiplication using Accelerate BLAS
/// Returns None if BLAS is not available
pub fn matmul_blas(a: &SimpleTensor, b: &SimpleTensor) -> Option<SimpleTensor> {
    #[cfg(target_os = "macos")]
    {
        // Validate shapes
        if a.shape.dims().len() != 2 || b.shape.dims().len() != 2 {
            return None;
        }
        
        let (m, k1) = (a.shape.dims()[0], a.shape.dims()[1]);
        let (k2, n) = (b.shape.dims()[0], b.shape.dims()[1]);
        
        if k1 != k2 {
            return None;
        }
        
        let k = k1;
        
        // Allocate output
        let mut output = vec![0.0f32; m * n];
        
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,  // alpha
                a.data.as_ptr(),
                k as i32,  // lda
                b.data.as_ptr(),
                n as i32,  // ldb
                0.0,  // beta
                output.as_mut_ptr(),
                n as i32,  // ldc
            );
        }
        
        Some(SimpleTensor {
            data: output,
            shape: Shape::matrix(m, n),
        })
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Matrix-vector multiplication using BLAS
pub fn matvec_blas(matrix: &SimpleTensor, vector: &SimpleTensor) -> Option<SimpleTensor> {
    #[cfg(target_os = "macos")]
    {
        // Validate shapes
        if matrix.shape.dims().len() != 2 || vector.shape.dims().len() != 1 {
            return None;
        }
        
        let (m, n) = (matrix.shape.dims()[0], matrix.shape.dims()[1]);
        let vec_len = vector.shape.dims()[0];
        
        if n != vec_len {
            return None;
        }
        
        // Allocate output
        let mut output = vec![0.0f32; m];
        
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                1.0,  // alpha
                matrix.data.as_ptr(),
                n as i32,  // lda
                vector.data.as_ptr(),
                1,  // incx
                0.0,  // beta
                output.as_mut_ptr(),
                1,  // incy
            );
        }
        
        Some(SimpleTensor {
            data: output,
            shape: Shape::vector(m),
        })
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Batched matrix multiplication for transformer operations
pub fn matmul_batch_blas(a: &SimpleTensor, b: &SimpleTensor, batch_size: usize) -> Option<SimpleTensor> {
    #[cfg(target_os = "macos")]
    {
        if a.shape.dims().len() != 3 || b.shape.dims().len() != 3 {
            return None;
        }
        
        let (batch_a, m, k1) = (a.shape.dims()[0], a.shape.dims()[1], a.shape.dims()[2]);
        let (batch_b, k2, n) = (b.shape.dims()[0], b.shape.dims()[1], b.shape.dims()[2]);
        
        if batch_a != batch_size || batch_b != batch_size || k1 != k2 {
            return None;
        }
        
        let k = k1;
        let mut output = vec![0.0f32; batch_size * m * n];
        
        // Process each batch
        for i in 0..batch_size {
            let a_offset = i * m * k;
            let b_offset = i * k * n;
            let c_offset = i * m * n;
            
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,  // alpha
                    a.data[a_offset..].as_ptr(),
                    k as i32,  // lda
                    b.data[b_offset..].as_ptr(),
                    n as i32,  // ldb
                    0.0,  // beta
                    output[c_offset..].as_mut_ptr(),
                    n as i32,  // ldc
                );
            }
        }
        
        Some(SimpleTensor {
            data: output,
            shape: Shape::from_slice(&[batch_size, m, n]),
        })
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_blas_availability() {
        let available = is_blas_available();
        #[cfg(target_os = "macos")]
        assert!(available);
        #[cfg(not(target_os = "macos"))]
        assert!(!available);
    }
    
    #[test]
    fn test_matmul_blas() {
        let a = SimpleTensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: Shape::matrix(2, 2),
        };
        
        let b = SimpleTensor {
            data: vec![5.0, 6.0, 7.0, 8.0],
            shape: Shape::matrix(2, 2),
        };
        
        if let Some(result) = matmul_blas(&a, &b) {
            // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
            //         = [[19, 22], [43, 50]]
            assert_eq!(result.data[0], 19.0);
            assert_eq!(result.data[1], 22.0);
            assert_eq!(result.data[2], 43.0);
            assert_eq!(result.data[3], 50.0);
        }
    }
}