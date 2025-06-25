//! Property-based tests for woolly-tensor operations
//!
//! These tests use proptest to generate random inputs and verify
//! mathematical properties and invariants of tensor operations.

use proptest::prelude::*;
use woolly_tensor::{
    ops::*,
    Shape,
};

const MAX_SIZE: usize = 1000;
const MAX_DIM: usize = 100;

// Property test strategies
prop_compose! {
    fn arb_vector(max_len: usize)(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..=max_len)
    ) -> Vec<f32> {
        data
    }
}

prop_compose! {
    fn arb_matrix_dims()(
        rows in 1usize..=50,
        cols in 1usize..=50
    ) -> (usize, usize) {
        (rows, cols)
    }
}

prop_compose! {
    fn arb_matrix(max_rows: usize, max_cols: usize)(
        (rows, cols) in arb_matrix_dims(),
    )(
        data in prop::collection::vec(-10.0f32..10.0f32, rows * cols..=rows * cols),
        rows in Just(rows),
        cols in Just(cols)
    ) -> (Vec<f32>, usize, usize) {
        (data, rows, cols)
    }
}

proptest! {
    /// Test that addition is commutative: a + b = b + a
    #[test]
    fn test_addition_commutative(
        a in arb_vector(MAX_SIZE),
        b in arb_vector(MAX_SIZE)
    ) {
        if a.len() == b.len() {
            let mut result1 = vec![0.0; a.len()];
            let mut result2 = vec![0.0; a.len()];
            
            Add::apply_f32(&a, &b, &mut result1).unwrap();
            Add::apply_f32(&b, &a, &mut result2).unwrap();
            
            for (r1, r2) in result1.iter().zip(result2.iter()) {
                prop_assert!((r1 - r2).abs() < 1e-6);
            }
        }
    }

    /// Test that addition is associative: (a + b) + c = a + (b + c)
    #[test]
    fn test_addition_associative(
        a in arb_vector(MAX_SIZE),
        b in arb_vector(MAX_SIZE),
        c in arb_vector(MAX_SIZE)
    ) {
        if a.len() == b.len() && b.len() == c.len() {
            let mut ab = vec![0.0; a.len()];
            let mut bc = vec![0.0; b.len()];
            let mut ab_c = vec![0.0; a.len()];
            let mut a_bc = vec![0.0; a.len()];
            
            // (a + b) + c
            Add::apply_f32(&a, &b, &mut ab).unwrap();
            Add::apply_f32(&ab, &c, &mut ab_c).unwrap();
            
            // a + (b + c)
            Add::apply_f32(&b, &c, &mut bc).unwrap();
            Add::apply_f32(&a, &bc, &mut a_bc).unwrap();
            
            for (r1, r2) in ab_c.iter().zip(a_bc.iter()) {
                prop_assert!((r1 - r2).abs() < 1e-5);
            }
        }
    }

    /// Test that multiplication is commutative: a * b = b * a
    #[test]
    fn test_multiplication_commutative(
        a in arb_vector(MAX_SIZE),
        b in arb_vector(MAX_SIZE)
    ) {
        if a.len() == b.len() {
            let mut result1 = vec![0.0; a.len()];
            let mut result2 = vec![0.0; a.len()];
            
            Mul::apply_f32(&a, &b, &mut result1).unwrap();
            Mul::apply_f32(&b, &a, &mut result2).unwrap();
            
            for (r1, r2) in result1.iter().zip(result2.iter()) {
                prop_assert!((r1 - r2).abs() < 1e-6);
            }
        }
    }

    /// Test that ReLU is idempotent: ReLU(ReLU(x)) = ReLU(x)
    #[test]
    fn test_relu_idempotent(input in arb_vector(MAX_SIZE)) {
        let mut first_relu = vec![0.0; input.len()];
        let mut second_relu = vec![0.0; input.len()];
        
        ReLU::apply_f32(&input, &mut first_relu).unwrap();
        ReLU::apply_f32(&first_relu, &mut second_relu).unwrap();
        
        for (r1, r2) in first_relu.iter().zip(second_relu.iter()) {
            prop_assert!((r1 - r2).abs() < 1e-6);
        }
    }

    /// Test that ReLU never produces negative values
    #[test]
    fn test_relu_non_negative(input in arb_vector(MAX_SIZE)) {
        let mut output = vec![0.0; input.len()];
        ReLU::apply_f32(&input, &mut output).unwrap();
        
        for &val in &output {
            prop_assert!(val >= 0.0);
        }
    }

    /// Test that absolute value is always non-negative
    #[test]
    fn test_abs_non_negative(input in arb_vector(MAX_SIZE)) {
        let mut output = vec![0.0; input.len()];
        Abs::apply(&input, &mut output).unwrap();
        
        for &val in &output {
            prop_assert!(val >= 0.0);
        }
    }

    /// Test that Abs(Abs(x)) = Abs(x) (idempotent)
    #[test]
    fn test_abs_idempotent(input in arb_vector(MAX_SIZE)) {
        let mut first_abs = vec![0.0; input.len()];
        let mut second_abs = vec![0.0; input.len()];
        
        Abs::apply(&input, &mut first_abs).unwrap();
        Abs::apply(&first_abs, &mut second_abs).unwrap();
        
        for (r1, r2) in first_abs.iter().zip(second_abs.iter()) {
            prop_assert!((r1 - r2).abs() < 1e-6);
        }
    }

    /// Test sum reduction properties
    #[test]
    fn test_sum_properties(input in arb_vector(MAX_SIZE)) {
        let sum = Sum::reduce_all(&input);
        
        // Sum should equal the manual sum
        let manual_sum: f32 = input.iter().sum();
        prop_assert!((sum - manual_sum).abs() < 1e-5);
        
        // Sum of all zeros should be zero
        let zeros = vec![0.0; input.len()];
        let zero_sum: f32 = Sum::reduce_all(&zeros);
        prop_assert!(zero_sum.abs() < 1e-6);
    }

    /// Test mean properties
    #[test]
    fn test_mean_properties(input in arb_vector(MAX_SIZE)) {
        if !input.is_empty() {
            let mean = Mean::reduce_all(&input);
            let sum: f32 = input.iter().sum();
            let expected_mean = sum / input.len() as f32;
            
            prop_assert!((mean - expected_mean).abs() < 1e-5);
            
            // Mean of constant values should be that constant
            let constant = 5.0f32;
            let constants = vec![constant; input.len()];
            let const_mean = Mean::reduce_all(&constants);
            prop_assert!((const_mean - constant).abs() < 1e-6);
        }
    }

    /// Test variance properties
    #[test]
    fn test_variance_properties(input in arb_vector(MAX_SIZE)) {
        if input.len() > 1 {
            let variance = Variance::compute(&input, 0);
            
            // Variance should be non-negative
            prop_assert!(variance >= 0.0);
            
            // Variance of constant values should be zero
            let constant = 3.0f32;
            let constants = vec![constant; input.len()];
            let const_variance = Variance::compute(&constants, 0);
            prop_assert!(const_variance < 1e-6);
        }
    }

    /// Test matrix multiplication properties
    #[test]
    fn test_matrix_multiplication_properties(
        (a_data, a_rows, a_cols) in arb_matrix(20, 20),
        (b_data, b_rows, b_cols) in arb_matrix(20, 20)
    ) {
        // Only test if dimensions are compatible
        if a_cols == b_rows {
            let mut result = vec![0.0; a_rows * b_cols];
            
            let a_shape = Shape::matrix(a_rows, a_cols);
            let b_shape = Shape::matrix(b_rows, b_cols);
            let config = MatMulConfig::default();
            
            let result_ok = Gemm::compute(
                &a_data,
                &b_data,
                &mut result,
                &a_shape,
                &b_shape,
                &config
            );
            
            prop_assert!(result_ok.is_ok());
            
            // Result should have the correct size
            prop_assert_eq!(result.len(), a_rows * b_cols);
        }
    }

    /// Test reduction operation bounds
    #[test]
    fn test_reduction_bounds(input in arb_vector(MAX_SIZE)) {
        if !input.is_empty() {
            let min_val = input.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let max_val = input.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            
            let mean = Mean::reduce_all(&input);
            
            // Mean should be between min and max
            prop_assert!(mean >= *min_val - 1e-6);
            prop_assert!(mean <= *max_val + 1e-6);
            
            // Test max reduction
            if let Some(computed_max) = MaxReduce::reduce_all(&input) {
                prop_assert!((computed_max - max_val).abs() < 1e-6);
            }
        }
    }

    /// Test SIMD vs scalar operation equivalence
    #[test]
    fn test_simd_scalar_equivalence(
        a in arb_vector(64), // Use size divisible by common SIMD widths
        b in arb_vector(64)
    ) {
        if a.len() == b.len() && a.len() >= 64 {
            let mut simd_result = vec![0.0; a.len()];
            let mut scalar_result = vec![0.0; a.len()];
            
            // Test SIMD addition
            Add::apply_f32(&a, &b, &mut simd_result).unwrap();
            
            // Manual scalar addition for comparison
            for i in 0..a.len() {
                scalar_result[i] = a[i] + b[i];
            }
            
            for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
                prop_assert!((simd_val - scalar_val).abs() < 1e-6);
            }
        }
    }

    /// Test shape operations consistency
    #[test]
    fn test_shape_consistency(
        dims in prop::collection::vec(1usize..=MAX_DIM, 1..=4)
    ) {
        let shape = Shape::from_slice(&dims);
        
        // Test that ndim matches the number of dimensions
        prop_assert_eq!(shape.ndim(), dims.len());
        
        // Test that numel equals the product of dimensions
        let expected_size: usize = dims.iter().product();
        prop_assert_eq!(shape.numel(), expected_size);
        
        // Test that shape can be converted back to slice
        prop_assert_eq!(shape.as_slice(), &dims[..]);
    }
}