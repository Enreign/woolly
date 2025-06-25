//! Integration tests for woolly-tensor operations

use woolly_tensor::ops::*;
use woolly_tensor::Result;

#[test]
fn test_comprehensive_operations() -> Result<()> {
    // Test unary operations
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut output = vec![0.0; 5];
    
    // ReLU
    ReLU::apply_f32(&input, &mut output)?;
    assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    
    // Sin and Cos
    let angles = vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI];
    let mut sin_result = vec![0.0; 3];
    let mut cos_result = vec![0.0; 3];
    
    Sin::apply(&angles, &mut sin_result)?;
    Cos::apply(&angles, &mut cos_result)?;
    
    assert!((sin_result[0] - 0.0).abs() < 1e-6);
    assert!((sin_result[1] - 1.0).abs() < 1e-6);
    assert!((cos_result[0] - 1.0).abs() < 1e-6);
    assert!((cos_result[2] - (-1.0)).abs() < 1e-6);
    
    // Test binary operations
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut result = vec![0.0; 4];
    
    // Addition with SIMD
    Add::apply_f32(&a, &b, &mut result)?;
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    
    // Multiplication with SIMD
    Mul::apply_f32(&a, &b, &mut result)?;
    assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    
    // Comparison operations
    let mut bool_result = vec![false; 4];
    Greater::apply(&a, &[2.0, 2.0, 2.0, 2.0], &mut bool_result)?;
    assert_eq!(bool_result, vec![false, false, true, true]);
    
    // Test reduction operations
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let sum = Sum::reduce_all(&data);
    assert_eq!(sum, 15.0);
    
    let mean = Mean::reduce_all(&data);
    assert_eq!(mean, 3.0);
    
    let max = MaxReduce::reduce_all(&data);
    assert_eq!(max, Some(5.0));
    
    let argmax = ArgMax::compute(&data);
    assert_eq!(argmax, Some(4));
    
    let variance = Variance::compute(&data, 0);
    assert!((variance - 2.0).abs() < 1e-6);
    
    // Test matrix multiplication
    let matrix_a = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ];
    let matrix_b = vec![
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0,
    ];
    let mut matrix_c = vec![0.0; 4];
    
    use woolly_tensor::Shape;
    
    Gemm::compute(
        &matrix_a,
        &matrix_b,
        &mut matrix_c,
        &Shape::matrix(2, 3),
        &Shape::matrix(3, 2),
        &MatMulConfig::default(),
    )?;
    
    // Expected: [[58, 64], [139, 154]]
    assert_eq!(matrix_c, vec![58.0, 64.0, 139.0, 154.0]);
    
    // Test batched matrix multiplication
    let batch_a = vec![
        // Batch 1
        1.0, 2.0,
        3.0, 4.0,
        // Batch 2
        5.0, 6.0,
        7.0, 8.0,
    ];
    let batch_b = vec![
        // Batch 1
        9.0, 10.0,
        11.0, 12.0,
        // Batch 2
        13.0, 14.0,
        15.0, 16.0,
    ];
    let mut batch_c = vec![0.0; 8];
    
    BatchedMatMul::compute(
        &batch_a,
        &batch_b,
        &mut batch_c,
        &Shape::from_slice(&[2, 2, 2]), // 2 batches of 2x2 matrices
        &Shape::from_slice(&[2, 2, 2]),
    )?;
    
    // Results should be two 2x2 matrices
    assert_eq!(batch_c.len(), 8);
    
    Ok(())
}

#[test]
fn test_simd_performance() -> Result<()> {
    // Test SIMD operations with larger arrays
    let size = 1000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    let mut result = vec![0.0; size];
    
    // Test SIMD addition
    Add::apply_f32(&a, &b, &mut result)?;
    for i in 0..size {
        assert_eq!(result[i], (i * 3) as f32);
    }
    
    // Test SIMD multiplication
    Mul::apply_f32(&a, &b, &mut result)?;
    for i in 0..size {
        assert_eq!(result[i], (i * i * 2) as f32);
    }
    
    // Test SIMD ReLU with negative numbers
    let mixed: Vec<f32> = (0..size).map(|i| i as f32 - 500.0).collect();
    let mut relu_result = vec![0.0; size];
    
    ReLU::apply_f32(&mixed, &mut relu_result)?;
    for i in 0..size {
        let expected = if i < 500 { 0.0 } else { (i - 500) as f32 };
        assert_eq!(relu_result[i], expected);
    }
    
    Ok(())
}

#[test]
fn test_activation_functions() -> Result<()> {
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let mut output = vec![0.0; 4];
    
    // Test GELU
    GeLU::apply(&input, &mut output)?;
    // GELU should be approximately: x * Φ(x)
    assert!(output[0] < 0.0); // negative input
    assert!(output[1] == 0.0); // zero input
    assert!(output[2] > 0.0); // positive input
    
    // Test Sigmoid
    Sigmoid::apply(&input, &mut output)?;
    // All outputs should be between 0 and 1
    for &val in &output {
        assert!(val > 0.0 && val < 1.0);
    }
    
    // Test Tanh
    Tanh::apply(&input, &mut output)?;
    // All outputs should be between -1 and 1
    for &val in &output {
        assert!(val > -1.0 && val < 1.0);
    }
    
    // Test Exp
    let small_input = vec![0.0, 1.0, 2.0];
    let mut exp_output = vec![0.0; 3];
    Exp::apply(&small_input, &mut exp_output)?;
    
    assert!((exp_output[0] - 1.0).abs() < 1e-6);
    assert!((exp_output[1] - std::f32::consts::E).abs() < 1e-6);
    
    Ok(())
}