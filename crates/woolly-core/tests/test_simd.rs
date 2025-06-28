use woolly_core::cpu_features::{CpuFeatures, SimdDispatcher};
use woolly_core::optimized_dequantization::OptimizedDequantizer;

#[test]
fn test_simd_detection() {
    println!("Testing SIMD Detection");
    
    let features = CpuFeatures::detect();
    println!("CPU Features:");
    println!("  x86_64: {}", features.is_x86_64);
    println!("  AVX2: {}", features.has_avx2);
    println!("  AVX: {}", features.has_avx);
    println!("  SSE2: {}", features.has_sse2);
    println!("  SIMD Level: {:?}", features.best_simd_level());
    
    // On x86_64, we should have at least SSE2
    #[cfg(target_arch = "x86_64")]
    assert!(features.has_sse2);
}

#[test]
fn test_q4k_dequantization_with_simd() {
    let dequantizer = OptimizedDequantizer::new();
    let dispatcher = SimdDispatcher::new();
    
    println!("Using SIMD level: {:?}", dispatcher.simd_level());
    
    // Create test data for one Q4_K block
    let test_size = 256; // One Q4_K block
    let mut output = vec![0.0f32; test_size];
    let mut input = vec![0u8; 144]; // Q4_K block size
    
    // Set some non-zero values
    input[12] = 0x00; // d (f16) 
    input[13] = 0x3C; // d = 1.0 in f16
    input[14] = 0x00; // dmin (f16)
    input[15] = 0x00; // dmin = 0.0
    
    // Set some quantized values
    for i in 16..144 {
        input[i] = 0x88; // Some 4-bit values
    }
    
    let result = dequantizer.dequantize_q4_k(&input, &mut output);
    assert!(result.is_ok(), "Q4_K dequantization failed: {:?}", result);
    
    // Check some values were dequantized
    let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
    println!("Non-zero values after dequantization: {}", non_zero_count);
    assert!(non_zero_count > 0, "All values are zero after dequantization");
}