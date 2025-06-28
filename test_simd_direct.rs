use woolly_core::cpu_features::{CpuFeatures, SimdDispatcher};
use woolly_core::optimized_dequantization::OptimizedDequantizer;

fn main() {
    println!("Direct SIMD Test");
    println!("================");
    
    // Check CPU features
    let features = CpuFeatures::detect();
    println!("CPU Features:");
    println!("  x86_64: {}", features.is_x86_64);
    println!("  AVX2: {}", features.has_avx2);
    println!("  AVX: {}", features.has_avx);
    println!("  SSE2: {}", features.has_sse2);
    println!("  SIMD Level: {:?}", features.best_simd_level());
    
    // Test dequantization
    let dequantizer = OptimizedDequantizer::new();
    println!("\nDequantizer ready, would use SIMD level: {:?}", 
             SimdDispatcher::new().simd_level());
    
    // Create small test data for Q4_K
    let test_size = 256; // One Q4_K block
    let mut output = vec![0.0f32; test_size];
    let input = vec![0u8; 144]; // Q4_K block size in bytes
    
    println!("\nTesting Q4_K dequantization...");
    match dequantizer.dequantize_q4_k(&input, &mut output) {
        Ok(_) => println!("✅ Q4_K dequantization successful"),
        Err(e) => println!("❌ Q4_K dequantization failed: {:?}", e),
    }
}