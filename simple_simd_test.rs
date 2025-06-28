use std::time::Instant;
use woolly_tensor::ops::simd::SimdF32;

fn main() {
    // Check SIMD status
    let simd_enabled = std::env::var("WOOLLY_DISABLE_SIMD")
        .map(|v| v != "1" && v.to_lowercase() != "true")
        .unwrap_or(true);
    
    println!("SIMD Status: {}", if simd_enabled { "ENABLED" } else { "DISABLED" });
    
    // Create test data
    let size = 1024 * 1024; // 1M elements
    let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();
    
    // Warm up
    for _ in 0..10 {
        let _ = SimdF32::dot_product(&a, &b);
    }
    
    // Benchmark
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _ = SimdF32::dot_product(&a, &b);
    }
    
    let duration = start.elapsed();
    let ops_per_sec = (iterations as f64) / duration.as_secs_f64();
    let ms_per_op = duration.as_millis() as f64 / iterations as f64;
    
    println!("Results:");
    println!("  Total time: {:?}", duration);
    println!("  Operations/sec: {:.2}", ops_per_sec);
    println!("  Time per operation: {:.2} ms", ms_per_op);
    println!("  Elements processed: {} per operation", size);
    
    // Save results
    let filename = if simd_enabled { "simd_enabled_results.txt" } else { "simd_disabled_results.txt" };
    std::fs::write(filename, format!("{:.2}", ms_per_op)).unwrap();
}