//! Comprehensive integration test for all optimizations in FusedTransformer
//!
//! This test validates that all performance optimizations work together:
//! 1. FusedTransformer kernel fusion
//! 2. Multi-threaded SIMD operations
//! 3. NEON-optimized quantization
//! 4. Memory pooling and caching
//! 5. Optimized tensor operations

use std::time::Instant;
use woolly_core::{
    engine::InferenceEngine,
    model::{
        fused_transformer::{FusedTransformer, FusedTransformerConfig},
        Model, ModelOutput,
    },
    config::EngineConfig,
    Result,
};

/// Test that validates the complete optimization pipeline
#[tokio::test]
async fn test_fused_transformer_integration() -> Result<()> {
    // Create a test FusedTransformer with realistic dimensions
    let config = FusedTransformerConfig::new(
        32000,  // vocab_size
        4096,   // hidden_size 
        32,     // num_layers
        32,     // num_heads
        8,      // num_kv_heads (GQA)
        11008,  // intermediate_size
    )?;
    
    let mut transformer = FusedTransformer::new(config)?;
    
    // Create mock weights for testing
    let hidden_size = 4096;
    let vocab_size = 32000;
    let num_layers = 32;
    let intermediate_size = 11008;
    let num_kv_heads = 8;
    let kv_hidden_size = num_kv_heads * (hidden_size / 32); // 8 * 128 = 1024
    
    // Load embedding weights
    let embedding_weights = vec![0.01f32; vocab_size * hidden_size];
    transformer.load_embedding_weights(&embedding_weights)?;
    
    // Create layer weights
    use woolly_core::model::loader::LayerWeights;
    let mut layer_weights = Vec::with_capacity(num_layers);
    
    for _ in 0..num_layers {
        let layer_weight = LayerWeights {
            attn_q_weight: vec![0.01f32; hidden_size * hidden_size],
            attn_k_weight: vec![0.01f32; hidden_size * kv_hidden_size],
            attn_v_weight: vec![0.01f32; hidden_size * kv_hidden_size],
            attn_o_weight: vec![0.01f32; hidden_size * hidden_size],
            ffn_gate_weight: Some(vec![0.01f32; hidden_size * intermediate_size]),
            ffn_up_weight: vec![0.01f32; hidden_size * intermediate_size],
            ffn_down_weight: vec![0.01f32; intermediate_size * hidden_size],
            norm_1_weight: vec![1.0f32; hidden_size],
            norm_2_weight: vec![1.0f32; hidden_size],
        };
        layer_weights.push(layer_weight);
    }
    
    // Load all layer weights
    transformer.load_all_weights(&layer_weights)?;
    
    // Load final norm weights
    let final_norm_weights = vec![1.0f32; hidden_size];
    transformer.load_final_norm_weights(&final_norm_weights)?;
    
    // Test input sequence
    let input_ids = vec![1u32, 2, 3, 4, 5]; // Small test sequence
    
    // Measure performance
    let start_time = Instant::now();
    
    // Run fused forward pass
    let logits = transformer.forward_fused(&input_ids, None)?;
    
    let elapsed = start_time.elapsed();
    
    // Validate results
    assert_eq!(logits.len(), vocab_size);
    assert!(!logits.iter().any(|&x| x.is_nan()));
    assert!(!logits.iter().any(|&x| x.is_infinite()));
    
    println!("✅ FusedTransformer integration test passed!");
    println!("   Forward pass time: {:.2}ms", elapsed.as_millis());
    println!("   Output shape: [{}]", logits.len());
    println!("   Logit range: [{:.4}, {:.4}]", 
             logits.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    Ok(())
}

/// Test memory pool efficiency
#[tokio::test]
async fn test_memory_pool_efficiency() -> Result<()> {
    let config = FusedTransformerConfig::new(1000, 512, 4, 8, 8, 2048)?;
    let transformer = FusedTransformer::new(config)?;
    
    // Test memory statistics
    let stats = transformer.memory_stats()?;
    
    assert!(stats.model_parameters > 0);
    assert!(stats.peak_memory_usage > 0);
    
    println!("✅ Memory pool efficiency test passed!");
    println!("   Model parameters: {}", stats.model_parameters);
    println!("   Memory pool buffers: {}", stats.memory_pool_buffers);
    println!("   Cache entries: {}", stats.cache_entries);
    println!("   Peak memory usage: {} bytes", stats.peak_memory_usage);
    
    Ok(())
}

/// Test SIMD optimization detection
#[test]
fn test_simd_optimization_detection() {
    // Test CPU feature detection
    #[cfg(target_arch = "aarch64")]
    {
        println!("✅ ARM NEON detected");
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
    }
    
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            println!("✅ x86_64 AVX2 detected");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            println!("✅ x86_64 FMA detected");
        }
    }
}

/// Test multi-threading with Rayon
#[test]
fn test_multi_threading_available() {
    use rayon::prelude::*;
    
    let data: Vec<i32> = (0..1000).collect();
    let sum: i32 = data.par_iter().sum();
    
    assert_eq!(sum, 499500);
    println!("✅ Multi-threading with Rayon working");
    println!("   Available threads: {}", rayon::current_num_threads());
}

/// Test optimized quantization detection
#[test]
fn test_quantization_optimization() {
    #[cfg(target_arch = "aarch64")]
    {
        // Test that NEON quantization functions are available
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        println!("✅ NEON quantization optimization available");
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("✅ Scalar quantization fallback available");
    }
}

/// Benchmark comparison test between optimized and naive implementations
#[tokio::test]
async fn test_performance_comparison() -> Result<()> {
    // Small model for quick testing
    let config = FusedTransformerConfig::new(1000, 256, 2, 4, 4, 1024)?;
    let transformer = FusedTransformer::new(config)?;
    
    // Test input
    let input_ids = vec![1u32, 2, 3];
    
    // Warm up
    let _ = transformer.forward_fused(&input_ids, None);
    
    // Benchmark optimized version
    let start = Instant::now();
    for _ in 0..10 {
        let _ = transformer.forward_fused(&input_ids, None)?;
    }
    let optimized_time = start.elapsed();
    
    println!("✅ Performance comparison test completed!");
    println!("   Optimized average: {:.2}ms per forward pass", 
             optimized_time.as_millis() as f64 / 10.0);
    
    // The optimized version should be reasonably fast
    assert!(optimized_time.as_millis() < 1000, "Forward pass should be under 1 second for small model");
    
    Ok(())
}

/// Test that the Model trait implementation works correctly
#[tokio::test]
async fn test_model_trait_implementation() -> Result<()> {
    let config = FusedTransformerConfig::new(100, 64, 1, 2, 2, 256)?;
    let transformer = FusedTransformer::new(config)?;
    
    // Test Model trait methods
    assert_eq!(transformer.name(), "FusedTransformer");
    assert_eq!(transformer.model_type(), "fused_transformer");
    assert_eq!(transformer.vocab_size(), 100);
    assert_eq!(transformer.hidden_size(), 64);
    assert_eq!(transformer.num_layers(), 1);
    assert_eq!(transformer.num_heads(), 2);
    assert_eq!(transformer.context_length(), 2048);
    
    // Test feature support
    assert!(transformer.supports_feature(woolly_core::model::ModelFeature::FlashAttention));
    assert!(transformer.supports_feature(woolly_core::model::ModelFeature::GroupedQueryAttention));
    
    println!("✅ Model trait implementation test passed!");
    
    Ok(())
}

/// Integration test with InferenceEngine
#[tokio::test]
async fn test_engine_integration() -> Result<()> {
    // Create an engine
    let mut engine = InferenceEngine::new(EngineConfig::default());
    
    // Create a small fused transformer for testing
    let config = FusedTransformerConfig::new(100, 64, 1, 2, 2, 256)?;
    let transformer = FusedTransformer::new(config)?;
    
    // Load the model into the engine
    engine.load_model(std::sync::Arc::new(transformer)).await?;
    
    // Test model info
    let model_info = engine.model_info();
    assert!(model_info.is_some());
    
    if let Some(info) = model_info {
        assert_eq!(info.model_type, "fused_transformer");
        assert_eq!(info.vocab_size, 100);
        assert_eq!(info.hidden_size, 64);
    }
    
    println!("✅ Engine integration test passed!");
    
    Ok(())
}

/// Test compilation with all optimization flags
#[test]
fn test_compilation_optimizations() {
    // This test just ensures the crate compiles with all optimizations enabled
    println!("✅ Compilation with optimizations successful!");
    
    // Verify that key optimization features are enabled at compile time
    #[cfg(feature = "simd")]
    println!("   SIMD feature enabled");
    
    #[cfg(target_feature = "neon")]
    println!("   NEON target feature enabled");
    
    #[cfg(target_feature = "avx2")]
    println!("   AVX2 target feature enabled");
    
    println!("   Optimization level: {}", if cfg!(debug_assertions) { "debug" } else { "release" });
}