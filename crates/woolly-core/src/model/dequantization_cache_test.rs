#[cfg(test)]
mod dequantization_cache_integration_tests {
    use super::super::dequantization_cache::*;
    use super::super::lazy_loader::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use woolly_gguf::{GGUFLoader, GGMLType, TensorInfo};
    
    /// Mock GGUF loader for testing
    struct MockGGUFLoader {
        tensors: std::collections::HashMap<String, (Vec<u8>, TensorInfo)>,
    }
    
    impl MockGGUFLoader {
        fn new() -> Self {
            let mut tensors = std::collections::HashMap::new();
            
            // Create some mock quantized tensors
            for layer in 0..4 {
                for weight_type in &["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up", "ffn_down"] {
                    let name = format!("blk.{}.{}.weight", layer, weight_type);
                    let size = 1024; // Small size for testing
                    let data = vec![0u8; size * 18]; // Q4_0 format size
                    
                    let info = TensorInfo {
                        name: name.clone(),
                        shape: vec![256, 256],
                        ggml_type: GGMLType::Q4_0,
                        offset: 0,
                    };
                    
                    tensors.insert(name, (data, info));
                }
            }
            
            Self { tensors }
        }
    }
    
    #[test]
    fn test_cache_integration_with_lazy_loader() {
        // This test would require a proper mock implementation
        // For now, we'll test the cache in isolation
        
        let config = DequantizationCacheConfig {
            max_memory_bytes: 10 * 1024 * 1024, // 10MB
            prefetch_ahead: 2,
            use_frequency_priority: true,
            frequency_window: Duration::from_secs(60),
            enable_async_prefetch: false, // Disable for testing
        };
        
        let cache = DequantizationCache::new(config);
        
        // Test basic caching
        let result1 = cache.get_or_dequantize("test_weight_1", || {
            // Simulate dequantization
            std::thread::sleep(Duration::from_millis(10));
            Ok((vec![1.0f32; 1000], Duration::from_millis(10)))
        }).unwrap();
        
        assert_eq!(result1.len(), 1000);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
        
        // Second access should hit cache
        let start = Instant::now();
        let result2 = cache.get_or_dequantize("test_weight_1", || {
            panic!("Should not dequantize - cache hit expected");
        }).unwrap();
        let cache_access_time = start.elapsed();
        
        assert_eq!(result2, result1);
        assert_eq!(cache.stats().hits, 1);
        assert!(cache_access_time < Duration::from_millis(1)); // Should be very fast
    }
    
    #[test]
    fn test_cache_eviction_and_memory_limits() {
        let config = DequantizationCacheConfig {
            max_memory_bytes: 1000, // Very small for testing eviction
            ..Default::default()
        };
        
        let cache = DequantizationCache::new(config);
        
        // Add entries that will cause eviction
        for i in 0..10 {
            let key = format!("weight_{}", i);
            cache.get_or_dequantize(&key, || {
                Ok((vec![i as f32; 50], Duration::from_millis(1)))
            }).unwrap();
        }
        
        // Check evictions occurred
        let stats = cache.stats();
        assert!(stats.evictions > 0);
        
        // Check memory limit is respected
        let (current, max, _) = cache.memory_info();
        assert!(current <= max);
    }
    
    #[test]
    fn test_access_pattern_tracking() {
        let tracker = WeightAccessTracker::new(Duration::from_secs(60));
        
        // Simulate access pattern
        for _ in 0..5 {
            tracker.record_access("blk.0.attn_q.weight", Duration::from_millis(10));
        }
        
        for _ in 0..3 {
            tracker.record_access("blk.0.ffn_gate.weight", Duration::from_millis(15));
        }
        
        tracker.record_access("blk.1.attn_k.weight", Duration::from_millis(8));
        
        // Get hot weights
        let hot_weights = tracker.get_hot_weights(3);
        
        // Should have 2 weights with >= 3 accesses
        assert_eq!(hot_weights.len(), 2);
        assert_eq!(hot_weights[0].0, "blk.0.attn_q.weight");
        assert_eq!(hot_weights[0].1, 5);
        assert_eq!(hot_weights[1].0, "blk.0.ffn_gate.weight");
        assert_eq!(hot_weights[1].1, 3);
    }
    
    #[test]
    fn test_prefetch_queue() {
        let config = DequantizationCacheConfig::default();
        let cache = DequantizationCache::new(config);
        
        // Test prefetch functionality
        let weight_names = vec![
            "blk.{}.attn_q.weight".to_string(),
            "blk.{}.attn_k.weight".to_string(),
            "blk.{}.attn_v.weight".to_string(),
        ];
        
        cache.prefetch_layer_weights(0, weight_names);
        
        // In a real implementation, we'd verify the prefetch queue
        // For now, just ensure it doesn't panic
    }
    
    #[test]
    fn test_cache_statistics_accuracy() {
        let config = DequantizationCacheConfig::default();
        let cache = DequantizationCache::new(config);
        
        // Perform various operations
        let test_data = vec![1.0f32; 100];
        let test_size = test_data.len() * std::mem::size_of::<f32>();
        
        // Miss
        cache.get_or_dequantize("weight1", || {
            Ok((test_data.clone(), Duration::from_millis(5)))
        }).unwrap();
        
        // Hit
        cache.get_or_dequantize("weight1", || {
            panic!("Should be cached");
        }).unwrap();
        
        // Another miss
        cache.get_or_dequantize("weight2", || {
            Ok((test_data.clone(), Duration::from_millis(3)))
        }).unwrap();
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hit_rate(), 1.0 / 3.0);
        assert_eq!(stats.total_bytes_cached, test_size * 2);
        assert_eq!(stats.total_dequantization_time_saved, Duration::from_millis(5));
    }
}