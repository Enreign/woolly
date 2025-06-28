//! Optimized quantization integration for the woolly inference engine
//!
//! This module provides a high-level interface to the optimized quantization
//! implementations, automatically selecting the best available method based
//! on the target architecture and available features.
//!
//! Performance Goals:
//! - Reduce Q4_K_M dequantization from 90s/token to 4-9s/token (10-20x speedup)
//! - Utilize NEON SIMD on ARM M4 processors
//! - Implement smart caching and bulk processing
//! - Minimize memory allocation overhead

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use crate::{CoreError, Result};

/// High-level quantization optimization manager
pub struct OptimizedQuantizationEngine {
    /// Cache for dequantized weights
    cache: Arc<RwLock<QuantizationCache>>,
    /// Performance statistics
    stats: Arc<RwLock<QuantizationStats>>,
    /// Configuration settings
    config: QuantizationConfig,
}

/// Configuration for quantization optimizations
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    /// Enable bulk processing for layers
    pub enable_bulk_processing: bool,
    /// Enable prefetching for next layers
    pub enable_prefetching: bool,
    /// Threshold for enabling SIMD (number of blocks)
    pub simd_threshold: usize,
    /// Enable performance monitoring
    pub enable_stats: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            max_cache_size_mb: 512,
            enable_bulk_processing: true,
            enable_prefetching: true,
            simd_threshold: 4,
            enable_stats: true,
        }
    }
}

/// Performance statistics for quantization operations
#[derive(Debug, Clone, Default)]
pub struct QuantizationStats {
    /// Total dequantization operations
    pub total_operations: u64,
    /// Total time spent in dequantization
    pub total_time: Duration,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Number of SIMD operations used
    pub simd_operations: u64,
    /// Number of scalar fallback operations
    pub scalar_operations: u64,
    /// Time saved by caching
    pub cache_time_saved: Duration,
    /// Average dequantization time per operation
    pub avg_time_per_op: Duration,
}

impl QuantizationStats {
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }
    
    pub fn simd_usage_rate(&self) -> f64 {
        let total = self.simd_operations + self.scalar_operations;
        if total == 0 { 0.0 } else { self.simd_operations as f64 / total as f64 }
    }
    
    pub fn update_avg_time(&mut self) {
        if self.total_operations > 0 {
            self.avg_time_per_op = self.total_time / self.total_operations as u32;
        }
    }
}

/// Internal cache implementation
struct QuantizationCache {
    /// Cached dequantized weights
    weights: HashMap<u64, CachedWeight>,
    /// LRU access order
    access_order: std::collections::VecDeque<u64>,
    /// Current memory usage in bytes
    current_memory: usize,
    /// Maximum memory usage in bytes
    max_memory: usize,
}

/// Cached weight entry
struct CachedWeight {
    /// Dequantized weight data
    data: Vec<f32>,
    /// Memory size in bytes
    size: usize,
    /// Last access time
    last_access: Instant,
    /// Access count
    access_count: u64,
    /// Time it took to dequantize
    dequantization_time: Duration,
}

impl OptimizedQuantizationEngine {
    /// Create a new optimization engine with the given configuration
    pub fn new(config: QuantizationConfig) -> Self {
        let cache = QuantizationCache {
            weights: HashMap::new(),
            access_order: std::collections::VecDeque::new(),
            current_memory: 0,
            max_memory: config.max_cache_size_mb * 1024 * 1024,
        };
        
        Self {
            cache: Arc::new(RwLock::new(cache)),
            stats: Arc::new(RwLock::new(QuantizationStats::default())),
            config,
        }
    }
    
    /// Dequantize Q4_K_M weights with optimal method selection
    pub fn dequantize_q4_k(
        &self,
        tensor_id: u64,
        data: &[u8],
        num_blocks: usize,
        output_size: usize,
    ) -> Result<Vec<f32>> {
        let start_time = Instant::now();
        
        // Check cache first
        if let Some(cached) = self.get_from_cache(tensor_id) {
            if self.config.enable_stats {
                let mut stats = self.stats.write().unwrap();
                stats.cache_hits += 1;
                stats.cache_time_saved += cached.dequantization_time;
            }
            return Ok(cached.data);
        }
        
        // Perform dequantization using optimal method
        let result = self.dequantize_optimized(data, num_blocks, output_size)?;
        let dequantization_time = start_time.elapsed();
        
        // Update statistics
        if self.config.enable_stats {
            let mut stats = self.stats.write().unwrap();
            stats.total_operations += 1;
            stats.total_time += dequantization_time;
            stats.cache_misses += 1;
            
            if num_blocks >= self.config.simd_threshold {
                stats.simd_operations += 1;
            } else {
                stats.scalar_operations += 1;
            }
            
            stats.update_avg_time();
        }
        
        // Cache the result
        self.add_to_cache(tensor_id, result.clone(), dequantization_time);
        
        Ok(result)
    }
    
    /// Bulk dequantize multiple tensors for a layer
    pub fn bulk_dequantize_layer(
        &self,
        tensors: &[(u64, &[u8], usize, usize)], // (id, data, num_blocks, output_size)
    ) -> Result<Vec<Vec<f32>>> {
        if !self.config.enable_bulk_processing || tensors.is_empty() {
            // Fallback to individual processing
            return tensors.iter()
                .map(|(id, data, num_blocks, output_size)| {
                    self.dequantize_q4_k(*id, data, *num_blocks, *output_size)
                })
                .collect();
        }
        
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(tensors.len());
        let mut cache_hits = 0;
        let mut processed_tensors = Vec::new();
        
        // Check cache for all tensors first
        for (id, data, num_blocks, output_size) in tensors {
            if let Some(cached) = self.get_from_cache(*id) {
                results.push(cached.data);
                cache_hits += 1;
            } else {
                results.push(Vec::new()); // Placeholder
                processed_tensors.push((results.len() - 1, *id, *data, *num_blocks, *output_size));
            }
        }
        
        // Bulk process uncached tensors
        if !processed_tensors.is_empty() {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    self.bulk_dequantize_neon(&processed_tensors, &mut results)?;
                } else {
                    self.bulk_dequantize_scalar(&processed_tensors, &mut results)?;
                }
            }
            
            #[cfg(not(target_arch = "aarch64"))]
            {
                self.bulk_dequantize_scalar(&processed_tensors, &mut results)?;
            }
        }
        
        // Update statistics
        if self.config.enable_stats {
            let mut stats = self.stats.write().unwrap();
            stats.total_operations += tensors.len() as u64;
            stats.total_time += start_time.elapsed();
            stats.cache_hits += cache_hits;
            stats.cache_misses += (tensors.len() - cache_hits as usize) as u64;
            stats.simd_operations += 1; // Bulk operations are always considered SIMD
            stats.update_avg_time();
        }
        
        Ok(results)
    }
    
    /// Prefetch weights for upcoming layers
    pub fn prefetch_layer_weights(&self, layer_indices: &[usize], weight_patterns: &[&str]) {
        if !self.config.enable_prefetching {
            return;
        }
        
        // This would typically trigger background loading of weights
        // For now, we just mark frequently accessed weights as high priority
        let cache = self.cache.read().unwrap();
        let frequent_weights: Vec<_> = cache.weights
            .iter()
            .filter(|(_, weight)| weight.access_count > 5)
            .map(|(id, _)| *id)
            .collect();
        
        // In a real implementation, we would use the layer_indices and weight_patterns
        // to predict which weights will be needed next and preload them
        drop(cache);
        
        for weight_id in frequent_weights {
            // Move frequently accessed weights to the back of the eviction queue
            let mut cache = self.cache.write().unwrap();
            if let Some(pos) = cache.access_order.iter().position(|&id| id == weight_id) {
                cache.access_order.remove(pos);
                cache.access_order.push_back(weight_id);
            }
        }
    }
    
    /// Get performance statistics
    pub fn stats(&self) -> QuantizationStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Clear the cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.weights.clear();
        cache.access_order.clear();
        cache.current_memory = 0;
    }
    
    /// Get current cache memory usage in MB
    pub fn cache_memory_usage_mb(&self) -> f64 {
        let cache = self.cache.read().unwrap();
        cache.current_memory as f64 / (1024.0 * 1024.0)
    }
    
    // Private methods
    
    fn dequantize_optimized(
        &self,
        data: &[u8],
        num_blocks: usize,
        output_size: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; output_size];
        
        #[cfg(target_arch = "aarch64")]
        {
            if num_blocks >= self.config.simd_threshold && 
               std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    woolly_gguf::simd::dequantize_q4_k_optimized(data, &mut output, num_blocks);
                }
                return Ok(output);
            }
        }
        
        // Fallback to scalar implementation
        let result = woolly_gguf::dequantize(data, woolly_gguf::GGMLType::Q4_K, output_size)
            .map_err(|e| CoreError::quantization("DEQUANTIZATION_FAILED", &e.to_string()))?;
        
        Ok(result)
    }
    
    #[cfg(target_arch = "aarch64")]
    fn bulk_dequantize_neon(
        &self,
        tensors: &[(usize, u64, &[u8], usize, usize)],
        results: &mut [Vec<f32>],
    ) -> Result<()> {
        unsafe {
            let mut bulk_data = Vec::new();
            for (result_idx, id, data, num_blocks, output_size) in tensors {
                let mut output = vec![0.0f32; *output_size];
                woolly_gguf::simd::dequantize_q4_k_optimized(data, &mut output, *num_blocks);
                
                // Cache the result
                let dequantization_time = Duration::from_millis(1); // Approximation for bulk
                self.add_to_cache(*id, output.clone(), dequantization_time);
                
                results[*result_idx] = output;
            }
        }
        Ok(())
    }
    
    fn bulk_dequantize_scalar(
        &self,
        tensors: &[(usize, u64, &[u8], usize, usize)],
        results: &mut [Vec<f32>],
    ) -> Result<()> {
        for (result_idx, id, data, _num_blocks, output_size) in tensors {
            let result = woolly_gguf::dequantize(data, woolly_gguf::GGMLType::Q4_K, *output_size)
                .map_err(|e| CoreError::quantization("DEQUANTIZATION_FAILED", &e.to_string()))?;
            
            // Cache the result
            let dequantization_time = Duration::from_millis(5); // Approximation for scalar
            self.add_to_cache(*id, result.clone(), dequantization_time);
            
            results[*result_idx] = result;
        }
        Ok(())
    }
    
    fn get_from_cache(&self, tensor_id: u64) -> Option<CachedWeight> {
        let mut cache = self.cache.write().unwrap();
        
        if let Some(weight) = cache.weights.get_mut(&tensor_id) {
            weight.last_access = Instant::now();
            weight.access_count += 1;
            
            // Move to end of LRU queue
            if let Some(pos) = cache.access_order.iter().position(|&id| id == tensor_id) {
                cache.access_order.remove(pos);
            }
            cache.access_order.push_back(tensor_id);
            
            return Some(CachedWeight {
                data: weight.data.clone(),
                size: weight.size,
                last_access: weight.last_access,
                access_count: weight.access_count,
                dequantization_time: weight.dequantization_time,
            });
        }
        
        None
    }
    
    fn add_to_cache(&self, tensor_id: u64, data: Vec<f32>, dequantization_time: Duration) {
        let mut cache = self.cache.write().unwrap();
        let size = data.len() * std::mem::size_of::<f32>();
        
        // Evict if necessary
        while cache.current_memory + size > cache.max_memory && !cache.access_order.is_empty() {
            if let Some(oldest_id) = cache.access_order.pop_front() {
                if let Some(evicted) = cache.weights.remove(&oldest_id) {
                    cache.current_memory -= evicted.size;
                }
            }
        }
        
        // Add new entry if it fits
        if cache.current_memory + size <= cache.max_memory {
            let weight = CachedWeight {
                data,
                size,
                last_access: Instant::now(),
                access_count: 1,
                dequantization_time,
            };
            
            cache.weights.insert(tensor_id, weight);
            cache.access_order.push_back(tensor_id);
            cache.current_memory += size;
        }
    }
}

/// Factory function to create an optimized quantization engine
pub fn create_quantization_engine() -> OptimizedQuantizationEngine {
    let config = QuantizationConfig::default();
    OptimizedQuantizationEngine::new(config)
}

/// Factory function to create an optimized quantization engine with custom config
pub fn create_quantization_engine_with_config(config: QuantizationConfig) -> OptimizedQuantizationEngine {
    OptimizedQuantizationEngine::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantization_engine_creation() {
        let engine = create_quantization_engine();
        let stats = engine.stats();
        
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.cache_hit_rate(), 0.0);
        assert_eq!(stats.simd_usage_rate(), 0.0);
    }
    
    #[test]
    fn test_cache_memory_usage() {
        let engine = create_quantization_engine();
        assert_eq!(engine.cache_memory_usage_mb(), 0.0);
        
        // Clear cache should work without panic
        engine.clear_cache();
        assert_eq!(engine.cache_memory_usage_mb(), 0.0);
    }
    
    #[test]
    fn test_stats_calculations() {
        let mut stats = QuantizationStats::default();
        stats.cache_hits = 80;
        stats.cache_misses = 20;
        stats.simd_operations = 60;
        stats.scalar_operations = 40;
        
        assert!((stats.cache_hit_rate() - 0.8).abs() < 0.001);
        assert!((stats.simd_usage_rate() - 0.6).abs() < 0.001);
    }
}