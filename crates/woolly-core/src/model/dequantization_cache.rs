//! Dequantization cache for optimizing quantized weight access
//!
//! This module provides an LRU cache for dequantized weights to reduce
//! repeated dequantization overhead during inference.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use crate::{CoreError, Result};

/// Statistics for cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_bytes_cached: usize,
    pub total_dequantization_time_saved: Duration,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Entry in the dequantization cache
struct CacheEntry {
    data: Vec<f32>,
    size: usize,
    last_access: Instant,
    access_count: u64,
    /// Time taken to dequantize this tensor
    dequantization_time: Duration,
}

/// Configuration for the dequantization cache
#[derive(Debug, Clone)]
pub struct DequantizationCacheConfig {
    /// Maximum memory size for the cache in bytes
    pub max_memory_bytes: usize,
    /// Number of entries to prefetch ahead
    pub prefetch_ahead: usize,
    /// Whether to prioritize frequently accessed weights
    pub use_frequency_priority: bool,
    /// Time window for access frequency calculation
    pub frequency_window: Duration,
    /// Enable async prefetching
    pub enable_async_prefetch: bool,
}

impl Default for DequantizationCacheConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            prefetch_ahead: 2,
            use_frequency_priority: true,
            frequency_window: Duration::from_secs(60),
            enable_async_prefetch: true,
        }
    }
}

/// LRU cache for dequantized weights with memory awareness
pub struct DequantizationCache {
    config: DequantizationCacheConfig,
    /// Cache storage with tensor name as key
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// LRU queue for eviction
    lru_queue: Arc<Mutex<VecDeque<String>>>,
    /// Current memory usage
    current_memory: Arc<RwLock<usize>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Prefetch queue for async loading
    prefetch_queue: Arc<Mutex<VecDeque<String>>>,
    /// Layer-specific cache priorities
    layer_priorities: Arc<RwLock<HashMap<usize, f32>>>,
}

impl DequantizationCache {
    pub fn new(config: DequantizationCacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            lru_queue: Arc::new(Mutex::new(VecDeque::new())),
            current_memory: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            prefetch_queue: Arc::new(Mutex::new(VecDeque::new())),
            layer_priorities: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a tensor from cache without dequantizing (simple lookup)
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        let cache = self.cache.read().unwrap();
        if let Some(entry) = cache.get(key) {
            // Update stats 
            let mut stats = self.stats.write().unwrap();
            stats.hits += 1;
            stats.total_dequantization_time_saved += entry.dequantization_time;
            eprintln!("📊 CACHE HIT for '{}' - Total hits: {}, Total misses: {}, Hit rate: {:.1}%", 
                key, stats.hits, stats.misses, stats.hit_rate() * 100.0);
            drop(stats);
            
            // Update LRU position
            self.update_lru(key);
            
            Some(entry.data.clone())
        } else {
            None
        }
    }

    /// Get a dequantized tensor from cache or dequantize and cache it
    pub fn get_or_dequantize<F>(
        &self,
        key: &str,
        dequantize_fn: F,
    ) -> Result<Vec<f32>>
    where
        F: FnOnce() -> Result<(Vec<f32>, Duration)>,
    {
        // Check cache first (read lock)
        {
            let cache = self.cache.read().unwrap();
            if let Some(entry) = cache.get(key) {
                // Update stats and LRU
                let mut stats = self.stats.write().unwrap();
                stats.hits += 1;
                stats.total_dequantization_time_saved += entry.dequantization_time;
                eprintln!("📊 CACHE HIT for '{}' - Total hits: {}, Total misses: {}, Hit rate: {:.1}%", 
                    key, stats.hits, stats.misses, stats.hit_rate() * 100.0);
                drop(stats);
                
                // Update LRU position
                self.update_lru(key);
                
                return Ok(entry.data.clone());
            }
        }

        // Cache miss - need to dequantize
        let mut stats = self.stats.write().unwrap();
        stats.misses += 1;
        eprintln!("📊 CACHE MISS for '{}' - Total hits: {}, Total misses: {}, Hit rate: {:.1}%", 
            key, stats.hits, stats.misses, stats.hit_rate() * 100.0);
        drop(stats);

        // Dequantize the tensor
        let (data, dequantization_time) = dequantize_fn()?;
        let data_size = data.len() * std::mem::size_of::<f32>();

        // Add to cache if it fits
        self.add_to_cache(key.to_string(), data.clone(), data_size, dequantization_time)?;

        Ok(data)
    }

    /// Add a tensor to the cache with LRU eviction if needed
    fn add_to_cache(
        &self,
        key: String,
        data: Vec<f32>,
        size: usize,
        dequantization_time: Duration,
    ) -> Result<()> {
        // Check if we need to evict entries
        self.evict_if_needed(size)?;

        // Add to cache
        let entry = CacheEntry {
            data,
            size,
            last_access: Instant::now(),
            access_count: 1,
            dequantization_time,
        };

        let mut cache = self.cache.write().unwrap();
        cache.insert(key.clone(), entry);
        
        // Update memory usage
        let mut current_memory = self.current_memory.write().unwrap();
        *current_memory += size;
        
        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.total_bytes_cached = *current_memory;
        
        // Add to LRU queue
        let mut lru_queue = self.lru_queue.lock().unwrap();
        lru_queue.push_back(key);

        Ok(())
    }

    /// Evict entries if needed to make room for new data
    fn evict_if_needed(&self, needed_size: usize) -> Result<()> {
        let current_mem = *self.current_memory.read().unwrap();
        
        if current_mem + needed_size <= self.config.max_memory_bytes {
            return Ok(());
        }

        let mut cache = self.cache.write().unwrap();
        let mut lru_queue = self.lru_queue.lock().unwrap();
        let mut current_memory = self.current_memory.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Evict entries until we have enough space
        while *current_memory + needed_size > self.config.max_memory_bytes {
            if let Some(key) = lru_queue.pop_front() {
                if let Some(entry) = cache.remove(&key) {
                    *current_memory -= entry.size;
                    stats.evictions += 1;
                }
            } else {
                return Err(CoreError::model(
                    "CACHE_EVICTION_FAILED",
                    "Cannot evict enough entries to make room",
                    "Adding to dequantization cache",
                    "Increase cache size or reduce tensor size"
                ));
            }
        }

        stats.total_bytes_cached = *current_memory;
        Ok(())
    }

    /// Update LRU position for a key
    fn update_lru(&self, key: &str) {
        let mut lru_queue = self.lru_queue.lock().unwrap();
        
        // Remove from current position
        if let Some(pos) = lru_queue.iter().position(|k| k == key) {
            lru_queue.remove(pos);
        }
        
        // Add to back (most recently used)
        lru_queue.push_back(key.to_string());
        
        // Update access time and count
        let mut cache = self.cache.write().unwrap();
        if let Some(entry) = cache.get_mut(key) {
            entry.last_access = Instant::now();
            entry.access_count += 1;
        }
    }

    /// Prefetch weights for upcoming layers
    pub fn prefetch_layer_weights(&self, layer_idx: usize, weight_names: Vec<String>) {
        if !self.config.enable_async_prefetch {
            return;
        }

        let mut prefetch_queue = self.prefetch_queue.lock().unwrap();
        
        // Add weights for current and next few layers
        for i in 0..self.config.prefetch_ahead {
            let target_layer = layer_idx + i;
            for name in &weight_names {
                let prefetch_key = name.replace("{}", &target_layer.to_string());
                if !prefetch_queue.contains(&prefetch_key) {
                    prefetch_queue.push_back(prefetch_key);
                }
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        
        let mut lru_queue = self.lru_queue.lock().unwrap();
        lru_queue.clear();
        
        let mut current_memory = self.current_memory.write().unwrap();
        *current_memory = 0;
        
        let mut stats = self.stats.write().unwrap();
        *stats = CacheStats::default();
    }

    /// Set layer priority for cache retention
    pub fn set_layer_priority(&self, layer_idx: usize, priority: f32) {
        let mut priorities = self.layer_priorities.write().unwrap();
        priorities.insert(layer_idx, priority);
    }

    /// Analyze weight access patterns and return frequently accessed weights
    pub fn analyze_access_patterns(&self) -> Vec<(String, u64, Duration)> {
        let cache = self.cache.read().unwrap();
        let now = Instant::now();
        
        let mut patterns: Vec<(String, u64, Duration)> = cache
            .iter()
            .filter(|(_, entry)| {
                now.duration_since(entry.last_access) < self.config.frequency_window
            })
            .map(|(key, entry)| {
                (key.clone(), entry.access_count, entry.dequantization_time)
            })
            .collect();
        
        // Sort by access count (descending)
        patterns.sort_by(|a, b| b.1.cmp(&a.1));
        
        patterns
    }

    /// Preload frequently accessed weights based on historical patterns
    pub fn preload_frequent_weights(&self, patterns: &[(String, u64, Duration)], top_n: usize) {
        let mut prefetch_queue = self.prefetch_queue.lock().unwrap();
        
        for (key, _, _) in patterns.iter().take(top_n) {
            if !prefetch_queue.contains(key) {
                prefetch_queue.push_back(key.clone());
            }
        }
    }

    /// Get memory usage information
    pub fn memory_info(&self) -> (usize, usize, f64) {
        let current = *self.current_memory.read().unwrap();
        let max = self.config.max_memory_bytes;
        let usage_percent = (current as f64 / max as f64) * 100.0;
        
        (current, max, usage_percent)
    }
}

/// Weight access tracker for identifying hot weights
pub struct WeightAccessTracker {
    access_counts: Arc<RwLock<HashMap<String, AccessInfo>>>,
    window_size: Duration,
}

#[derive(Clone)]
struct AccessInfo {
    count: u64,
    first_access: Instant,
    last_access: Instant,
    total_time: Duration,
}

impl WeightAccessTracker {
    pub fn new(window_size: Duration) -> Self {
        Self {
            access_counts: Arc::new(RwLock::new(HashMap::new())),
            window_size,
        }
    }

    pub fn record_access(&self, weight_name: &str, access_time: Duration) {
        let mut counts = self.access_counts.write().unwrap();
        let now = Instant::now();
        
        counts.entry(weight_name.to_string())
            .and_modify(|info| {
                info.count += 1;
                info.last_access = now;
                info.total_time += access_time;
            })
            .or_insert(AccessInfo {
                count: 1,
                first_access: now,
                last_access: now,
                total_time: access_time,
            });
    }

    pub fn get_hot_weights(&self, threshold: u64) -> Vec<(String, u64, Duration)> {
        let counts = self.access_counts.read().unwrap();
        let now = Instant::now();
        
        counts
            .iter()
            .filter(|(_, info)| {
                info.count >= threshold && 
                now.duration_since(info.last_access) < self.window_size
            })
            .map(|(name, info)| {
                (name.clone(), info.count, info.total_time / info.count as u32)
            })
            .collect()
    }

    pub fn clear_old_entries(&self) {
        let mut counts = self.access_counts.write().unwrap();
        let now = Instant::now();
        
        counts.retain(|_, info| {
            now.duration_since(info.last_access) < self.window_size
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cache_basic_operations() {
        let config = DequantizationCacheConfig {
            max_memory_bytes: 1024 * 100, // 100KB
            ..Default::default()
        };
        
        let cache = DequantizationCache::new(config);
        
        // Test cache miss and hit
        let result1 = cache.get_or_dequantize("test_weight", || {
            Ok((vec![1.0, 2.0, 3.0], Duration::from_millis(10)))
        }).unwrap();
        
        assert_eq!(result1, vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
        
        // Test cache hit
        let result2 = cache.get_or_dequantize("test_weight", || {
            panic!("Should not be called - cache hit expected");
        }).unwrap();
        
        assert_eq!(result2, vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = DequantizationCacheConfig {
            max_memory_bytes: 100, // Very small cache
            ..Default::default()
        };
        
        let cache = DequantizationCache::new(config);
        
        // Add entries that will cause eviction
        for i in 0..5 {
            let key = format!("weight_{}", i);
            let _ = cache.get_or_dequantize(&key, || {
                Ok((vec![i as f32; 10], Duration::from_millis(1)))
            });
        }
        
        // Check that evictions occurred
        assert!(cache.stats().evictions > 0);
        
        // Verify memory limit is respected
        let (current, max, _) = cache.memory_info();
        assert!(current <= max);
    }

    #[test]
    fn test_access_tracker() {
        let tracker = WeightAccessTracker::new(Duration::from_secs(60));
        
        // Record some accesses
        tracker.record_access("weight_1", Duration::from_millis(5));
        tracker.record_access("weight_1", Duration::from_millis(5));
        tracker.record_access("weight_2", Duration::from_millis(10));
        
        // Get hot weights
        let hot_weights = tracker.get_hot_weights(2);
        assert_eq!(hot_weights.len(), 1);
        assert_eq!(hot_weights[0].0, "weight_1");
        assert_eq!(hot_weights[0].1, 2);
    }
}