//! Model weight caching and sharing across sessions
//!
//! This module provides intelligent caching of model weights, tokenizer data,
//! and computation results to reduce memory usage and improve performance
//! when multiple sessions share the same model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Weak};
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};

use crate::{CoreError, Result};

/// Configuration for model caching
#[derive(Debug, Clone)]
pub struct ModelCacheConfig {
    /// Maximum memory for model weight cache (bytes)
    pub max_weight_cache: u64,
    /// Maximum memory for tokenizer cache (bytes)
    pub max_tokenizer_cache: u64,
    /// Maximum memory for computation result cache (bytes)
    pub max_computation_cache: u64,
    /// TTL for cached models
    pub model_ttl: Duration,
    /// TTL for tokenizer cache
    pub tokenizer_ttl: Duration,
    /// TTL for computation results
    pub computation_ttl: Duration,
    /// Enable cross-session weight sharing
    pub enable_weight_sharing: bool,
    /// Enable computation result caching
    pub enable_computation_cache: bool,
    /// Enable persistent disk cache
    pub enable_disk_cache: bool,
    /// Disk cache directory
    pub disk_cache_dir: Option<PathBuf>,
}

impl Default for ModelCacheConfig {
    fn default() -> Self {
        Self {
            max_weight_cache: 4 * 1024 * 1024 * 1024, // 4GB
            max_tokenizer_cache: 64 * 1024 * 1024, // 64MB
            max_computation_cache: 512 * 1024 * 1024, // 512MB
            model_ttl: Duration::from_secs(3600), // 1 hour
            tokenizer_ttl: Duration::from_secs(7200), // 2 hours
            computation_ttl: Duration::from_secs(300), // 5 minutes
            enable_weight_sharing: true,
            enable_computation_cache: true,
            enable_disk_cache: false,
            disk_cache_dir: None,
        }
    }
}

/// Cached model weights with reference counting
#[derive(Debug)]
pub struct CachedModelWeights {
    /// Model weights data
    pub weights: HashMap<String, Vec<f32>>,
    /// Model configuration
    pub config: serde_json::Value,
    /// Reference count for sharing
    pub ref_count: usize,
    /// Creation time
    pub created_at: Instant,
    /// Last access time
    pub last_accessed: Instant,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Model identifier (path or hash)
    pub model_id: String,
}

impl CachedModelWeights {
    /// Create new cached model weights
    pub fn new(
        model_id: String,
        weights: HashMap<String, Vec<f32>>,
        config: serde_json::Value,
    ) -> Self {
        let memory_usage = weights
            .values()
            .map(|w| w.len() * std::mem::size_of::<f32>())
            .sum::<usize>() as u64;
        
        let now = Instant::now();
        Self {
            weights,
            config,
            ref_count: 1,
            created_at: now,
            last_accessed: now,
            memory_usage,
            model_id,
        }
    }
    
    /// Increment reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
        self.last_accessed = Instant::now();
    }
    
    /// Decrement reference count
    pub fn remove_ref(&mut self) -> usize {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count
    }
    
    /// Check if cache entry is expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.last_accessed.elapsed() > ttl && self.ref_count == 0
    }
}

/// Cached tokenizer data
#[derive(Debug, Clone)]
pub struct CachedTokenizer {
    /// Tokenizer vocabulary
    pub vocab: HashMap<String, u32>,
    /// Reverse vocabulary (token_id -> token)
    pub reverse_vocab: HashMap<u32, String>,
    /// Special tokens
    pub special_tokens: HashMap<String, u32>,
    /// Tokenizer configuration
    pub config: serde_json::Value,
    /// Creation time
    pub created_at: Instant,
    /// Last access time
    pub last_accessed: Instant,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Tokenizer identifier
    pub tokenizer_id: String,
}

impl CachedTokenizer {
    /// Create new cached tokenizer
    pub fn new(
        tokenizer_id: String,
        vocab: HashMap<String, u32>,
        special_tokens: HashMap<String, u32>,
        config: serde_json::Value,
    ) -> Self {
        let reverse_vocab: HashMap<u32, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        
        let memory_usage = (vocab.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>()) +
                           reverse_vocab.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>()) +
                           special_tokens.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>())) as u64;
        
        let now = Instant::now();
        Self {
            vocab,
            reverse_vocab,
            special_tokens,
            config,
            created_at: now,
            last_accessed: now,
            memory_usage,
            tokenizer_id,
        }
    }
    
    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
    }
    
    /// Check if cache entry is expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.last_accessed.elapsed() > ttl
    }
}

/// Cached computation result
#[derive(Debug, Clone)]
pub struct CachedComputation {
    /// Input hash for cache key
    pub input_hash: u64,
    /// Computation result
    pub result: Vec<f32>,
    /// Computation type identifier
    pub computation_type: String,
    /// Creation time
    pub created_at: Instant,
    /// Access count
    pub access_count: u32,
    /// Memory usage in bytes
    pub memory_usage: u64,
}

impl CachedComputation {
    /// Create new cached computation
    pub fn new(
        input_hash: u64,
        result: Vec<f32>,
        computation_type: String,
    ) -> Self {
        let memory_usage = (result.len() * std::mem::size_of::<f32>()) as u64;
        
        Self {
            input_hash,
            result,
            computation_type,
            created_at: Instant::now(),
            access_count: 1,
            memory_usage,
        }
    }
    
    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
    }
    
    /// Check if cache entry is expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// Statistics for model cache
#[derive(Debug, Clone, Default)]
pub struct ModelCacheStats {
    /// Number of cached models
    pub cached_models: usize,
    /// Number of cached tokenizers
    pub cached_tokenizers: usize,
    /// Number of cached computations
    pub cached_computations: usize,
    /// Total memory used by model weights
    pub weight_cache_usage: u64,
    /// Total memory used by tokenizers
    pub tokenizer_cache_usage: u64,
    /// Total memory used by computations
    pub computation_cache_usage: u64,
    /// Model cache hits
    pub model_cache_hits: u64,
    /// Model cache misses
    pub model_cache_misses: u64,
    /// Tokenizer cache hits
    pub tokenizer_cache_hits: u64,
    /// Tokenizer cache misses
    pub tokenizer_cache_misses: u64,
    /// Computation cache hits
    pub computation_cache_hits: u64,
    /// Computation cache misses
    pub computation_cache_misses: u64,
    /// Number of evictions
    pub evictions: u64,
}

/// Global model cache for sharing across sessions
pub struct GlobalModelCache {
    /// Configuration
    config: ModelCacheConfig,
    
    /// Model weights cache (model_id -> weights)
    model_weights: RwLock<HashMap<String, Arc<RwLock<CachedModelWeights>>>>,
    
    /// Tokenizer cache (tokenizer_id -> tokenizer)
    tokenizers: RwLock<HashMap<String, CachedTokenizer>>,
    
    /// Computation result cache (cache_key -> result)
    computations: RwLock<HashMap<String, CachedComputation>>,
    
    /// Cache statistics
    stats: RwLock<ModelCacheStats>,
    
    /// Weak references to active models for cleanup
    active_models: RwLock<HashMap<String, Weak<RwLock<CachedModelWeights>>>>,
}

impl GlobalModelCache {
    /// Create new global model cache
    pub fn new(config: ModelCacheConfig) -> Self {
        info!("Initializing global model cache with {} MB weight limit, {} MB tokenizer limit",
              config.max_weight_cache / (1024 * 1024),
              config.max_tokenizer_cache / (1024 * 1024));
        
        Self {
            config,
            model_weights: RwLock::new(HashMap::new()),
            tokenizers: RwLock::new(HashMap::new()),
            computations: RwLock::new(HashMap::new()),
            stats: RwLock::new(ModelCacheStats::default()),
            active_models: RwLock::new(HashMap::new()),
        }
    }
    
    /// Get or load model weights with sharing
    pub fn get_or_load_model<F>(
        &self,
        model_id: &str,
        loader: F,
    ) -> Result<Arc<RwLock<CachedModelWeights>>>
    where
        F: FnOnce() -> Result<(HashMap<String, Vec<f32>>, serde_json::Value)>,
    {
        // Check if model is already cached
        {
            let models = self.model_weights.read().unwrap();
            if let Some(cached_model) = models.get(model_id) {
                // Update access statistics
                cached_model.write().unwrap().add_ref();
                
                let mut stats = self.stats.write().unwrap();
                stats.model_cache_hits += 1;
                
                debug!("Model cache hit for '{}'", model_id);
                return Ok(Arc::clone(cached_model));
            }
        }
        
        // Cache miss - load the model
        debug!("Model cache miss for '{}', loading...", model_id);
        
        let (weights, config) = loader()?;
        let cached_weights = CachedModelWeights::new(model_id.to_string(), weights, config);
        let memory_usage = cached_weights.memory_usage;
        
        // Check memory limits
        self.ensure_weight_cache_space(memory_usage)?;
        
        let cached_arc = Arc::new(RwLock::new(cached_weights));
        
        // Store in cache
        {
            let mut models = self.model_weights.write().unwrap();
            models.insert(model_id.to_string(), Arc::clone(&cached_arc));
            
            // Store weak reference for cleanup
            let mut active = self.active_models.write().unwrap();
            active.insert(model_id.to_string(), Arc::downgrade(&cached_arc));
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.model_cache_misses += 1;
            stats.cached_models += 1;
            stats.weight_cache_usage += memory_usage;
        }
        
        info!("Loaded and cached model '{}': {} MB", model_id, memory_usage / (1024 * 1024));
        
        Ok(cached_arc)
    }
    
    /// Release model reference
    pub fn release_model(&self, model_id: &str) {
        let should_remove = {
            let models = self.model_weights.read().unwrap();
            if let Some(cached_model) = models.get(model_id) {
                let ref_count = cached_model.write().unwrap().remove_ref();
                ref_count == 0
            } else {
                false
            }
        };
        
        if should_remove {
            debug!("Removing unreferenced model '{}' from cache", model_id);
            
            let memory_freed = {
                let mut models = self.model_weights.write().unwrap();
                if let Some(cached_model) = models.remove(model_id) {
                    cached_model.read().unwrap().memory_usage
                } else {
                    0
                }
            };
            
            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.cached_models = stats.cached_models.saturating_sub(1);
                stats.weight_cache_usage = stats.weight_cache_usage.saturating_sub(memory_freed);
            }
            
            // Remove from active models
            {
                let mut active = self.active_models.write().unwrap();
                active.remove(model_id);
            }
        }
    }
    
    /// Get or load tokenizer
    pub fn get_or_load_tokenizer<F>(
        &self,
        tokenizer_id: &str,
        loader: F,
    ) -> Result<CachedTokenizer>
    where
        F: FnOnce() -> Result<(HashMap<String, u32>, HashMap<String, u32>, serde_json::Value)>,
    {
        // Check if tokenizer is cached
        {
            let tokenizers = self.tokenizers.read().unwrap();
            if let Some(mut cached_tokenizer) = tokenizers.get(tokenizer_id).cloned() {
                cached_tokenizer.mark_accessed();
                
                let mut stats = self.stats.write().unwrap();
                stats.tokenizer_cache_hits += 1;
                
                debug!("Tokenizer cache hit for '{}'", tokenizer_id);
                return Ok(cached_tokenizer);
            }
        }
        
        // Cache miss - load the tokenizer
        debug!("Tokenizer cache miss for '{}', loading...", tokenizer_id);
        
        let (vocab, special_tokens, config) = loader()?;
        let cached_tokenizer = CachedTokenizer::new(
            tokenizer_id.to_string(),
            vocab,
            special_tokens,
            config,
        );
        
        let memory_usage = cached_tokenizer.memory_usage;
        
        // Check memory limits
        self.ensure_tokenizer_cache_space(memory_usage)?;
        
        // Store in cache
        {
            let mut tokenizers = self.tokenizers.write().unwrap();
            tokenizers.insert(tokenizer_id.to_string(), cached_tokenizer.clone());
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.tokenizer_cache_misses += 1;
            stats.cached_tokenizers += 1;
            stats.tokenizer_cache_usage += memory_usage;
        }
        
        info!("Loaded and cached tokenizer '{}': {} KB", 
              tokenizer_id, memory_usage / 1024);
        
        Ok(cached_tokenizer)
    }
    
    /// Cache computation result
    pub fn cache_computation(
        &self,
        cache_key: &str,
        input_hash: u64,
        result: Vec<f32>,
        computation_type: &str,
    ) -> Result<()> {
        if !self.config.enable_computation_cache {
            return Ok(());
        }
        
        let cached_computation = CachedComputation::new(
            input_hash,
            result,
            computation_type.to_string(),
        );
        
        let memory_usage = cached_computation.memory_usage;
        
        // Check memory limits
        self.ensure_computation_cache_space(memory_usage)?;
        
        // Store in cache
        {
            let mut computations = self.computations.write().unwrap();
            computations.insert(cache_key.to_string(), cached_computation);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cached_computations += 1;
            stats.computation_cache_usage += memory_usage;
        }
        
        trace!("Cached computation result for '{}': {} bytes", cache_key, memory_usage);
        
        Ok(())
    }
    
    /// Get cached computation result
    pub fn get_cached_computation(&self, cache_key: &str, input_hash: u64) -> Option<Vec<f32>> {
        if !self.config.enable_computation_cache {
            return None;
        }
        
        let mut computations = self.computations.write().unwrap();
        if let Some(cached) = computations.get_mut(cache_key) {
            if cached.input_hash == input_hash && !cached.is_expired(self.config.computation_ttl) {
                cached.mark_accessed();
                
                let mut stats = self.stats.write().unwrap();
                stats.computation_cache_hits += 1;
                
                trace!("Computation cache hit for '{}'", cache_key);
                return Some(cached.result.clone());
            }
        }
        
        // Update miss statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.computation_cache_misses += 1;
        }
        
        None
    }
    
    /// Ensure sufficient space in weight cache
    fn ensure_weight_cache_space(&self, required: u64) -> Result<()> {
        let current_usage = self.stats.read().unwrap().weight_cache_usage;
        
        if current_usage + required > self.config.max_weight_cache {
            // Need to evict some models
            self.evict_models(required)?;
        }
        
        Ok(())
    }
    
    /// Ensure sufficient space in tokenizer cache
    fn ensure_tokenizer_cache_space(&self, required: u64) -> Result<()> {
        let current_usage = self.stats.read().unwrap().tokenizer_cache_usage;
        
        if current_usage + required > self.config.max_tokenizer_cache {
            self.evict_tokenizers(required)?;
        }
        
        Ok(())
    }
    
    /// Ensure sufficient space in computation cache
    fn ensure_computation_cache_space(&self, required: u64) -> Result<()> {
        let current_usage = self.stats.read().unwrap().computation_cache_usage;
        
        if current_usage + required > self.config.max_computation_cache {
            self.evict_computations(required)?;
        }
        
        Ok(())
    }
    
    /// Evict models to free space
    fn evict_models(&self, target_free: u64) -> Result<()> {
        let mut freed = 0u64;
        let mut to_evict = Vec::new();
        
        // Find models to evict (unreferenced and oldest first)
        {
            let models = self.model_weights.read().unwrap();
            let mut candidates: Vec<_> = models
                .iter()
                .filter_map(|(id, model)| {
                    let model_ref = model.read().unwrap();
                    if model_ref.ref_count == 0 && model_ref.is_expired(self.config.model_ttl) {
                        Some((id.clone(), model_ref.last_accessed, model_ref.memory_usage))
                    } else {
                        None
                    }
                })
                .collect();
            
            // Sort by last access time (oldest first)
            candidates.sort_by_key(|(_, last_accessed, _)| *last_accessed);
            
            for (id, _, memory_usage) in candidates {
                to_evict.push(id);
                freed += memory_usage;
                if freed >= target_free {
                    break;
                }
            }
        }
        
        // Evict selected models
        {
            let mut models = self.model_weights.write().unwrap();
            let mut stats = self.stats.write().unwrap();
            
            for model_id in to_evict {
                if let Some(model) = models.remove(&model_id) {
                    let memory_usage = model.read().unwrap().memory_usage;
                    stats.weight_cache_usage = stats.weight_cache_usage.saturating_sub(memory_usage);
                    stats.cached_models = stats.cached_models.saturating_sub(1);
                    stats.evictions += 1;
                    
                    debug!("Evicted model '{}' ({} MB)", model_id, memory_usage / (1024 * 1024));
                }
            }
        }
        
        info!("Model cache eviction freed {} MB", freed / (1024 * 1024));
        Ok(())
    }
    
    /// Evict tokenizers to free space
    fn evict_tokenizers(&self, target_free: u64) -> Result<()> {
        let mut freed = 0u64;
        let mut to_evict = Vec::new();
        
        // Find tokenizers to evict (oldest first)
        {
            let tokenizers = self.tokenizers.read().unwrap();
            let mut candidates: Vec<_> = tokenizers
                .iter()
                .filter_map(|(id, tokenizer)| {
                    if tokenizer.is_expired(self.config.tokenizer_ttl) {
                        Some((id.clone(), tokenizer.last_accessed, tokenizer.memory_usage))
                    } else {
                        None
                    }
                })
                .collect();
            
            // Sort by last access time (oldest first)
            candidates.sort_by_key(|(_, last_accessed, _)| *last_accessed);
            
            for (id, _, memory_usage) in candidates {
                to_evict.push(id);
                freed += memory_usage;
                if freed >= target_free {
                    break;
                }
            }
        }
        
        // Evict selected tokenizers
        {
            let mut tokenizers = self.tokenizers.write().unwrap();
            let mut stats = self.stats.write().unwrap();
            
            for tokenizer_id in to_evict {
                if let Some(tokenizer) = tokenizers.remove(&tokenizer_id) {
                    stats.tokenizer_cache_usage = stats.tokenizer_cache_usage.saturating_sub(tokenizer.memory_usage);
                    stats.cached_tokenizers = stats.cached_tokenizers.saturating_sub(1);
                    stats.evictions += 1;
                    
                    debug!("Evicted tokenizer '{}' ({} KB)", tokenizer_id, tokenizer.memory_usage / 1024);
                }
            }
        }
        
        info!("Tokenizer cache eviction freed {} KB", freed / 1024);
        Ok(())
    }
    
    /// Evict computation results to free space
    fn evict_computations(&self, target_free: u64) -> Result<()> {
        let mut freed = 0u64;
        let mut to_evict = Vec::new();
        
        // Find computations to evict (expired and least accessed first)
        {
            let computations = self.computations.read().unwrap();
            let mut candidates: Vec<_> = computations
                .iter()
                .filter_map(|(key, comp)| {
                    if comp.is_expired(self.config.computation_ttl) {
                        Some((key.clone(), comp.access_count, comp.memory_usage))
                    } else {
                        None
                    }
                })
                .collect();
            
            // Sort by access count (least accessed first)
            candidates.sort_by_key(|(_, access_count, _)| *access_count);
            
            for (key, _, memory_usage) in candidates {
                to_evict.push(key);
                freed += memory_usage;
                if freed >= target_free {
                    break;
                }
            }
        }
        
        // Evict selected computations
        {
            let mut computations = self.computations.write().unwrap();
            let mut stats = self.stats.write().unwrap();
            
            for key in to_evict {
                if let Some(comp) = computations.remove(&key) {
                    stats.computation_cache_usage = stats.computation_cache_usage.saturating_sub(comp.memory_usage);
                    stats.cached_computations = stats.cached_computations.saturating_sub(1);
                    stats.evictions += 1;
                }
            }
        }
        
        info!("Computation cache eviction freed {} KB", freed / 1024);
        Ok(())
    }
    
    /// Clean up expired entries
    pub fn cleanup_expired(&self) {
        // Clean up unreferenced weak references
        {
            let mut active = self.active_models.write().unwrap();
            active.retain(|_, weak_ref| weak_ref.strong_count() > 0);
        }
        
        // Clean up expired tokenizers
        {
            let mut tokenizers = self.tokenizers.write().unwrap();
            let mut stats = self.stats.write().unwrap();
            
            let original_count = tokenizers.len();
            let mut freed_memory = 0u64;
            
            tokenizers.retain(|_, tokenizer| {
                let expired = tokenizer.is_expired(self.config.tokenizer_ttl);
                if expired {
                    freed_memory += tokenizer.memory_usage;
                }
                !expired
            });
            
            let removed = original_count - tokenizers.len();
            stats.cached_tokenizers = tokenizers.len();
            stats.tokenizer_cache_usage = stats.tokenizer_cache_usage.saturating_sub(freed_memory);
            
            if removed > 0 {
                debug!("Cleaned up {} expired tokenizers, freed {} KB", removed, freed_memory / 1024);
            }
        }
        
        // Clean up expired computations
        {
            let mut computations = self.computations.write().unwrap();
            let mut stats = self.stats.write().unwrap();
            
            let original_count = computations.len();
            let mut freed_memory = 0u64;
            
            computations.retain(|_, comp| {
                let expired = comp.is_expired(self.config.computation_ttl);
                if expired {
                    freed_memory += comp.memory_usage;
                }
                !expired
            });
            
            let removed = original_count - computations.len();
            stats.cached_computations = computations.len();
            stats.computation_cache_usage = stats.computation_cache_usage.saturating_sub(freed_memory);
            
            if removed > 0 {
                debug!("Cleaned up {} expired computations, freed {} KB", removed, freed_memory / 1024);
            }
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> ModelCacheStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Clear all caches
    pub fn clear_all(&self) {
        {
            let mut models = self.model_weights.write().unwrap();
            models.clear();
        }
        
        {
            let mut tokenizers = self.tokenizers.write().unwrap();
            tokenizers.clear();
        }
        
        {
            let mut computations = self.computations.write().unwrap();
            computations.clear();
        }
        
        {
            let mut active = self.active_models.write().unwrap();
            active.clear();
        }
        
        {
            let mut stats = self.stats.write().unwrap();
            *stats = ModelCacheStats::default();
        }
        
        info!("Cleared all model caches");
    }
}

/// Global cache instance
static GLOBAL_CACHE: std::sync::OnceLock<Arc<GlobalModelCache>> = std::sync::OnceLock::new();

/// Get or initialize the global model cache
pub fn global_cache() -> &'static Arc<GlobalModelCache> {
    GLOBAL_CACHE.get_or_init(|| Arc::new(GlobalModelCache::new(ModelCacheConfig::default())))
}

/// Initialize global cache with custom configuration
pub fn init_global_cache(config: ModelCacheConfig) -> &'static Arc<GlobalModelCache> {
    GLOBAL_CACHE.get_or_init(|| Arc::new(GlobalModelCache::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_model_cache_basic() {
        let config = ModelCacheConfig::default();
        let cache = GlobalModelCache::new(config);
        
        // Test model caching
        let weights = {
            let mut w = HashMap::new();
            w.insert("layer1".to_string(), vec![1.0; 1000]);
            w
        };
        let config_json = serde_json::json!({"hidden_size": 1000});
        
        let model_arc = cache.get_or_load_model("test_model", || {
            Ok((weights, config_json))
        }).unwrap();
        
        // Should cache hit on second access
        let model_arc2 = cache.get_or_load_model("test_model", || {
            panic!("Should not be called on cache hit");
        }).unwrap();
        
        assert_eq!(model_arc.read().unwrap().ref_count, 2);
        
        let stats = cache.stats();
        assert_eq!(stats.model_cache_hits, 1);
        assert_eq!(stats.model_cache_misses, 1);
    }
    
    #[test]
    fn test_tokenizer_cache() {
        let config = ModelCacheConfig::default();
        let cache = GlobalModelCache::new(config);
        
        let vocab = {
            let mut v = HashMap::new();
            v.insert("hello".to_string(), 1);
            v.insert("world".to_string(), 2);
            v
        };
        let special_tokens = HashMap::new();
        let config_json = serde_json::json!({"vocab_size": 1000});
        
        // First access should miss
        let tokenizer1 = cache.get_or_load_tokenizer("test_tokenizer", || {
            Ok((vocab.clone(), special_tokens.clone(), config_json.clone()))
        }).unwrap();
        
        // Second access should hit
        let tokenizer2 = cache.get_or_load_tokenizer("test_tokenizer", || {
            panic!("Should not be called on cache hit");
        }).unwrap();
        
        assert_eq!(tokenizer1.tokenizer_id, tokenizer2.tokenizer_id);
        
        let stats = cache.stats();
        assert_eq!(stats.tokenizer_cache_hits, 1);
        assert_eq!(stats.tokenizer_cache_misses, 1);
    }
    
    #[test]
    fn test_computation_cache() {
        let config = ModelCacheConfig::default();
        let cache = GlobalModelCache::new(config);
        
        let result = vec![1.0, 2.0, 3.0];
        let cache_key = "test_computation";
        let input_hash = 12345u64;
        
        // Cache the result
        cache.cache_computation(cache_key, input_hash, result.clone(), "test").unwrap();
        
        // Should hit on retrieval
        let cached_result = cache.get_cached_computation(cache_key, input_hash);
        assert_eq!(cached_result, Some(result));
        
        // Should miss with different hash
        let miss_result = cache.get_cached_computation(cache_key, 67890u64);
        assert_eq!(miss_result, None);
        
        let stats = cache.stats();
        assert_eq!(stats.computation_cache_hits, 1);
        assert_eq!(stats.computation_cache_misses, 1);
    }
}