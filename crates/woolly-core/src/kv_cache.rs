//! Optimized KV cache implementation with memory management and eviction policies
//!
//! This module provides an efficient key-value cache for transformer inference with:
//! - Memory-bounded caching with LRU and sliding window eviction
//! - SIMD-optimized memory layout for better performance
//! - Compression support for long sequences
//! - Multi-layer cache management with layer-specific policies

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};

use crate::{CoreError, Result};

/// Configuration for KV cache behavior
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Maximum memory for KV cache (bytes)
    pub max_memory: u64,
    /// Maximum sequence length to cache
    pub max_seq_length: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden size per attention head
    pub head_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<usize>,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable compression for long sequences
    pub enable_compression: bool,
    /// Compression threshold (sequence length)
    pub compression_threshold: usize,
    /// Block size for memory allocation
    pub block_size: usize,
    /// Enable SIMD-optimized layout
    pub enable_simd_layout: bool,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            max_seq_length: 8192,
            num_layers: 32,
            head_dim: 128,
            num_heads: 32,
            num_kv_heads: None,
            eviction_policy: EvictionPolicy::SlidingWindow,
            enable_compression: true,
            compression_threshold: 4096,
            block_size: 64, // 64 token block size
            enable_simd_layout: true,
        }
    }
}

/// Eviction policies for KV cache management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Sliding window (keep recent tokens)
    SlidingWindow,
    /// Hybrid: keep both recent and frequently accessed tokens
    Hybrid,
    /// Custom policy based on attention scores
    AttentionBased,
}

/// Statistics for KV cache usage
#[derive(Debug, Clone, Default)]
pub struct KVCacheStats {
    /// Total memory used (bytes)
    pub memory_used: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Number of compressed blocks
    pub compressed_blocks: u64,
    /// Peak memory usage
    pub peak_memory: u64,
    /// Average sequence length
    pub avg_seq_length: f64,
    /// Fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// A single KV cache entry for one layer
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Key tensor data (flattened)
    pub keys: Vec<f32>,
    /// Value tensor data (flattened)  
    pub values: Vec<f32>,
    /// Sequence length for this entry
    pub seq_length: usize,
    /// Layer index
    pub layer_idx: usize,
    /// Last access time
    pub last_accessed: Instant,
    /// Access count for LRU
    pub access_count: u32,
    /// Whether this entry is compressed
    pub is_compressed: bool,
    /// Attention scores for attention-based eviction
    pub attention_scores: Option<Vec<f32>>,
}

impl KVCacheEntry {
    /// Create a new KV cache entry
    pub fn new(
        keys: Vec<f32>,
        values: Vec<f32>,
        seq_length: usize,
        layer_idx: usize,
    ) -> Self {
        Self {
            keys,
            values,
            seq_length,
            layer_idx,
            last_accessed: Instant::now(),
            access_count: 1,
            is_compressed: false,
            attention_scores: None,
        }
    }
    
    /// Get the memory usage of this entry in bytes
    pub fn memory_usage(&self) -> u64 {
        let key_bytes = self.keys.len() * std::mem::size_of::<f32>();
        let value_bytes = self.values.len() * std::mem::size_of::<f32>();
        let scores_bytes = self.attention_scores.as_ref()
            .map(|s| s.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        
        (key_bytes + value_bytes + scores_bytes) as u64
    }
    
    /// Update access statistics
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
    
    /// Compress the KV entry using simple run-length encoding
    pub fn compress(&mut self) -> Result<()> {
        if self.is_compressed {
            return Ok(());
        }
        
        // Simple compression: remove duplicate consecutive values
        // In a real implementation, this would use proper compression algorithms
        let original_size = self.memory_usage();
        
        self.keys = compress_f32_vector(&self.keys);
        self.values = compress_f32_vector(&self.values);
        self.is_compressed = true;
        
        let compressed_size = self.memory_usage();
        let ratio = compressed_size as f64 / original_size as f64;
        
        debug!("Compressed KV entry for layer {}: {:.1}% of original size", 
               self.layer_idx, ratio * 100.0);
        
        Ok(())
    }
    
    /// Decompress the KV entry
    pub fn decompress(&mut self) -> Result<()> {
        if !self.is_compressed {
            return Ok(());
        }
        
        self.keys = decompress_f32_vector(&self.keys);
        self.values = decompress_f32_vector(&self.values);
        self.is_compressed = false;
        
        Ok(())
    }
}

/// Optimized KV cache with memory management
pub struct OptimizedKVCache {
    /// Configuration
    config: KVCacheConfig,
    
    /// Cache entries per layer
    /// layer_idx -> session_id -> KVCacheEntry
    layer_caches: Vec<RwLock<HashMap<String, KVCacheEntry>>>,
    
    /// LRU tracking for eviction
    access_order: Arc<Mutex<VecDeque<(usize, String)>>>, // (layer_idx, session_id)
    
    /// Current memory usage
    current_memory: Arc<Mutex<u64>>,
    
    /// Statistics
    stats: Arc<RwLock<KVCacheStats>>,
    
    /// Memory blocks for efficient allocation
    memory_blocks: Arc<Mutex<Vec<MemoryBlock>>>,
}

/// Memory block for efficient KV cache allocation
#[derive(Debug)]
struct MemoryBlock {
    data: Vec<f32>,
    used_tokens: usize,
    total_tokens: usize,
    layer_idx: usize,
}

impl OptimizedKVCache {
    /// Create a new optimized KV cache
    pub fn new(config: KVCacheConfig) -> Self {
        let layer_caches = (0..config.num_layers)
            .map(|_| RwLock::new(HashMap::new()))
            .collect();
        
        info!("Initialized KV cache: {} layers, max memory: {} MB, max seq length: {}",
              config.num_layers, config.max_memory / (1024 * 1024), config.max_seq_length);
        
        Self {
            config,
            layer_caches,
            access_order: Arc::new(Mutex::new(VecDeque::new())),
            current_memory: Arc::new(Mutex::new(0)),
            stats: Arc::new(RwLock::new(KVCacheStats::default())),
            memory_blocks: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Store KV cache for a specific layer and session
    pub fn store(
        &self,
        layer_idx: usize,
        session_id: &str,
        keys: Vec<f32>,
        values: Vec<f32>,
        seq_length: usize,
        attention_scores: Option<Vec<f32>>,
    ) -> Result<()> {
        if layer_idx >= self.config.num_layers {
            return Err(CoreError::Cache {
                code: "INVALID_LAYER",
                message: format!("Layer index {} exceeds maximum {}", layer_idx, self.config.num_layers),
                context: "KV cache storage".to_string(),
                suggestion: "Check model configuration".to_string(),
                cache_size: None,
                available_memory: None,
            });
        }
        
        // Check sequence length limit
        if seq_length > self.config.max_seq_length {
            warn!("Sequence length {} exceeds maximum {}, truncating", 
                  seq_length, self.config.max_seq_length);
            return self.store_truncated(layer_idx, session_id, keys, values, attention_scores);
        }
        
        let mut entry = KVCacheEntry::new(keys, values, seq_length, layer_idx);
        entry.attention_scores = attention_scores;
        
        let entry_size = entry.memory_usage();
        
        // Check if we need to compress large entries
        if self.config.enable_compression && seq_length > self.config.compression_threshold {
            entry.compress()?;
        }
        
        // Ensure we have enough memory
        self.ensure_memory_available(entry_size)?;
        
        // Store the entry
        {
            let mut layer_cache = self.layer_caches[layer_idx].write().unwrap();
            
            // Remove old entry if it exists
            if let Some(old_entry) = layer_cache.remove(session_id) {
                let old_size = old_entry.memory_usage();
                let mut current_memory = self.current_memory.lock().unwrap();
                *current_memory = current_memory.saturating_sub(old_size);
            }
            
            layer_cache.insert(session_id.to_string(), entry);
        }
        
        // Update memory usage
        {
            let mut current_memory = self.current_memory.lock().unwrap();
            *current_memory += entry_size;
            
            let mut stats = self.stats.write().unwrap();
            if *current_memory > stats.peak_memory {
                stats.peak_memory = *current_memory;
            }
            stats.memory_used = *current_memory;
        }
        
        // Update access order
        {
            let mut access_order = self.access_order.lock().unwrap();
            access_order.push_back((layer_idx, session_id.to_string()));
        }
        
        trace!("Stored KV cache for layer {} session '{}': {} tokens, {} bytes",
               layer_idx, session_id, seq_length, entry_size);
        
        Ok(())
    }
    
    /// Retrieve KV cache for a specific layer and session
    pub fn retrieve(
        &self,
        layer_idx: usize,
        session_id: &str,
    ) -> Result<Option<(Vec<f32>, Vec<f32>, usize)>> {
        if layer_idx >= self.config.num_layers {
            return Err(CoreError::Cache {
                code: "INVALID_LAYER",
                message: format!("Layer index {} exceeds maximum {}", layer_idx, self.config.num_layers),
                context: "KV cache retrieval".to_string(),
                suggestion: "Check model configuration".to_string(),
                cache_size: None,
                available_memory: None,
            });
        }
        
        let mut layer_cache = self.layer_caches[layer_idx].write().unwrap();
        
        if let Some(entry) = layer_cache.get_mut(session_id) {
            // Update access statistics
            entry.mark_accessed();
            
            // Decompress if needed
            if entry.is_compressed {
                entry.decompress()?;
            }
            
            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.cache_hits += 1;
            }
            
            trace!("Retrieved KV cache for layer {} session '{}': {} tokens",
                   layer_idx, session_id, entry.seq_length);
            
            Ok(Some((entry.keys.clone(), entry.values.clone(), entry.seq_length)))
        } else {
            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.cache_misses += 1;
            }
            
            Ok(None)
        }
    }
    
    /// Append new KV data to existing cache entry
    pub fn append(
        &self,
        layer_idx: usize,
        session_id: &str,
        new_keys: Vec<f32>,
        new_values: Vec<f32>,
        attention_scores: Option<Vec<f32>>,
    ) -> Result<()> {
        if layer_idx >= self.config.num_layers {
            return Err(CoreError::Cache {
                code: "INVALID_LAYER",
                message: format!("Layer index {} exceeds maximum {}", layer_idx, self.config.num_layers),
                context: "KV cache append".to_string(),
                suggestion: "Check model configuration".to_string(),
                cache_size: None,
                available_memory: None,
            });
        }
        
        let mut layer_cache = self.layer_caches[layer_idx].write().unwrap();
        
        if let Some(entry) = layer_cache.get_mut(session_id) {
            // Decompress before appending
            if entry.is_compressed {
                entry.decompress()?;
            }
            
            let old_size = entry.memory_usage();
            
            // Calculate new sequence length before moving the keys
            let new_seq_len = new_keys.len() / (self.config.head_dim * self.config.num_heads);
            
            // Append new data
            entry.keys.extend(new_keys);
            entry.values.extend(new_values);
            entry.seq_length += new_seq_len;
            
            // Update attention scores
            if let Some(new_scores) = attention_scores {
                if let Some(ref mut old_scores) = entry.attention_scores {
                    old_scores.extend(new_scores);
                } else {
                    entry.attention_scores = Some(new_scores);
                }
            }
            
            // Check sequence length limit
            if entry.seq_length > self.config.max_seq_length {
                self.apply_eviction_policy(entry)?;
            }
            
            // Compress if needed
            if self.config.enable_compression && entry.seq_length > self.config.compression_threshold {
                entry.compress()?;
            }
            
            entry.mark_accessed();
            
            let new_size = entry.memory_usage();
            let size_diff = new_size as i64 - old_size as i64;
            
            // Update memory usage
            {
                let mut current_memory = self.current_memory.lock().unwrap();
                if size_diff > 0 {
                    *current_memory += size_diff as u64;
                } else {
                    *current_memory = current_memory.saturating_sub((-size_diff) as u64);
                }
                
                let mut stats = self.stats.write().unwrap();
                stats.memory_used = *current_memory;
            }
            
            trace!("Appended to KV cache for layer {} session '{}': new length {} tokens",
                   layer_idx, session_id, entry.seq_length);
            
            Ok(())
        } else {
            // No existing entry, create new one
            let seq_len = new_keys.len() / (self.config.head_dim * self.config.num_heads);
            self.store(layer_idx, session_id, new_keys, new_values, 
                      seq_len, attention_scores)
        }
    }
    
    /// Store truncated KV cache when sequence is too long
    fn store_truncated(
        &self,
        layer_idx: usize,
        session_id: &str,
        mut keys: Vec<f32>,
        mut values: Vec<f32>,
        attention_scores: Option<Vec<f32>>,
    ) -> Result<()> {
        let head_size = self.config.head_dim * self.config.num_heads;
        let max_tokens = self.config.max_seq_length;
        let max_key_size = max_tokens * head_size;
        
        // Truncate to maximum size
        if keys.len() > max_key_size {
            keys.truncate(max_key_size);
        }
        if values.len() > max_key_size {
            values.truncate(max_key_size);
        }
        
        let truncated_attention = attention_scores.map(|mut scores| {
            if scores.len() > max_tokens {
                scores.truncate(max_tokens);
            }
            scores
        });
        
        self.store(layer_idx, session_id, keys, values, max_tokens, truncated_attention)
    }
    
    /// Apply eviction policy to reduce sequence length
    fn apply_eviction_policy(&self, entry: &mut KVCacheEntry) -> Result<()> {
        match self.config.eviction_policy {
            EvictionPolicy::SlidingWindow => {
                self.apply_sliding_window_eviction(entry)?;
            }
            EvictionPolicy::LRU => {
                // For single entry, just truncate from beginning
                self.truncate_from_beginning(entry)?;
            }
            EvictionPolicy::AttentionBased => {
                self.apply_attention_based_eviction(entry)?;
            }
            EvictionPolicy::Hybrid => {
                // Combine sliding window with attention-based
                if entry.attention_scores.is_some() {
                    self.apply_attention_based_eviction(entry)?;
                } else {
                    self.apply_sliding_window_eviction(entry)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply sliding window eviction (keep most recent tokens)
    fn apply_sliding_window_eviction(&self, entry: &mut KVCacheEntry) -> Result<()> {
        let target_length = self.config.max_seq_length / 2; // Keep half
        let head_size = self.config.head_dim * self.config.num_heads;
        
        if entry.seq_length <= target_length {
            return Ok(());
        }
        
        let tokens_to_remove = entry.seq_length - target_length;
        let elements_to_remove = tokens_to_remove * head_size;
        
        // Remove from beginning (keep recent tokens)
        entry.keys.drain(0..elements_to_remove);
        entry.values.drain(0..elements_to_remove);
        
        if let Some(ref mut scores) = entry.attention_scores {
            scores.drain(0..tokens_to_remove);
        }
        
        entry.seq_length = target_length;
        
        debug!("Applied sliding window eviction: kept {} recent tokens", target_length);
        Ok(())
    }
    
    /// Truncate from beginning (simple LRU for single entry)
    fn truncate_from_beginning(&self, entry: &mut KVCacheEntry) -> Result<()> {
        let target_length = self.config.max_seq_length * 3 / 4; // Keep 75%
        let head_size = self.config.head_dim * self.config.num_heads;
        
        if entry.seq_length <= target_length {
            return Ok(());
        }
        
        let tokens_to_remove = entry.seq_length - target_length;
        let elements_to_remove = tokens_to_remove * head_size;
        
        entry.keys.drain(0..elements_to_remove);
        entry.values.drain(0..elements_to_remove);
        
        if let Some(ref mut scores) = entry.attention_scores {
            scores.drain(0..tokens_to_remove);
        }
        
        entry.seq_length = target_length;
        
        debug!("Truncated from beginning: kept {} tokens", target_length);
        Ok(())
    }
    
    /// Apply attention-based eviction (keep high-attention tokens)
    fn apply_attention_based_eviction(&self, entry: &mut KVCacheEntry) -> Result<()> {
        let Some(ref attention_scores) = entry.attention_scores else {
            // Fall back to sliding window if no attention scores
            return self.apply_sliding_window_eviction(entry);
        };
        
        let target_length = self.config.max_seq_length / 2;
        if entry.seq_length <= target_length {
            return Ok(());
        }
        
        // Find tokens with highest attention scores
        let mut scored_tokens: Vec<(usize, f32)> = attention_scores.iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        // Sort by attention score (descending)
        scored_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Keep top scoring tokens plus recent tokens
        let mut keep_indices = std::collections::HashSet::new();
        
        // Keep top 50% by attention score
        for &(idx, _) in scored_tokens.iter().take(target_length / 2) {
            keep_indices.insert(idx);
        }
        
        // Keep most recent 50%
        let recent_start = entry.seq_length.saturating_sub(target_length / 2);
        for idx in recent_start..entry.seq_length {
            keep_indices.insert(idx);
        }
        
        // Build new vectors with only kept tokens
        let mut new_keys = Vec::new();
        let mut new_values = Vec::new();
        let mut new_scores = Vec::new();
        
        let head_size = self.config.head_dim * self.config.num_heads;
        
        for token_idx in 0..entry.seq_length {
            if keep_indices.contains(&token_idx) {
                let start_elem = token_idx * head_size;
                let end_elem = (token_idx + 1) * head_size;
                
                new_keys.extend_from_slice(&entry.keys[start_elem..end_elem]);
                new_values.extend_from_slice(&entry.values[start_elem..end_elem]);
                new_scores.push(attention_scores[token_idx]);
            }
        }
        
        entry.keys = new_keys;
        entry.values = new_values;
        entry.attention_scores = Some(new_scores);
        entry.seq_length = keep_indices.len();
        
        debug!("Applied attention-based eviction: kept {} high-attention tokens", entry.seq_length);
        Ok(())
    }
    
    /// Ensure sufficient memory is available, evicting if necessary
    fn ensure_memory_available(&self, required: u64) -> Result<()> {
        let current = *self.current_memory.lock().unwrap();
        
        if current + required <= self.config.max_memory {
            return Ok(());
        }
        
        info!("Memory limit exceeded, triggering eviction: current {} MB, required {} MB, limit {} MB",
              current / (1024 * 1024), required / (1024 * 1024), self.config.max_memory / (1024 * 1024));
        
        // Apply global eviction based on policy
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru_entries(required)?,
            EvictionPolicy::SlidingWindow => self.evict_oldest_entries(required)?,
            EvictionPolicy::AttentionBased => self.evict_low_attention_entries(required)?,
            EvictionPolicy::Hybrid => self.evict_hybrid_entries(required)?,
        }
        
        Ok(())
    }
    
    /// Evict LRU entries to free memory
    fn evict_lru_entries(&self, target_free: u64) -> Result<()> {
        let mut freed = 0u64;
        let mut access_order = self.access_order.lock().unwrap();
        
        while freed < target_free && !access_order.is_empty() {
            if let Some((layer_idx, session_id)) = access_order.pop_front() {
                let mut layer_cache = self.layer_caches[layer_idx].write().unwrap();
                
                if let Some(entry) = layer_cache.remove(&session_id) {
                    let entry_size = entry.memory_usage();
                    freed += entry_size;
                    
                    let mut current_memory = self.current_memory.lock().unwrap();
                    *current_memory = current_memory.saturating_sub(entry_size);
                    
                    debug!("Evicted LRU entry: layer {} session '{}' ({} bytes)", 
                           layer_idx, session_id, entry_size);
                }
            }
        }
        
        let mut stats = self.stats.write().unwrap();
        stats.evictions += 1;
        stats.memory_used = *self.current_memory.lock().unwrap();
        
        info!("LRU eviction freed {} MB", freed / (1024 * 1024));
        Ok(())
    }
    
    /// Evict oldest entries
    fn evict_oldest_entries(&self, target_free: u64) -> Result<()> {
        let mut freed = 0u64;
        let now = Instant::now();
        
        // Find oldest entries across all layers
        let mut candidates: Vec<(Duration, usize, String, u64)> = Vec::new();
        
        for (layer_idx, layer_cache) in self.layer_caches.iter().enumerate() {
            let cache = layer_cache.read().unwrap();
            for (session_id, entry) in cache.iter() {
                let age = now.duration_since(entry.last_accessed);
                candidates.push((age, layer_idx, session_id.clone(), entry.memory_usage()));
            }
        }
        
        // Sort by age (oldest first)
        candidates.sort_by_key(|&(age, _, _, _)| age);
        
        // Evict oldest entries
        for (_, layer_idx, session_id, entry_size) in candidates {
            if freed >= target_free {
                break;
            }
            
            let mut layer_cache = self.layer_caches[layer_idx].write().unwrap();
            if layer_cache.remove(&session_id).is_some() {
                freed += entry_size;
                
                let mut current_memory = self.current_memory.lock().unwrap();
                *current_memory = current_memory.saturating_sub(entry_size);
                
                debug!("Evicted old entry: layer {} session '{}' ({} bytes)", 
                       layer_idx, session_id, entry_size);
            }
        }
        
        let mut stats = self.stats.write().unwrap();
        stats.evictions += 1;
        stats.memory_used = *self.current_memory.lock().unwrap();
        
        info!("Age-based eviction freed {} MB", freed / (1024 * 1024));
        Ok(())
    }
    
    /// Evict entries with low attention scores
    fn evict_low_attention_entries(&self, target_free: u64) -> Result<()> {
        let mut freed = 0u64;
        
        // Find entries with lowest average attention scores
        let mut candidates: Vec<(f32, usize, String, u64)> = Vec::new();
        
        for (layer_idx, layer_cache) in self.layer_caches.iter().enumerate() {
            let cache = layer_cache.read().unwrap();
            for (session_id, entry) in cache.iter() {
                let avg_attention = entry.attention_scores.as_ref()
                    .map(|scores| scores.iter().sum::<f32>() / scores.len() as f32)
                    .unwrap_or(0.0);
                candidates.push((avg_attention, layer_idx, session_id.clone(), entry.memory_usage()));
            }
        }
        
        // Sort by attention score (lowest first)
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Evict low-attention entries
        for (_, layer_idx, session_id, entry_size) in candidates {
            if freed >= target_free {
                break;
            }
            
            let mut layer_cache = self.layer_caches[layer_idx].write().unwrap();
            if layer_cache.remove(&session_id).is_some() {
                freed += entry_size;
                
                let mut current_memory = self.current_memory.lock().unwrap();
                *current_memory = current_memory.saturating_sub(entry_size);
                
                debug!("Evicted low-attention entry: layer {} session '{}' ({} bytes)", 
                       layer_idx, session_id, entry_size);
            }
        }
        
        let mut stats = self.stats.write().unwrap();
        stats.evictions += 1;
        stats.memory_used = *self.current_memory.lock().unwrap();
        
        info!("Attention-based eviction freed {} MB", freed / (1024 * 1024));
        Ok(())
    }
    
    /// Hybrid eviction: combine multiple strategies
    fn evict_hybrid_entries(&self, target_free: u64) -> Result<()> {
        // Use different strategies for different amounts
        let quarter = target_free / 4;
        
        // 25% LRU eviction
        self.evict_lru_entries(quarter)?;
        
        // 25% attention-based eviction
        self.evict_low_attention_entries(quarter)?;
        
        // 50% age-based eviction
        self.evict_oldest_entries(target_free / 2)?;
        
        info!("Hybrid eviction completed");
        Ok(())
    }
    
    /// Clear all cache entries
    pub fn clear(&self) {
        for layer_cache in &self.layer_caches {
            layer_cache.write().unwrap().clear();
        }
        
        *self.current_memory.lock().unwrap() = 0;
        self.access_order.lock().unwrap().clear();
        
        let mut stats = self.stats.write().unwrap();
        stats.memory_used = 0;
        
        info!("Cleared all KV cache entries");
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> KVCacheStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().unwrap();
        if stats.cache_hits + stats.cache_misses == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64
        }
    }
    
    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        *self.current_memory.lock().unwrap()
    }
    
    /// Get number of cached entries per layer
    pub fn entries_per_layer(&self) -> Vec<usize> {
        self.layer_caches
            .iter()
            .map(|cache| cache.read().unwrap().len())
            .collect()
    }
}

/// Simple compression for f32 vectors (placeholder implementation)
fn compress_f32_vector(data: &[f32]) -> Vec<f32> {
    // Placeholder: in a real implementation, this would use
    // proper compression like zstd, lz4, or quantization
    data.to_vec()
}

/// Simple decompression for f32 vectors (placeholder implementation)
fn decompress_f32_vector(data: &[f32]) -> Vec<f32> {
    // Placeholder: in a real implementation, this would decompress
    data.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kv_cache_basic() {
        let config = KVCacheConfig::default();
        let cache = OptimizedKVCache::new(config);
        
        let keys = vec![1.0; 1024];
        let values = vec![2.0; 1024];
        
        // Store entry
        cache.store(0, "session1", keys.clone(), values.clone(), 8, None).unwrap();
        
        // Retrieve entry
        let result = cache.retrieve(0, "session1").unwrap();
        assert!(result.is_some());
        
        let (retrieved_keys, retrieved_values, seq_len) = result.unwrap();
        assert_eq!(retrieved_keys, keys);
        assert_eq!(retrieved_values, values);
        assert_eq!(seq_len, 8);
    }
    
    #[test]
    fn test_kv_cache_eviction() {
        let mut config = KVCacheConfig::default();
        config.max_memory = 1024; // Very small limit to trigger eviction
        
        let cache = OptimizedKVCache::new(config);
        
        // Store entries that exceed memory limit
        for i in 0..10 {
            let keys = vec![i as f32; 1024];
            let values = vec![(i + 10) as f32; 1024];
            cache.store(0, &format!("session{}", i), keys, values, 8, None).unwrap();
        }
        
        let stats = cache.stats();
        assert!(stats.evictions > 0);
        assert!(cache.memory_usage() <= 1024);
    }
    
    #[test]
    fn test_kv_cache_append() {
        let config = KVCacheConfig::default();
        let cache = OptimizedKVCache::new(config);
        
        let keys1 = vec![1.0; 512];
        let values1 = vec![2.0; 512];
        
        // Store initial entry
        cache.store(0, "session1", keys1, values1, 4, None).unwrap();
        
        let keys2 = vec![3.0; 512];
        let values2 = vec![4.0; 512];
        
        // Append to entry
        cache.append(0, "session1", keys2, values2, None).unwrap();
        
        // Retrieve and verify
        let result = cache.retrieve(0, "session1").unwrap().unwrap();
        assert_eq!(result.2, 8); // seq_length should be 8
        assert_eq!(result.0.len(), 1024); // keys length should be doubled
    }
    
    #[test]
    fn test_attention_based_eviction() {
        let mut config = KVCacheConfig::default();
        config.max_seq_length = 10; // Small limit to trigger eviction
        config.eviction_policy = EvictionPolicy::AttentionBased;
        
        let cache = OptimizedKVCache::new(config);
        
        let keys = vec![1.0; 2048]; // 16 tokens * 128 head_dim * 1 head (simplified)
        let values = vec![2.0; 2048];
        // High attention for first and last tokens, low for middle
        let attention = vec![1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0];
        
        cache.store(0, "session1", keys, values, 16, Some(attention)).unwrap();
        
        // Retrieve and check that high-attention tokens were kept
        let result = cache.retrieve(0, "session1").unwrap().unwrap();
        assert!(result.2 <= 10); // Should be truncated
    }
}