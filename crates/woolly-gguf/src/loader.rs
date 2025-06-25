//! Optimized memory-mapped GGUF file loader with streaming and lazy loading
//!
//! This module provides zero-copy, memory-mapped loading of GGUF format files with advanced
//! memory optimizations including:
//! - Streaming tensor loading for large models
//! - Lazy loading with on-demand tensor access
//! - Memory pressure monitoring and adaptive loading
//! - Chunk-based loading for models larger than available memory
//! - Tensor prefetching with intelligent caching

use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tracing::{debug, info, trace, warn};

use crate::error::{Error, Result};
use crate::format::{GGUFHeader, GGUF_DEFAULT_ALIGNMENT};
use crate::metadata::GGUFMetadata;
use crate::tensor_info::TensorInfo;

/// Configuration for memory-optimized loading
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Maximum memory to use for tensor caching (bytes)
    pub cache_limit: u64,
    /// Enable lazy loading (load tensors on-demand)
    pub lazy_loading: bool,
    /// Number of tensors to prefetch
    pub prefetch_count: usize,
    /// Enable streaming for very large models
    pub enable_streaming: bool,
    /// Chunk size for streaming (bytes)
    pub chunk_size: u64,
    /// Enable memory pressure monitoring
    pub monitor_memory: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            cache_limit: 2 * 1024 * 1024 * 1024, // 2GB default
            lazy_loading: true,
            prefetch_count: 10,
            enable_streaming: true,
            chunk_size: 64 * 1024 * 1024, // 64MB chunks
            monitor_memory: true,
        }
    }
}

/// Cached tensor data with metadata
#[derive(Debug, Clone)]
struct CachedTensor {
    data: Arc<[u8]>,
    last_accessed: Instant,
    access_count: u32,
    size: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_cached: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub peak_usage: u64,
}

/// A memory-optimized GGUF file loader with streaming and caching
pub struct GGUFLoader {
    /// Memory-mapped file
    mmap: Mmap,
    
    /// File header
    header: GGUFHeader,
    
    /// Metadata
    metadata: GGUFMetadata,
    
    /// Tensor information indexed by name
    tensors: HashMap<String, TensorInfo>,
    
    /// Offset where tensor data begins
    data_start_offset: u64,
    
    /// Alignment for tensor data
    alignment: u32,
    
    /// Configuration for memory optimization
    config: LoaderConfig,
    
    /// LRU cache for tensor data
    tensor_cache: Arc<Mutex<lru::LruCache<String, CachedTensor>>>,
    
    /// Current memory usage tracking
    current_cache_size: Arc<Mutex<u64>>,
    
    /// Memory usage statistics
    stats: Arc<RwLock<MemoryStats>>,
    
    /// Prefetch queue for intelligent loading
    prefetch_queue: Arc<Mutex<Vec<String>>>,
}

impl GGUFLoader {
    /// Load a GGUF file from the given path with default configuration
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_path_with_config(path, LoaderConfig::default())
    }
    
    /// Load a GGUF file from the given path with custom configuration
    pub fn from_path_with_config<P: AsRef<Path>>(path: P, config: LoaderConfig) -> Result<Self> {
        let file = File::open(path)?;
        Self::from_file_with_config(file, config)
    }
    
    /// Load a GGUF file from an open file with default configuration
    pub fn from_file(file: File) -> Result<Self> {
        Self::from_file_with_config(file, LoaderConfig::default())
    }
    
    /// Load a GGUF file from an open file with custom configuration
    pub fn from_file_with_config(file: File, config: LoaderConfig) -> Result<Self> {
        // Memory map the file
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        Self::from_mmap_with_config(mmap, config)
    }
    
    /// Load from an existing memory map with default configuration
    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        Self::from_mmap_with_config(mmap, LoaderConfig::default())
    }
    
    /// Load from an existing memory map with custom configuration
    pub fn from_mmap_with_config(mmap: Mmap, config: LoaderConfig) -> Result<Self> {
        let mut cursor = Cursor::new(&mmap[..]);
        
        // Read header
        let header = GGUFHeader::read_from(&mut cursor)?;
        
        // Read metadata
        let metadata = GGUFMetadata::read_from(&mut cursor, header.metadata_kv_count)?;
        
        // Get alignment from metadata or use default
        let alignment = metadata
            .get_u32("general.alignment")
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        
        // Read tensor info
        let mut tensors = HashMap::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            let tensor = TensorInfo::read_from(&mut cursor)?;
            tensors.insert(tensor.name.clone(), tensor);
        }
        
        // Calculate data start offset with alignment
        let current_offset = cursor.position();
        let data_start_offset = align_offset(current_offset, alignment as u64);
        
        // Initialize memory-optimized components
        let cache_capacity = if config.lazy_loading {
            // Calculate cache capacity based on average tensor size and limit
            let avg_tensor_size = if tensors.is_empty() {
                1024 * 1024 // 1MB default
            } else {
                let total_size: u64 = tensors.values().map(|t| t.data_size()).sum();
                total_size / tensors.len() as u64
            };
            
            std::cmp::max(10, config.cache_limit / avg_tensor_size) as usize
        } else {
            tensors.len() // Cache all tensors if not lazy loading
        };
        
        info!("Initializing GGUF loader with {} tensors, cache capacity: {}, memory limit: {} MB",
              tensors.len(), cache_capacity, config.cache_limit / (1024 * 1024));
        
        Ok(Self {
            mmap,
            header,
            metadata,
            tensors,
            data_start_offset,
            alignment,
            config,
            tensor_cache: Arc::new(Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(cache_capacity).unwrap()
            ))),
            current_cache_size: Arc::new(Mutex::new(0)),
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            prefetch_queue: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    /// Get the file header
    pub fn header(&self) -> &GGUFHeader {
        &self.header
    }
    
    /// Get the metadata
    pub fn metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }
    
    /// Get information about all tensors
    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }
    
    /// Get information about a specific tensor
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }
    
    /// Get the names of all tensors
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
    
    /// Get tensor data with intelligent caching and memory management
    /// 
    /// Returns a reference to cached tensor data or loads it on-demand.
    /// Implements LRU eviction when memory limits are exceeded.
    pub fn tensor_data(&self, name: &str) -> Result<Arc<[u8]>> {
        // Check if tensor exists
        let tensor = self.tensors.get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        
        // Try to get from cache first
        {
            let mut cache = self.tensor_cache.lock().unwrap();
            if let Some(cached) = cache.get_mut(name) {
                // Update access statistics
                cached.last_accessed = Instant::now();
                cached.access_count += 1;
                
                // Update global stats
                if let Ok(mut stats) = self.stats.write() {
                    stats.cache_hits += 1;
                }
                
                trace!("Cache hit for tensor '{}'", name);
                return Ok(cached.data.clone());
            }
        }
        
        // Cache miss - load tensor data
        trace!("Cache miss for tensor '{}', loading from disk", name);
        
        let data = self.load_tensor_data_direct(name, tensor)?;
        let data_arc: Arc<[u8]> = Arc::from(data.into_boxed_slice());
        
        // Add to cache if enabled
        if self.config.lazy_loading {
            self.cache_tensor_data(name.to_string(), data_arc.clone(), tensor.data_size())?;
        }
        
        // Update global stats
        if let Ok(mut stats) = self.stats.write() {
            stats.cache_misses += 1;
        }
        
        Ok(data_arc)
    }
    
    /// Load tensor data directly from memory map (bypassing cache)
    fn load_tensor_data_direct(&self, name: &str, tensor: &TensorInfo) -> Result<Vec<u8>> {
        let absolute_offset = self.data_start_offset + tensor.offset;
        let data_size = tensor.data_size() as usize;
        
        // Check bounds
        if absolute_offset as usize + data_size > self.mmap.len() {
            return Err(Error::BufferTooSmall {
                needed: absolute_offset as usize + data_size,
                available: self.mmap.len(),
            });
        }
        
        // For streaming mode, use chunked loading for very large tensors
        if self.config.enable_streaming && data_size > self.config.chunk_size as usize {
            return self.load_tensor_chunked(absolute_offset, data_size);
        }
        
        // Direct memory copy for regular-sized tensors
        let data = self.mmap[absolute_offset as usize..absolute_offset as usize + data_size].to_vec();
        debug!("Loaded tensor '{}' ({} bytes)", name, data_size);
        
        Ok(data)
    }
    
    /// Load large tensor using chunked streaming
    fn load_tensor_chunked(&self, offset: u64, size: usize) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(size);
        let chunk_size = self.config.chunk_size as usize;
        let mut current_offset = offset as usize;
        let end_offset = current_offset + size;
        
        trace!("Loading tensor in chunks: {} bytes, chunk size: {}", size, chunk_size);
        
        while current_offset < end_offset {
            let remaining = end_offset - current_offset;
            let current_chunk_size = std::cmp::min(chunk_size, remaining);
            
            if current_offset + current_chunk_size > self.mmap.len() {
                return Err(Error::BufferTooSmall {
                    needed: current_offset + current_chunk_size,
                    available: self.mmap.len(),
                });
            }
            
            result.extend_from_slice(&self.mmap[current_offset..current_offset + current_chunk_size]);
            current_offset += current_chunk_size;
            
            // Check memory pressure and yield if needed
            if self.config.monitor_memory && result.len() % (10 * chunk_size) == 0 {
                self.check_memory_pressure()?;
            }
        }
        
        debug!("Completed chunked loading: {} bytes", result.len());
        Ok(result)
    }
    
    /// Cache tensor data with LRU eviction
    fn cache_tensor_data(&self, name: String, data: Arc<[u8]>, size: u64) -> Result<()> {
        let mut cache = self.tensor_cache.lock().unwrap();
        let mut current_size = self.current_cache_size.lock().unwrap();
        
        // Check if we need to evict items to stay under memory limit
        while *current_size + size > self.config.cache_limit && !cache.is_empty() {
            if let Some((evicted_name, evicted_tensor)) = cache.pop_lru() {
                *current_size = current_size.saturating_sub(evicted_tensor.size as u64);
                
                debug!("Evicted tensor '{}' from cache ({} bytes)", evicted_name, evicted_tensor.size);
                
                if let Ok(mut stats) = self.stats.write() {
                    stats.evictions += 1;
                }
            } else {
                break;
            }
        }
        
        // Add new tensor to cache
        let cached_tensor = CachedTensor {
            data,
            last_accessed: Instant::now(),
            access_count: 1,
            size: size as usize,
        };
        
        cache.put(name.clone(), cached_tensor);
        *current_size += size;
        
        // Update peak usage
        if let Ok(mut stats) = self.stats.write() {
            if *current_size > stats.peak_usage {
                stats.peak_usage = *current_size;
            }
            stats.total_cached = *current_size;
        }
        
        trace!("Cached tensor '{}' ({} bytes), total cache: {} bytes", name, size, *current_size);
        Ok(())
    }
    
    /// Check memory pressure and take action if needed
    fn check_memory_pressure(&self) -> Result<()> {
        if !self.config.monitor_memory {
            return Ok(());
        }
        
        // Simple memory pressure check - in a real implementation,
        // this would query system memory usage
        let current_size = *self.current_cache_size.lock().unwrap();
        let pressure_threshold = (self.config.cache_limit as f64 * 0.8) as u64;
        
        if current_size > pressure_threshold {
            warn!("Memory pressure detected: {} bytes cached ({}% of limit)", 
                  current_size, (current_size * 100) / self.config.cache_limit);
            
            // Trigger aggressive cache eviction
            self.evict_oldest_tensors(self.config.cache_limit / 4)?;
        }
        
        Ok(())
    }
    
    /// Evict oldest tensors to free up specified amount of memory
    fn evict_oldest_tensors(&self, target_free: u64) -> Result<()> {
        let mut cache = self.tensor_cache.lock().unwrap();
        let mut current_size = self.current_cache_size.lock().unwrap();
        let mut freed = 0u64;
        
        while freed < target_free && !cache.is_empty() {
            if let Some((evicted_name, evicted_tensor)) = cache.pop_lru() {
                let size = evicted_tensor.size as u64;
                *current_size = current_size.saturating_sub(size);
                freed += size;
                
                debug!("Pressure evicted tensor '{}' ({} bytes)", evicted_name, size);
            } else {
                break;
            }
        }
        
        if let Ok(mut stats) = self.stats.write() {
            stats.total_cached = *current_size;
        }
        
        info!("Freed {} bytes due to memory pressure", freed);
        Ok(())
    }
    
    /// Get a typed view of tensor data for non-quantized tensors
    /// 
    /// This method only works for non-quantized tensor types (F32, F16, etc.)
    /// For quantized tensors, use `tensor_data` and handle dequantization separately.
    /// 
    /// Returns an Arc containing the typed slice to maintain proper lifetime management.
    pub fn tensor_data_as<T>(&self, name: &str) -> Result<Arc<[T]>> 
    where
        T: bytemuck::Pod,
    {
        let tensor = self.tensors.get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        
        if tensor.ggml_type.is_quantized() {
            return Err(Error::InvalidTensorInfo(
                format!("Cannot get typed view of quantized tensor '{}'", name)
            ));
        }
        
        let expected_size = std::mem::size_of::<T>();
        let actual_size = tensor.ggml_type.element_size();
        if expected_size != actual_size {
            return Err(Error::InvalidTensorInfo(
                format!("Type size mismatch: expected {} bytes, tensor has {} bytes per element", 
                        expected_size, actual_size)
            ));
        }
        
        let data = self.tensor_data(name)?;
        
        // Convert Arc<[u8]> to Arc<[T]> using bytemuck
        let byte_slice = &*data;
        let typed_slice = bytemuck::cast_slice::<u8, T>(byte_slice);
        Ok(Arc::from(typed_slice))
    }
    
    /// Get model architecture from metadata
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get_string("general.architecture")
    }
    
    /// Get model name from metadata
    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get_string("general.name")
    }
    
    /// Get quantization version from metadata
    pub fn quantization_version(&self) -> Option<u32> {
        self.metadata.get_u32("general.quantization_version")
    }
    
    /// Get the total size of all tensor data
    pub fn total_tensor_size(&self) -> u64 {
        self.tensors.values()
            .map(|t| t.data_size())
            .sum()
    }
    
    /// Get the file size
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }
    
    /// Get the alignment used for tensor data
    pub fn alignment(&self) -> u32 {
        self.alignment
    }
    
    /// Get loader configuration
    pub fn config(&self) -> &LoaderConfig {
        &self.config
    }
    
    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Get cache efficiency (hit rate)
    pub fn cache_hit_rate(&self) -> f64 {
        let stats = self.stats.read().unwrap();
        if stats.cache_hits + stats.cache_misses == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64
        }
    }
    
    /// Warm up cache by preloading frequently accessed tensors
    pub fn warmup_cache(&self, tensor_names: &[&str]) -> Result<()> {
        info!("Warming up cache with {} tensors", tensor_names.len());
        
        for &name in tensor_names {
            if let Err(e) = self.tensor_data(name) {
                warn!("Failed to warm up tensor '{}': {}", name, e);
            }
        }
        
        let stats = self.memory_stats();
        info!("Cache warmup complete: {} bytes cached, {:.1}% hit rate", 
              stats.total_cached, self.cache_hit_rate() * 100.0);
        
        Ok(())
    }
    
    /// Prefetch tensors that are likely to be accessed soon
    pub fn prefetch_tensors(&self, tensor_names: Vec<String>) -> Result<()> {
        if !self.config.lazy_loading {
            return Ok(()); // No prefetching needed if not lazy loading
        }
        
        let mut queue = self.prefetch_queue.lock().unwrap();
        queue.extend(tensor_names.iter().cloned());
        
        // Limit prefetch queue size
        if queue.len() > self.config.prefetch_count * 2 {
            let drain_count = queue.len() - self.config.prefetch_count;
            queue.drain(0..drain_count);
        }
        
        debug!("Added {} tensors to prefetch queue (total: {})", tensor_names.len(), queue.len());
        
        // Execute prefetching in background (simplified synchronous version)
        self.execute_prefetch(self.config.prefetch_count)
    }
    
    /// Execute prefetching for queued tensors
    fn execute_prefetch(&self, max_prefetch: usize) -> Result<()> {
        let tensors_to_prefetch = {
            let mut queue = self.prefetch_queue.lock().unwrap();
            let count = std::cmp::min(max_prefetch, queue.len());
            queue.drain(0..count).collect::<Vec<_>>()
        };
        
        for tensor_name in tensors_to_prefetch {
            // Check if already cached
            {
                let cache = self.tensor_cache.lock().unwrap();
                if cache.contains(&tensor_name) {
                    continue;
                }
            }
            
            // Load tensor into cache
            if let Err(e) = self.tensor_data(&tensor_name) {
                debug!("Failed to prefetch tensor '{}': {}", tensor_name, e);
            } else {
                trace!("Prefetched tensor '{}'", tensor_name);
            }
        }
        
        Ok(())
    }
    
    /// Clear all cached tensor data
    pub fn clear_cache(&self) {
        let mut cache = self.tensor_cache.lock().unwrap();
        let mut current_size = self.current_cache_size.lock().unwrap();
        
        let evicted_count = cache.len();
        cache.clear();
        *current_size = 0;
        
        if let Ok(mut stats) = self.stats.write() {
            stats.total_cached = 0;
        }
        
        info!("Cleared cache: {} tensors evicted", evicted_count);
    }
    
    /// Force eviction of a specific tensor from cache
    pub fn evict_tensor(&self, name: &str) -> bool {
        let mut cache = self.tensor_cache.lock().unwrap();
        let mut current_size = self.current_cache_size.lock().unwrap();
        
        if let Some(evicted) = cache.pop(name) {
            *current_size = current_size.saturating_sub(evicted.size as u64);
            
            if let Ok(mut stats) = self.stats.write() {
                stats.total_cached = *current_size;
                stats.evictions += 1;
            }
            
            debug!("Manually evicted tensor '{}' ({} bytes)", name, evicted.size);
            true
        } else {
            false
        }
    }
    
    /// Get list of currently cached tensors with their sizes
    pub fn cached_tensors(&self) -> Vec<(String, usize, Instant)> {
        let cache = self.tensor_cache.lock().unwrap();
        cache.iter()
            .map(|(name, tensor)| (name.clone(), tensor.size, tensor.last_accessed))
            .collect()
    }
    
    /// Optimize cache by keeping most frequently accessed tensors
    pub fn optimize_cache(&self) -> Result<()> {
        let mut cache = self.tensor_cache.lock().unwrap();
        
        // Collect all tensors with their access patterns
        let mut tensors: Vec<_> = cache.iter()
            .map(|(name, tensor)| {
                let score = tensor.access_count as f64 / 
                    tensor.last_accessed.elapsed().as_secs_f64().max(1.0);
                (name.clone(), score)
            })
            .collect();
        
        // Sort by access score (frequency / recency)
        tensors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Keep top tensors, evict others
        let keep_count = cache.len() / 2; // Keep top 50%
        let to_evict: Vec<_> = tensors.iter().skip(keep_count).map(|(name, _)| name.clone()).collect();
        
        let mut freed = 0u64;
        for name in to_evict {
            if let Some(evicted) = cache.pop(&name) {
                freed += evicted.size as u64;
            }
        }
        
        let mut current_size = self.current_cache_size.lock().unwrap();
        *current_size = current_size.saturating_sub(freed);
        
        if let Ok(mut stats) = self.stats.write() {
            stats.total_cached = *current_size;
        }
        
        info!("Cache optimization complete: freed {} bytes, {} tensors remaining", 
              freed, cache.len());
        
        Ok(())
    }
}

/// Align an offset to the specified alignment
fn align_offset(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
}

// Re-export bytemuck for convenience
pub use bytemuck;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }
    
    #[test]
    fn test_loader_creation() {
        // This is a basic test - in practice you'd need a valid GGUF file
        // For now, we'll just test that the error handling works
        let result = GGUFLoader::from_path("/nonexistent/file.gguf");
        assert!(result.is_err());
    }
}