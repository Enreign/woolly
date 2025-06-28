//! Memory pool for efficient tensor allocation and reuse
//!
//! This module provides a memory pool implementation that reduces allocation overhead
//! and memory fragmentation by reusing tensor storage buffers.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, trace};

use crate::backend::{DType, Device, Result as TensorResult};
use crate::shape::Shape;

/// Configuration for memory pool behavior
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum total memory to pool (bytes)
    pub max_pool_size: u64,
    /// Maximum size of individual allocations to pool
    pub max_allocation_size: u64,
    /// Time before unused allocations are freed
    pub ttl: Duration,
    /// Enable automatic defragmentation
    pub enable_defrag: bool,
    /// Defragmentation threshold (fragmentation percentage)
    pub defrag_threshold: f64,
    /// Enable allocation alignment
    pub enable_alignment: bool,
    /// Alignment boundary (bytes)
    pub alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1024 * 1024 * 1024, // 1GB
            max_allocation_size: 256 * 1024 * 1024, // 256MB
            ttl: Duration::from_secs(300), // 5 minutes
            enable_defrag: true,
            defrag_threshold: 0.3, // 30% fragmentation
            enable_alignment: true,
            alignment: 64, // 64-byte alignment for SIMD
        }
    }
}

/// Metadata for a pooled memory allocation
#[derive(Debug, Clone)]
struct PooledAllocation {
    /// Raw memory buffer
    buffer: Vec<u8>,
    /// Size of the allocation
    size: usize,
    /// Time when allocation was returned to pool
    returned_at: Instant,
    /// Number of times this allocation has been reused
    reuse_count: u32,
    /// Device where allocation resides
    _device: Device,
    /// Data type stored in allocation
    _dtype: DType,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    /// Total memory currently pooled (bytes)
    pub total_pooled: u64,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Number of pooled (free) allocations
    pub pooled_allocations: usize,
    /// Total allocations served
    pub total_allocations: u64,
    /// Cache hits (reused allocations)
    pub cache_hits: u64,
    /// Cache misses (new allocations)
    pub cache_misses: u64,
    /// Total bytes allocated
    pub bytes_allocated: u64,
    /// Peak memory usage
    pub peak_usage: u64,
    /// Number of defragmentation cycles
    pub defrag_cycles: u64,
    /// Total bytes freed during cleanup
    pub bytes_freed: u64,
}

/// A memory pool for efficient tensor storage allocation
pub struct MemoryPool {
    /// Configuration
    config: MemoryPoolConfig,
    
    /// Pools organized by size buckets for different devices/dtypes
    /// Key: (device, dtype, size_bucket), Value: queue of available allocations
    pools: Arc<Mutex<HashMap<(Device, DType, usize), VecDeque<PooledAllocation>>>>,
    
    /// Size buckets for efficient allocation matching
    size_buckets: Vec<usize>,
    
    /// Statistics
    stats: Arc<Mutex<MemoryPoolStats>>,
    
    /// Current total pooled memory
    current_usage: Arc<Mutex<u64>>,
    
    /// Track active allocations to prevent double-free
    active_allocations: Arc<Mutex<HashMap<usize, (usize, Device, DType)>>>,
    
    /// Next allocation ID
    next_allocation_id: Arc<Mutex<usize>>,
}

impl MemoryPool {
    /// Create a new memory pool with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }
    
    /// Create a new memory pool with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        // Create size buckets: powers of 2 from 1KB to max_allocation_size
        let mut size_buckets = Vec::new();
        let mut size = 1024; // Start at 1KB
        while size <= config.max_allocation_size as usize {
            size_buckets.push(size);
            size *= 2;
        }
        
        info!("Initialized memory pool with {} size buckets, max size: {} MB",
              size_buckets.len(), config.max_pool_size / (1024 * 1024));
        
        Self {
            config,
            pools: Arc::new(Mutex::new(HashMap::new())),
            size_buckets,
            stats: Arc::new(Mutex::new(MemoryPoolStats::default())),
            current_usage: Arc::new(Mutex::new(0)),
            active_allocations: Arc::new(Mutex::new(HashMap::new())),
            next_allocation_id: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Allocate tensor storage from the pool
    pub fn allocate(&self, shape: &Shape, dtype: DType, device: Device) -> TensorResult<PooledTensorStorage> {
        let element_size = dtype.size_in_bytes();
        let total_elements = shape.numel();
        let required_size = total_elements * element_size;
        
        // Apply alignment if enabled
        let aligned_size = if self.config.enable_alignment {
            align_size(required_size, self.config.alignment)
        } else {
            required_size
        };
        
        // Check if allocation is too large for pooling
        if aligned_size > self.config.max_allocation_size as usize {
            debug!("Allocation too large for pooling ({} bytes), using direct allocation", aligned_size);
            return self.allocate_direct(aligned_size, dtype, device);
        }
        
        // Find appropriate size bucket
        let bucket_size = self.find_size_bucket(aligned_size);
        let pool_key = (device, dtype, bucket_size);
        
        // Try to reuse from pool first
        if let Some(allocation) = self.try_reuse_allocation(&pool_key, aligned_size) {
            trace!("Reused allocation from pool: {} bytes", bucket_size);
            
            let mut stats = self.stats.lock().unwrap();
            stats.cache_hits += 1;
            stats.total_allocations += 1;
            
            return Ok(PooledTensorStorage::new(
                allocation.buffer,
                aligned_size,
                dtype,
                device,
                Some(self.clone_arc()),
            ));
        }
        
        // No suitable allocation found, create new one
        debug!("Creating new allocation: {} bytes (bucket: {})", aligned_size, bucket_size);
        
        let buffer = self.allocate_new_buffer(bucket_size, dtype, device)?;
        
        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_misses += 1;
            stats.total_allocations += 1;
            stats.bytes_allocated += bucket_size as u64;
            
            let current_usage = *self.current_usage.lock().unwrap();
            if current_usage > stats.peak_usage {
                stats.peak_usage = current_usage;
            }
        }
        
        Ok(PooledTensorStorage::new(
            buffer,
            aligned_size,
            dtype,
            device,
            Some(self.clone_arc()),
        ))
    }
    
    /// Try to reuse an allocation from the pool
    fn try_reuse_allocation(&self, pool_key: &(Device, DType, usize), required_size: usize) -> Option<PooledAllocation> {
        let mut pools = self.pools.lock().unwrap();
        
        if let Some(pool) = pools.get_mut(pool_key) {
            // Find the best-fit allocation (smallest that fits)
            let mut best_idx = None;
            let mut best_size = usize::MAX;
            
            for (idx, allocation) in pool.iter().enumerate() {
                if allocation.size >= required_size && allocation.size < best_size {
                    best_idx = Some(idx);
                    best_size = allocation.size;
                    
                    // Perfect fit, use it immediately
                    if allocation.size == required_size {
                        break;
                    }
                }
            }
            
            if let Some(idx) = best_idx {
                let mut allocation = pool.remove(idx).unwrap();
                allocation.reuse_count += 1;
                
                // Update current usage
                let mut current_usage = self.current_usage.lock().unwrap();
                *current_usage = current_usage.saturating_sub(allocation.size as u64);
                
                return Some(allocation);
            }
        }
        
        None
    }
    
    /// Allocate new buffer
    fn allocate_new_buffer(&self, size: usize, _dtype: DType, device: Device) -> TensorResult<Vec<u8>> {
        match device {
            Device::Cpu => {
                let mut buffer = Vec::with_capacity(size);
                buffer.resize(size, 0);
                Ok(buffer)
            }
            Device::Cuda(_) => {
                // For CUDA, we might need special allocation
                // For now, allocate on CPU and transfer later
                let mut buffer = Vec::with_capacity(size);
                buffer.resize(size, 0);
                Ok(buffer)
            }
            Device::Metal => {
                // For Metal, use unified memory allocation
                let mut buffer = Vec::with_capacity(size);
                buffer.resize(size, 0);
                Ok(buffer)
            }
        }
    }
    
    /// Allocate directly without pooling (for large allocations)
    fn allocate_direct(&self, size: usize, dtype: DType, device: Device) -> TensorResult<PooledTensorStorage> {
        let buffer = self.allocate_new_buffer(size, dtype, device)?;
        
        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_misses += 1;
            stats.total_allocations += 1;
            stats.bytes_allocated += size as u64;
        }
        
        Ok(PooledTensorStorage::new(buffer, size, dtype, device, None))
    }
    
    /// Return an allocation to the pool for reuse
    pub fn deallocate(&self, buffer: Vec<u8>, size: usize, dtype: DType, device: Device) {
        // Check if allocation should be pooled
        if size > self.config.max_allocation_size as usize {
            trace!("Allocation too large for pooling, freeing directly");
            return; // Let Vec::drop handle it
        }
        
        // Check pool capacity
        let current_usage = *self.current_usage.lock().unwrap();
        if current_usage >= self.config.max_pool_size {
            trace!("Pool at capacity, freeing allocation directly");
            return;
        }
        
        let bucket_size = self.find_size_bucket(size);
        let pool_key = (device, dtype, bucket_size);
        
        let allocation = PooledAllocation {
            buffer,
            size,
            returned_at: Instant::now(),
            reuse_count: 0,
            _device: device,
            _dtype: dtype,
        };
        
        // Add to appropriate pool
        {
            let mut pools = self.pools.lock().unwrap();
            let pool = pools.entry(pool_key).or_insert_with(VecDeque::new);
            pool.push_back(allocation);
            
            // Update usage tracking
            let mut current_usage = self.current_usage.lock().unwrap();
            *current_usage += size as u64;
        }
        
        trace!("Returned allocation to pool: {} bytes", size);
        
        // Trigger cleanup if needed
        if self.should_cleanup() {
            self.cleanup_expired();
        }
    }
    
    /// Find the appropriate size bucket for an allocation
    fn find_size_bucket(&self, size: usize) -> usize {
        // Find the smallest bucket that can fit the allocation
        for &bucket_size in &self.size_buckets {
            if bucket_size >= size {
                return bucket_size;
            }
        }
        
        // If no bucket fits, use the largest bucket
        // This shouldn't happen given max_allocation_size check
        *self.size_buckets.last().unwrap()
    }
    
    /// Check if cleanup is needed
    fn should_cleanup(&self) -> bool {
        let current_usage = *self.current_usage.lock().unwrap();
        current_usage > self.config.max_pool_size / 2 // Cleanup when over 50% full
    }
    
    /// Clean up expired allocations
    pub fn cleanup_expired(&self) {
        let now = Instant::now();
        let mut pools = self.pools.lock().unwrap();
        let mut freed_bytes = 0u64;
        let mut freed_count = 0;
        
        for pool in pools.values_mut() {
            // Remove expired allocations
            let _original_len = pool.len();
            pool.retain(|allocation| {
                let expired = now.duration_since(allocation.returned_at) > self.config.ttl;
                if expired {
                    freed_bytes += allocation.size as u64;
                    freed_count += 1;
                }
                !expired
            });
        }
        
        // Update usage tracking
        if freed_bytes > 0 {
            let mut current_usage = self.current_usage.lock().unwrap();
            *current_usage = current_usage.saturating_sub(freed_bytes);
            
            let mut stats = self.stats.lock().unwrap();
            stats.bytes_freed += freed_bytes;
            
            debug!("Cleaned up {} expired allocations, freed {} bytes", freed_count, freed_bytes);
        }
    }
    
    /// Force cleanup of all pooled allocations
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        let mut freed_bytes = 0u64;
        let mut freed_count = 0;
        
        for pool in pools.values_mut() {
            for allocation in pool.drain(..) {
                freed_bytes += allocation.size as u64;
                freed_count += 1;
            }
        }
        
        *self.current_usage.lock().unwrap() = 0;
        
        let mut stats = self.stats.lock().unwrap();
        stats.bytes_freed += freed_bytes;
        
        info!("Cleared memory pool: {} allocations freed, {} bytes", freed_count, freed_bytes);
    }
    
    /// Get memory pool statistics
    pub fn stats(&self) -> MemoryPoolStats {
        let mut stats = self.stats.lock().unwrap();
        let pools = self.pools.lock().unwrap();
        
        stats.total_pooled = *self.current_usage.lock().unwrap();
        stats.pooled_allocations = pools.values().map(|p| p.len()).sum();
        stats.active_allocations = self.active_allocations.lock().unwrap().len();
        
        stats.clone()
    }
    
    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        if stats.total_allocations == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / stats.total_allocations as f64
        }
    }
    
    /// Clone the Arc for sharing with storage
    fn clone_arc(&self) -> Arc<MemoryPool> {
        Arc::new(MemoryPool {
            config: self.config.clone(),
            pools: Arc::clone(&self.pools),
            size_buckets: self.size_buckets.clone(),
            stats: Arc::clone(&self.stats),
            current_usage: Arc::clone(&self.current_usage),
            active_allocations: Arc::clone(&self.active_allocations),
            next_allocation_id: Arc::clone(&self.next_allocation_id),
        })
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor storage that automatically returns memory to pool when dropped
pub struct PooledTensorStorage {
    buffer: Option<Vec<u8>>,
    size: usize,
    dtype: DType,
    device: Device,
    pool: Option<Arc<MemoryPool>>,
}

impl PooledTensorStorage {
    fn new(
        buffer: Vec<u8>,
        size: usize,
        dtype: DType,
        device: Device,
        pool: Option<Arc<MemoryPool>>,
    ) -> Self {
        Self {
            buffer: Some(buffer),
            size,
            dtype,
            device,
            pool,
        }
    }
    
    /// Get a reference to the underlying buffer
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().unwrap()
    }
    
    /// Get a mutable reference to the underlying buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut().unwrap()
    }
    
    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    /// Get the device
    pub fn device(&self) -> Device {
        self.device
    }
}

impl Drop for PooledTensorStorage {
    fn drop(&mut self) {
        if let (Some(buffer), Some(pool)) = (self.buffer.take(), &self.pool) {
            pool.deallocate(buffer, self.size, self.dtype, self.device);
        }
    }
}

/// Align size to specified boundary
fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Global memory pool instance
static GLOBAL_POOL: std::sync::OnceLock<Arc<MemoryPool>> = std::sync::OnceLock::new();

/// Get or initialize the global memory pool
pub fn global_pool() -> &'static Arc<MemoryPool> {
    GLOBAL_POOL.get_or_init(|| Arc::new(MemoryPool::new()))
}

/// Initialize global pool with custom configuration
pub fn init_global_pool(config: MemoryPoolConfig) -> &'static Arc<MemoryPool> {
    GLOBAL_POOL.get_or_init(|| Arc::new(MemoryPool::with_config(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;
    
    #[test]
    fn test_memory_pool_basic() {
        let pool = MemoryPool::new();
        let shape = Shape::vector(1000);
        
        // Allocate storage
        let storage = pool.allocate(&shape, DType::F32, Device::Cpu).unwrap();
        assert_eq!(storage.size(), 4000); // 1000 * 4 bytes
        
        let stats = pool.stats();
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.total_allocations, 1);
    }
    
    #[test]
    fn test_memory_pool_reuse() {
        let pool = MemoryPool::new();
        let shape = Shape::vector(1000);
        
        // Allocate and drop storage
        {
            let _storage = pool.allocate(&shape, DType::F32, Device::Cpu).unwrap();
        }
        
        // Allocate again - should reuse
        let _storage2 = pool.allocate(&shape, DType::F32, Device::Cpu).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.total_allocations, 2);
    }
    
    #[test]
    fn test_size_buckets() {
        let pool = MemoryPool::new();
        
        // Test various sizes map to correct buckets
        assert_eq!(pool.find_size_bucket(500), 1024);
        assert_eq!(pool.find_size_bucket(1024), 1024);
        assert_eq!(pool.find_size_bucket(1500), 2048);
        assert_eq!(pool.find_size_bucket(2048), 2048);
    }
    
    #[test]
    fn test_alignment() {
        assert_eq!(align_size(100, 64), 128);
        assert_eq!(align_size(64, 64), 64);
        assert_eq!(align_size(65, 64), 128);
        assert_eq!(align_size(1, 64), 64);
    }
    
    #[test]
    fn test_pool_cleanup() {
        let mut config = MemoryPoolConfig::default();
        config.ttl = Duration::from_millis(10); // Very short TTL for testing
        
        let pool = MemoryPool::with_config(config);
        let shape = Shape::vector(1000);
        
        // Allocate and drop storage
        {
            let _storage = pool.allocate(&shape, DType::F32, Device::Cpu).unwrap();
        }
        
        // Wait for TTL and cleanup
        std::thread::sleep(Duration::from_millis(20));
        pool.cleanup_expired();
        
        let stats = pool.stats();
        assert!(stats.bytes_freed > 0);
    }
}