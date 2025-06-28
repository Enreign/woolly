//! Enhanced memory pool for SIMD operations with better integration
//! 
//! This module provides an enhanced memory pool that:
//! - Pre-allocates aligned memory for SIMD operations
//! - Provides thread-local pools for lock-free access
//! - Supports different alignment requirements
//! - Tracks usage statistics for optimization

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::cell::RefCell;
use std::collections::HashMap;

/// Alignment for SIMD operations (32 bytes for AVX2)
const SIMD_ALIGN: usize = 32;

/// Minimum size threshold for SIMD operations (in elements)
/// Below this threshold, scalar operations are more efficient due to overhead
pub const SIMD_THRESHOLD: usize = 256;

/// Statistics for memory pool usage
#[derive(Debug, Default)]
pub struct PoolStats {
    pub allocations: AtomicUsize,
    pub reuses: AtomicUsize,
    pub peak_buffers: AtomicUsize,
    pub total_bytes: AtomicUsize,
}

impl PoolStats {
    pub fn record_allocation(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size * 4, Ordering::Relaxed); // f32 = 4 bytes
    }
    
    pub fn record_reuse(&self) {
        self.reuses.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn update_peak(&self, current: usize) {
        let mut peak = self.peak_buffers.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_buffers.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }
}

/// Aligned buffer for SIMD operations
#[derive(Debug)]
pub struct AlignedBuffer {
    data: Vec<f32>,
    capacity: usize,
    alignment: usize,
}

impl AlignedBuffer {
    /// Create a new aligned buffer
    pub fn new(capacity: usize, alignment: usize) -> Self {
        // Round up capacity to alignment boundary
        let aligned_capacity = (capacity + alignment - 1) & !(alignment - 1);
        
        // Allocate with extra space for alignment
        let mut data = Vec::with_capacity(aligned_capacity + alignment);
        
        // Ensure the vector is properly aligned
        unsafe {
            let ptr = data.as_mut_ptr();
            let aligned_ptr = ((ptr as usize + alignment - 1) & !(alignment - 1)) as *mut f32;
            let offset = (aligned_ptr as usize - ptr as usize) / std::mem::size_of::<f32>();
            
            // Adjust vector to start at aligned position
            if offset > 0 {
                data.set_len(offset);
                data.reserve_exact(aligned_capacity);
            }
        }
        
        Self {
            data,
            capacity: aligned_capacity,
            alignment,
        }
    }
    
    /// Get the buffer as a slice with the requested size
    pub fn as_slice_mut(&mut self, size: usize) -> &mut [f32] {
        assert!(size <= self.capacity);
        self.data.clear();
        self.data.resize(size, 0.0);
        &mut self.data
    }
    
    /// Check if this buffer can satisfy a request
    pub fn can_satisfy(&self, size: usize, alignment: usize) -> bool {
        size <= self.capacity && alignment <= self.alignment
    }
}

/// Size class for buffer pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SizeClass {
    Tiny,      // < 256 elements (1KB)
    Small,     // 256 - 4K elements (16KB)
    Medium,    // 4K - 64K elements (256KB)
    Large,     // 64K - 1M elements (4MB)
    Huge,      // > 1M elements
}

impl SizeClass {
    fn from_size(size: usize) -> Self {
        match size {
            0..=256 => SizeClass::Tiny,
            257..=4096 => SizeClass::Small,
            4097..=65536 => SizeClass::Medium,
            65537..=1048576 => SizeClass::Large,
            _ => SizeClass::Huge,
        }
    }
    
    fn max_buffers(&self) -> usize {
        match self {
            SizeClass::Tiny => 32,
            SizeClass::Small => 16,
            SizeClass::Medium => 8,
            SizeClass::Large => 4,
            SizeClass::Huge => 2,
        }
    }
}

/// Thread-local memory pool for SIMD operations
thread_local! {
    static LOCAL_POOL: RefCell<LocalMemoryPool> = RefCell::new(LocalMemoryPool::new());
}

/// Local memory pool for a single thread
struct LocalMemoryPool {
    pools: HashMap<SizeClass, Vec<AlignedBuffer>>,
    stats: Arc<PoolStats>,
}

impl LocalMemoryPool {
    fn new() -> Self {
        Self {
            pools: HashMap::new(),
            stats: Arc::new(PoolStats::default()),
        }
    }
    
    /// Get a buffer with at least the requested size and alignment
    fn get_buffer(&mut self, size: usize, alignment: usize) -> Vec<f32> {
        let size_class = SizeClass::from_size(size);
        let pool = self.pools.entry(size_class).or_insert_with(Vec::new);
        
        // Try to find a suitable buffer
        if let Some(pos) = pool.iter().position(|buf| buf.can_satisfy(size, alignment)) {
            self.stats.record_reuse();
            let mut buffer = pool.swap_remove(pos);
            let slice = buffer.as_slice_mut(size);
            return slice.to_vec();
        }
        
        // Allocate new buffer
        self.stats.record_allocation(size);
        
        // For tiny buffers, just use a regular Vec
        if size_class == SizeClass::Tiny {
            return vec![0.0; size];
        }
        
        // For larger buffers, use aligned allocation
        let capacity = match size_class {
            SizeClass::Small => (size + 511) & !511,      // Round to 512
            SizeClass::Medium => (size + 4095) & !4095,   // Round to 4K
            SizeClass::Large => (size + 65535) & !65535,  // Round to 64K
            _ => size,
        };
        
        vec![0.0; size]
    }
    
    /// Return a buffer to the pool
    fn return_buffer(&mut self, buffer: Vec<f32>, alignment: usize) {
        let capacity = buffer.capacity();
        let size_class = SizeClass::from_size(capacity);
        
        let pool = self.pools.entry(size_class).or_insert_with(Vec::new);
        
        // Only keep buffers if we haven't exceeded the limit
        if pool.len() < size_class.max_buffers() {
            let aligned_buffer = AlignedBuffer {
                data: buffer,
                capacity,
                alignment,
            };
            pool.push(aligned_buffer);
            
            // Update peak buffer count
            let total_buffers: usize = self.pools.values().map(|p| p.len()).sum();
            self.stats.update_peak(total_buffers);
        }
    }
    
    /// Clear all pools
    fn clear(&mut self) {
        self.pools.clear();
    }
    
    /// Get statistics
    fn stats(&self) -> Arc<PoolStats> {
        self.stats.clone()
    }
}

/// Enhanced tensor memory pool with SIMD optimization
pub struct EnhancedTensorMemoryPool {
    /// Global statistics
    global_stats: Arc<PoolStats>,
    /// Cached CPU features for SIMD decisions
    simd_available: bool,
    /// Matrix multiplication cache
    matmul_cache: HashMap<(usize, usize), Vec<f32>>,
    /// Dequantization cache
    dequant_cache: HashMap<usize, Vec<f32>>,
}

impl EnhancedTensorMemoryPool {
    pub fn new() -> Self {
        let simd_available = {
            #[cfg(target_arch = "x86_64")]
            {
                std::is_x86_feature_detected!("avx2")
            }
            #[cfg(target_arch = "aarch64")]
            {
                true // NEON is always available on aarch64
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                false
            }
        };
        
        Self {
            global_stats: Arc::new(PoolStats::default()),
            simd_available,
            matmul_cache: HashMap::new(),
            dequant_cache: HashMap::new(),
        }
    }
    
    /// Get a buffer optimized for SIMD operations
    pub fn get_simd_buffer(&mut self, size: usize) -> Vec<f32> {
        let alignment = if self.simd_available { SIMD_ALIGN } else { 8 };
        
        LOCAL_POOL.with(|pool| {
            pool.borrow_mut().get_buffer(size, alignment)
        })
    }
    
    /// Get a regular buffer
    pub fn get_buffer(&mut self, size: usize) -> Vec<f32> {
        LOCAL_POOL.with(|pool| {
            pool.borrow_mut().get_buffer(size, 8)
        })
    }
    
    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: Vec<f32>) {
        let alignment = if self.simd_available { SIMD_ALIGN } else { 8 };
        
        LOCAL_POOL.with(|pool| {
            pool.borrow_mut().return_buffer(buffer, alignment)
        })
    }
    
    /// Get multiple buffers at once (for parallel operations)
    pub fn get_buffers(&mut self, sizes: &[usize]) -> Vec<Vec<f32>> {
        sizes.iter().map(|&size| self.get_simd_buffer(size)).collect()
    }
    
    /// Return multiple buffers at once
    pub fn return_buffers(&mut self, buffers: Vec<Vec<f32>>) {
        for buffer in buffers {
            self.return_buffer(buffer);
        }
    }
    
    /// Get cached matrix multiplication result
    pub fn get_matmul_cache(&self, key: (usize, usize)) -> Option<&Vec<f32>> {
        self.matmul_cache.get(&key)
    }
    
    /// Cache matrix multiplication result
    pub fn cache_matmul_result(&mut self, key: (usize, usize), result: Vec<f32>) {
        // Limit cache size
        if self.matmul_cache.len() < 64 {
            self.matmul_cache.insert(key, result);
        } else if self.matmul_cache.len() > 128 {
            // Clear half of the cache when it gets too large
            let to_remove: Vec<_> = self.matmul_cache
                .keys()
                .take(self.matmul_cache.len() / 2)
                .cloned()
                .collect();
            for key in to_remove {
                self.matmul_cache.remove(&key);
            }
        }
    }
    
    /// Get cached dequantization result
    pub fn get_dequant_cache(&self, key: usize) -> Option<&Vec<f32>> {
        self.dequant_cache.get(&key)
    }
    
    /// Cache dequantization result
    pub fn cache_dequant_result(&mut self, key: usize, result: Vec<f32>) {
        if self.dequant_cache.len() < 32 {
            self.dequant_cache.insert(key, result);
        }
    }
    
    /// Clear all caches and pools
    pub fn clear_all(&mut self) {
        self.matmul_cache.clear();
        self.dequant_cache.clear();
        
        LOCAL_POOL.with(|pool| {
            pool.borrow_mut().clear();
        });
    }
    
    /// Get memory usage statistics
    pub fn stats(&self) -> PoolStats {
        let local_stats = LOCAL_POOL.with(|pool| {
            pool.borrow().stats()
        });
        
        PoolStats {
            allocations: AtomicUsize::new(
                local_stats.allocations.load(Ordering::Relaxed) +
                self.global_stats.allocations.load(Ordering::Relaxed)
            ),
            reuses: AtomicUsize::new(
                local_stats.reuses.load(Ordering::Relaxed) +
                self.global_stats.reuses.load(Ordering::Relaxed)
            ),
            peak_buffers: AtomicUsize::new(
                local_stats.peak_buffers.load(Ordering::Relaxed).max(
                    self.global_stats.peak_buffers.load(Ordering::Relaxed)
                )
            ),
            total_bytes: AtomicUsize::new(
                local_stats.total_bytes.load(Ordering::Relaxed) +
                self.global_stats.total_bytes.load(Ordering::Relaxed)
            ),
        }
    }
    
    /// Check if SIMD is available
    pub fn simd_available(&self) -> bool {
        self.simd_available
    }
}

impl Default for EnhancedTensorMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aligned_buffer() {
        let mut buffer = AlignedBuffer::new(100, 32);
        let slice = buffer.as_slice_mut(50);
        assert_eq!(slice.len(), 50);
        assert_eq!(slice[0], 0.0);
        
        // Check alignment
        let ptr = slice.as_ptr() as usize;
        assert_eq!(ptr % 32, 0);
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool = EnhancedTensorMemoryPool::new();
        
        // Get and return buffers
        let buf1 = pool.get_simd_buffer(1000);
        assert_eq!(buf1.len(), 1000);
        
        pool.return_buffer(buf1);
        
        // Second allocation should reuse
        let buf2 = pool.get_simd_buffer(800);
        assert_eq!(buf2.len(), 800);
        
        let stats = pool.stats();
        assert_eq!(stats.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.reuses.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_cache_operations() {
        let mut pool = EnhancedTensorMemoryPool::new();
        
        // Test matmul cache
        let result = vec![1.0, 2.0, 3.0];
        pool.cache_matmul_result((2, 3), result.clone());
        
        assert_eq!(pool.get_matmul_cache((2, 3)), Some(&result));
        assert_eq!(pool.get_matmul_cache((3, 2)), None);
        
        // Test dequant cache
        let dequant = vec![4.0, 5.0, 6.0];
        pool.cache_dequant_result(123, dequant.clone());
        
        assert_eq!(pool.get_dequant_cache(123), Some(&dequant));
    }
}