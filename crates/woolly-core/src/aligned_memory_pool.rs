//! Aligned memory pool for zero-allocation inference
//! 
//! This module provides a high-performance memory pool with aligned allocations
//! to maximize SIMD performance and eliminate allocations during inference.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::ops::{Deref, DerefMut};

/// Alignment for SIMD operations (32 bytes for AVX, works for NEON too)
const SIMD_ALIGNMENT: usize = 32;

/// Common buffer sizes used during inference
const COMMON_SIZES: &[usize] = &[
    256,       // Single block
    1024,      // 1K elements
    4096,      // Hidden size
    16384,     // 4K * 4 (attention scores)
    65536,     // 16K * 4 (QKV projections)
    262144,    // 64K * 4 (FFN intermediate)
    1048576,   // 256K * 4 (large FFN)
    4194304,   // 1M * 4 (embedding table)
    16777216,  // 4M * 4 (large weight matrices)
];

/// Aligned buffer with SIMD-friendly memory layout
pub struct AlignedBuffer {
    ptr: *mut f32,
    capacity: usize,
    layout: Layout,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Create a new aligned buffer
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size * std::mem::size_of::<f32>(), SIMD_ALIGNMENT)
            .expect("Invalid layout");
        
        let ptr = unsafe { alloc(layout) as *mut f32 };
        
        if ptr.is_null() {
            panic!("Failed to allocate aligned memory");
        }
        
        // Initialize to zero for safety
        unsafe {
            ptr::write_bytes(ptr, 0, size);
        }
        
        AlignedBuffer {
            ptr,
            capacity: size,
            layout,
        }
    }
    
    /// Get a mutable slice to the buffer
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }
    
    /// Get a slice to the buffer
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity) }
    }
    
    /// Get the raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

/// Pool of aligned buffers for a specific size
struct BufferPool {
    size: usize,
    buffers: Vec<AlignedBuffer>,
    available: Vec<bool>,
}

impl BufferPool {
    fn new(size: usize, count: usize) -> Self {
        let mut buffers = Vec::with_capacity(count);
        let mut available = Vec::with_capacity(count);
        
        for _ in 0..count {
            buffers.push(AlignedBuffer::new(size));
            available.push(true);
        }
        
        BufferPool {
            size,
            buffers,
            available,
        }
    }
    
    fn acquire(&mut self) -> Option<&mut AlignedBuffer> {
        for (i, avail) in self.available.iter_mut().enumerate() {
            if *avail {
                *avail = false;
                return Some(&mut self.buffers[i]);
            }
        }
        
        // All buffers in use, allocate a new one
        self.buffers.push(AlignedBuffer::new(self.size));
        self.available.push(false);
        self.buffers.last_mut()
    }
    
    fn release(&mut self, ptr: *mut f32) {
        for (i, buffer) in self.buffers.iter().enumerate() {
            if buffer.ptr == ptr {
                self.available[i] = true;
                return;
            }
        }
    }
}

/// High-performance aligned memory pool
pub struct AlignedMemoryPool {
    pools: Arc<Mutex<Vec<BufferPool>>>,
    allocation_count: AtomicUsize,
    hit_count: AtomicUsize,
}

impl AlignedMemoryPool {
    /// Create a new memory pool with pre-allocated buffers
    pub fn new() -> Self {
        let mut pools = Vec::new();
        
        // Pre-allocate common sizes
        for &size in COMMON_SIZES {
            // Allocate multiple buffers for each size
            let count = match size {
                s if s <= 4096 => 16,      // Many small buffers
                s if s <= 65536 => 8,      // Moderate medium buffers
                s if s <= 1048576 => 4,    // Few large buffers
                _ => 2,                     // Very few huge buffers
            };
            
            pools.push(BufferPool::new(size, count));
        }
        
        AlignedMemoryPool {
            pools: Arc::new(Mutex::new(pools)),
            allocation_count: AtomicUsize::new(0),
            hit_count: AtomicUsize::new(0),
        }
    }
    
    /// Get a buffer of at least the specified size
    pub fn acquire(&self, min_size: usize) -> AlignedBufferHandle {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        let mut pools = self.pools.lock();
        
        // Find the smallest pool that fits
        for pool in pools.iter_mut() {
            if pool.size >= min_size {
                if let Some(buffer) = pool.acquire() {
                    self.hit_count.fetch_add(1, Ordering::Relaxed);
                    let ptr = buffer.as_mut_ptr();
                    let capacity = buffer.capacity;
                    return AlignedBufferHandle {
                        ptr,
                        capacity,
                        actual_size: min_size,
                        pool: Arc::clone(&self.pools),
                    };
                }
            }
        }
        
        // No suitable pool found, create a new one
        let size = min_size.next_power_of_two();
        let new_pool_idx = pools.len();
        pools.push(BufferPool::new(size, 1));
        
        if let Some(buffer) = pools[new_pool_idx].acquire() {
            let ptr = buffer.as_mut_ptr();
            let capacity = buffer.capacity;
            AlignedBufferHandle {
                ptr,
                capacity,
                actual_size: min_size,
                pool: Arc::clone(&self.pools),
            }
        } else {
            panic!("Failed to acquire buffer from new pool");
        }
    }
    
    /// Get statistics about pool usage
    pub fn stats(&self) -> (usize, usize, f64) {
        let allocations = self.allocation_count.load(Ordering::Relaxed);
        let hits = self.hit_count.load(Ordering::Relaxed);
        let hit_rate = if allocations > 0 {
            hits as f64 / allocations as f64
        } else {
            0.0
        };
        (allocations, hits, hit_rate)
    }
}

/// Handle to an aligned buffer that returns it to the pool on drop
pub struct AlignedBufferHandle {
    ptr: *mut f32,
    capacity: usize,
    actual_size: usize,
    pool: Arc<Mutex<Vec<BufferPool>>>,
}

impl AlignedBufferHandle {
    /// Get a mutable slice to the buffer
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.actual_size) }
    }
    
    /// Get a slice to the buffer
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.actual_size) }
    }
    
    /// Get the raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }
    
    /// Convert to a Vec without copying (takes ownership)
    pub fn into_vec(self) -> Vec<f32> {
        let slice = unsafe { std::slice::from_raw_parts(self.ptr, self.actual_size) };
        slice.to_vec()
    }
}

impl Drop for AlignedBufferHandle {
    fn drop(&mut self) {
        let mut pools = self.pool.lock();
        for pool in pools.iter_mut() {
            if pool.size == self.capacity {
                pool.release(self.ptr);
                return;
            }
        }
    }
}

impl Deref for AlignedBufferHandle {
    type Target = [f32];
    
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedBufferHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

// Global memory pool instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_MEMORY_POOL: AlignedMemoryPool = AlignedMemoryPool::new();
}

/// Get a buffer from the global memory pool
pub fn get_buffer(size: usize) -> AlignedBufferHandle {
    GLOBAL_MEMORY_POOL.acquire(size)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aligned_allocation() {
        let mut buffer = AlignedBuffer::new(1024);
        let ptr = buffer.as_mut_ptr() as usize;
        assert_eq!(ptr % SIMD_ALIGNMENT, 0, "Buffer should be aligned");
        
        let slice = buffer.as_mut_slice();
        assert_eq!(slice.len(), 1024);
        
        // Test writing
        slice[0] = 1.0;
        slice[1023] = 2.0;
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1023], 2.0);
    }
    
    #[test]
    fn test_memory_pool() {
        let pool = AlignedMemoryPool::new();
        
        // Get multiple buffers
        let mut handles = vec![];
        for i in 0..5 {
            let mut handle = pool.acquire(1024);
            handle.as_mut_slice()[0] = i as f32;
            handles.push(handle);
        }
        
        // Check they're different
        for (i, handle) in handles.iter().enumerate() {
            assert_eq!(handle.as_slice()[0], i as f32);
        }
        
        // Drop and check stats
        drop(handles);
        let (allocations, hits, _) = pool.stats();
        assert_eq!(allocations, 5);
        assert!(hits > 0);
    }
}