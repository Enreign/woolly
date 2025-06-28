//! Memory pool for tensor operations to reduce allocations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use parking_lot::RwLock;

/// Thread-safe memory pool for tensor operations to reduce allocations
pub struct TensorMemoryPool {
    /// Reusable buffers for different tensor sizes (thread-safe)
    small_buffers: Arc<Mutex<Vec<Vec<f32>>>>,    // < 1KB
    medium_buffers: Arc<Mutex<Vec<Vec<f32>>>>,   // 1KB - 1MB  
    large_buffers: Arc<Mutex<Vec<Vec<f32>>>>,    // > 1MB
    /// Matrix multiplication result cache (read-heavy, write-light)
    matmul_cache: Arc<RwLock<HashMap<(usize, usize), Vec<f32>>>>,
    /// Kernel-specific buffer pools for fused operations
    qkv_buffers: Arc<Mutex<Vec<Vec<f32>>>>,      // QKV projection buffers
    attention_buffers: Arc<Mutex<Vec<Vec<f32>>>>, // Attention computation buffers
    ffn_buffers: Arc<Mutex<Vec<Vec<f32>>>>,      // FFN intermediate buffers
    /// Pre-allocated working memory for common sizes
    working_memory: Arc<RwLock<HashMap<usize, Vec<Vec<f32>>>>>,
}

/// Legacy single-threaded memory pool for backwards compatibility
pub struct LegacyTensorMemoryPool {
    /// Reusable buffers for different tensor sizes
    small_buffers: Vec<Vec<f32>>,    // < 1KB
    medium_buffers: Vec<Vec<f32>>,   // 1KB - 1MB  
    large_buffers: Vec<Vec<f32>>,    // > 1MB
    /// Matrix multiplication result cache
    matmul_cache: HashMap<(usize, usize), Vec<f32>>,
    /// Kernel-specific buffer pools for fused operations
    qkv_buffers: Vec<Vec<f32>>,      // QKV projection buffers
    attention_buffers: Vec<Vec<f32>>, // Attention computation buffers
    ffn_buffers: Vec<Vec<f32>>,      // FFN intermediate buffers
    /// Pre-allocated working memory for common sizes
    working_memory: HashMap<usize, Vec<Vec<f32>>>,
}

impl TensorMemoryPool {
    pub fn new() -> Self {
        Self {
            small_buffers: Arc::new(Mutex::new(Vec::new())),
            medium_buffers: Arc::new(Mutex::new(Vec::new())),
            large_buffers: Arc::new(Mutex::new(Vec::new())),
            matmul_cache: Arc::new(RwLock::new(HashMap::new())),
            qkv_buffers: Arc::new(Mutex::new(Vec::new())),
            attention_buffers: Arc::new(Mutex::new(Vec::new())),
            ffn_buffers: Arc::new(Mutex::new(Vec::new())),
            working_memory: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a clone of this memory pool that shares the same underlying buffers
    pub fn clone_shared(&self) -> Self {
        Self {
            small_buffers: Arc::clone(&self.small_buffers),
            medium_buffers: Arc::clone(&self.medium_buffers),
            large_buffers: Arc::clone(&self.large_buffers),
            matmul_cache: Arc::clone(&self.matmul_cache),
            qkv_buffers: Arc::clone(&self.qkv_buffers),
            attention_buffers: Arc::clone(&self.attention_buffers),
            ffn_buffers: Arc::clone(&self.ffn_buffers),
            working_memory: Arc::clone(&self.working_memory),
        }
    }
    
    /// Pre-allocate working memory for common tensor sizes to eliminate allocations
    pub fn preallocate_for_model(&self, config: &crate::model::fused_kernels::FusedKernelConfig, max_seq_len: usize) {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let kv_size = num_kv_heads * config.head_dim;
        
        // Pre-allocate buffers for different sequence lengths
        for seq_len in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048].iter() {
            if *seq_len > max_seq_len { break; }
            
            let seq_len = *seq_len;
            
            // Normalized input buffer
            self.preallocate_buffer(seq_len * hidden_size);
            
            // QKV projection buffer
            let qkv_size = hidden_size + 2 * kv_size;
            self.preallocate_buffer(seq_len * qkv_size);
            
            // Attention scores and weights
            self.preallocate_buffer(seq_len * seq_len * num_heads);
            
            // Attention output
            self.preallocate_buffer(seq_len * hidden_size);
            
            // FFN gate+up buffer
            self.preallocate_buffer(seq_len * 2 * intermediate_size);
            
            // FFN intermediate buffer
            self.preallocate_buffer(seq_len * intermediate_size);
        }
    }
    
    /// Pre-allocate a buffer of specific size (thread-safe)
    fn preallocate_buffer(&self, size: usize) {
        let mut working_memory = self.working_memory.write();
        let buffers = working_memory.entry(size).or_insert_with(Vec::new);
        if buffers.len() < 4 { // Keep up to 4 buffers per size
            buffers.push(vec![0.0; size]);
        }
    }
    
    /// Get a buffer optimized for fused operations (thread-safe)
    pub fn get_fused_buffer(&self, size: usize, buffer_type: FusedBufferType) -> Vec<f32> {
        // Try pre-allocated working memory first
        {
            let mut working_memory = self.working_memory.write();
            if let Some(buffers) = working_memory.get_mut(&size) {
                if let Some(buffer) = buffers.pop() {
                    return buffer;
                }
            }
        }
        
        // Try kernel-specific pools
        let pool_arc = match buffer_type {
            FusedBufferType::QKV => &self.qkv_buffers,
            FusedBufferType::Attention => &self.attention_buffers,
            FusedBufferType::FFN => &self.ffn_buffers,
            FusedBufferType::General => return self.get_buffer(size),
        };
        
        // Find suitable buffer in kernel-specific pool
        {
            let mut pool = pool_arc.lock().unwrap();
            for (i, buf) in pool.iter().enumerate() {
                if buf.capacity() >= size {
                    let mut reused_buf = pool.swap_remove(i);
                    reused_buf.clear();
                    reused_buf.resize(size, 0.0);
                    return reused_buf;
                }
            }
        }
        
        // Fallback to general allocation
        self.get_buffer(size)
    }
    
    /// Return a buffer to the appropriate pool (thread-safe)
    pub fn return_fused_buffer(&self, mut buffer: Vec<f32>, buffer_type: FusedBufferType) {
        let size = buffer.capacity();
        buffer.clear();
        
        // Try to return to working memory first
        {
            let mut working_memory = self.working_memory.write();
            if let Some(buffers) = working_memory.get_mut(&size) {
                if buffers.len() < 4 {
                    buffers.push(buffer);
                    return;
                }
            }
        }
        
        // Return to kernel-specific pool
        let pool_arc = match buffer_type {
            FusedBufferType::QKV => &self.qkv_buffers,
            FusedBufferType::Attention => &self.attention_buffers,
            FusedBufferType::FFN => &self.ffn_buffers,
            FusedBufferType::General => {
                self.return_buffer(buffer);
                return;
            }
        };
        
        let mut pool = pool_arc.lock().unwrap();
        if pool.len() < 8 {
            pool.push(buffer);
        }
    }
    
    /// Get multiple buffers atomically for fused operations (thread-safe)
    pub fn get_fused_buffers(&self, sizes: &[(usize, FusedBufferType)]) -> Vec<Vec<f32>> {
        sizes.iter()
            .map(|(size, buffer_type)| self.get_fused_buffer(*size, *buffer_type))
            .collect()
    }
    
    /// Return multiple buffers atomically (thread-safe)
    pub fn return_fused_buffers(&self, buffers: Vec<Vec<f32>>, buffer_types: &[FusedBufferType]) {
        for (buffer, buffer_type) in buffers.into_iter().zip(buffer_types.iter()) {
            self.return_fused_buffer(buffer, *buffer_type);
        }
    }
    
    /// Get a buffer of the requested size, reusing if possible (thread-safe)
    pub fn get_buffer(&self, size: usize) -> Vec<f32> {
        let buffer_pool_arc = if size < 256 {  // < 1KB
            &self.small_buffers
        } else if size < 262_144 {  // < 1MB
            &self.medium_buffers
        } else {
            &self.large_buffers
        };
        
        // Try to reuse existing buffer
        {
            let mut buffer_pool = buffer_pool_arc.lock().unwrap();
            for (i, buf) in buffer_pool.iter().enumerate() {
                if buf.capacity() >= size {
                    let mut reused_buf = buffer_pool.swap_remove(i);
                    reused_buf.clear();
                    reused_buf.resize(size, 0.0);
                    return reused_buf;
                }
            }
        }
        
        // Allocate new buffer if no suitable one found
        vec![0.0; size]
    }
    
    /// Return a buffer to the pool for reuse (thread-safe)
    pub fn return_buffer(&self, mut buffer: Vec<f32>) {
        let size = buffer.capacity();
        buffer.clear();
        
        let buffer_pool_arc = if size < 256 {
            &self.small_buffers
        } else if size < 262_144 {
            &self.medium_buffers
        } else {
            &self.large_buffers
        };
        
        // Only keep a reasonable number of buffers per size
        let mut buffer_pool = buffer_pool_arc.lock().unwrap();
        if buffer_pool.len() < 8 {
            buffer_pool.push(buffer);
        }
    }
    
    /// Get cached matrix multiplication result (thread-safe)
    pub fn get_matmul_cache(&self, key: (usize, usize)) -> Option<Vec<f32>> {
        let cache = self.matmul_cache.read();
        cache.get(&key).cloned()
    }
    
    /// Cache matrix multiplication result (thread-safe)
    pub fn cache_matmul_result(&self, key: (usize, usize), result: Vec<f32>) {
        let mut cache = self.matmul_cache.write();
        // Limit cache size to prevent memory bloat
        if cache.len() < 32 {
            cache.insert(key, result);
        }
    }
    
    /// Clear all cached data to free memory (thread-safe)
    pub fn clear_cache(&self) {
        {
            let mut cache = self.matmul_cache.write();
            cache.clear();
        }
        {
            let mut buffers = self.small_buffers.lock().unwrap();
            buffers.clear();
        }
        {
            let mut buffers = self.medium_buffers.lock().unwrap();
            buffers.clear();
        }
        {
            let mut buffers = self.large_buffers.lock().unwrap();
            buffers.clear();
        }
        {
            let mut buffers = self.qkv_buffers.lock().unwrap();
            buffers.clear();
        }
        {
            let mut buffers = self.attention_buffers.lock().unwrap();
            buffers.clear();
        }
        {
            let mut buffers = self.ffn_buffers.lock().unwrap();
            buffers.clear();
        }
        {
            let mut working_memory = self.working_memory.write();
            working_memory.clear();
        }
    }
}

/// Buffer type for kernel-specific optimization
#[derive(Debug, Clone, Copy)]
pub enum FusedBufferType {
    QKV,       // QKV projection buffers
    Attention, // Attention computation buffers  
    FFN,       // FFN intermediate buffers
    General,   // General purpose buffers
}

// Thread-safe memory pool traits
unsafe impl Send for TensorMemoryPool {}
unsafe impl Sync for TensorMemoryPool {}

impl Clone for TensorMemoryPool {
    fn clone(&self) -> Self {
        self.clone_shared()
    }
}

impl Default for TensorMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}