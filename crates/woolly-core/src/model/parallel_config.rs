//! Configuration for parallel inference execution
//! 
//! This module provides configuration and utilities for optimizing
//! parallel inference across CPU cores with proper load balancing.

use std::sync::Arc;
use rayon::{ThreadPoolBuilder, ThreadPool};

/// Configuration for parallel inference execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use for computation (defaults to num_cpus)
    pub num_threads: usize,
    /// Whether to enable parallel processing
    pub enable_parallel: bool,
    /// Thread pool for inference operations
    pub thread_pool: Option<Arc<ThreadPool>>,
    /// Minimum work size to justify parallelization
    pub parallel_threshold: ParallelThreshold,
    /// Work distribution strategy
    pub distribution_strategy: WorkDistributionStrategy,
}

/// Thresholds for determining when to use parallel processing
#[derive(Debug, Clone)]
pub struct ParallelThreshold {
    /// Minimum sequence length for parallel attention
    pub attention_seq_len: usize,
    /// Minimum matrix size for parallel matrix multiplication  
    pub matmul_size: usize,
    /// Minimum number of heads for parallel attention
    pub attention_heads: usize,
    /// Minimum batch size for parallel batch processing
    pub batch_size: usize,
    /// Minimum FFN intermediate size for parallel processing
    pub ffn_size: usize,
}

/// Strategy for distributing work across threads
#[derive(Debug, Clone, Copy)]
pub enum WorkDistributionStrategy {
    /// Distribute work evenly across all available threads
    Even,
    /// Use dynamic scheduling to balance load
    Dynamic,
    /// Optimize for cache locality
    CacheOptimized,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration with sensible defaults
    pub fn new() -> Self {
        let num_threads = num_cpus::get();
        
        Self {
            num_threads,
            enable_parallel: true,
            thread_pool: None,
            parallel_threshold: ParallelThreshold::default(),
            distribution_strategy: WorkDistributionStrategy::Hybrid,
        }
    }
    
    /// Create configuration optimized for a specific model size
    pub fn for_model(hidden_size: usize, num_heads: usize, num_layers: usize) -> Self {
        let mut config = Self::new();
        
        // Adjust thresholds based on model size
        if hidden_size >= 4096 {
            // Large model - more aggressive parallelization
            config.parallel_threshold = ParallelThreshold {
                attention_seq_len: 16,
                matmul_size: 1024,
                attention_heads: 2,
                batch_size: 1,
                ffn_size: 2048,
            };
        } else if hidden_size >= 2048 {
            // Medium model
            config.parallel_threshold = ParallelThreshold {
                attention_seq_len: 32,
                matmul_size: 2048,
                attention_heads: 4,
                batch_size: 2,
                ffn_size: 4096,
            };
        } else {
            // Small model - less parallelization overhead
            config.parallel_threshold = ParallelThreshold {
                attention_seq_len: 64,
                matmul_size: 4096,
                attention_heads: 8,
                batch_size: 4,
                ffn_size: 8192,
            };
        }
        
        // Choose strategy based on model characteristics
        if num_layers > 32 && num_heads > 16 {
            config.distribution_strategy = WorkDistributionStrategy::CacheOptimized;
        } else if num_heads > 8 {
            config.distribution_strategy = WorkDistributionStrategy::Dynamic;
        }
        
        config
    }
    
    /// Create configuration for CPU with specific characteristics
    pub fn for_cpu(num_cores: usize, has_smt: bool, cache_size_kb: usize) -> Self {
        let mut config = Self::new();
        
        // Adjust thread count based on SMT and workload characteristics
        config.num_threads = if has_smt {
            // For CPU-bound tasks like inference, often better to use physical cores only
            std::cmp::max(1, num_cores / 2)
        } else {
            num_cores
        };
        
        // Adjust strategy based on cache size
        if cache_size_kb > 8192 {
            config.distribution_strategy = WorkDistributionStrategy::CacheOptimized;
        } else {
            config.distribution_strategy = WorkDistributionStrategy::Dynamic;
        }
        
        config
    }
    
    /// Initialize the thread pool with the current configuration
    pub fn init_thread_pool(&mut self) -> Result<(), String> {
        if !self.enable_parallel {
            return Ok(());
        }
        
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .thread_name(|i| format!("woolly-inference-{}", i))
            .build()
            .map_err(|e| format!("Failed to create thread pool: {}", e))?;
            
        self.thread_pool = Some(Arc::new(pool));
        Ok(())
    }
    
    /// Get the thread pool, initializing if necessary
    pub fn get_thread_pool(&mut self) -> Result<Arc<ThreadPool>, String> {
        if self.thread_pool.is_none() {
            self.init_thread_pool()?;
        }
        
        self.thread_pool.as_ref()
            .cloned()
            .ok_or_else(|| "Thread pool not initialized".to_string())
    }
    
    /// Check if parallel processing should be used for attention computation
    pub fn should_parallelize_attention(&self, seq_len: usize, num_heads: usize) -> bool {
        self.enable_parallel &&
        seq_len >= self.parallel_threshold.attention_seq_len &&
        num_heads >= self.parallel_threshold.attention_heads
    }
    
    /// Check if parallel processing should be used for matrix multiplication
    pub fn should_parallelize_matmul(&self, m: usize, n: usize, k: usize) -> bool {
        self.enable_parallel &&
        (m * n * k) >= self.parallel_threshold.matmul_size
    }
    
    /// Check if parallel processing should be used for FFN computation
    pub fn should_parallelize_ffn(&self, intermediate_size: usize) -> bool {
        self.enable_parallel &&
        intermediate_size >= self.parallel_threshold.ffn_size
    }
    
    /// Check if parallel processing should be used for batch processing
    pub fn should_parallelize_batch(&self, batch_size: usize) -> bool {
        self.enable_parallel &&
        batch_size >= self.parallel_threshold.batch_size
    }
    
    /// Get the optimal chunk size for parallel processing
    pub fn get_chunk_size(&self, total_work: usize, work_type: WorkType) -> usize {
        let base_chunk_size = total_work / self.num_threads;
        
        match work_type {
            WorkType::Attention => {
                // For attention, prefer chunks that align with head boundaries
                std::cmp::max(1, base_chunk_size)
            }
            WorkType::MatMul => {
                // For matrix multiplication, prefer cache-aligned chunks
                let cache_line_size = 64; // bytes
                let f32_per_cache_line = cache_line_size / 4;
                ((base_chunk_size + f32_per_cache_line - 1) / f32_per_cache_line) * f32_per_cache_line
            }
            WorkType::FFN => {
                // For FFN, balance between parallelism and overhead
                std::cmp::max(32, base_chunk_size)
            }
            WorkType::Quantization => {
                // For quantization, align with block boundaries
                std::cmp::max(4, base_chunk_size)
            }
        }
    }
    
    /// Configure Rayon thread pool for current configuration
    pub fn configure_rayon(&self) -> Result<(), String> {
        if !self.enable_parallel {
            return Ok(());
        }
        
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .thread_name(|i| format!("woolly-rayon-{}", i))
            .build_global()
            .map_err(|e| format!("Failed to configure Rayon: {}", e))
    }
}

/// Type of work being performed for optimization purposes
#[derive(Debug, Clone, Copy)]
pub enum WorkType {
    /// Attention computation
    Attention,
    /// Matrix multiplication
    MatMul,
    /// Feed-forward network
    FFN,
    /// Quantization/dequantization
    Quantization,
}

impl Default for ParallelThreshold {
    fn default() -> Self {
        Self {
            attention_seq_len: 32,
            matmul_size: 4096,
            attention_heads: 4,
            batch_size: 2,
            ffn_size: 8192,
        }
    }
}

/// Utilities for load balancing and work distribution
pub mod load_balancing {
    use super::*;
    
    /// Calculate optimal work distribution across threads
    pub fn distribute_work(
        total_items: usize,
        num_threads: usize,
        strategy: WorkDistributionStrategy,
    ) -> Vec<(usize, usize)> {
        match strategy {
            WorkDistributionStrategy::Even => distribute_even(total_items, num_threads),
            WorkDistributionStrategy::Dynamic => distribute_dynamic(total_items, num_threads),
            WorkDistributionStrategy::CacheOptimized => distribute_cache_optimized(total_items, num_threads),
            WorkDistributionStrategy::Hybrid => distribute_hybrid(total_items, num_threads),
        }
    }
    
    /// Distribute work evenly across threads
    fn distribute_even(total_items: usize, num_threads: usize) -> Vec<(usize, usize)> {
        let chunk_size = total_items / num_threads;
        let remainder = total_items % num_threads;
        
        let mut ranges = Vec::new();
        let mut start = 0;
        
        for i in 0..num_threads {
            let current_chunk_size = chunk_size + if i < remainder { 1 } else { 0 };
            if current_chunk_size > 0 {
                ranges.push((start, start + current_chunk_size));
                start += current_chunk_size;
            }
        }
        
        ranges
    }
    
    /// Distribute work with dynamic load balancing considerations
    fn distribute_dynamic(total_items: usize, num_threads: usize) -> Vec<(usize, usize)> {
        // For dynamic distribution, use smaller chunks that can be stolen
        let min_chunk_size = 32;
        let optimal_chunks = total_items / min_chunk_size;
        let chunks_per_thread = std::cmp::max(1, optimal_chunks / num_threads);
        
        let mut ranges = Vec::new();
        let chunk_size = total_items / (chunks_per_thread * num_threads);
        let remainder = total_items % (chunks_per_thread * num_threads);
        
        let mut start = 0;
        for i in 0..(chunks_per_thread * num_threads) {
            let current_chunk_size = chunk_size + if i < remainder { 1 } else { 0 };
            if current_chunk_size > 0 && start < total_items {
                let end = std::cmp::min(start + current_chunk_size, total_items);
                ranges.push((start, end));
                start = end;
            }
        }
        
        ranges
    }
    
    /// Distribute work optimized for cache locality
    fn distribute_cache_optimized(total_items: usize, num_threads: usize) -> Vec<(usize, usize)> {
        // Align chunks to cache line boundaries
        const CACHE_LINE_SIZE: usize = 64; // bytes
        const F32_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 4;
        
        let base_chunk_size = total_items / num_threads;
        let aligned_chunk_size = ((base_chunk_size + F32_PER_CACHE_LINE - 1) / F32_PER_CACHE_LINE) * F32_PER_CACHE_LINE;
        
        let mut ranges = Vec::new();
        let mut start = 0;
        
        for _ in 0..num_threads {
            if start >= total_items {
                break;
            }
            let end = std::cmp::min(start + aligned_chunk_size, total_items);
            ranges.push((start, end));
            start = end;
        }
        
        ranges
    }
    
    /// Hybrid distribution combining multiple strategies
    fn distribute_hybrid(total_items: usize, num_threads: usize) -> Vec<(usize, usize)> {
        // Use cache-optimized for large workloads, dynamic for smaller ones
        if total_items > 16384 {
            distribute_cache_optimized(total_items, num_threads)
        } else {
            distribute_dynamic(total_items, num_threads)
        }
    }
    
    /// Calculate the optimal number of threads for a given workload
    pub fn optimal_thread_count(
        workload_size: usize,
        work_type: WorkType,
        available_threads: usize,
    ) -> usize {
        let min_work_per_thread = match work_type {
            WorkType::Attention => 64,   // Minimum sequence length per thread
            WorkType::MatMul => 1024,    // Minimum matrix elements per thread
            WorkType::FFN => 512,        // Minimum FFN elements per thread
            WorkType::Quantization => 32, // Minimum blocks per thread
        };
        
        let max_useful_threads = workload_size / min_work_per_thread;
        std::cmp::min(available_threads, std::cmp::max(1, max_useful_threads))
    }
}

/// Performance monitoring for parallel execution
pub mod performance {
    use std::time::{Duration, Instant};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    
    /// Performance metrics for parallel operations
    #[derive(Debug)]
    pub struct ParallelMetrics {
        /// Total time spent in parallel operations
        pub total_time: AtomicU64,
        /// Number of parallel operations performed
        pub operation_count: AtomicU64,
        /// Time spent in serial fallback operations
        pub serial_fallback_time: AtomicU64,
        /// Number of times serial fallback was used
        pub serial_fallback_count: AtomicU64,
    }
    
    impl ParallelMetrics {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                total_time: AtomicU64::new(0),
                operation_count: AtomicU64::new(0),
                serial_fallback_time: AtomicU64::new(0),
                serial_fallback_count: AtomicU64::new(0),
            })
        }
        
        /// Record a parallel operation
        pub fn record_parallel_op(&self, duration: Duration) {
            self.total_time.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            self.operation_count.fetch_add(1, Ordering::Relaxed);
        }
        
        /// Record a serial fallback operation
        pub fn record_serial_fallback(&self, duration: Duration) {
            self.serial_fallback_time.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            self.serial_fallback_count.fetch_add(1, Ordering::Relaxed);
        }
        
        /// Get average parallel operation time
        pub fn avg_parallel_time(&self) -> Duration {
            let total = self.total_time.load(Ordering::Relaxed);
            let count = self.operation_count.load(Ordering::Relaxed);
            
            if count > 0 {
                Duration::from_nanos(total / count)
            } else {
                Duration::ZERO
            }
        }
        
        /// Get parallel efficiency ratio
        pub fn parallel_efficiency(&self) -> f64 {
            let parallel_ops = self.operation_count.load(Ordering::Relaxed);
            let serial_ops = self.serial_fallback_count.load(Ordering::Relaxed);
            let total_ops = parallel_ops + serial_ops;
            
            if total_ops > 0 {
                parallel_ops as f64 / total_ops as f64
            } else {
                0.0
            }
        }
    }
    
    impl Default for ParallelMetrics {
        fn default() -> Self {
            Self {
                total_time: AtomicU64::new(0),
                operation_count: AtomicU64::new(0),
                serial_fallback_time: AtomicU64::new(0),
                serial_fallback_count: AtomicU64::new(0),
            }
        }
    }
    
    /// Timer for measuring parallel operation performance
    pub struct ParallelTimer {
        start: Instant,
        metrics: Arc<ParallelMetrics>,
        is_parallel: bool,
    }
    
    impl ParallelTimer {
        pub fn new(metrics: Arc<ParallelMetrics>, is_parallel: bool) -> Self {
            Self {
                start: Instant::now(),
                metrics,
                is_parallel,
            }
        }
    }
    
    impl Drop for ParallelTimer {
        fn drop(&mut self) {
            let duration = self.start.elapsed();
            if self.is_parallel {
                self.metrics.record_parallel_op(duration);
            } else {
                self.metrics.record_serial_fallback(duration);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_config_creation() {
        let config = ParallelConfig::new();
        assert!(config.enable_parallel);
        assert!(config.num_threads > 0);
    }
    
    #[test]
    fn test_model_specific_config() {
        let large_config = ParallelConfig::for_model(4096, 32, 48);
        let small_config = ParallelConfig::for_model(768, 12, 12);
        
        assert!(large_config.parallel_threshold.attention_seq_len <= small_config.parallel_threshold.attention_seq_len);
        assert!(large_config.parallel_threshold.matmul_size <= small_config.parallel_threshold.matmul_size);
    }
    
    #[test]
    fn test_work_distribution() {
        let ranges = load_balancing::distribute_work(100, 4, WorkDistributionStrategy::Even);
        assert_eq!(ranges.len(), 4);
        
        let total_items: usize = ranges.iter().map(|(start, end)| end - start).sum();
        assert_eq!(total_items, 100);
    }
    
    #[test]
    fn test_optimal_thread_count() {
        let threads = load_balancing::optimal_thread_count(1000, WorkType::MatMul, 8);
        assert!(threads <= 8);
        assert!(threads > 0);
        
        let threads_small = load_balancing::optimal_thread_count(10, WorkType::MatMul, 8);
        assert_eq!(threads_small, 1);
    }
    
    #[test]
    fn test_parallel_thresholds() {
        let config = ParallelConfig::new();
        
        assert!(config.should_parallelize_attention(64, 8));
        assert!(!config.should_parallelize_attention(8, 2));
        
        assert!(config.should_parallelize_matmul(100, 100, 100));
        assert!(!config.should_parallelize_matmul(10, 10, 10));
    }
}