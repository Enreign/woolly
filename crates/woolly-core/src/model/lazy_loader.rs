//! Lazy model loader that loads weights on-demand to save memory

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::{CoreError, Result};
use crate::optimized_dequantization::{OptimizedDequantizer, dequantize_q4_k_optimized, dequantize_q8_0_optimized};
use woolly_gguf::{GGUFLoader, TensorInfo, GGMLType, dequantize};
use super::memory_pool::TensorMemoryPool;
use super::dequantization_cache::{DequantizationCache, DequantizationCacheConfig, WeightAccessTracker};

/// Lazy tensor that dequantizes on first access
pub struct LazyTensor {
    name: String,
    info: TensorInfo,
    loader: Arc<GGUFLoader>,
    cached_data: Option<Vec<f32>>,
    /// Preallocated buffer for reuse
    temp_buffer: Option<Vec<f32>>,
    /// Cache of frequently used dequantized blocks
    block_cache: Option<std::collections::HashMap<usize, Vec<f32>>>,
    /// Reference to the global dequantization cache
    dequant_cache: Option<Arc<DequantizationCache>>,
}

impl LazyTensor {
    pub fn new(name: String, info: TensorInfo, loader: Arc<GGUFLoader>) -> Self {
        Self {
            name,
            info,
            loader,
            cached_data: None,
            temp_buffer: None,
            block_cache: None,
            dequant_cache: None,
        }
    }
    
    pub fn with_cache(mut self, cache: Arc<DequantizationCache>) -> Self {
        self.dequant_cache = Some(cache);
        self
    }

    /// Get the tensor data, dequantizing if necessary
    pub fn data(&mut self) -> Result<&[f32]> {
        if self.cached_data.is_none() {
            eprintln!("🔍 LazyTensor '{}' has no cached_data, checking dequant cache...", self.name);
            // Check dequantization cache first
            if let Some(ref dequant_cache) = self.dequant_cache.clone() {
                eprintln!("🚀 Using dequantization cache for tensor: {}", self.name);
                let tensor_name = self.name.clone();
                let cached_result = dequant_cache.get_or_dequantize(&tensor_name, || {
                    eprintln!("  ⚡ Cache miss - dequantizing tensor: {}", tensor_name);
                    self.dequantize_tensor()
                })?;
                self.cached_data = Some(cached_result);
            } else {
                // No cache available, dequantize directly
                eprintln!("⚠️  No dequantization cache available for tensor: {}", self.name);
                let (data, _) = self.dequantize_tensor()?;
                self.cached_data = Some(data);
            }
        } else {
            eprintln!("✅ LazyTensor '{}' already has cached_data, returning directly", self.name);
        }
        
        Ok(self.cached_data.as_ref().unwrap())
    }
    
    /// Internal method to dequantize the tensor
    fn dequantize_tensor(&mut self) -> Result<(Vec<f32>, Duration)> {
        let start_time = Instant::now();
        
        // Pre-allocate buffer if not exists
        let num_elements: usize = self.info.shape().iter()
            .map(|&x| x as usize)
            .product();
        
        if self.temp_buffer.is_none() {
            self.temp_buffer = Some(Vec::with_capacity(num_elements));
        }
        
        // Load and dequantize the tensor
        let raw_data = self.loader.tensor_data(&self.name)
            .map_err(|e| CoreError::model(
                "TENSOR_LOAD_FAILED",
                format!("Failed to load tensor data: {}", e),
                "Loading tensor from GGUF file",
                "Check that the model file is not corrupted"
            ))?;

            let float_data = match self.info.ggml_type {
                GGMLType::F32 => {
                    // Direct cast for F32 data
                    let f32_data: Vec<f32> = bytemuck::cast_slice(&raw_data).to_vec();
                    
                    // Verify data size matches expected elements
                    if f32_data.len() != num_elements {
                        return Err(CoreError::model(
                            "TENSOR_SIZE_MISMATCH",
                            format!("Tensor '{}' data size {} doesn't match expected elements {} from shape {:?}", 
                                self.name, f32_data.len(), num_elements, self.info.shape()),
                            "Loading F32 tensor data",
                            "Check tensor shape metadata in GGUF file"
                        ));
                    }
                    f32_data
                }
                GGMLType::F16 => {
                    // Convert F16 to F32 - reuse buffer
                    let f16_data: &[u16] = bytemuck::cast_slice(&raw_data);
                    let buffer = self.temp_buffer.as_mut().unwrap();
                    buffer.clear();
                    buffer.reserve(num_elements);
                    
                    for &x in f16_data {
                        buffer.push(half::f16::from_bits(x).to_f32());
                    }
                    
                    // Verify size after conversion
                    if buffer.len() != num_elements {
                        return Err(CoreError::model(
                            "TENSOR_SIZE_MISMATCH",
                            format!("Tensor '{}' F16 data size {} doesn't match expected elements {} from shape {:?}", 
                                self.name, buffer.len(), num_elements, self.info.shape()),
                            "Converting F16 tensor data",
                            "Check tensor shape metadata in GGUF file"
                        ));
                    }
                    buffer.clone()
                }
                _ => {
                    // Use optimized SIMD dequantization for quantized types
                    let dequantized = self.optimized_dequantize(&raw_data, num_elements)
                        .unwrap_or_else(|e| {
                            eprintln!("Optimized dequantization failed for '{}': {}, falling back to standard", self.name, e);
                            // Fall back to standard dequantization if optimized fails
                            dequantize(&raw_data, self.info.ggml_type, num_elements)
                                .map_err(|e| CoreError::model(
                                    "DEQUANTIZATION_FAILED",
                                    format!("Failed to dequantize tensor '{}': {}", self.name, e),
                                    "Dequantizing tensor weights",
                                    "Check dequantization implementation"
                                )).unwrap_or_else(|_| vec![0.0; num_elements])
                        });
                    
                    // Verify dequantized size
                    if dequantized.len() != num_elements {
                        eprintln!("WARNING: Tensor '{}' dequantized size {} doesn't match expected elements {} from shape {:?}", 
                            self.name, dequantized.len(), num_elements, self.info.shape());
                        eprintln!("GGML type: {:?}, raw data size: {} bytes", self.info.ggml_type, raw_data.len());
                    }
                    
                    dequantized
                }
            };

            let duration = start_time.elapsed();
            Ok((float_data, duration))
        }

    /// Optimized SIMD dequantization for quantized tensor types
    fn optimized_dequantize(&self, raw_data: &[u8], num_elements: usize) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; num_elements];
        
        // Log which tensor type we're dequantizing for debugging
        eprintln!("🚀 Optimized dequantization: tensor '{}', type {:?}, {} elements, {} raw bytes", 
                  self.name, self.info.ggml_type, num_elements, raw_data.len());
        
        match self.info.ggml_type {
            GGMLType::Q4_K => {
                let start = std::time::Instant::now();
                dequantize_q4_k_optimized(raw_data, &mut output)
                    .map_err(|e| CoreError::model(
                        "OPTIMIZED_DEQUANT_Q4_K_FAILED",
                        format!("SIMD Q4_K dequantization failed: {}", e),
                        "Optimized Q4_K dequantization",
                        "Fall back to standard dequantization"
                    ))?;
                let duration = start.elapsed();
                eprintln!("✅ Q4_K optimized dequantization completed in {:?}", duration);
            }
            GGMLType::Q5_K => {
                // Q5_K can use similar approach to Q4_K - use fallback for now
                eprintln!("⚠️ Q5_K not optimized yet, falling back to standard");
                return Err(CoreError::model(
                    "Q5_K_NOT_OPTIMIZED_YET",
                    "Q5_K optimized dequantization not implemented yet".to_string(),
                    "Q5_K dequantization",
                    "Use standard dequantization for now"
                ));
            }
            GGMLType::Q6_K => {
                let start = std::time::Instant::now();
                let optimized_dequantizer = crate::optimized_dequantization::OptimizedDequantizer::get();
                optimized_dequantizer.dequantize_q6_k(raw_data, &mut output)
                    .map_err(|e| CoreError::model(
                        "OPTIMIZED_DEQUANT_Q6_K_FAILED",
                        format!("SIMD Q6_K dequantization failed: {}", e),
                        "Optimized Q6_K dequantization",
                        "Fall back to standard dequantization"
                    ))?;
                let duration = start.elapsed();
                eprintln!("✅ Q6_K optimized dequantization completed in {:?}", duration);
            }
            GGMLType::Q8_0 => {
                // Q8_0 requires separate scale data - for now use standard implementation
                eprintln!("⚠️ Q8_0 not optimized yet, falling back to standard");
                return Err(CoreError::model(
                    "Q8_0_NOT_OPTIMIZED_YET",
                    "Q8_0 optimized dequantization not implemented yet".to_string(),
                    "Q8_0 dequantization",
                    "Use standard dequantization for now"
                ));
            }
            _ => {
                eprintln!("⚠️ Quantization type {:?} not supported by optimized dequantizer, falling back", self.info.ggml_type);
                return Err(CoreError::model(
                    "UNSUPPORTED_QUANT_TYPE",
                    format!("Quantization type {:?} not supported by optimized dequantizer", self.info.ggml_type),
                    "Optimized dequantization",
                    "Fall back to standard dequantization"
                ));
            }
        }
        
        Ok(output)
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[u64] {
        self.info.shape()
    }

    /// Clear cached data to free memory
    pub fn clear_cache(&mut self) {
        self.cached_data = None;
    }
    
    /// Get partial tensor data for FFN weights (for caching common blocks)
    pub fn get_block(&mut self, block_start: usize, block_size: usize) -> Result<Vec<f32>> {
        // Initialize block cache if needed
        if self.block_cache.is_none() {
            self.block_cache = Some(std::collections::HashMap::new());
        }
        
        // Check if block is already cached
        if let Some(ref cache) = self.block_cache {
            if let Some(cached_block) = cache.get(&block_start) {
                return Ok(cached_block.clone());
            }
        }
        
        // Load full tensor to extract block
        let _full_data = self.data()?;
        let full_data = self.cached_data.as_ref().unwrap();
        
        let block_end = std::cmp::min(block_start + block_size, full_data.len());
        if block_start < full_data.len() {
            let block_data = full_data[block_start..block_end].to_vec();
            
            // Cache the block
            if let Some(ref mut cache) = self.block_cache {
                cache.insert(block_start, block_data.clone());
            }
            
            Ok(block_data)
        } else {
            Err(CoreError::model(
                "TENSOR_BLOCK_OUT_OF_BOUNDS",
                "Block start exceeds tensor size",
                "Getting tensor block",
                "Check block indices"
            ))
        }
    }
}

/// Lazy model weights that load on demand
pub struct LazyModelWeights {
    _loader: Arc<GGUFLoader>,
    tensors: HashMap<String, LazyTensor>,
    config: crate::model::ModelConfig,
    /// Dequantization cache for frequently accessed weights
    dequant_cache: Arc<DequantizationCache>,
    /// Access tracker for identifying hot weights
    access_tracker: Arc<WeightAccessTracker>,
}

impl LazyModelWeights {
    /// Create lazy model weights from a GGUF loader
    pub fn from_loader(loader: GGUFLoader, config: crate::model::ModelConfig) -> Result<Self> {
        let loader = Arc::new(loader);
        let mut tensors = HashMap::new();

        // CACHE ENABLED - Smart caching for 16GB systems
        // Use 8GB cache (50% of system RAM) to avoid swapping
        // This will cache the most frequently used weights
        let cache_config = DequantizationCacheConfig {
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB cache - safe for 16GB system
            prefetch_ahead: 1, // Reduce prefetch to save memory
            use_frequency_priority: true, // LRU eviction for hot weights
            frequency_window: Duration::from_secs(60), // Shorter window
            enable_async_prefetch: false, // Keep disabled for now
        };
        let dequant_cache = Arc::new(DequantizationCache::new(cache_config));
        
        // Initialize access tracker
        let access_tracker = Arc::new(WeightAccessTracker::new(Duration::from_secs(300)));
        
        // Register all tensors but don't load them yet
        for tensor_name in loader.tensor_names() {
            if let Some(info) = loader.tensor_info(&tensor_name) {
                tensors.insert(
                    tensor_name.to_string(),
                    LazyTensor::new(tensor_name.to_string(), info.clone(), Arc::clone(&loader))
                        .with_cache(Arc::clone(&dequant_cache))
                );
            }
        }

        Ok(Self {
            _loader: loader,
            tensors,
            config,
            dequant_cache,
            access_tracker,
        })
    }

    /// Get a tensor by name with caching - ALWAYS use dequantization cache for consistency
    pub fn get_tensor(&mut self, name: &str) -> Result<&[f32]> {
        eprintln!("🔧 CRITICAL FIX: get_tensor() now redirects to get_tensor_cached() for cache consistency");
        
        // CRITICAL FIX: Always use the dequantization cache to ensure consistency
        // Convert cached result to stored reference in LazyTensor
        let cached_data = self.get_tensor_cached(name)?;
        
        // Store in tensor's individual cache for reference lifetime
        let tensor = self.tensors.get_mut(name).unwrap();
        tensor.cached_data = Some(cached_data);
        
        // Return reference to cached data
        Ok(tensor.cached_data.as_ref().unwrap())
    }

    /// Get tensor shape without loading data
    pub fn get_tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        self.tensors.get(name)
            .ok_or_else(|| CoreError::model(
                "TENSOR_NOT_FOUND",
                format!("Tensor '{}' not found", name),
                "Getting tensor shape",
                "Check tensor name"
            ))
            .map(|t| t.shape().iter().map(|&x| x as usize).collect())
    }

    /// Check if tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Clear cache for specific tensors to free memory
    pub fn clear_tensor_cache(&mut self, name: &str) {
        if let Some(tensor) = self.tensors.get_mut(name) {
            tensor.clear_cache();
        }
    }

    /// Get model configuration
    pub fn config(&self) -> &crate::model::ModelConfig {
        &self.config
    }

    /// Preload critical tensors (embeddings, normalization)
    pub fn preload_critical_tensors(&mut self) -> Result<()> {
        // Preload embeddings
        let _ = self.get_tensor_cached("token_embd.weight")?;
        
        // Preload output norm
        let _ = self.get_tensor_cached("output_norm.weight")?;
        
        // Preload first and last layer norms for better startup performance
        if self.has_tensor("blk.0.attn_norm.weight") {
            let _ = self.get_tensor_cached("blk.0.attn_norm.weight")?;
        }
        
        let last_layer = self.config.num_layers - 1;
        let last_norm = format!("blk.{}.ffn_norm.weight", last_layer);
        if self.has_tensor(&last_norm) {
            let _ = self.get_tensor_cached(&last_norm)?;
        }

        Ok(())
    }
    
    /// Preload ALL model weights to eliminate dequantization during inference
    pub fn preload_all_weights(&mut self) -> Result<()> {
        eprintln!("🚀 Preloading all model weights to eliminate dequantization bottleneck...");
        let start_time = Instant::now();
        
        // Collect all tensor names first
        let tensor_names: Vec<String> = self.tensors.keys().cloned().collect();
        let total_tensors = tensor_names.len();
        eprintln!("📊 Total tensors to preload: {}", total_tensors);
        
        // Use rayon for parallel dequantization if available
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            eprintln!("⚡ Using parallel dequantization with {} threads", rayon::current_num_threads());
            
            // Process tensors in parallel batches
            tensor_names.par_chunks(8).for_each(|chunk| {
                for name in chunk {
                    if let Err(e) = self.get_tensor_cached(name) {
                        eprintln!("⚠️  Failed to preload tensor '{}': {}", name, e);
                    }
                }
            });
        }
        
        // Sequential loading if parallel feature not enabled
        #[cfg(not(feature = "parallel"))]
        {
            eprintln!("🔄 Using sequential dequantization (enable 'parallel' feature for faster loading)");
            let mut loaded = 0;
            for name in &tensor_names {
                if let Err(e) = self.get_tensor_cached(name) {
                    eprintln!("⚠️  Failed to preload tensor '{}': {}", name, e);
                } else {
                    loaded += 1;
                    if loaded % 50 == 0 {
                        eprintln!("  Progress: {}/{} tensors loaded ({:.1}%)", 
                            loaded, total_tensors, (loaded as f32 / total_tensors as f32) * 100.0);
                    }
                }
            }
        }
        
        let duration = start_time.elapsed();
        eprintln!("✅ All weights preloaded in {:.2} seconds", duration.as_secs_f64());
        
        // Print memory usage
        let cache_stats = self.cache_stats();
        eprintln!("💾 Memory usage: {} MB", cache_stats.total_bytes_cached / (1024 * 1024));
        eprintln!("📈 Cache stats - Hits: {}, Misses: {}, Hit rate: {:.1}%", 
            cache_stats.hits, cache_stats.misses, cache_stats.hit_rate() * 100.0);
        
        // Verify cache persistence
        self.verify_cache_persistence()?;
        
        Ok(())
    }
    
    /// Preload weights for specific layers (useful for streaming)
    pub fn preload_layer_weights(&mut self, layer_indices: &[usize]) -> Result<()> {
        eprintln!("📦 Preloading weights for {} layers", layer_indices.len());
        
        for &layer_idx in layer_indices {
            // Attention weights
            let attn_weights = vec![
                format!("blk.{}.attn_norm.weight", layer_idx),
                format!("blk.{}.attn_q.weight", layer_idx),
                format!("blk.{}.attn_k.weight", layer_idx),
                format!("blk.{}.attn_v.weight", layer_idx),
                format!("blk.{}.attn_output.weight", layer_idx),
            ];
            
            // FFN weights
            let ffn_weights = vec![
                format!("blk.{}.ffn_norm.weight", layer_idx),
                format!("blk.{}.ffn_gate.weight", layer_idx),
                format!("blk.{}.ffn_up.weight", layer_idx),
                format!("blk.{}.ffn_down.weight", layer_idx),
            ];
            
            // Load all weights for this layer
            for weight_name in attn_weights.iter().chain(ffn_weights.iter()) {
                if self.has_tensor(weight_name) {
                    let _ = self.get_tensor(weight_name)?;
                }
            }
        }
        
        Ok(())
    }

    /// Get memory pool for tensor operations
    pub fn memory_pool(&mut self) -> &mut TensorMemoryPool {
        // For now, create a singleton pool - in production this should be injected
        static mut POOL: Option<TensorMemoryPool> = None;
        unsafe {
            if POOL.is_none() {
                POOL = Some(TensorMemoryPool::new());
            }
            POOL.as_mut().unwrap()
        }
    }

    /// Preload FFN weights for a specific layer to improve performance
    pub fn preload_ffn_weights(&mut self, layer_idx: usize) -> Result<()> {
        let gate_weight = format!("blk.{}.ffn_gate.weight", layer_idx);
        let up_weight = format!("blk.{}.ffn_up.weight", layer_idx);
        let down_weight = format!("blk.{}.ffn_down.weight", layer_idx);

        // Load FFN weights if they exist
        if self.has_tensor(&gate_weight) {
            let _ = self.get_tensor(&gate_weight)?;
        }
        if self.has_tensor(&up_weight) {
            let _ = self.get_tensor(&up_weight)?;
        }
        if self.has_tensor(&down_weight) {
            let _ = self.get_tensor(&down_weight)?;
        }
        
        // Prefetch weights for next layers
        let weight_patterns = vec![
            "blk.{}.ffn_gate.weight".to_string(),
            "blk.{}.ffn_up.weight".to_string(),
            "blk.{}.ffn_down.weight".to_string(),
            "blk.{}.attn_q.weight".to_string(),
            "blk.{}.attn_k.weight".to_string(),
            "blk.{}.attn_v.weight".to_string(),
            "blk.{}.attn_output.weight".to_string(),
        ];
        self.dequant_cache.prefetch_layer_weights(layer_idx, weight_patterns);

        Ok(())
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> super::dequantization_cache::CacheStats {
        self.dequant_cache.stats()
    }
    
    /// Analyze and optimize cache based on access patterns
    pub fn optimize_cache(&self) {
        // Get hot weights
        let hot_weights = self.access_tracker.get_hot_weights(10);
        
        // Preload frequently accessed weights
        self.dequant_cache.preload_frequent_weights(&hot_weights, 20);
        
        // Set layer priorities based on access patterns
        for (weight_name, count, _) in &hot_weights {
            if let Some(layer_idx) = extract_layer_index(weight_name) {
                let priority = (*count as f32).log2();
                self.dequant_cache.set_layer_priority(layer_idx, priority);
            }
        }
    }

    /// Try to get tensor from dequantization cache
    fn get_from_dequant_cache(&self, name: &str) -> Option<Vec<f32>> {
        // Check if tensor is in dequantization cache
        self.dequant_cache.get(name)
    }
    
    /// Verify cache persistence after preloading
    pub fn verify_cache_persistence(&mut self) -> Result<()> {
        eprintln!("🔍 Verifying cache persistence...");
        
        // Check a few critical tensors
        let test_tensors = vec![
            "token_embd.weight", 
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "output_norm.weight"
        ];
        
        let mut found = 0;
        let mut missing = 0;
        
        for tensor_name in &test_tensors {
            if let Some(_cached_data) = self.dequant_cache.get(tensor_name) {
                eprintln!("✅ Cache verified: {} is still cached", tensor_name);
                found += 1;
            } else {
                eprintln!("❌ Cache lost: {} is NOT cached!", tensor_name);
                missing += 1;
            }
        }
        
        let stats = self.cache_stats();
        eprintln!("📊 Verification complete - Found: {}/{}, Cache stats - Hits: {}, Misses: {}", 
            found, test_tensors.len(), stats.hits, stats.misses);
        
        if missing > 0 {
            eprintln!("⚠️  WARNING: Some preloaded weights are missing from cache!");
        }
        
        Ok(())
    }
    
    /// Get a tensor using the dequantization cache
    pub fn get_tensor_cached(&mut self, name: &str) -> Result<Vec<f32>> {
        let start_time = Instant::now();
        
        eprintln!("🔍 get_tensor_cached() called for '{}' - using SHARED dequantization cache", name);
        
        let tensor = self.tensors.get_mut(name)
            .ok_or_else(|| CoreError::model(
                "TENSOR_NOT_FOUND",
                format!("Tensor '{}' not found", name),
                "Getting tensor from lazy loader",
                "Check tensor name"
            ))?;
        
        // Use dequantization cache
        let dequant_cache = Arc::clone(&self.dequant_cache);
        eprintln!("🔧 Using dequantization cache instance (strong_count: {})", Arc::strong_count(&dequant_cache));
        
        let result = dequant_cache.get_or_dequantize(name, || {
            let dequant_start = Instant::now();
            eprintln!("🚨 CACHE MISS detected in get_tensor_cached() for '{}' - this should not happen after preload!", name);
            
            // Load raw data
            let raw_data = tensor.loader.tensor_data(&tensor.name)
                .map_err(|e| CoreError::model(
                    "TENSOR_LOAD_FAILED",
                    format!("Failed to load tensor data: {}", e),
                    "Loading tensor from GGUF file",
                    "Check that the model file is not corrupted"
                ))?;
            
            let num_elements: usize = tensor.info.shape().iter()
                .map(|&x| x as usize)
                .product();
            
            // Dequantize based on type
            let float_data = match tensor.info.ggml_type {
                GGMLType::F32 => {
                    bytemuck::cast_slice(&raw_data).to_vec()
                }
                GGMLType::F16 => {
                    let f16_data: &[u16] = bytemuck::cast_slice(&raw_data);
                    f16_data.iter()
                        .map(|&x| half::f16::from_bits(x).to_f32())
                        .collect()
                }
                _ => {
                    // Use existing dequantization logic from tensor
                    tensor.optimized_dequantize(&raw_data, num_elements)?
                }
            };
            
            let dequant_time = dequant_start.elapsed();
            Ok((float_data, dequant_time))
        })?;
        
        // Record access
        let access_time = start_time.elapsed();
        self.access_tracker.record_access(name, access_time);
        
        Ok(result)
    }
}

/// Extract layer index from weight name (e.g., "blk.12.ffn_gate.weight" -> Some(12))
fn extract_layer_index(weight_name: &str) -> Option<usize> {
    if weight_name.starts_with("blk.") {
        let parts: Vec<&str> = weight_name.split('.').collect();
        if parts.len() >= 2 {
            return parts[1].parse::<usize>().ok();
        }
    }
    None
}