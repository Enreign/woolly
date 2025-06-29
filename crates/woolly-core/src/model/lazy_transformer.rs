//! Memory-efficient transformer that loads weights on demand

use std::sync::{Arc, Mutex};
use std::path::Path;
use async_trait::async_trait;
use crate::{CoreError, Result};
use crate::tensor_utils::{embedding_lookup, layer_norm, rms_norm, matmul, swiglu, tensor_from_slice};
use crate::aligned_memory_pool::get_buffer;
use crate::blas_attention::grouped_query_attention_blas;
use crate::tensor_utils_simd::{simd_matvec, simd_rms_norm, simd_swiglu, simd_residual_add, simd_attention_projections, simd_ffn_forward};
use crate::model::{Model, ModelOutput};
use crate::kv_cache::{KVCacheConfig, OptimizedKVCache};
use super::lazy_loader::LazyModelWeights;
use super::transformer::{TransformerConfig, NormType};
use super::memory_pool::TensorMemoryPool;
use woolly_tensor::Shape;

/// Lazy-loading transformer model
pub struct LazyTransformer {
    config: TransformerConfig,
    weights: Arc<Mutex<LazyModelWeights>>,
    name: String,
    kv_cache: Option<Arc<OptimizedKVCache>>,
    memory_pool: Arc<Mutex<TensorMemoryPool>>,
}

impl LazyTransformer {
    /// Create a new lazy transformer from a GGUF file
    pub async fn from_gguf(path: &Path, config: TransformerConfig) -> Result<Self> {
        use woolly_gguf::GGUFLoader;
        use crate::model::loader::{GGUFModelLoader, ModelLoader};
        
        // Load GGUF file
        let loader = GGUFModelLoader::from_path(path)?;
        let model_config = loader.config()?;
        
        // Create lazy weights
        let gguf_loader = GGUFLoader::from_path(path)
            .map_err(|e| CoreError::model(
                "GGUF_LOAD_FAILED",
                format!("Failed to load GGUF file: {}", e),
                "Loading GGUF model",
                "Check file path and format"
            ))?;
            
        let mut lazy_weights = LazyModelWeights::from_loader(gguf_loader, model_config.clone())?;
        
        // Smart preloading for 16GB systems - only preload critical weights
        if config.preload_weights {
            eprintln!("🔄 Smart preloading for 16GB system - caching critical weights...");
            // For limited memory, just preload the most important weights
            lazy_weights.preload_critical_tensors()?;
            eprintln!("✅ Critical weights cached - remaining weights will use on-demand loading with LRU cache");
        } else {
            eprintln!("⚡ Lazy loading enabled - weights will be loaded on demand");
        }
        
        // Initialize KV cache if enabled
        let kv_cache = if config.use_kv_cache {
            let cache_config = KVCacheConfig {
                max_memory: 512 * 1024 * 1024, // 512MB for cache
                max_seq_length: model_config.context_length.min(2048), // Limit for performance
                num_layers: model_config.num_layers,
                head_dim: model_config.hidden_size / model_config.num_heads,
                num_heads: model_config.num_heads,
                num_kv_heads: model_config.num_key_value_heads,
                ..Default::default()
            };
            Some(Arc::new(OptimizedKVCache::new(cache_config)))
        } else {
            None
        };
        
        Ok(Self {
            config: TransformerConfig {
                model_config: model_config.clone(),
                norm_type: config.norm_type,
                position_encoding: config.position_encoding,
                activation: config.activation,
                tie_embeddings: config.tie_embeddings,
                pre_norm: config.pre_norm,
                dropout: config.dropout,
                attention_dropout: config.attention_dropout,
                use_kv_cache: config.use_kv_cache,
                preload_weights: config.preload_weights,
            },
            weights: Arc::new(Mutex::new(lazy_weights)),
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("lazy_transformer")
                .to_string(),
            kv_cache,
            memory_pool: Arc::new(Mutex::new(TensorMemoryPool::new())),
        })
    }

    /// Process a single transformer layer
    fn process_layer(
        &self,
        layer_idx: usize,
        hidden_states: &mut Vec<f32>,
        _attention_mask: Option<&[f32]>,
        _past_kv: Option<(&[f32], &[f32])>,
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let layer_prefix = format!("blk.{}", layer_idx);
        let hidden_size = self.config.model_config.hidden_size;
        let seq_len = hidden_states.len() / hidden_size;
        
        // Lock weights for this layer processing
        let mut weights = self.weights.lock().map_err(|poisoned| {
            // Handle poisoned mutex by recovering the data
            CoreError::model(
                "WEIGHTS_MUTEX_POISONED",
                "Weights mutex was poisoned due to a previous panic",
                "lazy transformer process layer",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        
        // Get memory pool
        let mut pool = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        
        // Apply attention normalization (use RMSNorm for modern models like Granite)
        let attn_norm_weight_ref = weights.get_tensor(&format!("{}.attn_norm.weight", layer_prefix))?;
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        let norm_tensor = tensor_from_slice(attn_norm_weight_ref, Shape::vector(hidden_size))?;
        let normed_result = rms_norm(&hidden_tensor, &norm_tensor, self.config.model_config.layer_norm_epsilon)?;
        
        // Get buffer from pool and copy result
        let mut normed_hidden = pool.get_buffer(seq_len * hidden_size);
        normed_hidden.copy_from_slice(&normed_result.to_vec());
        
        eprintln!("DEBUG process_layer: normed_hidden.len()={}, expected={}", 
                  normed_hidden.len(), seq_len * hidden_size);
        
        // Ensure we only pass the correct size to SIMD function
        let normed_slice = &normed_hidden[..seq_len * hidden_size];
        
        // Drop pool lock before calling compute methods to avoid deadlock
        drop(pool);
        
        // Check if SIMD is disabled via environment variable
        let use_simd = std::env::var("WOOLLY_DISABLE_SIMD")
            .map(|v| v != "1" && v.to_lowercase() != "true")
            .unwrap_or(true);
        
        let attn_output = if use_simd {
            // Use SIMD-optimized GQA attention
            self.compute_simd_gqa_attention(
                normed_slice,
                layer_idx,
                seq_len,
                hidden_size,
                &mut weights,
                &layer_prefix,
            )?
        } else {
            // Use non-SIMD optimized GQA attention
            self.compute_optimized_gqa_attention(
                normed_slice,
                layer_idx,
                seq_len,
                hidden_size,
                &mut weights,
                &layer_prefix,
            )?
        };
        
        // Add residual connection
        for (h, &a) in hidden_states.iter_mut().zip(attn_output.iter()) {
            *h += a;
        }
        
        // FFN normalization (use RMSNorm for modern models like Granite)
        let mut pool = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        let ffn_norm_weight_ref = weights.get_tensor(&format!("{}.ffn_norm.weight", layer_prefix))?;
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        let norm_tensor = tensor_from_slice(ffn_norm_weight_ref, Shape::vector(hidden_size))?;
        let normed_result = rms_norm(&hidden_tensor, &norm_tensor, self.config.model_config.layer_norm_epsilon)?;
        
        // Get buffer from pool and copy result
        let mut normed_for_ffn = pool.get_buffer(seq_len * hidden_size);
        normed_for_ffn.copy_from_slice(&normed_result.to_vec());
        
        // Drop pool lock before calling compute methods
        drop(pool);
        
        // Check if SIMD is disabled via environment variable
        let use_simd = std::env::var("WOOLLY_DISABLE_SIMD")
            .map(|v| v != "1" && v.to_lowercase() != "true")
            .unwrap_or(true);
        
        let ffn_output = if use_simd {
            // SIMD-optimized SwiGLU FFN computation
            self.compute_simd_swiglu_ffn(
                &normed_for_ffn,
                seq_len,
                hidden_size,
                &mut weights,
                &layer_prefix,
            )?
        } else {
            // Non-SIMD SwiGLU FFN computation
            self.compute_swiglu_ffn(
                &normed_for_ffn,
                seq_len,
                hidden_size,
                &mut weights,
                &layer_prefix,
            )?
        };
        
        // Add residual
        for (h, &f) in hidden_states.iter_mut().zip(ffn_output.iter()) {
            *h += f;
        }
        
        // Return buffers to pool
        let mut pool = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        pool.return_buffer(normed_hidden);
        pool.return_buffer(normed_for_ffn);
        pool.return_buffer(attn_output);
        pool.return_buffer(ffn_output);
        drop(pool);
        
        // DON'T clear layer weights from cache - we want to reuse them!
        // This was causing repeated dequantization on every token
        // weights.clear_tensor_cache(&format!("{}.attn_q.weight", layer_prefix));
        // weights.clear_tensor_cache(&format!("{}.attn_k.weight", layer_prefix));
        // weights.clear_tensor_cache(&format!("{}.attn_v.weight", layer_prefix));
        // weights.clear_tensor_cache(&format!("{}.attn_output.weight", layer_prefix));
        // weights.clear_tensor_cache(&format!("{}.ffn_up.weight", layer_prefix));
        // weights.clear_tensor_cache(&format!("{}.ffn_down.weight", layer_prefix));
        // weights.clear_tensor_cache(&format!("{}.ffn_gate.weight", layer_prefix));
        
        Ok(None) // No KV cache for now
    }
    
    /// Compute optimized grouped query attention with efficient KV caching and memory pooling
    fn compute_optimized_gqa_attention(
        &self,
        hidden_states: &[f32],
        layer_idx: usize,
        seq_len: usize,
        hidden_size: usize,
        weights: &mut LazyModelWeights,
        layer_prefix: &str,
    ) -> Result<Vec<f32>> {
        let num_heads = self.config.model_config.num_heads;
        let head_dim = hidden_size / num_heads;
        
        // Get memory pool
        let mut pool = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        
        // Get tensor shapes for GQA first
        let k_shape = weights.get_tensor_shape(&format!("{}.attn_k.weight", layer_prefix))?;
        let kv_hidden_size = k_shape[1];
        let num_kv_heads = kv_hidden_size / head_dim;
        let head_groups = num_heads / num_kv_heads;
        
        // Create tensors by getting each weight tensor individually to avoid borrowing conflicts
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        
        let q_tensor = {
            let q_weight_ref = weights.get_tensor(&format!("{}.attn_q.weight", layer_prefix))?;
            tensor_from_slice(q_weight_ref, Shape::matrix(hidden_size, hidden_size))?
        };
        
        let k_tensor = {
            let k_weight_ref = weights.get_tensor(&format!("{}.attn_k.weight", layer_prefix))?;
            tensor_from_slice(k_weight_ref, Shape::matrix(hidden_size, kv_hidden_size))?
        };
        
        let v_tensor = {
            let v_weight_ref = weights.get_tensor(&format!("{}.attn_v.weight", layer_prefix))?;
            tensor_from_slice(v_weight_ref, Shape::matrix(hidden_size, kv_hidden_size))?
        };
        
        // Get buffers from pool for Q, K, V projections
        let mut queries = pool.get_buffer(seq_len * hidden_size);
        let mut keys = pool.get_buffer(seq_len * kv_hidden_size);
        let mut values = pool.get_buffer(seq_len * kv_hidden_size);
        
        // Perform matrix multiplications into pooled buffers
        let q_result = matmul(&hidden_tensor, &q_tensor)?;
        let k_result = matmul(&hidden_tensor, &k_tensor)?;
        let v_result = matmul(&hidden_tensor, &v_tensor)?;
        
        // Copy results to pooled buffers
        queries.copy_from_slice(&q_result.to_vec());
        keys.copy_from_slice(&k_result.to_vec());
        values.copy_from_slice(&v_result.to_vec());
        
        // Handle KV cache
        let session_id = "default";
        let (cached_keys, cached_values, past_seq_len) = if let Some(ref kv_cache) = self.kv_cache {
            match kv_cache.retrieve(layer_idx, session_id) {
                Ok(Some((cached_k, cached_v, cached_len))) => {
                    // Get buffer from pool for concatenated KV
                    let mut full_k = pool.get_buffer((seq_len + cached_len) * kv_hidden_size);
                    let mut full_v = pool.get_buffer((seq_len + cached_len) * kv_hidden_size);
                    
                    // Copy cached and new keys/values
                    full_k[..cached_k.len()].copy_from_slice(&cached_k);
                    full_k[cached_k.len()..].copy_from_slice(&keys);
                    full_v[..cached_v.len()].copy_from_slice(&cached_v);
                    full_v[cached_v.len()..].copy_from_slice(&values);
                    
                    // Update cache
                    let _ = kv_cache.store(layer_idx, session_id, full_k.clone(), full_v.clone(), seq_len + cached_len, None);
                    (full_k, full_v, cached_len)
                }
                _ => {
                    // First time - store current KV
                    let _ = kv_cache.store(layer_idx, session_id, keys.clone(), values.clone(), seq_len, None);
                    (keys.clone(), values.clone(), 0)
                }
            }
        } else {
            (keys.clone(), values.clone(), 0)
        };
        
        let total_seq_len = seq_len + past_seq_len;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Use BLAS-optimized attention computation
        let output = if crate::blas_matmul::is_blas_available() {
            eprintln!("  🚀 Using BLAS-optimized GQA attention");
            grouped_query_attention_blas(
                &queries,
                &cached_keys,
                &cached_values,
                seq_len,
                total_seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                scale
            )?
        } else {
            eprintln!("  ⚠️ BLAS not available, using fallback attention");
            // Fallback implementation
            let mut output = pool.get_buffer(seq_len * hidden_size);
            let mut scores_buffer = get_buffer(total_seq_len);
            
            // Optimized GQA computation - avoid nested memory allocations
            for pos in 0..seq_len {
                let current_abs_pos = past_seq_len + pos;
                
                // For each KV head group - process all Q heads in group together
                for kv_head in 0..num_kv_heads {
                    let kv_start = kv_head * head_dim;
                    
                    // Process all Q heads in this group efficiently
                    for group_idx in 0..head_groups {
                        let q_head = kv_head * head_groups + group_idx;
                        let q_start = pos * hidden_size + q_head * head_dim;
                        
                        // Compute attention scores using direct indexing (no copying)
                        for key_pos in 0..=current_abs_pos { // Causal masking built-in
                            let key_start = key_pos * kv_hidden_size + kv_start;
                            
                            // Vectorized dot product without inner loops
                            let score = queries[q_start..q_start + head_dim]
                                .iter()
                                .zip(&cached_keys[key_start..key_start + head_dim])
                                .map(|(&q, &k)| q * k)
                                .sum::<f32>();
                            
                            scores_buffer[key_pos] = score * scale;
                        }
                        
                        // Zero out future positions (causal mask)
                        for key_pos in (current_abs_pos + 1)..total_seq_len {
                            scores_buffer[key_pos] = f32::NEG_INFINITY;
                        }
                        
                        // Efficient softmax using pre-allocated buffer
                        let max_score = scores_buffer[..=current_abs_pos].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        if max_score == f32::NEG_INFINITY {
                            continue; // Skip if all masked
                        }
                        
                        let mut sum_exp = 0.0f32;
                        for i in 0..=current_abs_pos {
                            scores_buffer[i] = (scores_buffer[i] - max_score).exp();
                            sum_exp += scores_buffer[i];
                        }
                        
                        if sum_exp > 0.0 {
                            let inv_sum = 1.0 / sum_exp;
                            // Apply attention to values using direct indexing
                            let output_start = pos * hidden_size + q_head * head_dim;
                            for val_pos in 0..=current_abs_pos {
                                let attention_weight = scores_buffer[val_pos] * inv_sum;
                                let val_start = val_pos * kv_hidden_size + kv_start;
                                
                                for d in 0..head_dim {
                                    output[output_start + d] += attention_weight * cached_values[val_start + d];
                                }
                            }
                        }
                    }
                }
            }
            output
        };
        
        // Apply output projection
        let output_tensor = tensor_from_slice(&output, Shape::matrix(seq_len, hidden_size))?;
        let o_tensor = {
            let o_weight_ref = weights.get_tensor(&format!("{}.attn_output.weight", layer_prefix))?;
            tensor_from_slice(o_weight_ref, Shape::matrix(hidden_size, hidden_size))?
        };
        let final_output = matmul(&output_tensor, &o_tensor)?;
        
        // Get final output buffer and copy result
        let mut result = pool.get_buffer(seq_len * hidden_size);
        result.copy_from_slice(&final_output.to_vec());
        
        // Return used buffers to pool
        pool.return_buffer(queries);
        pool.return_buffer(keys);
        pool.return_buffer(values);
        pool.return_buffer(output);
        
        Ok(result)
    }
    
    /// Compute SwiGLU Feed-Forward Network with memory pooling
    fn compute_swiglu_ffn(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        hidden_size: usize,
        weights: &mut LazyModelWeights,
        layer_prefix: &str,
    ) -> Result<Vec<f32>> {
        // SwiGLU FFN: hidden → [gate_proj, up_proj] → SwiGLU → down_proj
        
        // Get memory pool
        let mut pool = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        
        // Get FFN weights as references to avoid allocations
        // Get intermediate size from weight shapes first
        let gate_shape = weights.get_tensor_shape(&format!("{}.ffn_gate.weight", layer_prefix))?;
        
        // Get weight tensors one at a time to avoid borrow checker issues
        let gate_weight_ref = weights.get_tensor(&format!("{}.ffn_gate.weight", layer_prefix))?;
        let gate_weight_data = gate_weight_ref.to_vec(); // Clone to avoid borrow issues
        
        let up_weight_ref = weights.get_tensor(&format!("{}.ffn_up.weight", layer_prefix))?;
        let up_weight_data = up_weight_ref.to_vec(); // Clone to avoid borrow issues
        
        let down_weight_ref = weights.get_tensor(&format!("{}.ffn_down.weight", layer_prefix))?;
        let down_weight_data = down_weight_ref.to_vec(); // Clone to avoid borrow issues
        let intermediate_size = gate_shape[1]; // [hidden_size, intermediate_size]
        
        eprintln!("DEBUG: SwiGLU FFN - hidden_size: {}, intermediate_size: {}, seq_len: {}", 
                  hidden_size, intermediate_size, seq_len);
        
        // Project to gate and up
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        let gate_tensor = tensor_from_slice(&gate_weight_data, Shape::matrix(hidden_size, intermediate_size))?;
        let up_tensor = tensor_from_slice(&up_weight_data, Shape::matrix(hidden_size, intermediate_size))?;
        
        // Get buffers from pool for intermediate results
        let mut gate_proj_buf = pool.get_buffer(seq_len * intermediate_size);
        let mut up_proj_buf = pool.get_buffer(seq_len * intermediate_size);
        
        // Perform matrix multiplications
        let gate_proj = matmul(&hidden_tensor, &gate_tensor)?;
        let up_proj = matmul(&hidden_tensor, &up_tensor)?;
        
        // Copy results to pooled buffers
        gate_proj_buf.copy_from_slice(&gate_proj.to_vec());
        up_proj_buf.copy_from_slice(&up_proj.to_vec());
        
        // Create tensors from pooled buffers
        let gate_proj_tensor = tensor_from_slice(&gate_proj_buf, Shape::matrix(seq_len, intermediate_size))?;
        let up_proj_tensor = tensor_from_slice(&up_proj_buf, Shape::matrix(seq_len, intermediate_size))?;
        
        // Apply SwiGLU activation: SiLU(gate) * up
        let activated = swiglu(&gate_proj_tensor, &up_proj_tensor)?;
        
        // Project down to hidden_size
        let down_tensor = tensor_from_slice(&down_weight_data, Shape::matrix(intermediate_size, hidden_size))?;
        let output = matmul(&activated, &down_tensor)?;
        
        // Get final output buffer and copy result
        let mut result = pool.get_buffer(seq_len * hidden_size);
        result.copy_from_slice(&output.to_vec());
        
        // Return intermediate buffers to pool
        pool.return_buffer(gate_proj_buf);
        pool.return_buffer(up_proj_buf);
        
        Ok(result)
    }
    
    /// Legacy compute grouped query attention with KV cache optimization
    #[allow(dead_code)]
    fn compute_gqa_attention(
        &self,
        hidden_states: &[f32],
        layer_idx: usize,
        seq_len: usize,
        hidden_size: usize,
        weights: &mut LazyModelWeights,
        layer_prefix: &str,
    ) -> Result<Vec<f32>> {
        let num_heads = self.config.model_config.num_heads;
        let head_dim = hidden_size / num_heads;
        
        // Get Q, K, V projection weights with shape checking
        let q_shape = weights.get_tensor_shape(&format!("{}.attn_q.weight", layer_prefix))?;
        let k_shape = weights.get_tensor_shape(&format!("{}.attn_k.weight", layer_prefix))?;
        let v_shape = weights.get_tensor_shape(&format!("{}.attn_v.weight", layer_prefix))?;
        
        // Determine number of KV heads from tensor shapes (for GQA)
        let kv_hidden_size = k_shape[1]; // Second dimension of K/V weights
        let num_kv_heads = kv_hidden_size / head_dim;
        let head_groups = num_heads / num_kv_heads; // How many Q heads per KV head
        
        eprintln!("DEBUG: Layer {} GQA - Q heads: {}, KV heads: {}, groups: {}", 
                  layer_idx, num_heads, num_kv_heads, head_groups);
        
        let q_weight = weights.get_tensor(&format!("{}.attn_q.weight", layer_prefix))?.to_vec();
        let k_weight = weights.get_tensor(&format!("{}.attn_k.weight", layer_prefix))?.to_vec();
        let v_weight = weights.get_tensor(&format!("{}.attn_v.weight", layer_prefix))?.to_vec();
        
        // Project to Q, K, V with proper shapes
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        let q_tensor = tensor_from_slice(&q_weight, Shape::from_slice(&q_shape))?;
        let k_tensor = tensor_from_slice(&k_weight, Shape::from_slice(&k_shape))?;
        let v_tensor = tensor_from_slice(&v_weight, Shape::from_slice(&v_shape))?;
        
        let queries = matmul(&hidden_tensor, &q_tensor)?; // [seq_len, hidden_size]
        let keys = matmul(&hidden_tensor, &k_tensor)?;    // [seq_len, kv_hidden_size]
        let values = matmul(&hidden_tensor, &v_tensor)?;  // [seq_len, kv_hidden_size]
        
        let q_data = queries.to_vec();
        let k_data = keys.to_vec();
        let v_data = values.to_vec();
        
        // Handle KV cache for this layer
        let session_id = "default";
        let (k_all, v_all, past_seq_len) = if let Some(ref kv_cache) = self.kv_cache {
            if let Ok(Some((cached_k, cached_v, cached_len))) = kv_cache.retrieve(layer_idx, session_id) {
                // Concatenate cached KV with new KV
                let mut full_k = cached_k;
                full_k.extend_from_slice(&k_data);
                let mut full_v = cached_v;
                full_v.extend_from_slice(&v_data);
                
                // Store updated cache (append new tokens)
                let _ = kv_cache.store(layer_idx, session_id, full_k.clone(), full_v.clone(), seq_len + cached_len, None);
                
                (full_k, full_v, cached_len)
            } else {
                // First time, cache the current KV
                let _ = kv_cache.store(layer_idx, session_id, k_data.clone(), v_data.clone(), seq_len, None);
                (k_data.clone(), v_data.clone(), 0)
            }
        } else {
            (k_data.clone(), v_data.clone(), 0)
        };
        
        let total_seq_len = seq_len + past_seq_len;
        
        // Compute GQA attention
        let mut output = get_buffer(seq_len * hidden_size);
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // For each query position in the current sequence
        for pos in 0..seq_len {
            let q_offset = pos * hidden_size;
            
            // For each query head group
            for kv_head in 0..num_kv_heads {
                let kv_offset_per_pos = kv_head * head_dim;
                
                // Process all query heads in this group
                for group_idx in 0..head_groups {
                    let q_head = kv_head * head_groups + group_idx;
                    let q_head_offset = q_offset + q_head * head_dim;
                    
                    // Compute attention scores for this query head against all key positions
                    let mut scores = get_buffer(total_seq_len);
                    
                    for key_pos in 0..total_seq_len {
                        let k_pos_offset = key_pos * kv_hidden_size + kv_offset_per_pos;
                        
                        // Dot product between query and key
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += q_data[q_head_offset + d] * k_all[k_pos_offset + d];
                        }
                        scores[key_pos] = score * scale;
                    }
                    
                    // Apply causal mask (can only attend to previous and current positions)
                    let current_abs_pos = past_seq_len + pos;
                    for key_pos in (current_abs_pos + 1)..total_seq_len {
                        scores[key_pos] = f32::NEG_INFINITY;
                    }
                    
                    // Softmax
                    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    
                    if sum_exp > 0.0 {
                        // Apply attention weights to values
                        for key_pos in 0..total_seq_len {
                            let weight = exp_scores[key_pos] / sum_exp;
                            let v_pos_offset = key_pos * kv_hidden_size + kv_offset_per_pos;
                            
                            for d in 0..head_dim {
                                output[q_head_offset + d] += weight * v_all[v_pos_offset + d];
                            }
                        }
                    }
                }
            }
        }
        
        // Apply output projection
        let o_weight = weights.get_tensor(&format!("{}.attn_output.weight", layer_prefix))?.to_vec();
        let o_shape = weights.get_tensor_shape(&format!("{}.attn_output.weight", layer_prefix))?;
        let output_tensor = tensor_from_slice(&output, Shape::matrix(seq_len, hidden_size))?;
        let o_tensor = tensor_from_slice(&o_weight, Shape::from_slice(&o_shape))?;
        let final_output = matmul(&output_tensor, &o_tensor)?;
        
        Ok(final_output.to_vec())
    }
    
    /// Legacy compute multi-head attention with optional KV cache
    #[allow(dead_code)]
    fn compute_attention(
        &self,
        hidden_states: &[f32],
        layer_idx: usize,
        seq_len: usize,
        hidden_size: usize,
        weights: &mut LazyModelWeights,
        layer_prefix: &str,
    ) -> Result<Vec<f32>> {
        let num_heads = self.config.model_config.num_heads;
        let head_dim = hidden_size / num_heads;
        
        // Get Q, K, V projection weights with shape checking
        let q_shape = weights.get_tensor_shape(&format!("{}.attn_q.weight", layer_prefix))?;
        let k_shape = weights.get_tensor_shape(&format!("{}.attn_k.weight", layer_prefix))?;
        let v_shape = weights.get_tensor_shape(&format!("{}.attn_v.weight", layer_prefix))?;
        
        // Determine number of KV heads from tensor shapes (for GQA)
        let kv_hidden_size = k_shape[1]; // Second dimension of K/V weights
        let num_kv_heads = kv_hidden_size / head_dim;
        
        eprintln!("DEBUG: Layer {} - Q shape: {:?}, K shape: {:?}, V shape: {:?}", 
                  layer_idx, q_shape, k_shape, v_shape);
        
        let q_weight = weights.get_tensor(&format!("{}.attn_q.weight", layer_prefix))?.to_vec();
        let k_weight = weights.get_tensor(&format!("{}.attn_k.weight", layer_prefix))?.to_vec();
        let v_weight = weights.get_tensor(&format!("{}.attn_v.weight", layer_prefix))?.to_vec();
        
        // Project to Q, K, V with proper shapes
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        let q_tensor = tensor_from_slice(&q_weight, Shape::from_slice(&q_shape))?;
        let k_tensor = tensor_from_slice(&k_weight, Shape::from_slice(&k_shape))?;
        let v_tensor = tensor_from_slice(&v_weight, Shape::from_slice(&v_shape))?;
        
        eprintln!("DEBUG: hidden_tensor shape: {:?}", hidden_tensor.shape());
        eprintln!("DEBUG: q_tensor shape: {:?}", q_tensor.shape());
        eprintln!("DEBUG: k_tensor shape: {:?}", k_tensor.shape());
        eprintln!("DEBUG: v_tensor shape: {:?}", v_tensor.shape());
        
        let queries = matmul(&hidden_tensor, &q_tensor)?;
        eprintln!("DEBUG: queries shape after matmul: {:?}", queries.shape());
        
        let keys = matmul(&hidden_tensor, &k_tensor)?;
        eprintln!("DEBUG: keys shape after matmul: {:?}", keys.shape());
        
        let values = matmul(&hidden_tensor, &v_tensor)?;
        eprintln!("DEBUG: values shape after matmul: {:?}", values.shape());
        
        // Reshape for multi-head attention: [seq_len, num_heads, head_dim]
        let q_data = queries.to_vec();
        let k_data = keys.to_vec();
        let v_data = values.to_vec();
        
        // Check if we have cached KV for this layer
        // For now, we'll use a default session ID since we don't have session tracking yet
        let session_id = "default";
        let (k_to_use, v_to_use, cache_len) = if let Some(ref kv_cache) = self.kv_cache {
            // Try to get cached KV
            if let Ok(Some((cached_k, cached_v, cached_len))) = kv_cache.retrieve(layer_idx, session_id) {
                // Concatenate cached KV with new KV
                let mut full_k = cached_k;
                full_k.extend_from_slice(&k_data);
                let mut full_v = cached_v;
                full_v.extend_from_slice(&v_data);
                
                // Store updated cache
                let _ = kv_cache.store(layer_idx, session_id, full_k.clone(), full_v.clone(), seq_len + cached_len, None);
                
                (full_k, full_v, cached_len)
            } else {
                // First time, just cache the current KV
                let _ = kv_cache.store(layer_idx, session_id, k_data.clone(), v_data.clone(), seq_len, None);
                (k_data.clone(), v_data.clone(), 0)
            }
        } else {
            (k_data.clone(), v_data.clone(), 0)
        };
        
        // Compute scaled dot-product attention
        // For simplicity, compute for each head separately
        eprintln!("DEBUG: seq_len={}, hidden_size={}, num_heads={}, num_kv_heads={}, head_dim={}", 
                  seq_len, hidden_size, num_heads, num_kv_heads, head_dim);
        eprintln!("DEBUG: q_data len={}, k_data len={}, v_data len={}", 
                  q_data.len(), k_to_use.len(), v_to_use.len());
        
        let mut output = get_buffer(seq_len * hidden_size);
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Compute attention for each head with proper GQA mapping
        for h in 0..num_heads {
            for i in 0..seq_len {
                // Query for this position and head - queries have full hidden_size
                let q_start = i * hidden_size + h * head_dim;
                let q_end = q_start + head_dim;
                
                // Verify query bounds
                if q_end > q_data.len() {
                    return Err(CoreError::tensor(
                        "QUERY_INDEX_OUT_OF_BOUNDS",
                        format!("Query index out of bounds: {} > {}", q_end, q_data.len()),
                        "attention computation",
                        "Check query dimensions"
                    ));
                }
                
                // Compute attention scores with all keys (including cached)
                let total_seq_len = seq_len + cache_len;
                let mut scores = get_buffer(total_seq_len);
                
                for j in 0..total_seq_len {
                    // For GQA, map query head to KV head (e.g., 32 Q heads -> 8 KV heads)
                    let kv_h = h / (num_heads / num_kv_heads);
                    // Keys have reduced kv_hidden_size dimensions
                    let k_start = j * kv_hidden_size + kv_h * head_dim;
                    let k_end = k_start + head_dim;
                    
                    if k_end > k_to_use.len() {
                        eprintln!("ERROR: k_start={}, head_dim={}, k_to_use.len()={}", k_start, head_dim, k_to_use.len());
                        eprintln!("ERROR: j={}, kv_hidden_size={}, kv_h={}, num_kv_heads={}", 
                                  j, kv_hidden_size, kv_h, num_kv_heads);
                        return Err(CoreError::tensor(
                            "KEY_INDEX_OUT_OF_BOUNDS",
                            format!("Key tensor index out of bounds: {} > {}", k_end, k_to_use.len()),
                            "attention computation",
                            "Check key tensor dimensions"
                        ));
                    }
                    
                    // Dot product between query and key
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        score += q_data[q_start + d] * k_to_use[k_start + d];
                    }
                    scores[j] = score * scale;
                }
                
                // Apply causal mask for autoregressive generation
                for j in (i + cache_len + 1)..total_seq_len {
                    scores[j] = f32::NEG_INFINITY;
                }
                
                // Apply softmax to get attention weights
                let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attention_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();
                
                // Apply attention weights to values
                for j in 0..total_seq_len {
                    // For GQA, map query head to KV head
                    let kv_h = h / (num_heads / num_kv_heads);
                    // Values have reduced kv_hidden_size dimensions
                    let v_start = j * kv_hidden_size + kv_h * head_dim;
                    let v_end = v_start + head_dim;
                    let weight = attention_weights[j];
                    
                    if v_end > v_to_use.len() {
                        eprintln!("ERROR: v_start={}, head_dim={}, v_to_use.len()={}", v_start, head_dim, v_to_use.len());
                        eprintln!("ERROR: j={}, kv_hidden_size={}, kv_h={}, num_kv_heads={}", 
                                  j, kv_hidden_size, kv_h, num_kv_heads);
                        return Err(CoreError::tensor(
                            "VALUE_INDEX_OUT_OF_BOUNDS",
                            format!("Value tensor index out of bounds: {} > {}", v_end, v_to_use.len()),
                            "attention computation",
                            "Check value tensor dimensions"
                        ));
                    }
                    
                    for d in 0..head_dim {
                        output[q_start + d] += weight * v_to_use[v_start + d];
                    }
                }
            }
        }
        
        // Apply output projection
        let o_weight = weights.get_tensor(&format!("{}.attn_output.weight", layer_prefix))?.to_vec();
        let output_tensor = tensor_from_slice(&output, Shape::matrix(seq_len, hidden_size))?;
        let o_tensor = tensor_from_slice(&o_weight, Shape::matrix(hidden_size, hidden_size))?;
        let final_output = matmul(&output_tensor, &o_tensor)?;
        
        Ok(final_output.to_vec())
    }

    /// SIMD-optimized grouped query attention computation
    fn compute_simd_gqa_attention(
        &self,
        hidden_states: &[f32],
        layer_idx: usize,
        seq_len: usize,
        hidden_size: usize,
        weights: &mut LazyModelWeights,
        layer_prefix: &str,
    ) -> Result<Vec<f32>> {
        let num_heads = self.config.model_config.num_heads;
        let head_dim = hidden_size / num_heads;
        
        // Get memory pool
        let mut pool = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        
        // Create tensors by getting each weight tensor individually to avoid borrowing conflicts
        eprintln!("DEBUG SIMD: hidden_states.len()={}, seq_len={}, hidden_size={}", 
                  hidden_states.len(), seq_len, hidden_size);
        eprintln!("DEBUG SIMD: Expected size={}, actual size={}", 
                  seq_len * hidden_size, hidden_states.len());
        
        // Check if hidden_states has the expected size
        if hidden_states.len() != seq_len * hidden_size {
            return Err(CoreError::tensor(
                "TENSOR_SIZE_MISMATCH",
                format!("Hidden states size mismatch: expected {}, got {}", 
                        seq_len * hidden_size, hidden_states.len()),
                "SIMD attention computation",
                "Check hidden states dimensions"
            ));
        }
        
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        
        let q_tensor = {
            let q_weight_ref = weights.get_tensor(&format!("{}.attn_q.weight", layer_prefix))?;
            tensor_from_slice(q_weight_ref, Shape::matrix(hidden_size, hidden_size))?
        };
        
        let k_tensor = {
            let k_shape = weights.get_tensor_shape(&format!("{}.attn_k.weight", layer_prefix))?;
            let k_weight_ref = weights.get_tensor(&format!("{}.attn_k.weight", layer_prefix))?;
            tensor_from_slice(k_weight_ref, Shape::from_slice(&k_shape))?
        };
        
        let v_tensor = {
            let v_shape = weights.get_tensor_shape(&format!("{}.attn_v.weight", layer_prefix))?;
            let v_weight_ref = weights.get_tensor(&format!("{}.attn_v.weight", layer_prefix))?;
            tensor_from_slice(v_weight_ref, Shape::from_slice(&v_shape))?
        };
        
        let o_tensor = {
            let o_weight_ref = weights.get_tensor(&format!("{}.attn_output.weight", layer_prefix))?;
            tensor_from_slice(o_weight_ref, Shape::matrix(hidden_size, hidden_size))?
        };
        
        // Drop pool lock before expensive operations
        drop(pool);
        
        // Get number of KV heads from tensor shapes
        let kv_hidden_size = k_tensor.shape().as_slice()[1];
        let num_kv_heads = kv_hidden_size / head_dim;
        
        eprintln!("DEBUG SIMD: num_heads={}, num_kv_heads={}, head_dim={}, kv_hidden_size={}",
                  num_heads, num_kv_heads, head_dim, kv_hidden_size);
        
        // Use SIMD-optimized projections
        let mut pool_ref = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        let (queries, keys, values) = simd_attention_projections(
            &hidden_tensor,
            &q_tensor,
            &k_tensor,
            &v_tensor,
            &mut pool_ref,
        )?;
        drop(pool_ref);
        
        // Use BLAS-optimized attention instead of manual loops
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Use BLAS for attention computation
        let blas_available = crate::blas_matmul::is_blas_available();
        eprintln!("  🔍 Checking BLAS availability in SIMD attention: {}", blas_available);
        let output = if blas_available {
            eprintln!("  🚀 SIMD path: Using BLAS-optimized GQA attention");
            grouped_query_attention_blas(
                &queries.data,
                &keys.data,
                &values.data,
                seq_len,
                seq_len,  // total_seq_len = seq_len for now (no past KV)
                num_heads,
                num_kv_heads,
                head_dim,
                scale
            )?
        } else {
            eprintln!("  ⚠️ SIMD path: BLAS not available, using fallback");
            // Keep the old manual loop implementation as fallback
            let mut output = get_buffer(seq_len * hidden_size);
            
            for h in 0..num_heads {
                for i in 0..seq_len {
                    let q_start = i * hidden_size + h * head_dim;
                
                    if q_start + head_dim > queries.data.len() {
                        eprintln!("ERROR SIMD: q_start={}, head_dim={}, queries.len()={}", 
                                  q_start, head_dim, queries.data.len());
                        eprintln!("ERROR SIMD: i={}, h={}, hidden_size={}", i, h, hidden_size);
                        return Err(CoreError::tensor(
                            "TENSOR_INDEX_OUT_OF_BOUNDS",
                            format!("Query tensor index out of bounds: {} + {} > {}", 
                                    q_start, head_dim, queries.data.len()),
                            "SIMD attention computation",
                            "Check tensor dimensions"
                        ));
                    }
                    
                    let mut scores = get_buffer(seq_len);
                    
                    // Compute attention scores Q @ K^T
                    for j in 0..seq_len {
                        // For GQA, map query head to KV head (4:1 ratio for 32->8 heads)
                        let kv_h = h / (num_heads / num_kv_heads);
                        // Keys indexing: keys are seq_len * kv_hidden_size (reduced dimensions)
                        let k_start = j * kv_hidden_size + kv_h * head_dim;
                    
                        if k_start + head_dim > keys.data.len() {
                            eprintln!("ERROR SIMD K: k_start={}, head_dim={}, keys.len()={}", 
                                      k_start, head_dim, keys.data.len());
                            eprintln!("ERROR SIMD K: j={}, kv_h={}, kv_hidden_size={}", j, kv_h, kv_hidden_size);
                            return Err(CoreError::tensor(
                                "TENSOR_K_INDEX_OUT_OF_BOUNDS",
                                format!("Key tensor index out of bounds: {} + {} > {}", 
                                        k_start, head_dim, keys.data.len()),
                                "SIMD attention computation",
                                "Check key tensor dimensions"
                            ));
                        }
                    
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += queries.data[q_start + d] * keys.data[k_start + d];
                        }
                        scores[j] = score * scale;
                    }
                    
                    // Apply causal mask for autoregressive generation
                    for j in (i + 1)..seq_len {
                        scores[j] = f32::NEG_INFINITY;
                    }
                    
                    // Softmax over scores
                    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    if max_score != f32::NEG_INFINITY {
                        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                        let sum_exp: f32 = exp_scores.iter().sum();
                        
                        if sum_exp > 0.0 {
                            // Apply attention weights to values
                            for j in 0..seq_len {
                                // For GQA, map query head to KV head
                                let kv_h = h / (num_heads / num_kv_heads);
                                // Values indexing: values are seq_len * kv_hidden_size (reduced dimensions)
                                let v_start = j * kv_hidden_size + kv_h * head_dim;
                                
                                if v_start + head_dim > values.data.len() {
                                    eprintln!("ERROR SIMD V: v_start={}, head_dim={}, values.len()={}", 
                                              v_start, head_dim, values.data.len());
                                    eprintln!("ERROR SIMD V: j={}, kv_h={}, kv_hidden_size={}", j, kv_h, kv_hidden_size);
                                    return Err(CoreError::tensor(
                                        "TENSOR_V_INDEX_OUT_OF_BOUNDS",
                                        format!("Value tensor index out of bounds: {} + {} > {}", 
                                                v_start, head_dim, values.data.len()),
                                        "SIMD attention computation",
                                        "Check value tensor dimensions"
                                    ));
                                }
                                
                                let weight = exp_scores[j] / sum_exp;
                                for d in 0..head_dim {
                                    output[q_start + d] += weight * values.data[v_start + d];
                                }
                            }
                        }
                    }
                }
            }
            output.to_vec()
        };
        
        // Apply output projection using SIMD
        let output_tensor = tensor_from_slice(&output, Shape::matrix(seq_len, hidden_size))?;
        let projected = simd_matvec(&output_tensor, &o_tensor, false, 1.0, 0.0)?;
        
        Ok(projected.to_vec())
    }

    /// SIMD-optimized SwiGLU FFN computation
    fn compute_simd_swiglu_ffn(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        hidden_size: usize,
        weights: &mut LazyModelWeights,
        layer_prefix: &str,
    ) -> Result<Vec<f32>> {
        // Get shapes first to determine intermediate size
        let gate_shape = weights.get_tensor_shape(&format!("{}.ffn_gate.weight", layer_prefix))?;
        let intermediate_size = gate_shape[1];
        
        // Create tensors by getting each weight tensor individually to avoid borrowing conflicts
        let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
        
        // Create each tensor by copying the data immediately to avoid borrowing conflicts
        let gate_tensor = {
            let gate_weight_ref = weights.get_tensor(&format!("{}.ffn_gate.weight", layer_prefix))?;
            tensor_from_slice(gate_weight_ref, Shape::matrix(hidden_size, intermediate_size))?
        };
        
        let up_tensor = {
            let up_weight_ref = weights.get_tensor(&format!("{}.ffn_up.weight", layer_prefix))?;
            tensor_from_slice(up_weight_ref, Shape::matrix(hidden_size, intermediate_size))?
        };
        
        let down_tensor = {
            let down_weight_ref = weights.get_tensor(&format!("{}.ffn_down.weight", layer_prefix))?;
            tensor_from_slice(down_weight_ref, Shape::matrix(intermediate_size, hidden_size))?
        };
        
        // Use SIMD-optimized FFN computation
        let mut pool = self.memory_pool.lock().map_err(|_| {
            CoreError::model(
                "MEMORY_POOL_MUTEX_POISONED",
                "Memory pool mutex was poisoned due to a previous panic",
                "lazy transformer memory pool access",
                "This is likely due to a previous error in tensor operations"
            )
        })?;
        let result = simd_ffn_forward(
            &hidden_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            &mut pool,
        )?;
        drop(pool);
        
        Ok(result.to_vec())
    }
}

impl Model for LazyTransformer {
    fn name(&self) -> &str {
        &self.name
    }

    fn model_type(&self) -> &str {
        "lazy_transformer"
    }

    fn vocab_size(&self) -> usize {
        self.config.model_config.vocab_size
    }

    fn context_length(&self) -> usize {
        self.config.model_config.context_length
    }

    fn hidden_size(&self) -> usize {
        self.config.model_config.hidden_size
    }

    fn num_layers(&self) -> usize {
        self.config.model_config.num_layers
    }

    fn num_heads(&self) -> usize {
        self.config.model_config.num_heads
    }

    fn forward(
        &self,
        input_ids: &[u32],
    ) -> Result<Vec<f32>> {
        let seq_len = input_ids.len();
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;
        
        eprintln!("LazyTransformer: Processing {} tokens", seq_len);
        
        // Get embeddings and initial hidden states
        let (mut hidden_vec, embeddings_shape) = {
            // Lock weights in a scope
            let mut weights = self.weights.lock().map_err(|poisoned| {
                // Handle poisoned mutex by recovering the data
                CoreError::model(
                    "WEIGHTS_MUTEX_POISONED",
                    "Weights mutex was poisoned due to a previous panic",
                    "lazy transformer forward",
                    "This is likely due to a previous error in tensor operations"
                )
            })?;
            
            // Skip critical tensor preloading if all weights were already preloaded
            if !self.config.preload_weights {
                eprintln!("LazyTransformer: Preloading critical tensors...");
                weights.preload_critical_tensors()?;
            } else {
                eprintln!("LazyTransformer: Skipping critical tensor preload (all weights already preloaded)");
                // Verify cache is still populated
                eprintln!("LazyTransformer: Verifying cache before inference...");
                weights.verify_cache_persistence()?;
            }
            
            // Get embeddings
            eprintln!("LazyTransformer: Loading embeddings...");
            let embeddings_shape = weights.get_tensor_shape("token_embd.weight")?;
            eprintln!("LazyTransformer: Got embedding shape: {:?}", embeddings_shape);
            let embeddings_data = weights.get_tensor("token_embd.weight")?.to_vec();
            eprintln!("LazyTransformer: Loaded embedding data, length: {}", embeddings_data.len());
            
            // GGUF stores embeddings as [hidden_size, vocab_size], but embedding_lookup expects [vocab_size, hidden_size]
            eprintln!("LazyTransformer: Embedding shape from GGUF: {:?}", embeddings_shape);
            eprintln!("LazyTransformer: Actual embedding data length: {}", embeddings_data.len());
            
            // Check if the actual data size matches the expected shape
            let expected_elements: usize = embeddings_shape.iter().product();
            if embeddings_data.len() != expected_elements {
                // The shape metadata might be incorrect - try to infer the correct shape
                eprintln!("WARNING: Embedding data size {} doesn't match expected {} from shape {:?}", 
                    embeddings_data.len(), expected_elements, embeddings_shape);
                
                // Common case: embeddings are [hidden_size, actual_vocab_size] where actual_vocab_size is smaller
                if embeddings_shape.len() == 2 && embeddings_shape[0] == hidden_size {
                    let actual_vocab_size = embeddings_data.len() / hidden_size;
                    if embeddings_data.len() % hidden_size == 0 {
                        eprintln!("Inferring actual vocab size: {} (shape reports {})", actual_vocab_size, embeddings_shape[1]);
                        let corrected_shape = vec![hidden_size, actual_vocab_size];
                        eprintln!("LazyTransformer: Creating tensor with corrected shape: {:?}", corrected_shape);
                        let embedding_tensor = tensor_from_slice(
                            &embeddings_data,
                            Shape::from_slice(&corrected_shape)
                        )?;
                        eprintln!("LazyTransformer: Tensor created successfully");
                        
                        // Transpose to [vocab_size, hidden_size]
                        eprintln!("LazyTransformer: Transposing tensor...");
                        let embedding_tensor_transposed = embedding_tensor.transpose(&[1, 0])?;
                        eprintln!("LazyTransformer: Transposed embedding shape: {:?}", embedding_tensor_transposed.shape());
                        
                        // Do embedding lookup
                        eprintln!("LazyTransformer: Performing embedding lookup for {} tokens...", input_ids.len());
                        let hidden_states = embedding_lookup(input_ids, &embedding_tensor_transposed)?;
                        eprintln!("LazyTransformer: Embedding lookup complete");
                        let hidden_vec = hidden_states.to_vec();
                        eprintln!("LazyTransformer: Hidden states vector created, length: {}", hidden_vec.len());
                        
                        (hidden_vec, corrected_shape)
                    } else {
                        return Err(CoreError::model(
                            "EMBEDDING_SIZE_MISMATCH",
                            format!("Cannot infer correct embedding shape: data size {} is not divisible by hidden_size {}", 
                                embeddings_data.len(), hidden_size),
                            "Loading embeddings",
                            "Check model file integrity"
                        ));
                    }
                } else {
                    return Err(CoreError::model(
                        "EMBEDDING_SHAPE_MISMATCH",
                        format!("Embedding shape {:?} doesn't match data size {}", embeddings_shape, embeddings_data.len()),
                        "Loading embeddings", 
                        "Check model file integrity"
                    ));
                }
            } else {
                // Normal case - shape matches data
                let embedding_tensor = tensor_from_slice(
                    &embeddings_data,
                    Shape::from_slice(&embeddings_shape)
                )?;
                
                // Transpose to [vocab_size, hidden_size]
                let embedding_tensor_transposed = embedding_tensor.transpose(&[1, 0])?;
                eprintln!("LazyTransformer: Transposed embedding shape: {:?}", embedding_tensor_transposed.shape());
                
                // Do embedding lookup
                let hidden_states = embedding_lookup(input_ids, &embedding_tensor_transposed)?;
                let hidden_vec = hidden_states.to_vec();
                
                (hidden_vec, embeddings_shape)
            }
        }; // weights lock is released here
        
        // Process layers one at a time
        // Process layers (reduced logging for performance)
        for layer_idx in 0..self.config.model_config.num_layers {
            
            self.process_layer(
                layer_idx,
                &mut hidden_vec,
                None,
                None,
            )?;
        }
        
        // Final layer norm and logits
        eprintln!("LazyTransformer: Applying final normalization and computing logits...");
        let logits = {
            let mut weights = self.weights.lock().map_err(|poisoned| {
                // Handle poisoned mutex by recovering the data
                CoreError::model(
                    "WEIGHTS_MUTEX_POISONED",
                    "Weights mutex was poisoned due to a previous panic",
                    "lazy transformer final logits",
                    "This is likely due to a previous error in tensor operations"
                )
            })?;
            let final_norm_weight = weights.get_tensor("output_norm.weight")?.to_vec();
            
            let hidden_tensor = tensor_from_slice(&hidden_vec, Shape::matrix(seq_len, hidden_size))?;
            let norm_tensor = tensor_from_slice(&final_norm_weight, Shape::vector(hidden_size))?;
            
            let normed_hidden = rms_norm(&hidden_tensor, &norm_tensor, self.config.model_config.layer_norm_epsilon)?;
            
            // Get logits
            if weights.has_tensor("output.weight") {
                // Use output projection
                let output_shape = weights.get_tensor_shape("output.weight")?;
                let output_weight = weights.get_tensor("output.weight")?.to_vec();
                let weight_tensor = tensor_from_slice(&output_weight, Shape::from_slice(&output_shape))?;
                matmul(&normed_hidden, &weight_tensor)?
            } else {
                // Use tied embeddings - need to reload embeddings
                let embeddings_data = weights.get_tensor("token_embd.weight")?.to_vec();
                let embedding_tensor = tensor_from_slice(
                    &embeddings_data,
                    Shape::from_slice(&embeddings_shape)
                )?;
                // For output projection, we need [hidden_size, vocab_size] which is the original GGUF format
                matmul(&normed_hidden, &embedding_tensor)?
            }
        };
        
        eprintln!("LazyTransformer: Forward pass complete!");
        
        // Print cache statistics
        {
            let weights = self.weights.lock().unwrap();
            let cache_stats = weights.cache_stats();
            eprintln!("🎯 FINAL CACHE STATS - Hits: {}, Misses: {}, Hit rate: {:.1}%, Evictions: {}, Memory: {} MB", 
                cache_stats.hits, 
                cache_stats.misses, 
                cache_stats.hit_rate() * 100.0,
                cache_stats.evictions,
                cache_stats.total_bytes_cached / (1024 * 1024)
            );
            eprintln!("⏱️  Total dequantization time saved: {:?}", cache_stats.total_dequantization_time_saved);
        }
        
        Ok(logits.to_vec())
    }

}