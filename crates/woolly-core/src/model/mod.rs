//! Model trait and implementations

pub mod attention;
pub mod embedding;
pub mod feedforward;
pub mod layer_norm;
pub mod lazy_loader;
pub mod lazy_transformer;
pub mod loader;
pub mod transformer;
pub mod memory_pool;
pub mod memory_pool_benchmark;
pub mod memory_pool_enhanced;
pub mod optimized_transformer;
pub mod dequantization_cache;
pub mod fused_kernels;
pub mod fused_transformer;
pub mod fused_benchmark;
pub mod parallel_config;
// pub mod ultra_optimized_transformer;
// pub mod fast_initialization;
// pub mod fast_transformer;

#[cfg(test)]
mod dequantization_cache_test;

use crate::{CoreError, Result};
use async_trait::async_trait;
use std::path::Path;

/// Core trait for language models
pub trait Model: Send + Sync {
    /// Get the model name
    fn name(&self) -> &str;

    /// Get the model type (e.g., "llama", "mistral", "gpt")
    fn model_type(&self) -> &str;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get maximum context length
    fn context_length(&self) -> usize;

    /// Get hidden dimension size
    fn hidden_size(&self) -> usize;

    /// Get number of layers
    fn num_layers(&self) -> usize;

    /// Get number of attention heads
    fn num_heads(&self) -> usize;

    /// Forward pass through the model
    /// Returns logits for the last token only (for generation)
    fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>>;

    /// Check if the model supports a specific feature
    fn supports_feature(&self, feature: ModelFeature) -> bool {
        match feature {
            ModelFeature::FlashAttention => false,
            ModelFeature::GroupedQueryAttention => false,
            ModelFeature::SlidingWindowAttention => false,
            ModelFeature::RoPE => false,
            ModelFeature::ALiBi => false,
        }
    }
}

/// Output from a model forward pass
pub struct ModelOutput {
    /// Logits as a flattened vector
    pub logits: Vec<f32>,
    /// Shape of the logits [batch_size, seq_len, vocab_size]
    pub logits_shape: Vec<usize>,
    /// Updated key-value cache (type-erased for now)
    pub past_kv_cache: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// Hidden states if requested (flattened)
    pub hidden_states: Option<Vec<Vec<f32>>>,
    /// Attention weights if requested (flattened)
    pub attentions: Option<Vec<Vec<f32>>>,
}

/// Key-value cache for efficient inference
/// Temporarily simplified until tensor backend is fully integrated
pub struct KVCache {
    /// Cached keys for each layer (flattened)
    pub keys: Vec<Vec<f32>>,
    /// Cached values for each layer (flattened)
    pub values: Vec<Vec<f32>>,
    /// Shape information for each layer's cache
    pub cache_shapes: Vec<Vec<usize>>,
    /// Current sequence length in cache
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl KVCache {
    /// Create a new empty KV cache
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            keys: Vec::with_capacity(num_layers),
            values: Vec::with_capacity(num_layers),
            cache_shapes: Vec::with_capacity(num_layers),
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Update cache with new key-value pairs
    pub fn update(&mut self, layer_idx: usize, new_keys: Vec<f32>, new_values: Vec<f32>, shape: Vec<usize>) -> Result<()> {
        if layer_idx >= self.keys.capacity() {
            return Err(CoreError::cache(
                "INVALID_LAYER_INDEX",
                format!("Invalid layer index: {}", layer_idx),
                "Storing key-value cache for layer",
                "Check that the layer index is within the valid range"
            ));
        }

        // For now, just store the flattened data
        if layer_idx >= self.keys.len() {
            self.keys.resize(layer_idx + 1, Vec::new());
            self.values.resize(layer_idx + 1, Vec::new());
            self.cache_shapes.resize(layer_idx + 1, Vec::new());
        }

        self.keys[layer_idx] = new_keys;
        self.values[layer_idx] = new_values;
        self.cache_shapes[layer_idx] = shape;

        Ok(())
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.keys.clear();
        self.values.clear();
        self.cache_shapes.clear();
    }
}

/// Model features that can be supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFeature {
    /// Flash Attention optimization
    FlashAttention,
    /// Grouped Query Attention (GQA)
    GroupedQueryAttention,
    /// Sliding Window Attention
    SlidingWindowAttention,
    /// Rotary Position Embeddings (RoPE)
    RoPE,
    /// Attention with Linear Biases (ALiBi)
    ALiBi,
}

/// Base implementation for common model functionality
pub struct BaseModel {
    pub name: String,
    pub model_type: String,
    pub config: ModelConfig,
    // Backend will be handled differently - not as a trait object
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub context_length: usize,
    pub intermediate_size: usize,
    pub num_key_value_heads: Option<usize>,
    pub rope_theta: Option<f32>,
    pub layer_norm_epsilon: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            context_length: 2048,
            intermediate_size: 11008,
            num_key_value_heads: None,
            rope_theta: Some(10000.0),
            layer_norm_epsilon: 1e-5,
        }
    }
}

/// Builder for loading models
pub struct ModelBuilder {
    config: Option<ModelConfig>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            config: None,
        }
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Load a model from GGUF format
    pub async fn from_gguf<M: Model>(self, _path: &Path) -> Result<M> {
        // This would integrate with woolly-gguf
        todo!("GGUF loading integration")
    }

    /// Load a model from safetensors format
    pub async fn from_safetensors<M: Model>(self, _path: &Path) -> Result<M> {
        todo!("Safetensors loading integration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(32, 2048);
        assert_eq!(cache.keys.capacity(), 32);
        assert_eq!(cache.values.capacity(), 32);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.max_seq_len, 2048);
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
    }
}