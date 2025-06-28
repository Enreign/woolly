//! Fast FP32 model initialization bypassing quantization
//!
//! This module provides a fast path for testing the optimized kernels
//! by generating random FP32 weights instead of loading GGUF files.
//! This bypasses the 90s GGUF dequantization bottleneck to demonstrate
//! the real performance capability of the optimized inference kernels.

use crate::{CoreError, Result};
use crate::model::{ModelConfig, loader::LayerWeights};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Fast FP32 weight generator that creates random weights for testing
pub struct FastFP32WeightGenerator {
    /// Random number generator with fixed seed for reproducibility
    rng: StdRng,
    /// Model configuration
    config: ModelConfig,
}

impl FastFP32WeightGenerator {
    /// Create a new fast weight generator with the given configuration
    pub fn new(config: ModelConfig) -> Self {
        // Use fixed seed for reproducible performance testing
        let rng = StdRng::seed_from_u64(42);
        Self { rng, config }
    }

    /// Generate random embedding weights [hidden_size, vocab_size]
    pub fn generate_embeddings(&mut self) -> Result<Vec<f32>> {
        let size = self.config.hidden_size * self.config.vocab_size;
        let mut weights = Vec::with_capacity(size);
        
        // Use Xavier/Glorot initialization for stability
        let scale = (2.0 / (self.config.hidden_size + self.config.vocab_size) as f32).sqrt();
        
        for _ in 0..size {
            weights.push(self.rng.gen::<f32>() * scale * 2.0 - scale);
        }
        
        eprintln!("Generated embedding weights: {} elements", size);
        Ok(weights)
    }

    /// Generate weights for a single transformer layer
    pub fn generate_layer_weights(&mut self, layer_idx: usize) -> Result<LayerWeights> {
        eprintln!("Generating fast FP32 weights for layer {}", layer_idx);
        
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_size / num_heads;
        let num_kv_heads = self.config.num_key_value_heads.unwrap_or(num_heads);
        
        // Attention weights
        let attn_q_weight = self.generate_weight_matrix(hidden_size, hidden_size)?;
        let attn_k_weight = self.generate_weight_matrix(hidden_size, num_kv_heads * head_dim)?;
        let attn_v_weight = self.generate_weight_matrix(hidden_size, num_kv_heads * head_dim)?;
        let attn_o_weight = self.generate_weight_matrix(hidden_size, hidden_size)?;
        let attn_norm_weight = self.generate_norm_weights(hidden_size)?;
        
        // Feed-forward weights (SwiGLU)
        let ffn_gate_weight = self.generate_weight_matrix(hidden_size, intermediate_size)?;
        let ffn_up_weight = self.generate_weight_matrix(hidden_size, intermediate_size)?;
        let ffn_down_weight = self.generate_weight_matrix(intermediate_size, hidden_size)?;
        let ffn_norm_weight = self.generate_norm_weights(hidden_size)?;
        
        Ok(LayerWeights {
            attn_q_weight,
            attn_k_weight,
            attn_v_weight,
            attn_o_weight,
            attn_norm_weight,
            ffn_gate_weight: Some(ffn_gate_weight),
            ffn_up_weight,
            ffn_down_weight,
            ffn_norm_weight,
            // Shape information
            attn_q_shape: vec![hidden_size, hidden_size],
            attn_k_shape: vec![hidden_size, num_kv_heads * head_dim],
            attn_v_shape: vec![hidden_size, num_kv_heads * head_dim],
            attn_o_shape: vec![hidden_size, hidden_size],
            ffn_up_shape: vec![hidden_size, intermediate_size],
            ffn_down_shape: vec![intermediate_size, hidden_size],
            ffn_gate_shape: Some(vec![hidden_size, intermediate_size]),
        })
    }

    /// Generate final normalization weights
    pub fn generate_final_norm(&mut self) -> Result<Vec<f32>> {
        self.generate_norm_weights(self.config.hidden_size)
    }

    /// Generate LM head weights (if not tied to embeddings)
    pub fn generate_lm_head(&mut self) -> Result<Vec<f32>> {
        self.generate_weight_matrix(self.config.hidden_size, self.config.vocab_size)
    }

    /// Generate a weight matrix with proper initialization
    fn generate_weight_matrix(&mut self, input_dim: usize, output_dim: usize) -> Result<Vec<f32>> {
        let size = input_dim * output_dim;
        let mut weights = Vec::with_capacity(size);
        
        // Use Xavier/Glorot initialization
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        
        for _ in 0..size {
            weights.push(self.rng.gen::<f32>() * scale * 2.0 - scale);
        }
        
        Ok(weights)
    }

    /// Generate normalization weights (typically initialized to 1.0)
    fn generate_norm_weights(&mut self, size: usize) -> Result<Vec<f32>> {
        // LayerNorm weights are typically initialized to 1.0
        // Add small random variation for more realistic testing
        let mut weights = Vec::with_capacity(size);
        
        for _ in 0..size {
            weights.push(1.0 + (self.rng.gen::<f32>() - 0.5) * 0.01);
        }
        
        Ok(weights)
    }
}

/// Fast FP32 model that generates weights without loading GGUF
pub struct FastFP32Model {
    /// Model configuration
    config: ModelConfig,
    /// Whether the model has been initialized
    initialized: bool,
    /// Generated layer weights
    layer_weights: Vec<LayerWeights>,
    /// Generated embedding weights
    embedding_weights: Vec<f32>,
    /// Generated final norm weights
    final_norm_weights: Vec<f32>,
    /// Generated LM head weights
    lm_head_weights: Option<Vec<f32>>,
}

impl FastFP32Model {
    /// Create a new fast FP32 model with the given configuration
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            initialized: false,
            layer_weights: Vec::new(),
            embedding_weights: Vec::new(),
            final_norm_weights: Vec::new(),
            lm_head_weights: None,
        }
    }

    /// Initialize the model with random FP32 weights
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        eprintln!("FastFP32Model: Initializing with random weights (bypassing GGUF)");
        let start_time = std::time::Instant::now();
        
        let mut generator = FastFP32WeightGenerator::new(self.config.clone());
        
        // Generate embedding weights
        self.embedding_weights = generator.generate_embeddings()?;
        
        // Generate layer weights
        self.layer_weights.clear();
        self.layer_weights.reserve(self.config.num_layers);
        
        for layer_idx in 0..self.config.num_layers {
            let layer_weight = generator.generate_layer_weights(layer_idx)?;
            self.layer_weights.push(layer_weight);
        }
        
        // Generate final normalization weights
        self.final_norm_weights = generator.generate_final_norm()?;
        
        // Generate LM head weights (not tied to embeddings)
        self.lm_head_weights = Some(generator.generate_lm_head()?);
        
        self.initialized = true;
        
        let elapsed = start_time.elapsed();
        eprintln!("FastFP32Model: Initialization complete in {:.2}ms (vs 90s+ for GGUF dequantization)", 
                  elapsed.as_millis());
        
        Ok(())
    }

    /// Check if the model is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the embedding weights
    pub fn embedding_weights(&self) -> Result<&[f32]> {
        if !self.initialized {
            return Err(CoreError::model(
                "MODEL_NOT_INITIALIZED",
                "Model not initialized",
                "Getting embedding weights",
                "Call initialize() first"
            ));
        }
        Ok(&self.embedding_weights)
    }

    /// Get the layer weights
    pub fn layer_weights(&self) -> Result<&[LayerWeights]> {
        if !self.initialized {
            return Err(CoreError::model(
                "MODEL_NOT_INITIALIZED",
                "Model not initialized",
                "Getting layer weights",
                "Call initialize() first"
            ));
        }
        Ok(&self.layer_weights)
    }

    /// Get the final normalization weights
    pub fn final_norm_weights(&self) -> Result<&[f32]> {
        if !self.initialized {
            return Err(CoreError::model(
                "MODEL_NOT_INITIALIZED",
                "Model not initialized",
                "Getting final norm weights",
                "Call initialize() first"
            ));
        }
        Ok(&self.final_norm_weights)
    }

    /// Get the LM head weights
    pub fn lm_head_weights(&self) -> Result<Option<&[f32]>> {
        if !self.initialized {
            return Err(CoreError::model(
                "MODEL_NOT_INITIALIZED",
                "Model not initialized",
                "Getting LM head weights",
                "Call initialize() first"
            ));
        }
        Ok(self.lm_head_weights.as_deref())
    }
}

/// Check if fast FP32 mode is enabled via environment variable
pub fn is_fast_fp32_enabled() -> bool {
    std::env::var("WOOLLY_FAST_FP32")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Get the fast FP32 configuration from environment variables
pub fn get_fast_fp32_config() -> ModelConfig {
    // Allow customization via environment variables for testing
    let vocab_size = std::env::var("WOOLLY_FAST_VOCAB_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32000);
    
    let hidden_size = std::env::var("WOOLLY_FAST_HIDDEN_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    
    let num_layers = std::env::var("WOOLLY_FAST_NUM_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    
    let num_heads = std::env::var("WOOLLY_FAST_NUM_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    
    let context_length = std::env::var("WOOLLY_FAST_CONTEXT_LENGTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    
    let intermediate_size = std::env::var("WOOLLY_FAST_INTERMEDIATE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(11008);

    ModelConfig {
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        context_length,
        intermediate_size,
        num_key_value_heads: Some(num_heads), // Default to MHA
        rope_theta: Some(10000.0),
        layer_norm_epsilon: 1e-5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_weight_generator() {
        let config = ModelConfig::default();
        let mut generator = FastFP32WeightGenerator::new(config.clone());
        
        // Test embedding generation
        let embeddings = generator.generate_embeddings().unwrap();
        assert_eq!(embeddings.len(), config.hidden_size * config.vocab_size);
        
        // Test layer weight generation
        let layer_weights = generator.generate_layer_weights(0).unwrap();
        assert_eq!(layer_weights.attn_q_weight.len(), config.hidden_size * config.hidden_size);
    }

    #[test]
    fn test_fast_fp32_model() {
        let config = ModelConfig::default();
        let mut model = FastFP32Model::new(config.clone());
        
        assert!(!model.is_initialized());
        
        model.initialize().unwrap();
        assert!(model.is_initialized());
        
        // Test access to weights
        let embeddings = model.embedding_weights().unwrap();
        assert_eq!(embeddings.len(), config.hidden_size * config.vocab_size);
        
        let layer_weights = model.layer_weights().unwrap();
        assert_eq!(layer_weights.len(), config.num_layers);
    }

    #[test]
    fn test_environment_variable_detection() {
        // This test depends on environment setup
        // is_fast_fp32_enabled() should return false by default
        let config = get_fast_fp32_config();
        assert!(config.vocab_size > 0);
        assert!(config.hidden_size > 0);
    }
}