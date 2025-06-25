//! Transformer model implementation

use crate::{CoreError, Result};
use crate::model::{Model, ModelOutput, ModelFeature, ModelConfig, KVCache};
use crate::tensor_utils::{
    SimpleTensor, tensor_from_slice, zeros_tensor, embedding_lookup,
    layer_norm, rms_norm, matmul
};
use async_trait::async_trait;
use std::path::Path;
use woolly_tensor::Shape;

use super::attention::{MultiHeadAttention, AttentionConfig};
use super::embedding::{TokenEmbedding, RotaryEmbedding, SinusoidalPositionEmbedding};
use super::feedforward::{FeedForward, FeedForwardConfig, ActivationType};
use super::layer_norm::{LayerNorm, RMSNorm};

/// Transformer model configuration
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Base model configuration
    pub model_config: ModelConfig,
    /// Type of normalization to use
    pub norm_type: NormType,
    /// Type of position encoding
    pub position_encoding: PositionEncodingType,
    /// Activation function for FFN
    pub activation: ActivationType,
    /// Whether to tie input and output embeddings
    pub tie_embeddings: bool,
    /// Whether to use pre-normalization (like GPT) or post-normalization (like BERT)
    pub pre_norm: bool,
    /// Dropout probability
    pub dropout: f32,
    /// Attention dropout probability
    pub attention_dropout: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::default(),
            norm_type: NormType::LayerNorm,
            position_encoding: PositionEncodingType::Rotary,
            activation: ActivationType::SiLU,
            tie_embeddings: true,
            pre_norm: true,
            dropout: 0.1,
            attention_dropout: 0.1,
        }
    }
}

/// Normalization type
#[derive(Debug, Clone, Copy)]
pub enum NormType {
    LayerNorm,
    RMSNorm,
}

/// Position encoding type
#[derive(Debug, Clone, Copy)]
pub enum PositionEncodingType {
    Rotary,
    Sinusoidal,
    Learned,
    None,
}

/// Transformer decoder layer
struct TransformerLayer {
    /// Self-attention
    self_attn: MultiHeadAttention,
    /// Feed-forward network
    ffn: FeedForward,
    /// Attention normalization
    attn_norm: Box<dyn Norm>,
    /// FFN normalization
    ffn_norm: Box<dyn Norm>,
    /// Whether to use pre-normalization
    pre_norm: bool,
}

impl TransformerLayer {
    fn new(config: &TransformerConfig) -> Self {
        let attn_config = AttentionConfig {
            hidden_size: config.model_config.hidden_size,
            num_heads: config.model_config.num_heads,
            num_kv_heads: config.model_config.num_key_value_heads,
            head_dim: config.model_config.hidden_size / config.model_config.num_heads,
            max_seq_len: config.model_config.context_length,
            attention_dropout: config.attention_dropout,
            use_bias: false,
            use_flash_attention: false,
        };

        let ffn_config = FeedForwardConfig {
            hidden_size: config.model_config.hidden_size,
            intermediate_size: config.model_config.intermediate_size,
            activation: config.activation,
            dropout: config.dropout,
            use_bias: false,
            use_glu: matches!(config.activation, ActivationType::SwiGLU | ActivationType::GeGLU),
        };

        let attn_norm: Box<dyn Norm> = match config.norm_type {
            NormType::LayerNorm => Box::new(LayerNorm::new(
                config.model_config.hidden_size,
                config.model_config.layer_norm_epsilon,
                false,
            )),
            NormType::RMSNorm => Box::new(RMSNorm::new(
                config.model_config.hidden_size,
                config.model_config.layer_norm_epsilon,
            )),
        };

        let ffn_norm: Box<dyn Norm> = match config.norm_type {
            NormType::LayerNorm => Box::new(LayerNorm::new(
                config.model_config.hidden_size,
                config.model_config.layer_norm_epsilon,
                false,
            )),
            NormType::RMSNorm => Box::new(RMSNorm::new(
                config.model_config.hidden_size,
                config.model_config.layer_norm_epsilon,
            )),
        };

        Self {
            self_attn: MultiHeadAttention::new(attn_config),
            ffn: FeedForward::new(ffn_config),
            attn_norm,
            ffn_norm,
            pre_norm: config.pre_norm,
        }
    }

    fn forward(
        &self,
        hidden_states: &[f32],
        attention_mask: Option<&[f32]>,
        past_kv: Option<(&[f32], &[f32])>,
        use_cache: bool,
    ) -> Result<(Vec<f32>, Option<(Vec<f32>, Vec<f32>)>)> {
        let residual = hidden_states.to_vec();
        
        // Self-attention block
        let normed_hidden = if self.pre_norm {
            self.attn_norm.forward(hidden_states)?
        } else {
            hidden_states.to_vec()
        };

        let (attn_output, new_kv, _) = self.self_attn.forward(
            &normed_hidden,
            attention_mask,
            past_kv,
            use_cache,
        )?;

        // Add residual
        let mut hidden_states = if self.pre_norm {
            add_vectors(&residual, &attn_output)?
        } else {
            let normed = self.attn_norm.forward(&add_vectors(&residual, &attn_output)?)?;
            normed
        };

        // FFN block
        let residual = hidden_states.clone();
        
        let normed_hidden = if self.pre_norm {
            self.ffn_norm.forward(&hidden_states)?
        } else {
            hidden_states.clone()
        };

        let ffn_output = self.ffn.forward(&normed_hidden)?;

        hidden_states = if self.pre_norm {
            add_vectors(&residual, &ffn_output)?
        } else {
            self.ffn_norm.forward(&add_vectors(&residual, &ffn_output)?)?
        };

        Ok((hidden_states, new_kv))
    }

    /// Load weights for this transformer layer
    fn load_weights(&mut self, weights: &super::loader::LayerWeights) -> Result<()> {
        // Load attention weights
        self.self_attn.load_weights(weights)?;
        
        // Load feedforward weights  
        self.ffn.load_weights(weights)?;
        
        // Load normalization weights
        self.attn_norm.as_mut().load_weights(&weights.attn_norm_weight)?;
        self.ffn_norm.as_mut().load_weights(&weights.ffn_norm_weight)?;
        
        Ok(())
    }
}

/// Main transformer model
pub struct TransformerModel {
    config: TransformerConfig,
    /// Token embeddings as tensor
    embedding_weights: Option<SimpleTensor>,
    /// Token embeddings (legacy)
    embeddings: TokenEmbedding,
    /// Position embeddings
    position_encoder: Box<dyn PositionEncoder>,
    /// Transformer layers
    layers: Vec<TransformerLayer>,
    /// Final layer norm
    final_norm: Box<dyn Norm>,
    /// Output projection (language model head)
    lm_head: Option<Linear>,
}

impl TransformerModel {
    /// Create a new transformer model
    pub fn new(config: TransformerConfig) -> Self {
        let embeddings = TokenEmbedding::new(
            config.model_config.vocab_size,
            config.model_config.hidden_size,
        );

        let position_encoder: Box<dyn PositionEncoder> = match config.position_encoding {
            PositionEncodingType::Rotary => Box::new(
                RotaryEmbedding::new(
                    config.model_config.context_length,
                    config.model_config.hidden_size / config.model_config.num_heads,
                    config.model_config.rope_theta.unwrap_or(10000.0),
                ).unwrap()
            ),
            PositionEncodingType::Sinusoidal => Box::new(
                SinusoidalPositionEmbedding::new(
                    config.model_config.context_length,
                    config.model_config.hidden_size,
                )
            ),
            _ => Box::new(NoOpPositionEncoder),
        };

        let layers = (0..config.model_config.num_layers)
            .map(|_| TransformerLayer::new(&config))
            .collect();

        let final_norm: Box<dyn Norm> = match config.norm_type {
            NormType::LayerNorm => Box::new(LayerNorm::new(
                config.model_config.hidden_size,
                config.model_config.layer_norm_epsilon,
                false,
            )),
            NormType::RMSNorm => Box::new(RMSNorm::new(
                config.model_config.hidden_size,
                config.model_config.layer_norm_epsilon,
            )),
        };

        let lm_head = if config.tie_embeddings {
            None // Will use embedding weights transposed
        } else {
            Some(Linear::new(
                config.model_config.hidden_size,
                config.model_config.vocab_size,
                false,
            ))
        };

        Self {
            config,
            embedding_weights: None,
            embeddings,
            position_encoder,
            layers,
            final_norm,
            lm_head,
        }
    }

    /// Get logits from hidden states
    fn get_logits(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(hidden_states)
        } else {
            // Use tied embeddings - multiply by embedding matrix transpose
            let batch_size = hidden_states.len() / self.config.model_config.hidden_size;
            let vocab_size = self.config.model_config.vocab_size;
            let hidden_size = self.config.model_config.hidden_size;
            let mut logits = vec![0.0; batch_size * vocab_size];

            let embed_weights = self.embeddings.weights();

            for b in 0..batch_size {
                for v in 0..vocab_size {
                    let mut sum = 0.0;
                    for h in 0..hidden_size {
                        let hidden_idx = b * hidden_size + h;
                        let embed_idx = v * hidden_size + h;
                        sum += hidden_states[hidden_idx] * embed_weights[embed_idx];
                    }
                    logits[b * vocab_size + v] = sum;
                }
            }

            Ok(logits)
        }
    }
}

#[async_trait]
impl Model for TransformerModel {
    fn name(&self) -> &str {
        "TransformerModel"
    }

    fn model_type(&self) -> &str {
        "transformer"
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

    async fn forward(
        &self,
        input_ids: &[u32],
        past_kv_cache: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<ModelOutput> {
        let seq_len = input_ids.len();
        
        // Get embeddings using tensor operations
        let mut hidden_states = if let Some(ref embedding_weights) = self.embedding_weights {
            // Use tensor-based embedding lookup
            embedding_lookup(input_ids, embedding_weights)?
        } else {
            // Fallback to legacy embedding method
            let embedding_data = self.embeddings.forward(input_ids)?;
            tensor_from_slice(&embedding_data, Shape::matrix(seq_len, self.config.model_config.hidden_size))?
        };

        // Add position embeddings (for now, convert tensor to vec for position encoder)
        if !matches!(self.config.position_encoding, PositionEncodingType::Rotary) {
            let mut hidden_data = hidden_states.to_vec();
            self.position_encoder.encode(&mut hidden_data, seq_len)?;
            hidden_states = tensor_from_slice(&hidden_data, hidden_states.shape().clone())?;
        }

        // Process through layers
        let mut all_kv_cache = Vec::new();
        
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let past_kv = if let Some(cache) = past_kv_cache {
                if let Some(kv_cache) = cache.downcast_ref::<KVCache>() {
                    if layer_idx < kv_cache.keys.len() {
                        Some((kv_cache.keys[layer_idx].as_slice(), kv_cache.values[layer_idx].as_slice()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Convert tensor to vec for layer processing (temporary until layers are tensorized)
            let hidden_data = hidden_states.to_vec();
            
            let (new_hidden, new_kv) = layer.forward(&hidden_data, None, past_kv, true)?;
            
            // Convert back to tensor
            hidden_states = tensor_from_slice(&new_hidden, Shape::matrix(seq_len, self.config.model_config.hidden_size))?;

            if let Some((k, v)) = new_kv {
                all_kv_cache.push((k, v));
            }
        }

        // Final normalization using tensor operations 
        let final_norm_weights = zeros_tensor(Shape::vector(self.config.model_config.hidden_size))?; // Placeholder - should be loaded weights
        hidden_states = match self.config.norm_type {
            NormType::LayerNorm => layer_norm(&hidden_states, &final_norm_weights, None, self.config.model_config.layer_norm_epsilon)?,
            NormType::RMSNorm => rms_norm(&hidden_states, &final_norm_weights, self.config.model_config.layer_norm_epsilon)?,
        };

        // Get logits using tensor operations
        let logits_tensor = if let Some(ref lm_head) = self.lm_head {
            // Use LM head projection (would need to convert to tensor)
            let hidden_data = hidden_states.to_vec();
            let logits_data = lm_head.forward(&hidden_data)?;
            tensor_from_slice(&logits_data, Shape::matrix(seq_len, self.vocab_size()))?
        } else {
            // Use tied embeddings
            if let Some(ref embedding_weights) = self.embedding_weights {
                // Transpose embedding weights for tied head: hidden_states @ embed_weights^T
                let embed_transposed = embedding_weights.transpose(&[1, 0])
                    .map_err(|e| CoreError::Tensor(format!("Failed to transpose embeddings: {}", e)))?;
                matmul(&hidden_states, &embed_transposed)?
            } else {
                // Fallback to legacy method
                let hidden_data = hidden_states.to_vec();
                let logits_data = self.get_logits(&hidden_data)?;
                tensor_from_slice(&logits_data, Shape::matrix(seq_len, self.vocab_size()))?
            }
        };

        // Convert logits tensor to vec for output
        let logits = logits_tensor.to_vec();

        // Build KV cache
        let new_cache = if !all_kv_cache.is_empty() {
            let mut cache = KVCache::new(self.num_layers(), self.context_length());
            for (idx, (k, v)) in all_kv_cache.into_iter().enumerate() {
                cache.update(idx, k, v, vec![seq_len, self.num_heads(), self.hidden_size() / self.num_heads()])?;
            }
            Some(Box::new(cache) as Box<dyn std::any::Any + Send + Sync>)
        } else {
            None
        };

        Ok(ModelOutput {
            logits,
            logits_shape: vec![1, seq_len, self.vocab_size()],
            past_kv_cache: new_cache,
            hidden_states: None,
            attentions: None,
        })
    }

    async fn load_weights(&mut self, path: &Path) -> Result<()> {
        use super::loader::{GGUFModelLoader, ModelLoader, load_transformer_weights};
        
        // Load using GGUF loader
        let loader = GGUFModelLoader::from_path(path)?;
        let weights = load_transformer_weights(&loader, &self.config.model_config)?;
        
        // Load weights into the model components
        self.load_weights_from_model_weights(weights)?;
        
        Ok(())
    }

}

impl TransformerModel {
    /// Load weights from a ModelWeights structure into the transformer components
    fn load_weights_from_model_weights(&mut self, weights: super::loader::ModelWeights) -> Result<()> {
        // Load token embeddings as tensor
        let shape = &weights.embedding_shape;
        self.embedding_weights = Some(tensor_from_slice(&weights.embeddings, Shape::from_slice(shape))?);
        
        // Also load into legacy embedding for backward compatibility
        self.embeddings.load_weights(&weights.embeddings, &weights.embedding_shape)?;
        
        // Load layer weights
        if weights.layers.len() != self.layers.len() {
            return Err(CoreError::Model(format!(
                "Weight layer count {} doesn't match model layer count {}",
                weights.layers.len(),
                self.layers.len()
            )));
        }
        
        for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
            self.layers[layer_idx].load_weights(layer_weights)?;
        }
        
        // Load final normalization weights
        self.final_norm.as_mut().load_weights(&weights.final_norm)?;
        
        // Load language model head weights if present and not using tied embeddings
        if let Some(ref mut lm_head) = self.lm_head {
            if let (Some(ref lm_head_weights), Some(ref lm_head_shape)) = 
                (&weights.lm_head, &weights.lm_head_shape) {
                lm_head.load_weights(lm_head_weights, lm_head_shape)?;
            } else if !self.config.tie_embeddings {
                return Err(CoreError::Model(
                    "LM head weights not found but model not configured for tied embeddings".to_string()
                ));
            }
        }
        
        Ok(())
    }

    #[allow(dead_code)]
    fn supports_feature(&self, feature: ModelFeature) -> bool {
        match feature {
            ModelFeature::FlashAttention => false, // Could be enabled
            ModelFeature::GroupedQueryAttention => self.config.model_config.num_key_value_heads.is_some(),
            ModelFeature::SlidingWindowAttention => false,
            ModelFeature::RoPE => matches!(self.config.position_encoding, PositionEncodingType::Rotary),
            ModelFeature::ALiBi => false,
        }
    }
}

// Helper traits and implementations

trait Norm: Send + Sync {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>>;
    fn load_weights(&mut self, weights: &[f32]) -> Result<()>;
}

impl Norm for LayerNorm {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.forward(input)
    }
    
    fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        self.load_weights(weights)
    }
}

impl Norm for RMSNorm {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.forward(input)
    }
    
    fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        self.load_weights(weights)
    }
}

trait PositionEncoder: Send + Sync {
    fn encode(&self, embeddings: &mut [f32], seq_len: usize) -> Result<()>;
}

struct NoOpPositionEncoder;

impl PositionEncoder for NoOpPositionEncoder {
    fn encode(&self, _embeddings: &mut [f32], _seq_len: usize) -> Result<()> {
        Ok(())
    }
}

impl PositionEncoder for RotaryEmbedding {
    fn encode(&self, _embeddings: &mut [f32], _seq_len: usize) -> Result<()> {
        // RoPE is applied during attention, not to embeddings
        Ok(())
    }
}

impl PositionEncoder for SinusoidalPositionEmbedding {
    fn encode(&self, embeddings: &mut [f32], seq_len: usize) -> Result<()> {
        self.add_to_embeddings(embeddings, seq_len)
    }
}

/// Simple linear layer
struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let weight = vec![0.0; in_features * out_features];
        let bias = if use_bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };

        let mut layer = Self {
            in_features,
            out_features,
            weight,
            bias,
        };

        layer.init_weights();
        layer
    }

    fn init_weights(&mut self) {
        let scale = (2.0 / self.in_features as f32).sqrt();
        for w in self.weight.iter_mut() {
            *w = (rand::random::<f32>() * 2.0 - 1.0) * scale;
        }
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        let mut output = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0.0;
                for i in 0..self.in_features {
                    let input_idx = b * self.in_features + i;
                    let weight_idx = o * self.in_features + i;
                    sum += input[input_idx] * self.weight[weight_idx];
                }
                
                if let Some(ref bias) = self.bias {
                    sum += bias[o];
                }
                
                output[b * self.out_features + o] = sum;
            }
        }

        Ok(output)
    }

    /// Load weights from external data
    fn load_weights(&mut self, weights: &[f32], shape: &[usize]) -> Result<()> {
        // Validate shape
        if shape.len() != 2 {
            return Err(CoreError::InvalidInput(format!(
                "Expected 2D weight shape, got {}D", shape.len()
            )));
        }

        let expected_out_features = shape[0];
        let expected_in_features = shape[1];
        
        if expected_out_features != self.out_features {
            return Err(CoreError::InvalidInput(format!(
                "Output features mismatch: expected {}, got {}", 
                self.out_features, expected_out_features
            )));
        }
        
        if expected_in_features != self.in_features {
            return Err(CoreError::InvalidInput(format!(
                "Input features mismatch: expected {}, got {}", 
                self.in_features, expected_in_features
            )));
        }

        if weights.len() != self.out_features * self.in_features {
            return Err(CoreError::InvalidInput(format!(
                "Weight size mismatch: expected {}, got {}", 
                self.out_features * self.in_features, weights.len()
            )));
        }

        // Copy weights - note the transpose from [out_features, in_features] to [in_features, out_features]
        // GGUF format is typically [out_features, in_features] but we store as [in_features, out_features]
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                let src_idx = o * self.in_features + i;
                let dst_idx = o * self.in_features + i; // Same layout for now
                self.weight[dst_idx] = weights[src_idx];
            }
        }
        
        Ok(())
    }
}

// Helper functions

fn add_vectors(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(CoreError::InvalidInput("Vector lengths must match".to_string()));
    }
    
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_creation() {
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config);
        
        assert_eq!(model.name(), "TransformerModel");
        assert_eq!(model.model_type(), "transformer");
        assert_eq!(model.vocab_size(), 32000);
    }

    #[tokio::test]
    async fn test_transformer_forward() {
        let mut config = TransformerConfig::default();
        config.model_config.num_layers = 2; // Use fewer layers for testing
        config.model_config.hidden_size = 64;
        config.model_config.num_heads = 4;
        config.model_config.vocab_size = 100;
        
        let model = TransformerModel::new(config);
        
        let input_ids = vec![1, 2, 3, 4, 5];
        let output = model.forward(&input_ids, None).await.unwrap();
        
        assert_eq!(output.logits_shape, vec![1, 5, 100]);
        assert_eq!(output.logits.len(), 500); // 5 * 100
    }
}