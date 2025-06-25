//! Multi-head attention implementation for transformer models

use crate::{CoreError, Result};
use crate::tensor_utils::{SimpleTensor, tensor_from_slice, matmul, add_tensors, softmax};
use woolly_tensor::Shape;
use std::f32;

/// Multi-head attention configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<usize>,
    /// Head dimension (hidden_size / num_heads)
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Attention dropout probability
    pub attention_dropout: f32,
    /// Whether to use bias in linear projections
    pub use_bias: bool,
    /// Whether to use flash attention
    pub use_flash_attention: bool,
}

impl AttentionConfig {
    /// Create a new attention configuration
    pub fn new(hidden_size: usize, num_heads: usize, max_seq_len: usize) -> Result<Self> {
        if hidden_size % num_heads != 0 {
            return Err(CoreError::InvalidInput(
                "Hidden size must be divisible by number of heads".to_string(),
            ));
        }

        Ok(Self {
            hidden_size,
            num_heads,
            num_kv_heads: None,
            head_dim: hidden_size / num_heads,
            max_seq_len,
            attention_dropout: 0.0,
            use_bias: false,
            use_flash_attention: false,
        })
    }

    /// Enable grouped query attention with specified number of KV heads
    pub fn with_gqa(mut self, num_kv_heads: usize) -> Result<Self> {
        if self.num_heads % num_kv_heads != 0 {
            return Err(CoreError::InvalidInput(
                "Number of heads must be divisible by number of KV heads".to_string(),
            ));
        }
        self.num_kv_heads = Some(num_kv_heads);
        Ok(self)
    }
}

/// Multi-head attention layer
pub struct MultiHeadAttention {
    config: AttentionConfig,
    /// Query projection weights [hidden_size, hidden_size]
    q_proj: TensorLinear,
    /// Key projection weights [hidden_size, kv_hidden_size]
    k_proj: TensorLinear,
    /// Value projection weights [hidden_size, kv_hidden_size]
    v_proj: TensorLinear,
    /// Output projection weights [hidden_size, hidden_size]
    o_proj: TensorLinear,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(config: AttentionConfig) -> Self {
        let kv_hidden_size = if let Some(num_kv_heads) = config.num_kv_heads {
            num_kv_heads * config.head_dim
        } else {
            config.hidden_size
        };

        Self {
            q_proj: TensorLinear::new(config.hidden_size, config.hidden_size, config.use_bias),
            k_proj: TensorLinear::new(config.hidden_size, kv_hidden_size, config.use_bias),
            v_proj: TensorLinear::new(config.hidden_size, kv_hidden_size, config.use_bias),
            o_proj: TensorLinear::new(config.hidden_size, config.hidden_size, config.use_bias),
            config,
        }
    }

    /// Tensor-based forward pass through multi-head attention  
    pub fn forward_tensor(
        &self,
        hidden_states: &SimpleTensor,
        attention_mask: Option<&SimpleTensor>,
        _past_key_value: Option<(&SimpleTensor, &SimpleTensor)>,
        use_cache: bool,
    ) -> Result<(SimpleTensor, Option<(SimpleTensor, SimpleTensor)>, Option<SimpleTensor>)> {
        
        // Project to Q, K, V using tensor operations
        // TODO: Implement actual tensor-based linear projections
        let query_states = hidden_states.clone();
        let key_states = hidden_states.clone();
        let value_states = hidden_states.clone();

        // Apply scaled dot-product attention with tensors
        let (attention_output, attention_weights) = self.scaled_dot_product_attention_tensor(
            &query_states,
            &key_states, 
            &value_states,
            attention_mask,
        )?;

        // Project output
        let output = attention_output; // TODO: Implement actual tensor-based linear projection

        // Prepare cache if requested
        let cache = if use_cache {
            Some((key_states, value_states))
        } else {
            None
        };

        Ok((output, cache, attention_weights))
    }

    /// Legacy forward pass through multi-head attention
    /// Input shape: [batch_size * seq_len, hidden_size] (flattened)
    /// Returns: (output, attention_weights)
    pub fn forward(
        &self,
        hidden_states: &[f32],
        attention_mask: Option<&[f32]>,
        _past_key_value: Option<(&[f32], &[f32])>,
        use_cache: bool,
    ) -> Result<(Vec<f32>, Option<(Vec<f32>, Vec<f32>)>, Option<Vec<f32>>)> {
        let total_elements = hidden_states.len();
        let hidden_size = self.config.hidden_size;
        let seq_len = total_elements / hidden_size;

        // Project to Q, K, V using legacy linear layers
        // TODO: Convert to proper tensor operations
        let query_states = hidden_states.to_vec(); // Placeholder - should use actual linear projection
        let key_states = hidden_states.to_vec(); // Placeholder
        let value_states = hidden_states.to_vec(); // Placeholder

        // Reshape for multi-head attention
        let (query_states, key_states, value_states) = self.reshape_for_attention(
            &query_states,
            &key_states,
            &value_states,
            seq_len,
        )?;

        // Apply attention
        let (attention_output, attention_weights) = self.scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            seq_len,
        )?;

        // Reshape back and project output
        let attention_output = self.reshape_from_attention(&attention_output, seq_len);
        let output = attention_output; // Placeholder - should use actual linear projection

        // Prepare cache if requested
        let cache = if use_cache {
            Some((key_states, value_states))
        } else {
            None
        };

        Ok((output, cache, attention_weights))
    }

    /// Reshape tensors for multi-head attention
    fn reshape_for_attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads.unwrap_or(num_heads);
        let head_dim = self.config.head_dim;

        // Query: [seq_len, num_heads, head_dim]
        let mut q_reshaped = vec![0.0; query.len()];
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx = s * self.config.hidden_size + h * head_dim + d;
                    let dst_idx = (s * num_heads + h) * head_dim + d;
                    q_reshaped[dst_idx] = query[src_idx];
                }
            }
        }

        // Key and Value: [seq_len, num_kv_heads, head_dim]
        let kv_size = seq_len * num_kv_heads * head_dim;
        let mut k_reshaped = vec![0.0; kv_size];
        let mut v_reshaped = vec![0.0; kv_size];

        for s in 0..seq_len {
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let src_idx = s * num_kv_heads * head_dim + h * head_dim + d;
                    let dst_idx = (s * num_kv_heads + h) * head_dim + d;
                    k_reshaped[dst_idx] = key[src_idx];
                    v_reshaped[dst_idx] = value[src_idx];
                }
            }
        }

        Ok((q_reshaped, k_reshaped, v_reshaped))
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        attention_mask: Option<&[f32]>,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads.unwrap_or(num_heads);
        let head_dim = self.config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Handle grouped query attention
        let kv_repeat = num_heads / num_kv_heads;

        let mut attention_scores = vec![0.0; seq_len * seq_len * num_heads];
        let mut attention_weights = vec![0.0; seq_len * seq_len * num_heads];
        let mut output = vec![0.0; seq_len * num_heads * head_dim];

        // Compute attention scores for each head
        for h in 0..num_heads {
            let kv_h = h / kv_repeat; // Map to KV head

            // Compute Q @ K^T for this head
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        let q_idx = (i * num_heads + h) * head_dim + d;
                        let k_idx = (j * num_kv_heads + kv_h) * head_dim + d;
                        score += query[q_idx] * key[k_idx];
                    }
                    score *= scale;

                    let score_idx = h * seq_len * seq_len + i * seq_len + j;
                    attention_scores[score_idx] = score;
                }
            }
        }

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let idx = h * seq_len * seq_len + i * seq_len + j;
                        let mask_idx = i * seq_len + j;
                        attention_scores[idx] += mask[mask_idx];
                    }
                }
            }
        }

        // Apply causal mask (for autoregressive models)
        if self.config.use_flash_attention {
            // Placeholder for flash attention
            // In real implementation, this would use optimized kernels
        } else {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        let idx = h * seq_len * seq_len + i * seq_len + j;
                        attention_scores[idx] = -f32::INFINITY;
                    }
                }
            }
        }

        // Softmax over last dimension
        for h in 0..num_heads {
            for i in 0..seq_len {
                let row_start = h * seq_len * seq_len + i * seq_len;
                let row_end = row_start + seq_len;
                let row = &mut attention_scores[row_start..row_end];

                // Find max for numerical stability
                let max_score = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp and sum
                let mut sum = 0.0;
                for score in row.iter_mut() {
                    *score = (*score - max_score).exp();
                    sum += *score;
                }

                // Normalize
                for (idx, score) in row.iter().enumerate() {
                    attention_weights[row_start + idx] = score / sum;
                }
            }
        }

        // Apply attention weights to values
        for h in 0..num_heads {
            let kv_h = h / kv_repeat;

            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        let weight_idx = h * seq_len * seq_len + i * seq_len + j;
                        let v_idx = (j * num_kv_heads + kv_h) * head_dim + d;
                        sum += attention_weights[weight_idx] * value[v_idx];
                    }
                    let out_idx = (i * num_heads + h) * head_dim + d;
                    output[out_idx] = sum;
                }
            }
        }

        Ok((output, Some(attention_weights)))
    }

    /// Tensor-based scaled dot-product attention
    fn scaled_dot_product_attention_tensor(
        &self,
        query: &SimpleTensor,
        key: &SimpleTensor,
        value: &SimpleTensor,
        attention_mask: Option<&SimpleTensor>,
    ) -> Result<(SimpleTensor, Option<SimpleTensor>)> {
        let head_dim = self.config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Transpose key for attention computation: Q @ K^T
        let key_transposed = key.transpose(&[1, 0])?;
        
        // Compute attention scores: Q @ K^T
        let mut attention_scores = matmul(query, &key_transposed)?;
        
        // Scale by sqrt(head_dim)
        let scores_data = attention_scores.to_vec();
        let scaled_scores: Vec<f32> = scores_data.iter().map(|&x| x * scale).collect();
        attention_scores = tensor_from_slice(&scaled_scores, attention_scores.shape().clone())?;
        
        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            attention_scores = add_tensors(&attention_scores, mask)?;
        }
        
        // Apply causal mask for autoregressive generation  
        // For now, we'll handle this at the Vec level
        let mut scores_data = attention_scores.to_vec();
        
        let seq_len = query.shape().as_slice()[0];
        let num_heads = self.config.num_heads;
        
        // Apply causal mask
        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    if idx < scores_data.len() {
                        scores_data[idx] = -f32::INFINITY;
                    }
                }
            }
        }
        
        attention_scores = tensor_from_slice(&scores_data, attention_scores.shape().clone())?;
        
        // Apply softmax
        let attention_weights = softmax(&attention_scores)?;
        
        // Apply attention weights to values: Attention @ V
        let attention_output = matmul(&attention_weights, value)?;
        
        Ok((attention_output, Some(attention_weights)))
    }

    /// Reshape output from attention format back to original
    fn reshape_from_attention(&self, attention_output: &[f32], seq_len: usize) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = self.config.hidden_size;

        let mut output = vec![0.0; seq_len * hidden_size];

        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx = (s * num_heads + h) * head_dim + d;
                    let dst_idx = s * hidden_size + h * head_dim + d;
                    output[dst_idx] = attention_output[src_idx];
                }
            }
        }

        output
    }

    /// Load weights for the attention layer
    pub fn load_weights(&mut self, weights: &super::loader::LayerWeights) -> Result<()> {
        // Load query projection weights
        self.q_proj.load_weights(&weights.attn_q_weight, &weights.attn_q_shape)?;
        
        // Load key projection weights
        self.k_proj.load_weights(&weights.attn_k_weight, &weights.attn_k_shape)?;
        
        // Load value projection weights
        self.v_proj.load_weights(&weights.attn_v_weight, &weights.attn_v_shape)?;
        
        // Load output projection weights
        self.o_proj.load_weights(&weights.attn_o_weight, &weights.attn_o_shape)?;
        
        Ok(())
    }
}

/// Tensor-based linear layer for efficient projections
#[allow(dead_code)]
struct TensorLinear {
    in_features: usize,
    out_features: usize,
    weight: Option<SimpleTensor>,
    bias: Option<SimpleTensor>,
}

#[allow(dead_code)]
impl TensorLinear {
    fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        Self {
            in_features,
            out_features,
            weight: None,
            bias: if use_bias { None } else { None }, // Will be loaded from external weights
        }
    }

    fn forward(&self, input: &SimpleTensor) -> Result<SimpleTensor> {
        let weight = self.weight.as_ref()
            .ok_or_else(|| CoreError::Tensor("Linear layer weights not loaded".to_string()))?;
        
        // Perform matrix multiplication: input @ weight^T
        let output = matmul(input, weight)?;
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            add_tensors(&output, bias)
        } else {
            Ok(output)
        }
    }

    fn load_weights(&mut self, weights: &[f32], shape: &[usize]) -> Result<()> {
        if shape.len() != 2 {
            return Err(CoreError::InvalidInput(format!(
                "Expected 2D weight shape, got {}D", shape.len()
            )));
        }

        let expected_out_features = shape[0];
        let expected_in_features = shape[1];
        
        if expected_out_features != self.out_features || expected_in_features != self.in_features {
            return Err(CoreError::InvalidInput(format!(
                "Weight shape mismatch: expected {}x{}, got {}x{}", 
                self.out_features, self.in_features, expected_out_features, expected_in_features
            )));
        }

        // Store weights as tensor (transposed for efficient matrix multiplication)
        let weight_shape = Shape::matrix(self.in_features, self.out_features);
        let mut transposed_weights = vec![0.0f32; weights.len()];
        
        // Transpose from [out_features, in_features] to [in_features, out_features]
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                let src_idx = o * self.in_features + i;
                let dst_idx = i * self.out_features + o;
                transposed_weights[dst_idx] = weights[src_idx];
            }
        }
        
        self.weight = Some(tensor_from_slice(&transposed_weights, weight_shape)?);
        Ok(())
    }
}

/// Simple linear layer for projections (legacy)
#[allow(dead_code)]
struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

#[allow(dead_code)]
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

        // Initialize weights
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

        // Matrix multiplication: output = input @ weight.T + bias
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

        // Copy weights
        self.weight.copy_from_slice(weights);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(512, 8, 128).unwrap();
        assert_eq!(config.head_dim, 64);

        let gqa_config = config.with_gqa(4).unwrap();
        assert_eq!(gqa_config.num_kv_heads, Some(4));
    }

    #[test]
    fn test_multi_head_attention() {
        let config = AttentionConfig::new(64, 4, 32).unwrap();
        let mha = MultiHeadAttention::new(config);

        let input = vec![0.1; 10 * 64]; // seq_len=10, hidden_size=64
        let (output, _, _) = mha.forward(&input, None, None, false).unwrap();
        
        assert_eq!(output.len(), input.len());
    }
}