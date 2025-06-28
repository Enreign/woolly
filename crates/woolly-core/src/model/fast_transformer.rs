//! Fast FP32 Transformer for performance testing
//!
//! This transformer uses random FP32 weights to bypass GGUF loading
//! and demonstrate the real performance capabilities of the optimized kernels.

use crate::{CoreError, Result};
use crate::model::{Model, ModelConfig, ModelFeature, KVCache, fast_initialization::FastFP32Model};
use crate::tensor_utils_simd;
use std::collections::HashMap;

/// Fast transformer implementation for testing optimized kernels
pub struct FastTransformer {
    /// Model name
    name: String,
    /// Fast FP32 model with random weights
    fast_model: FastFP32Model,
    /// Current KV cache
    kv_cache: Option<KVCache>,
    /// Whether to use SIMD optimizations
    use_simd: bool,
}

impl FastTransformer {
    /// Create a new fast transformer with the given configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        let fast_model = FastFP32Model::new(config);
        
        Ok(Self {
            name: "FastTransformer".to_string(),
            fast_model,
            kv_cache: None,
            use_simd: !std::env::var("WOOLLY_DISABLE_SIMD")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
        })
    }

    /// Initialize the transformer with random weights
    pub fn initialize(&mut self) -> Result<()> {
        eprintln!("FastTransformer: Initializing with random FP32 weights");
        self.fast_model.initialize()?;
        
        // Initialize KV cache
        let config = self.fast_model.config();
        self.kv_cache = Some(KVCache::new(config.num_layers, config.context_length));
        
        eprintln!("FastTransformer: Ready for high-performance inference");
        Ok(())
    }

    /// Ensure the model is initialized
    fn ensure_initialized(&mut self) -> Result<()> {
        if !self.fast_model.is_initialized() {
            self.initialize()?;
        }
        Ok(())
    }

    /// Forward pass with optimized attention computation
    fn forward_attention(&self, layer_idx: usize, hidden_states: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let config = self.fast_model.config();
        let layer_weights = self.fast_model.layer_weights()?;
        let layer = &layer_weights[layer_idx];
        
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = hidden_size / num_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);
        
        // Q, K, V projections
        let q = self.matmul(hidden_states, &layer.attn_q_weight, seq_len, hidden_size, hidden_size)?;
        let k = self.matmul(hidden_states, &layer.attn_k_weight, seq_len, hidden_size, num_kv_heads * head_dim)?;
        let v = self.matmul(hidden_states, &layer.attn_v_weight, seq_len, hidden_size, num_kv_heads * head_dim)?;
        
        // Apply RoPE if configured
        let (q_rope, k_rope) = if config.rope_theta.is_some() {
            self.apply_rope(&q, &k, seq_len, head_dim)?
        } else {
            (q, k)
        };
        
        // Compute attention
        let attn_output = self.compute_attention(&q_rope, &k_rope, &v, seq_len, num_heads, head_dim, num_kv_heads)?;
        
        // Output projection
        self.matmul(&attn_output, &layer.attn_o_weight, seq_len, hidden_size, hidden_size)
    }

    /// Forward pass with optimized feed-forward computation
    fn forward_feedforward(&self, layer_idx: usize, hidden_states: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let config = self.fast_model.config();
        let layer_weights = self.fast_model.layer_weights()?;
        let layer = &layer_weights[layer_idx];
        
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        
        // SwiGLU: gate_proj(x) * silu(up_proj(x))
        let gate_output = self.matmul(hidden_states, layer.ffn_gate_weight.as_ref().unwrap(), 
                                     seq_len, hidden_size, intermediate_size)?;
        let up_output = self.matmul(hidden_states, &layer.ffn_up_weight, 
                                   seq_len, hidden_size, intermediate_size)?;
        
        // Apply SiLU activation to gate
        let gate_activated = self.apply_silu(&gate_output)?;
        
        // Element-wise multiplication
        let gated_output = self.elementwise_multiply(&gate_activated, &up_output)?;
        
        // Down projection
        self.matmul(&gated_output, &layer.ffn_down_weight, seq_len, intermediate_size, hidden_size)
    }

    /// Optimized matrix multiplication
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if self.use_simd {
            // Use SIMD-optimized matrix multiplication
            // Convert slices to SimpleTensor
            let a_tensor = crate::tensor_utils::SimpleTensor::new(a.to_vec(), woolly_tensor::Shape::matrix(m, k))?;
            let b_tensor = crate::tensor_utils::SimpleTensor::new(b.to_vec(), woolly_tensor::Shape::matrix(k, n))?;
            tensor_utils_simd::optimized_matmul(&a_tensor, &b_tensor, m, k, n)
                .map(|t| t.data)
                .map_err(|e| CoreError::tensor(
                    "MATMUL_FAILED",
                    format!("SIMD matmul failed: {}", e),
                    "Computing matrix multiplication",
                    "Check tensor dimensions and SIMD availability"
                ))
        } else {
            // Fallback to basic implementation
            self.basic_matmul(a, b, m, k, n)
        }
    }

    /// Basic matrix multiplication fallback
    fn basic_matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(CoreError::tensor(
                "MATMUL_DIMENSION_MISMATCH",
                format!("Matrix dimensions don't match: a={}x{}, b={}x{}", m, k, k, n),
                "Basic matrix multiplication",
                "Check input tensor dimensions"
            ));
        }
        
        let mut result = vec![0.0f32; m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        
        Ok(result)
    }

    /// Apply RoPE (Rotary Position Embedding)
    fn apply_rope(&self, q: &[f32], k: &[f32], seq_len: usize, head_dim: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let config = self.fast_model.config();
        let theta = config.rope_theta.unwrap_or(10000.0);
        
        let mut q_rope = q.to_vec();
        let mut k_rope = k.to_vec();
        
        let num_heads = q.len() / (seq_len * head_dim);
        
        for pos in 0..seq_len {
            for head in 0..num_heads {
                for i in (0..head_dim).step_by(2) {
                    let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
                    let angle = pos as f32 * freq;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();
                    
                    let q_offset = head * seq_len * head_dim + pos * head_dim + i;
                    let k_offset = head * seq_len * head_dim + pos * head_dim + i;
                    
                    if q_offset + 1 < q_rope.len() && k_offset + 1 < k_rope.len() {
                        let q_real = q[q_offset];
                        let q_imag = q[q_offset + 1];
                        let k_real = k[k_offset];
                        let k_imag = k[k_offset + 1];
                        
                        q_rope[q_offset] = q_real * cos_val - q_imag * sin_val;
                        q_rope[q_offset + 1] = q_real * sin_val + q_imag * cos_val;
                        k_rope[k_offset] = k_real * cos_val - k_imag * sin_val;
                        k_rope[k_offset + 1] = k_real * sin_val + k_imag * cos_val;
                    }
                }
            }
        }
        
        Ok((q_rope, k_rope))
    }

    /// Compute scaled dot-product attention
    fn compute_attention(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize, 
                        num_heads: usize, head_dim: usize, num_kv_heads: usize) -> Result<Vec<f32>> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * num_heads * head_dim];
        
        // Simplified attention computation for performance testing
        for h in 0..num_heads {
            let kv_head = h * num_kv_heads / num_heads; // Handle GQA
            
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j <= i { // Causal mask
                        let mut score = 0.0;
                        
                        // Compute attention score
                        for d in 0..head_dim {
                            let q_idx = h * seq_len * head_dim + i * head_dim + d;
                            let k_idx = kv_head * seq_len * head_dim + j * head_dim + d;
                            score += q[q_idx] * k[k_idx];
                        }
                        
                        score *= scale;
                        let attn_weight = score.exp(); // Simplified softmax
                        
                        // Apply attention to values
                        for d in 0..head_dim {
                            let v_idx = kv_head * seq_len * head_dim + j * head_dim + d;
                            let out_idx = h * seq_len * head_dim + i * head_dim + d;
                            output[out_idx] += attn_weight * v[v_idx];
                        }
                    }
                }
            }
        }
        
        Ok(output)
    }

    /// Apply SiLU activation function
    fn apply_silu(&self, input: &[f32]) -> Result<Vec<f32>> {
        if self.use_simd {
            // Use SIMD-optimized SiLU
            // Convert slice to SimpleTensor
            let input_tensor = crate::tensor_utils::SimpleTensor::new(input.to_vec(), woolly_tensor::Shape::vector(input.len()))?;
            tensor_utils_simd::silu(&input_tensor)
                .map(|t| t.data)
                .map_err(|e| CoreError::tensor(
                    "SILU_FAILED",
                    format!("SIMD SiLU failed: {}", e),
                    "Applying SiLU activation",
                    "Check SIMD availability"
                ))
        } else {
            // Basic SiLU implementation
            Ok(input.iter().map(|&x| x / (1.0 + (-x).exp())).collect())
        }
    }

    /// Element-wise multiplication
    fn elementwise_multiply(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(CoreError::tensor(
                "ELEMENTWISE_DIMENSION_MISMATCH",
                format!("Vector lengths don't match: {} vs {}", a.len(), b.len()),
                "Element-wise multiplication",
                "Check input tensor dimensions"
            ));
        }
        
        if self.use_simd {
            // Use SIMD-optimized element-wise multiplication
            // Convert slices to SimpleTensor
            let a_tensor = crate::tensor_utils::SimpleTensor::new(a.to_vec(), woolly_tensor::Shape::vector(a.len()))?;
            let b_tensor = crate::tensor_utils::SimpleTensor::new(b.to_vec(), woolly_tensor::Shape::vector(b.len()))?;
            tensor_utils_simd::elementwise_multiply(&a_tensor, &b_tensor)
                .map(|t| t.data)
                .map_err(|e| CoreError::tensor(
                    "ELEMENTWISE_FAILED",
                    format!("SIMD elementwise multiply failed: {}", e),
                    "Element-wise multiplication",
                    "Check SIMD availability"
                ))
        } else {
            // Basic implementation
            Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect())
        }
    }

    /// Apply layer normalization
    fn apply_layer_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        if input.len() != weight.len() {
            return Err(CoreError::tensor(
                "LAYERNORM_DIMENSION_MISMATCH",
                format!("Input and weight dimensions don't match: {} vs {}", input.len(), weight.len()),
                "Layer normalization",
                "Check input tensor dimensions"
            ));
        }
        
        let n = input.len() as f32;
        let mean = input.iter().sum::<f32>() / n;
        let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = (var + eps).sqrt();
        
        let normalized: Vec<f32> = input.iter()
            .zip(weight.iter())
            .map(|(&x, &w)| ((x - mean) / std) * w)
            .collect();
        
        Ok(normalized)
    }
}

impl Model for FastTransformer {
    fn name(&self) -> &str {
        &self.name
    }

    fn model_type(&self) -> &str {
        "fast_transformer"
    }

    fn vocab_size(&self) -> usize {
        self.fast_model.config().vocab_size
    }

    fn context_length(&self) -> usize {
        self.fast_model.config().context_length
    }

    fn hidden_size(&self) -> usize {
        self.fast_model.config().hidden_size
    }

    fn num_layers(&self) -> usize {
        self.fast_model.config().num_layers
    }

    fn num_heads(&self) -> usize {
        self.fast_model.config().num_heads
    }

    fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        // Assume model is already initialized
        
        let seq_len = input_ids.len();
        let config = self.fast_model.config();
        let hidden_size = config.hidden_size;
        
        eprintln!("FastTransformer: Processing {} tokens", seq_len);
        let start_time = std::time::Instant::now();
        
        // Embedding lookup
        let embedding_weights = self.fast_model.embedding_weights()?;
        let mut hidden_states = Vec::with_capacity(seq_len * hidden_size);
        
        for &token_id in input_ids {
            let token_idx = (token_id as usize).min(config.vocab_size - 1);
            let start_idx = token_idx * hidden_size;
            let end_idx = start_idx + hidden_size;
            hidden_states.extend_from_slice(&embedding_weights[start_idx..end_idx]);
        }
        
        // Transformer layers
        for layer_idx in 0..config.num_layers {
            // Pre-attention layer norm
            let layer_weights = self.fast_model.layer_weights()?;
            let normalized_hidden = self.apply_layer_norm(&hidden_states, &layer_weights[layer_idx].attn_norm_weight, config.layer_norm_epsilon)?;
            
            // Self-attention
            let attn_output = self.forward_attention(layer_idx, &normalized_hidden, seq_len)?;
            
            // Residual connection
            for i in 0..hidden_states.len() {
                hidden_states[i] += attn_output[i];
            }
            
            // Pre-FFN layer norm
            let normalized_hidden = self.apply_layer_norm(&hidden_states, &layer_weights[layer_idx].ffn_norm_weight, config.layer_norm_epsilon)?;
            
            // Feed-forward
            let ffn_output = self.forward_feedforward(layer_idx, &normalized_hidden, seq_len)?;
            
            // Residual connection
            for i in 0..hidden_states.len() {
                hidden_states[i] += ffn_output[i];
            }
        }
        
        // Final layer norm
        let final_norm_weights = self.fast_model.final_norm_weights()?;
        let normalized_output = self.apply_layer_norm(&hidden_states, final_norm_weights, config.layer_norm_epsilon)?;
        
        // LM head projection (only for the last token)
        let last_token_hidden = &normalized_output[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let lm_head_weights = self.fast_model.lm_head_weights()?.unwrap_or(embedding_weights);
        
        let logits = self.matmul(last_token_hidden, lm_head_weights, 1, hidden_size, config.vocab_size)?;
        
        let elapsed = start_time.elapsed();
        eprintln!("FastTransformer: Forward pass completed in {:.2}ms ({} tokens)", 
                  elapsed.as_millis(), seq_len);
        
        Ok(logits)
    }

    fn supports_feature(&self, feature: ModelFeature) -> bool {
        match feature {
            ModelFeature::RoPE => true,
            ModelFeature::GroupedQueryAttention => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_transformer_creation() {
        let config = ModelConfig::default();
        let transformer = FastTransformer::new(config).unwrap();
        
        assert_eq!(transformer.name(), "FastTransformer");
        assert_eq!(transformer.model_type(), "fast_transformer");
    }

    #[test]
    fn test_fast_transformer_forward() {
        let config = ModelConfig {
            vocab_size: 1000,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 8,
            context_length: 128,
            intermediate_size: 2048,
            num_key_value_heads: Some(8),
            rope_theta: Some(10000.0),
            layer_norm_epsilon: 1e-5,
        };
        
        let mut transformer = FastTransformer::new(config).unwrap();
        
        // Test forward pass
        let input_tokens = vec![1, 2, 3, 4, 5];
        let logits = transformer.forward(&input_tokens).unwrap();
        
        assert_eq!(logits.len(), 1000); // vocab_size
    }
}