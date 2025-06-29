//! Multi-head attention implementation for transformer models

use crate::{CoreError, Result};
use crate::tensor_utils::{SimpleTensor, tensor_from_slice, matmul, add_tensors, softmax};
use crate::blas_matmul::{matmul_blas, is_blas_available};
use crate::blas_attention::{scaled_dot_product_attention_blas, grouped_query_attention_blas};
use woolly_tensor::Shape;
use std::f32;
use rayon::prelude::*;

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
            return Err(CoreError::invalid_input(
                "INVALID_HEAD_CONFIG",
                "Hidden size must be divisible by number of heads",
                "attention configuration",
                "Ensure hidden_size is divisible by num_heads"
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
            return Err(CoreError::invalid_input(
                "INVALID_GQA_CONFIG",
                "Number of heads must be divisible by number of KV heads",
                "grouped query attention configuration",
                "Ensure num_heads is divisible by num_kv_heads"
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

    /// BLAS-optimized forward pass through multi-head attention
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

        eprintln!("🚀 Using BLAS-optimized attention (seq_len={}, hidden_size={})", seq_len, hidden_size);

        // Project to Q, K, V using BLAS-accelerated matrix multiplication
        let query_states = self.project_queries(hidden_states, seq_len)?;
        let key_states = self.project_keys(hidden_states, seq_len)?;
        let value_states = self.project_values(hidden_states, seq_len)?;

        // Apply BLAS-optimized attention
        let (attention_output, attention_weights) = if let Some(num_kv_heads) = self.config.num_kv_heads {
            // Use Grouped Query Attention
            eprintln!("🔄 Using Grouped Query Attention with {} KV heads", num_kv_heads);
            let output = grouped_query_attention_blas(
                &query_states,
                &key_states,
                &value_states,
                seq_len,
                seq_len, // total_seq_len = seq_len for now
                self.config.num_heads,
                num_kv_heads,
                self.config.head_dim,
                1.0 / (self.config.head_dim as f32).sqrt(),
            )?;
            (output, None) // GQA doesn't return attention weights for now
        } else {
            // Use standard multi-head attention
            eprintln!("🔄 Using standard multi-head attention");
            let output = scaled_dot_product_attention_blas(
                &query_states,
                &key_states,
                &value_states,
                seq_len,
                seq_len, // total_seq_len = seq_len for now
                self.config.num_heads,
                self.config.head_dim,
                1.0 / (self.config.head_dim as f32).sqrt(),
            )?;
            (output, None) // Standard doesn't return attention weights for now
        };

        // Project output using BLAS
        let output = self.project_output(&attention_output, seq_len)?;

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

        // Compute attention scores for each head in parallel
        let score_chunks: Vec<_> = (0..num_heads).into_par_iter().map(|h| {
            let kv_h = h / kv_repeat; // Map to KV head
            let mut head_scores = vec![0.0f32; seq_len * seq_len];

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
                    head_scores[i * seq_len + j] = score;
                }
            }
            (h, head_scores)
        }).collect();

        // Copy results back to attention_scores
        for (h, head_scores) in score_chunks {
            let offset = h * seq_len * seq_len;
            attention_scores[offset..offset + seq_len * seq_len].copy_from_slice(&head_scores);
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

        // Parallel softmax computation over last dimension
        let weight_chunks: Vec<_> = (0..num_heads).into_par_iter().map(|h| {
            let mut head_weights = vec![0.0f32; seq_len * seq_len];
            
            for i in 0..seq_len {
                let row_start = h * seq_len * seq_len + i * seq_len;
                let row_end = row_start + seq_len;
                let row = &attention_scores[row_start..row_end];

                // Find max for numerical stability
                let max_score = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp and sum
                let mut sum = 0.0;
                let mut exp_values = vec![0.0f32; seq_len];
                for (j, &score) in row.iter().enumerate() {
                    exp_values[j] = (score - max_score).exp();
                    sum += exp_values[j];
                }

                // Normalize
                for (j, exp_val) in exp_values.iter().enumerate() {
                    head_weights[i * seq_len + j] = exp_val / sum;
                }
            }
            (h, head_weights)
        }).collect();

        // Copy results back to attention_weights
        for (h, head_weights) in weight_chunks {
            let offset = h * seq_len * seq_len;
            attention_weights[offset..offset + seq_len * seq_len].copy_from_slice(&head_weights);
        }

        // Apply attention weights to values in parallel
        let output_chunks: Vec<_> = (0..num_heads).into_par_iter().map(|h| {
            let kv_h = h / kv_repeat;
            let mut head_output = vec![0.0f32; seq_len * head_dim];

            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        let weight_idx = h * seq_len * seq_len + i * seq_len + j;
                        let v_idx = (j * num_kv_heads + kv_h) * head_dim + d;
                        sum += attention_weights[weight_idx] * value[v_idx];
                    }
                    head_output[i * head_dim + d] = sum;
                }
            }
            (h, head_output)
        }).collect();

        // Copy results back to output
        for (h, head_output) in output_chunks {
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let src_idx = i * head_dim + d;
                    let out_idx = (i * num_heads + h) * head_dim + d;
                    output[out_idx] = head_output[src_idx];
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

    /// Project input to query states using BLAS
    fn project_queries(&self, hidden_states: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        
        if let Some(ref weight) = self.q_proj.weight {
            let input_tensor = SimpleTensor {
                data: hidden_states.to_vec(),
                shape: Shape::matrix(seq_len, hidden_size),
            };
            
            if is_blas_available() {
                if let Some(result) = matmul_blas(&input_tensor, weight) {
                    eprintln!("✅ Q projection completed with BLAS acceleration");
                    return Ok(result.data);
                }
            }
            
            // Fallback to manual computation
            eprintln!("⚠️  Falling back to manual Q projection");
            let result = matmul(&input_tensor, weight)?;
            Ok(result.data)
        } else {
            Err(CoreError::model("Q_PROJ_NOT_LOADED", "Q projection weights not loaded", "", ""))
        }
    }
    
    /// Project input to key states using BLAS
    fn project_keys(&self, hidden_states: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        
        if let Some(ref weight) = self.k_proj.weight {
            let input_tensor = SimpleTensor {
                data: hidden_states.to_vec(),
                shape: Shape::matrix(seq_len, hidden_size),
            };
            
            if is_blas_available() {
                if let Some(result) = matmul_blas(&input_tensor, weight) {
                    eprintln!("✅ K projection completed with BLAS acceleration");
                    return Ok(result.data);
                }
            }
            
            // Fallback to manual computation
            eprintln!("⚠️  Falling back to manual K projection");
            let result = matmul(&input_tensor, weight)?;
            Ok(result.data)
        } else {
            Err(CoreError::model("K_PROJ_NOT_LOADED", "K projection weights not loaded", "", ""))
        }
    }
    
    /// Project input to value states using BLAS
    fn project_values(&self, hidden_states: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        
        if let Some(ref weight) = self.v_proj.weight {
            let input_tensor = SimpleTensor {
                data: hidden_states.to_vec(),
                shape: Shape::matrix(seq_len, hidden_size),
            };
            
            if is_blas_available() {
                if let Some(result) = matmul_blas(&input_tensor, weight) {
                    eprintln!("✅ V projection completed with BLAS acceleration");
                    return Ok(result.data);
                }
            }
            
            // Fallback to manual computation
            eprintln!("⚠️  Falling back to manual V projection");
            let result = matmul(&input_tensor, weight)?;
            Ok(result.data)
        } else {
            Err(CoreError::model("V_PROJ_NOT_LOADED", "V projection weights not loaded", "", ""))
        }
    }
    
    /// Project attention output using BLAS
    fn project_output(&self, attention_output: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        
        if let Some(ref weight) = self.o_proj.weight {
            let input_tensor = SimpleTensor {
                data: attention_output.to_vec(),
                shape: Shape::matrix(seq_len, hidden_size),
            };
            
            if is_blas_available() {
                if let Some(result) = matmul_blas(&input_tensor, weight) {
                    eprintln!("✅ Output projection completed with BLAS acceleration");
                    return Ok(result.data);
                }
            }
            
            // Fallback to manual computation
            eprintln!("⚠️  Falling back to manual output projection");
            let result = matmul(&input_tensor, weight)?;
            Ok(result.data)
        } else {
            Err(CoreError::model("O_PROJ_NOT_LOADED", "Output projection weights not loaded", "", ""))
        }
    }

    /// Load weights for the attention layer
    pub fn load_weights(&mut self, weights: &super::loader::LayerWeights) -> Result<()> {
        // Validate Q projection dimensions
        if weights.attn_q_shape.len() != 2 {
            return Err(CoreError::invalid_input(
                "INVALID_Q_WEIGHT_DIMS",
                format!("Q weight must be 2D, got {} dimensions", weights.attn_q_shape.len()),
                "attention weight loading",
                "Ensure Q weight is a 2D tensor"
            ));
        }
        let q_expected_shape = [self.config.hidden_size, self.config.hidden_size];
        if weights.attn_q_shape != q_expected_shape {
            return Err(CoreError::invalid_input(
                "Q_WEIGHT_SHAPE_MISMATCH",
                format!("Q weight shape mismatch: expected {:?}, got {:?}", 
                        q_expected_shape, weights.attn_q_shape),
                "attention weight loading",
                "Check Q weight dimensions"
            ));
        }
        
        // Validate K/V projection dimensions for GQA
        let kv_hidden_size = if let Some(num_kv_heads) = self.config.num_kv_heads {
            num_kv_heads * self.config.head_dim
        } else {
            self.config.hidden_size
        };
        
        if weights.attn_k_shape.len() != 2 {
            return Err(CoreError::invalid_input(
                "INVALID_K_WEIGHT_DIMS",
                format!("K weight must be 2D, got {} dimensions", weights.attn_k_shape.len()),
                "attention weight loading",
                "Ensure K weight is a 2D tensor"
            ));
        }
        let k_expected_shape = [self.config.hidden_size, kv_hidden_size];
        if weights.attn_k_shape != k_expected_shape {
            return Err(CoreError::invalid_input(
                "K_WEIGHT_SHAPE_MISMATCH",
                format!("K weight shape mismatch: expected {:?}, got {:?}", 
                        k_expected_shape, weights.attn_k_shape),
                "attention weight loading",
                "Check K weight dimensions for GQA"
            ));
        }
        
        if weights.attn_v_shape.len() != 2 {
            return Err(CoreError::invalid_input(
                "INVALID_V_WEIGHT_DIMS",
                format!("V weight must be 2D, got {} dimensions", weights.attn_v_shape.len()),
                "attention weight loading", 
                "Ensure V weight is a 2D tensor"
            ));
        }
        let v_expected_shape = [self.config.hidden_size, kv_hidden_size];
        if weights.attn_v_shape != v_expected_shape {
            return Err(CoreError::invalid_input(
                "V_WEIGHT_SHAPE_MISMATCH",
                format!("V weight shape mismatch: expected {:?}, got {:?}", 
                        v_expected_shape, weights.attn_v_shape),
                "attention weight loading",
                "Check V weight dimensions for GQA"
            ));
        }
        
        // Validate output projection dimensions
        if weights.attn_o_shape.len() != 2 {
            return Err(CoreError::invalid_input(
                "INVALID_O_WEIGHT_DIMS",
                format!("Output weight must be 2D, got {} dimensions", weights.attn_o_shape.len()),
                "attention weight loading",
                "Ensure output weight is a 2D tensor"
            ));
        }
        let o_expected_shape = [self.config.hidden_size, self.config.hidden_size];
        if weights.attn_o_shape != o_expected_shape {
            return Err(CoreError::invalid_input(
                "O_WEIGHT_SHAPE_MISMATCH",
                format!("Output weight shape mismatch: expected {:?}, got {:?}", 
                        o_expected_shape, weights.attn_o_shape),
                "attention weight loading",
                "Check output weight dimensions"
            ));
        }
        
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
            .ok_or_else(|| CoreError::tensor("TENSOR_ERROR", "Linear layer weights not loaded", "", "Check tensor operations"))?;
        
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
            return Err(CoreError::invalid_input(
                "INVALID_WEIGHT_SHAPE",
                &format!("Expected 2D weight shape, got {}D", shape.len()),
                "linear layer weight loading",
                "Provide 2D weight tensor"
            ));
        }

        let expected_out_features = shape[0];
        let expected_in_features = shape[1];
        
        if expected_out_features != self.out_features || expected_in_features != self.in_features {
            return Err(CoreError::invalid_input(
                "WEIGHT_SHAPE_MISMATCH",
                &format!("Weight shape mismatch: expected {}x{}, got {}x{}", 
                    self.out_features, self.in_features, expected_out_features, expected_in_features),
                "linear layer weight loading",
                "Check weight tensor dimensions"
            ));
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
            return Err(CoreError::invalid_input(
                "INVALID_WEIGHT_SHAPE",
                &format!("Expected 2D weight shape, got {}D", shape.len()),
                "linear layer weight loading",
                "Provide 2D weight tensor"
            ));
        }

        let expected_out_features = shape[0];
        let expected_in_features = shape[1];
        
        if expected_out_features != self.out_features {
            return Err(CoreError::invalid_input(
                "ATTENTION_OUTPUT_FEATURES_MISMATCH",
                format!("Output features mismatch: expected {}, got {}", 
                    self.out_features, expected_out_features),
                "Attention weight loading",
                "Ensure weight tensor has the correct output feature dimensions"
            ));
        }
        
        if expected_in_features != self.in_features {
            return Err(CoreError::invalid_input(
                "ATTENTION_INPUT_FEATURES_MISMATCH",
                format!("Input features mismatch: expected {}, got {}", 
                    self.in_features, expected_in_features),
                "Attention weight loading",
                "Ensure weight tensor has the correct input feature dimensions"
            ));
        }

        if weights.len() != self.out_features * self.in_features {
            return Err(CoreError::invalid_input(
                "ATTENTION_WEIGHT_SIZE_MISMATCH",
                format!("Weight size mismatch: expected {}, got {}", 
                    self.out_features * self.in_features, weights.len()),
                "Attention weight loading",
                "Ensure weight tensor size matches out_features * in_features"
            ));
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