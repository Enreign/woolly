//! Token and position embeddings for transformer models

use crate::{CoreError, Result};

/// Token embedding layer that converts token IDs to dense vectors
pub struct TokenEmbedding {
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Embedding weights [vocab_size, embed_dim]
    weights: Vec<f32>,
}

impl TokenEmbedding {
    /// Create a new token embedding layer
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        // Initialize with small random values
        let weights = vec![0.0; vocab_size * embed_dim];
        Self {
            vocab_size,
            embed_dim,
            weights,
        }
    }

    /// Initialize embeddings with xavier uniform initialization
    pub fn init_xavier(&mut self) {
        let scale = (6.0 / (self.vocab_size + self.embed_dim) as f32).sqrt();
        for weight in self.weights.iter_mut() {
            // Simple pseudo-random for now
            *weight = (rand::random::<f32>() * 2.0 - 1.0) * scale;
        }
    }

    /// Forward pass: convert token IDs to embeddings
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = input_ids.len();
        let mut output = vec![0.0; batch_size * self.embed_dim];

        for (i, &token_id) in input_ids.iter().enumerate() {
            if token_id as usize >= self.vocab_size {
                return Err(CoreError::InvalidInput(format!(
                    "Token ID {} exceeds vocabulary size {}",
                    token_id, self.vocab_size
                )));
            }

            let token_id = token_id as usize;
            let embed_start = token_id * self.embed_dim;
            let out_start = i * self.embed_dim;

            // Copy embedding vector
            output[out_start..out_start + self.embed_dim]
                .copy_from_slice(&self.weights[embed_start..embed_start + self.embed_dim]);
        }

        Ok(output)
    }

    /// Get embedding weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get mutable embedding weights
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Load weights from external data
    pub fn load_weights(&mut self, weights: &[f32], shape: &[usize]) -> Result<()> {
        // Validate shape
        if shape.len() != 2 {
            return Err(CoreError::InvalidInput(format!(
                "Expected 2D embedding shape, got {}D", shape.len()
            )));
        }

        let expected_vocab_size = shape[0];
        let expected_embed_dim = shape[1];
        
        if expected_vocab_size != self.vocab_size {
            return Err(CoreError::InvalidInput(format!(
                "Vocabulary size mismatch: expected {}, got {}", 
                self.vocab_size, expected_vocab_size
            )));
        }
        
        if expected_embed_dim != self.embed_dim {
            return Err(CoreError::InvalidInput(format!(
                "Embedding dimension mismatch: expected {}, got {}", 
                self.embed_dim, expected_embed_dim
            )));
        }

        if weights.len() != self.vocab_size * self.embed_dim {
            return Err(CoreError::InvalidInput(format!(
                "Weight size mismatch: expected {}, got {}", 
                self.vocab_size * self.embed_dim, weights.len()
            )));
        }

        // Copy weights
        self.weights.copy_from_slice(weights);
        
        Ok(())
    }
}

/// Rotary Position Embeddings (RoPE)
#[allow(dead_code)]
pub struct RotaryEmbedding {
    /// Maximum sequence length
    max_seq_len: usize,
    /// Embedding dimension (must be even)
    embed_dim: usize,
    /// Base frequency
    base: f32,
    /// Precomputed cos values [max_seq_len, embed_dim/2]
    cos_cached: Vec<f32>,
    /// Precomputed sin values [max_seq_len, embed_dim/2]
    sin_cached: Vec<f32>,
}

impl RotaryEmbedding {
    /// Create new rotary embeddings
    pub fn new(max_seq_len: usize, embed_dim: usize, base: f32) -> Result<Self> {
        if embed_dim % 2 != 0 {
            return Err(CoreError::InvalidInput(
                "Embedding dimension must be even for RoPE".to_string(),
            ));
        }

        let half_dim = embed_dim / 2;
        let mut cos_cached = vec![0.0; max_seq_len * half_dim];
        let mut sin_cached = vec![0.0; max_seq_len * half_dim];

        // Precompute frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / embed_dim as f32))
            .collect();

        // Precompute cos and sin values
        for pos in 0..max_seq_len {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                let idx = pos * half_dim + i;
                cos_cached[idx] = angle.cos();
                sin_cached[idx] = angle.sin();
            }
        }

        Ok(Self {
            max_seq_len,
            embed_dim,
            base,
            cos_cached,
            sin_cached,
        })
    }

    /// Apply rotary embeddings to query and key tensors
    /// Input shape: [batch_size * seq_len * num_heads * head_dim]
    pub fn apply_rotary_pos_emb(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        seq_len: usize,
        num_heads: usize,
    ) -> Result<()> {
        if seq_len > self.max_seq_len {
            return Err(CoreError::InvalidInput(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_seq_len
            )));
        }

        let head_dim = self.embed_dim;
        let half_dim = head_dim / 2;

        // Apply RoPE to queries
        for pos in 0..seq_len {
            for h in 0..num_heads {
                let offset = (pos * num_heads + h) * head_dim;
                
                for i in 0..half_dim {
                    let cos_idx = pos * half_dim + i;
                    let cos_val = self.cos_cached[cos_idx];
                    let sin_val = self.sin_cached[cos_idx];

                    let q1 = q[offset + i];
                    let q2 = q[offset + i + half_dim];

                    q[offset + i] = q1 * cos_val - q2 * sin_val;
                    q[offset + i + half_dim] = q1 * sin_val + q2 * cos_val;
                }
            }
        }

        // Apply RoPE to keys
        for pos in 0..seq_len {
            for h in 0..num_heads {
                let offset = (pos * num_heads + h) * head_dim;
                
                for i in 0..half_dim {
                    let cos_idx = pos * half_dim + i;
                    let cos_val = self.cos_cached[cos_idx];
                    let sin_val = self.sin_cached[cos_idx];

                    let k1 = k[offset + i];
                    let k2 = k[offset + i + half_dim];

                    k[offset + i] = k1 * cos_val - k2 * sin_val;
                    k[offset + i + half_dim] = k1 * sin_val + k2 * cos_val;
                }
            }
        }

        Ok(())
    }
}

/// Sinusoidal position embeddings (classic transformer style)
pub struct SinusoidalPositionEmbedding {
    /// Maximum sequence length
    max_seq_len: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Precomputed embeddings [max_seq_len, embed_dim]
    embeddings: Vec<f32>,
}

impl SinusoidalPositionEmbedding {
    /// Create new sinusoidal position embeddings
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        let mut embeddings = vec![0.0; max_seq_len * embed_dim];

        for pos in 0..max_seq_len {
            for i in 0..embed_dim {
                let angle = if i % 2 == 0 {
                    // Even indices: sin
                    (pos as f32 / 10000f32.powf((i / 2) as f32 * 2.0 / embed_dim as f32)).sin()
                } else {
                    // Odd indices: cos
                    (pos as f32 / 10000f32.powf(((i - 1) / 2) as f32 * 2.0 / embed_dim as f32)).cos()
                };
                embeddings[pos * embed_dim + i] = angle;
            }
        }

        Self {
            max_seq_len,
            embed_dim,
            embeddings,
        }
    }

    /// Get position embeddings for a sequence
    pub fn forward(&self, seq_len: usize) -> Result<Vec<f32>> {
        if seq_len > self.max_seq_len {
            return Err(CoreError::InvalidInput(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_seq_len
            )));
        }

        let output_size = seq_len * self.embed_dim;
        Ok(self.embeddings[..output_size].to_vec())
    }

    /// Add position embeddings to input embeddings
    pub fn add_to_embeddings(&self, embeddings: &mut [f32], seq_len: usize) -> Result<()> {
        if seq_len > self.max_seq_len {
            return Err(CoreError::InvalidInput(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_seq_len
            )));
        }

        if embeddings.len() != seq_len * self.embed_dim {
            return Err(CoreError::InvalidInput(
                "Embedding size mismatch".to_string(),
            ));
        }

        for (i, emb) in embeddings.iter_mut().enumerate() {
            *emb += self.embeddings[i];
        }

        Ok(())
    }
}

/// Learnable position embeddings
pub struct LearnedPositionEmbedding {
    /// Maximum sequence length
    max_seq_len: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Position embedding weights [max_seq_len, embed_dim]
    weights: Vec<f32>,
}

impl LearnedPositionEmbedding {
    /// Create new learned position embeddings
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        let weights = vec![0.0; max_seq_len * embed_dim];
        Self {
            max_seq_len,
            embed_dim,
            weights,
        }
    }

    /// Initialize with small random values
    pub fn init_random(&mut self) {
        let scale = (1.0 / self.embed_dim as f32).sqrt();
        for weight in self.weights.iter_mut() {
            *weight = (rand::random::<f32>() * 2.0 - 1.0) * scale;
        }
    }

    /// Get position embeddings for a sequence
    pub fn forward(&self, seq_len: usize, start_pos: usize) -> Result<Vec<f32>> {
        if start_pos + seq_len > self.max_seq_len {
            return Err(CoreError::InvalidInput(format!(
                "Position range {}-{} exceeds maximum {}",
                start_pos,
                start_pos + seq_len,
                self.max_seq_len
            )));
        }

        let start_idx = start_pos * self.embed_dim;
        let end_idx = (start_pos + seq_len) * self.embed_dim;
        Ok(self.weights[start_idx..end_idx].to_vec())
    }

    /// Get weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get mutable weights
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding() {
        let mut embed = TokenEmbedding::new(100, 32);
        embed.init_xavier();

        let tokens = vec![1, 5, 10];
        let output = embed.forward(&tokens).unwrap();
        assert_eq!(output.len(), tokens.len() * 32);
    }

    #[test]
    fn test_rotary_embedding() {
        let rope = RotaryEmbedding::new(128, 64, 10000.0).unwrap();
        
        let mut q = vec![1.0; 10 * 8 * 64]; // seq_len=10, num_heads=8, head_dim=64
        let mut k = vec![1.0; 10 * 8 * 64];
        
        rope.apply_rotary_pos_emb(&mut q, &mut k, 10, 8).unwrap();
        
        // Verify that some values changed after applying rotary embedding
        let q_changed = q.iter().any(|&val| val != 1.0);
        let k_changed = k.iter().any(|&val| val != 1.0);
        assert!(q_changed, "Query values should change after applying rotary embedding");
        assert!(k_changed, "Key values should change after applying rotary embedding");
        
        // Test with different input values to ensure proper transformation
        let mut q2 = vec![0.5; 10 * 8 * 64];
        let mut k2 = vec![2.0; 10 * 8 * 64];
        
        rope.apply_rotary_pos_emb(&mut q2, &mut k2, 10, 8).unwrap();
        
        let q2_changed = q2.iter().any(|&val| val != 0.5);
        let k2_changed = k2.iter().any(|&val| val != 2.0);
        assert!(q2_changed, "Query values should change with different input values");
        assert!(k2_changed, "Key values should change with different input values");
    }

    #[test]
    fn test_sinusoidal_position_embedding() {
        let pos_embed = SinusoidalPositionEmbedding::new(128, 64);
        
        let output = pos_embed.forward(10).unwrap();
        assert_eq!(output.len(), 10 * 64);
        
        // Check that embeddings are bounded
        for &val in &output {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
}