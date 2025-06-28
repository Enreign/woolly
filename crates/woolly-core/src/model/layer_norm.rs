//! Layer normalization implementations for transformer models

use crate::Result;

/// Standard Layer Normalization
pub struct LayerNorm {
    /// Normalized dimension
    normalized_shape: usize,
    /// Small epsilon value for numerical stability
    epsilon: f32,
    /// Learned scale parameter (gamma)
    weight: Vec<f32>,
    /// Learned shift parameter (beta)
    bias: Vec<f32>,
    /// Whether to use bias
    use_bias: bool,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    pub fn new(normalized_shape: usize, epsilon: f32, use_bias: bool) -> Self {
        Self {
            normalized_shape,
            epsilon,
            weight: vec![1.0; normalized_shape],
            bias: vec![0.0; normalized_shape],
            use_bias,
        }
    }

    /// Forward pass through layer normalization
    /// Input: [batch_size * seq_len, hidden_size] (flattened)
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let total_elements = input.len();
        if total_elements % self.normalized_shape != 0 {
            return Err(crate::CoreError::invalid_input(
                "LAYERNORM_INPUT_SIZE_MISMATCH",
                "Input size must be divisible by normalized_shape",
                "layer normalization forward pass",
                "Check input tensor dimensions"
            ));
        }

        let batch_size = total_elements / self.normalized_shape;
        let mut output = vec![0.0; total_elements];

        // Process each sample in the batch
        for b in 0..batch_size {
            let start = b * self.normalized_shape;
            let end = start + self.normalized_shape;
            let sample = &input[start..end];

            // Compute mean
            let mean: f32 = sample.iter().sum::<f32>() / self.normalized_shape as f32;

            // Compute variance
            let variance: f32 = sample
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.normalized_shape as f32;

            // Normalize and apply affine transform
            let std_inv = 1.0 / (variance + self.epsilon).sqrt();
            
            for i in 0..self.normalized_shape {
                let normalized = (sample[i] - mean) * std_inv;
                output[start + i] = normalized * self.weight[i];
                if self.use_bias {
                    output[start + i] += self.bias[i];
                }
            }
        }

        Ok(output)
    }

    /// Get weight parameters
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Get mutable weight parameters
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// Get bias parameters
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Get mutable bias parameters
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }

    /// Load weights from external data
    pub fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        if weights.len() != self.normalized_shape {
            return Err(crate::CoreError::invalid_input(
                "LAYER_NORM_WEIGHT_SIZE_MISMATCH",
                format!("Weight size mismatch: expected {}, got {}", 
                    self.normalized_shape, weights.len()),
                "Layer normalization weight loading",
                "Ensure weight tensor has the correct dimensions"
            ));
        }

        self.weight.copy_from_slice(weights);
        Ok(())
    }
}

/// RMS Normalization (Root Mean Square Layer Normalization)
/// Used in models like LLaMA for better efficiency
pub struct RMSNorm {
    /// Normalized dimension
    normalized_shape: usize,
    /// Small epsilon value for numerical stability
    epsilon: f32,
    /// Learned scale parameter
    weight: Vec<f32>,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    pub fn new(normalized_shape: usize, epsilon: f32) -> Self {
        Self {
            normalized_shape,
            epsilon,
            weight: vec![1.0; normalized_shape],
        }
    }

    /// Forward pass through RMS normalization
    /// Input: [batch_size * seq_len, hidden_size] (flattened)
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let total_elements = input.len();
        if total_elements % self.normalized_shape != 0 {
            return Err(crate::CoreError::invalid_input(
                "LAYERNORM_INPUT_SIZE_MISMATCH",
                "Input size must be divisible by normalized_shape",
                "layer normalization forward pass",
                "Check input tensor dimensions"
            ));
        }

        let batch_size = total_elements / self.normalized_shape;
        let mut output = vec![0.0; total_elements];

        // Process each sample in the batch
        for b in 0..batch_size {
            let start = b * self.normalized_shape;
            let end = start + self.normalized_shape;
            let sample = &input[start..end];

            // Compute RMS
            let sum_squares: f32 = sample.iter().map(|&x| x * x).sum();
            let rms = (sum_squares / self.normalized_shape as f32 + self.epsilon).sqrt();
            let scale = 1.0 / rms;

            // Normalize and apply weight
            for i in 0..self.normalized_shape {
                output[start + i] = sample[i] * scale * self.weight[i];
            }
        }

        Ok(output)
    }

    /// Get weight parameters
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Get mutable weight parameters
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// Load weights from external data
    pub fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        if weights.len() != self.normalized_shape {
            return Err(crate::CoreError::invalid_input(
                "LAYER_NORM_WEIGHT_SIZE_MISMATCH",
                format!("Weight size mismatch: expected {}, got {}", 
                    self.normalized_shape, weights.len()),
                "Layer normalization weight loading",
                "Ensure weight tensor has the correct dimensions"
            ));
        }

        self.weight.copy_from_slice(weights);
        Ok(())
    }
}

/// Group Normalization
/// Divides channels into groups and normalizes within each group
pub struct GroupNorm {
    /// Number of groups
    num_groups: usize,
    /// Number of channels
    num_channels: usize,
    /// Small epsilon value for numerical stability
    epsilon: f32,
    /// Learned scale parameter
    weight: Vec<f32>,
    /// Learned shift parameter
    bias: Vec<f32>,
    /// Whether to use bias
    use_bias: bool,
}

impl GroupNorm {
    /// Create a new GroupNorm layer
    pub fn new(num_groups: usize, num_channels: usize, epsilon: f32, use_bias: bool) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(crate::CoreError::invalid_input(
                "GROUPNORM_CHANNELS_GROUPS_MISMATCH",
                "Number of channels must be divisible by number of groups",
                "group normalization initialization",
                "Ensure num_channels is divisible by num_groups"
            ));
        }

        Ok(Self {
            num_groups,
            num_channels,
            epsilon,
            weight: vec![1.0; num_channels],
            bias: vec![0.0; num_channels],
            use_bias,
        })
    }

    /// Forward pass through group normalization
    /// Input shape: [batch_size, num_channels, *] (flattened)
    pub fn forward(&self, input: &[f32], batch_size: usize, spatial_size: usize) -> Result<Vec<f32>> {
        let expected_size = batch_size * self.num_channels * spatial_size;
        if input.len() != expected_size {
            return Err(crate::CoreError::invalid_input(
                "GROUP_NORM_INPUT_SIZE_MISMATCH",
                format!("Expected input size {}, got {}", expected_size, input.len()),
                "Group normalization forward pass",
                "Ensure input tensor dimensions match expected batch size and spatial dimensions"
            ));
        }

        let channels_per_group = self.num_channels / self.num_groups;
        let group_size = channels_per_group * spatial_size;
        let mut output = vec![0.0; input.len()];

        for b in 0..batch_size {
            for g in 0..self.num_groups {
                // Calculate mean and variance for this group
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * self.num_channels * spatial_size + channel_idx * spatial_size + s;
                        let val = input[idx];
                        sum += val;
                        sum_sq += val * val;
                    }
                }

                let mean = sum / group_size as f32;
                let variance = sum_sq / group_size as f32 - mean * mean;
                let std_inv = 1.0 / (variance + self.epsilon).sqrt();

                // Normalize and apply affine transform
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = b * self.num_channels * spatial_size + channel_idx * spatial_size + s;
                        let normalized = (input[idx] - mean) * std_inv;
                        output[idx] = normalized * self.weight[channel_idx];
                        if self.use_bias {
                            output[idx] += self.bias[channel_idx];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get weight parameters
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Get mutable weight parameters
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// Get bias parameters
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Get mutable bias parameters
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(10, 1e-5, true);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let output = ln.forward(&input).unwrap();
        
        // Check output has same length
        assert_eq!(output.len(), input.len());
        
        // Check that output is normalized (mean ≈ 0, std ≈ 1)
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        assert!((mean).abs() < 0.1);
    }

    #[test]
    fn test_rms_norm() {
        let rms = RMSNorm::new(10, 1e-5);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let output = rms.forward(&input).unwrap();
        
        assert_eq!(output.len(), input.len());
        
        // Check that output maintains relative magnitudes
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    fn test_group_norm() {
        let gn = GroupNorm::new(2, 4, 1e-5, true).unwrap();
        let input = vec![1.0; 4 * 5]; // batch_size=1, channels=4, spatial=5
        let output = gn.forward(&input, 1, 5).unwrap();
        
        assert_eq!(output.len(), input.len());
    }
}