//! Feed-forward network implementations for transformer models

use crate::{CoreError, Result};

/// Feed-forward network configuration
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    /// Input/output dimension (model hidden size)
    pub hidden_size: usize,
    /// Intermediate dimension
    pub intermediate_size: usize,
    /// Activation function type
    pub activation: ActivationType,
    /// Dropout probability
    pub dropout: f32,
    /// Whether to use bias in linear layers
    pub use_bias: bool,
    /// Whether to use gated linear units
    pub use_glu: bool,
}

impl FeedForwardConfig {
    /// Create a new feed-forward configuration
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            activation: ActivationType::ReLU,
            dropout: 0.0,
            use_bias: true,
            use_glu: false,
        }
    }

    /// Set the activation function
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Enable gated linear units
    pub fn with_glu(mut self) -> Self {
        self.use_glu = true;
        self
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU, // Also known as Swish
    GeGLU,
    SwiGLU,
}

/// Standard feed-forward network (MLP)
pub struct FeedForward {
    config: FeedForwardConfig,
    /// First linear layer [hidden_size, intermediate_size]
    w1: Linear,
    /// Second linear layer [intermediate_size, hidden_size]
    w2: Linear,
    /// Gate linear layer for GLU variants [hidden_size, intermediate_size]
    w_gate: Option<Linear>,
}

impl FeedForward {
    /// Create a new feed-forward network
    pub fn new(config: FeedForwardConfig) -> Self {
        let w_gate = if config.use_glu {
            Some(Linear::new(
                config.hidden_size,
                config.intermediate_size,
                config.use_bias,
            ))
        } else {
            None
        };

        Self {
            w1: Linear::new(config.hidden_size, config.intermediate_size, config.use_bias),
            w2: Linear::new(config.intermediate_size, config.hidden_size, config.use_bias),
            w_gate,
            config,
        }
    }

    /// Forward pass through the feed-forward network
    /// Input shape: [batch_size * seq_len, hidden_size] (flattened)
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        // First projection
        let mut intermediate = self.w1.forward(hidden_states)?;

        // Apply activation and gating if enabled
        if self.config.use_glu {
            match self.config.activation {
                ActivationType::GeGLU => {
                    let gate = self.w_gate.as_ref().unwrap().forward(hidden_states)?;
                    self.apply_geglu(&mut intermediate, &gate);
                }
                ActivationType::SwiGLU => {
                    let gate = self.w_gate.as_ref().unwrap().forward(hidden_states)?;
                    self.apply_swiglu(&mut intermediate, &gate);
                }
                _ => {
                    // Standard GLU with specified activation
                    let gate = self.w_gate.as_ref().unwrap().forward(hidden_states)?;
                    self.apply_activation(&mut intermediate, self.config.activation);
                    self.apply_glu(&mut intermediate, &gate);
                }
            }
        } else {
            // Standard activation without gating
            self.apply_activation(&mut intermediate, self.config.activation);
        }

        // Apply dropout if enabled (placeholder - would need RNG)
        if self.config.dropout > 0.0 {
            // In real implementation, apply dropout here
        }

        // Second projection
        self.w2.forward(&intermediate)
    }

    /// Apply activation function in-place
    fn apply_activation(&self, tensor: &mut [f32], activation: ActivationType) {
        match activation {
            ActivationType::ReLU => {
                for x in tensor.iter_mut() {
                    *x = x.max(0.0);
                }
            }
            ActivationType::GELU => {
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                const SQRT_2_OVER_PI: f32 = 0.797_884_56;
                for x in tensor.iter_mut() {
                    let x_val = *x;
                    let tanh_arg = SQRT_2_OVER_PI * (x_val + 0.044715 * x_val * x_val * x_val);
                    *x = 0.5 * x_val * (1.0 + tanh_arg.tanh());
                }
            }
            ActivationType::SiLU => {
                // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                for x in tensor.iter_mut() {
                    *x = *x / (1.0 + (-*x).exp());
                }
            }
            _ => {} // GeGLU and SwiGLU are handled separately
        }
    }

    /// Apply gated linear unit
    fn apply_glu(&self, tensor: &mut [f32], gate: &[f32]) {
        for (x, &g) in tensor.iter_mut().zip(gate.iter()) {
            *x *= g;
        }
    }

    /// Apply GeGLU (GELU-gated linear unit)
    fn apply_geglu(&self, tensor: &mut [f32], gate: &[f32]) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_56;
        for (x, &g) in tensor.iter_mut().zip(gate.iter()) {
            let tanh_arg = SQRT_2_OVER_PI * (g + 0.044715 * g * g * g);
            let gelu_g = 0.5 * g * (1.0 + tanh_arg.tanh());
            *x *= gelu_g;
        }
    }

    /// Apply SwiGLU (Swish-gated linear unit)
    fn apply_swiglu(&self, tensor: &mut [f32], gate: &[f32]) {
        for (x, &g) in tensor.iter_mut().zip(gate.iter()) {
            let swish_g = g / (1.0 + (-g).exp());
            *x *= swish_g;
        }
    }

    /// Load weights for the feed-forward layer
    pub fn load_weights(&mut self, weights: &super::loader::LayerWeights) -> Result<()> {
        // Load up projection weights (w1)
        self.w1.load_weights(&weights.ffn_up_weight, &weights.ffn_up_shape)?;
        
        // Load down projection weights (w2)
        self.w2.load_weights(&weights.ffn_down_weight, &weights.ffn_down_shape)?;
        
        // Load gate weights if present (for GLU variants)
        if let Some(ref mut w_gate) = self.w_gate {
            if let (Some(ref gate_weights), Some(ref gate_shape)) = 
                (&weights.ffn_gate_weight, &weights.ffn_gate_shape) {
                w_gate.load_weights(gate_weights, gate_shape)?;
            } else {
                return Err(CoreError::Model(
                    "Gate weights expected but not found in loaded weights".to_string()
                ));
            }
        }
        
        Ok(())
    }
}

/// Linear layer implementation
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

        // Initialize weights with Kaiming initialization
        layer.init_weights();
        layer
    }

    fn init_weights(&mut self) {
        // Kaiming uniform initialization
        let fan_in = self.in_features as f32;
        let bound = (3.0 / fan_in).sqrt();
        
        for w in self.weight.iter_mut() {
            *w = (rand::random::<f32>() * 2.0 - 1.0) * bound;
        }
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(CoreError::InvalidInput(
                "Input size must be divisible by in_features".to_string(),
            ));
        }

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

    /// Get weight parameters
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Get mutable weight parameters
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// Get bias parameters
    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Get mutable bias parameters
    pub fn bias_mut(&mut self) -> Option<&mut [f32]> {
        self.bias.as_deref_mut()
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

/// Expert feed-forward network for Mixture of Experts (MoE)
pub struct ExpertFeedForward {
    /// Number of experts
    num_experts: usize,
    /// Individual expert networks
    experts: Vec<FeedForward>,
    /// Router network to select experts
    router: Linear,
}

impl ExpertFeedForward {
    /// Create a new mixture of experts feed-forward network
    pub fn new(config: FeedForwardConfig, num_experts: usize) -> Self {
        let experts = (0..num_experts)
            .map(|_| FeedForward::new(config.clone()))
            .collect();

        let router = Linear::new(config.hidden_size, num_experts, false);

        Self {
            num_experts,
            experts,
            router,
        }
    }

    /// Forward pass with top-k expert selection
    pub fn forward(&self, hidden_states: &[f32], _top_k: usize) -> Result<Vec<f32>> {
        let batch_size = hidden_states.len() / self.experts[0].config.hidden_size;
        
        // Get router logits
        let router_logits = self.router.forward(hidden_states)?;
        
        // For simplicity, we'll just use a weighted average of all experts
        // In a real implementation, you'd select top-k experts per token
        let mut output = vec![0.0; hidden_states.len()];
        
        // Compute softmax weights for each expert
        for b in 0..batch_size {
            let logits_start = b * self.num_experts;
            let logits_end = logits_start + self.num_experts;
            let logits = &router_logits[logits_start..logits_end];
            
            // Compute softmax
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            let mut exp_values = vec![0.0; self.num_experts];
            
            for (i, &logit) in logits.iter().enumerate() {
                exp_values[i] = (logit - max_logit).exp();
                exp_sum += exp_values[i];
            }
            
            // Normalize to get weights
            for exp_val in exp_values.iter_mut() {
                *exp_val /= exp_sum;
            }
            
            // Apply each expert weighted by router output
            let hidden_start = b * self.experts[0].config.hidden_size;
            let hidden_end = hidden_start + self.experts[0].config.hidden_size;
            let token_hidden = &hidden_states[hidden_start..hidden_end];
            
            for (expert_idx, expert) in self.experts.iter().enumerate() {
                let expert_output = expert.forward(token_hidden)?;
                let weight = exp_values[expert_idx];
                
                for (i, &val) in expert_output.iter().enumerate() {
                    output[hidden_start + i] += weight * val;
                }
            }
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward() {
        let config = FeedForwardConfig::new(64, 256);
        let ff = FeedForward::new(config);

        let input = vec![0.1; 10 * 64]; // batch_size * seq_len = 10, hidden_size = 64
        let output = ff.forward(&input).unwrap();
        
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_glu_feedforward() {
        let config = FeedForwardConfig::new(64, 256)
            .with_activation(ActivationType::SwiGLU)
            .with_glu();
        let ff = FeedForward::new(config);

        let input = vec![0.1; 10 * 64];
        let output = ff.forward(&input).unwrap();
        
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_expert_feedforward() {
        let config = FeedForwardConfig::new(64, 256);
        let moe = ExpertFeedForward::new(config, 4);

        let input = vec![0.1; 10 * 64];
        let output = moe.forward(&input, 2).unwrap();
        
        assert_eq!(output.len(), input.len());
    }
}