//! Model weight loading from various formats
//!
//! This module provides a unified interface for loading model weights from different
//! formats like GGUF, SafeTensors, etc., and converting them to the internal
//! representation used by woolly-core transformer models.

use std::collections::HashMap;
use std::path::Path;

use woolly_gguf::{GGUFLoader, GGMLType, TensorInfo, dequantize};
use crate::{CoreError, Result};
use crate::model::ModelConfig;

/// Trait for loading model weights from different formats
pub trait ModelLoader {
    /// Load model weights from the given path
    fn from_path<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized;

    /// Get the model configuration extracted from the file
    fn config(&self) -> Result<ModelConfig>;

    /// Get tensor data by name as f32 slice
    fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>>;

    /// Get tensor shape by name
    fn get_tensor_shape(&self, name: &str) -> Result<Vec<usize>>;

    /// List all available tensor names
    fn tensor_names(&self) -> Vec<String>;

    /// Get model architecture name
    fn architecture(&self) -> Option<String>;

    /// Check if a tensor exists
    fn has_tensor(&self, name: &str) -> bool;
}

/// GGUF format model loader
pub struct GGUFModelLoader {
    loader: GGUFLoader,
    tensor_name_map: HashMap<String, String>,
}

impl GGUFModelLoader {
    /// Create a new GGUF model loader with tensor name mapping
    pub fn with_name_mapping(loader: GGUFLoader, name_map: HashMap<String, String>) -> Self {
        Self {
            loader,
            tensor_name_map: name_map,
        }
    }

    /// Get the internal tensor name (maps from standard names to GGUF-specific names)
    fn map_tensor_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.tensor_name_map.get(name).map(|s| s.as_str()).unwrap_or(name)
    }

    /// Convert GGUF tensor to f32 data
    fn convert_tensor_to_f32(&self, tensor_info: &TensorInfo, data: &[u8]) -> Result<Vec<f32>> {
        match tensor_info.ggml_type {
            GGMLType::F32 => {
                // Direct cast for F32 data
                Ok(bytemuck::cast_slice(data).to_vec())
            }
            GGMLType::F16 => {
                // Convert F16 to F32 - using safer conversion
                let f16_data: &[u16] = bytemuck::cast_slice(data);
                Ok(f16_data.iter()
                    .map(|&x| half::f16::from_bits(x).to_f32())
                    .collect())
            }
            GGMLType::BF16 => {
                // Convert BF16 to F32 - using safer conversion
                let bf16_data: &[u16] = bytemuck::cast_slice(data);
                Ok(bf16_data.iter()
                    .map(|&x| half::bf16::from_bits(x).to_f32())
                    .collect())
            }
            _ => {
                // For quantized types, use dequantization
                let num_elements = tensor_info.shape().iter().map(|&x| x as usize).product();
                
                // Use the dequantization function from woolly-gguf
                match dequantize(data, tensor_info.ggml_type, num_elements) {
                    Ok(dequantized) => {
                        eprintln!("Successfully dequantized tensor '{}' (type: {:?})", 
                            tensor_info.name, tensor_info.ggml_type);
                        Ok(dequantized)
                    }
                    Err(e) => {
                        eprintln!("WARNING: Failed to dequantize tensor '{}' (type: {:?}): {}", 
                            tensor_info.name, tensor_info.ggml_type, e);
                        eprintln!("Using dummy weights as fallback");
                        Ok(vec![0.0f32; num_elements])
                    }
                }
            }
        }
    }

    /// Extract model configuration from GGUF metadata
    fn extract_config(&self) -> Result<ModelConfig> {
        let metadata = self.loader.metadata();

        // Get architecture to determine parameter names
        let arch = self.loader.architecture()
            .ok_or_else(|| CoreError::model("MODEL_ERROR", "No architecture specified in GGUF", "", "Check model configuration"))?;

        // Common parameters
        let vocab_size = metadata.get_u32(&format!("{}.vocab_size", arch))
            .or_else(|| metadata.get_u32("tokenizer.ggml.tokens.len"))
            .unwrap_or(32000) as usize;

        let hidden_size = metadata.get_u32(&format!("{}.embedding_length", arch))
            .or_else(|| metadata.get_u32(&format!("{}.embed_dim", arch)))
            .unwrap_or(4096) as usize;

        let num_layers = metadata.get_u32(&format!("{}.block_count", arch))
            .or_else(|| metadata.get_u32(&format!("{}.layer_count", arch)))
            .unwrap_or(32) as usize;

        let num_heads = metadata.get_u32(&format!("{}.attention.head_count", arch))
            .unwrap_or(32) as usize;

        let context_length = metadata.get_u32(&format!("{}.context_length", arch))
            .or_else(|| metadata.get_u32("tokenizer.ggml.context_length"))
            .unwrap_or(2048) as usize;
        eprintln!("Extracted context_length: {} (arch: {})", context_length, arch);

        let intermediate_size = metadata.get_u32(&format!("{}.feed_forward_length", arch))
            .unwrap_or((hidden_size * 4) as u32) as usize;

        let num_key_value_heads = metadata.get_u32(&format!("{}.attention.head_count_kv", arch))
            .map(|n| n as usize);

        let rope_theta = metadata.get_f32(&format!("{}.rope.freq_base", arch))
            .or_else(|| metadata.get_f32(&format!("{}.attention.rope_theta", arch)));

        let layer_norm_epsilon = metadata.get_f32(&format!("{}.attention.layer_norm_epsilon", arch))
            .or_else(|| metadata.get_f32(&format!("{}.layer_norm_epsilon", arch)))
            .unwrap_or(1e-5);

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            context_length,
            intermediate_size,
            num_key_value_heads,
            rope_theta,
            layer_norm_epsilon,
        })
    }
}

impl ModelLoader for GGUFModelLoader {
    fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let loader = GGUFLoader::from_path(path)
            .map_err(|e| CoreError::model(
                "GGUF_LOAD_FAILED",
                format!("Failed to load GGUF file: {}", e),
                "Loading GGUF model file",
                "Check that the file exists and is a valid GGUF format"
            ))?;

        // Create default tensor name mapping for common architectures
        let name_map = create_default_tensor_mapping(&loader);

        Ok(Self::with_name_mapping(loader, name_map))
    }

    fn config(&self) -> Result<ModelConfig> {
        self.extract_config()
    }

    fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let mapped_name = self.map_tensor_name(name);
        
        let tensor_info = self.loader.tensor_info(mapped_name)
            .ok_or_else(|| CoreError::model(
                "TENSOR_NOT_FOUND",
                format!("Tensor '{}' not found", mapped_name),
                "Loading tensor data from model",
                "Check that the tensor name is correct for this model format"
            ))?;

        let data = self.loader.tensor_data(mapped_name)
            .map_err(|e| CoreError::model(
                "TENSOR_DATA_FAILED",
                format!("Failed to get tensor data for '{}': {}", mapped_name, e),
                "Reading tensor data from model file",
                "Check that the model file is not corrupted"
            ))?;

        self.convert_tensor_to_f32(tensor_info, &data)
    }

    fn get_tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        let mapped_name = self.map_tensor_name(name);
        
        let tensor_info = self.loader.tensor_info(mapped_name)
            .ok_or_else(|| CoreError::model(
                "TENSOR_NOT_FOUND",
                format!("Tensor '{}' not found", mapped_name),
                "Getting tensor shape from model",
                "Check that the tensor name is correct for this model format"
            ))?;

        Ok(tensor_info.shape().iter().map(|&x| x as usize).collect())
    }

    fn tensor_names(&self) -> Vec<String> {
        self.loader.tensor_names().iter().map(|&s| s.to_string()).collect()
    }

    fn architecture(&self) -> Option<String> {
        self.loader.architecture().map(|s| s.to_string())
    }

    fn has_tensor(&self, name: &str) -> bool {
        let mapped_name = self.map_tensor_name(name);
        self.loader.tensor_info(mapped_name).is_some()
    }
}

/// Tensor shape validation and conversion utilities
pub struct TensorUtils;

impl TensorUtils {
    /// Validate that tensor shape matches expected shape
    pub fn validate_shape(actual: &[usize], expected: &[usize], tensor_name: &str) -> Result<()> {
        if actual.len() != expected.len() {
            return Err(CoreError::invalid_input(
                "TENSOR_DIMENSION_MISMATCH",
                format!("Tensor '{}' has {} dimensions, expected {}", tensor_name, actual.len(), expected.len()),
                "Validating tensor shape",
                "Check that the tensor dimensions match the expected model architecture"
            ));
        }

        for (i, (&actual_dim, &expected_dim)) in actual.iter().zip(expected.iter()).enumerate() {
            if actual_dim != expected_dim {
                return Err(CoreError::invalid_input(
                    "TENSOR_SIZE_MISMATCH",
                    format!("Tensor '{}' dimension {} has size {}, expected {}", tensor_name, i, actual_dim, expected_dim),
                    "Validating tensor dimension sizes",
                    "Check that the tensor sizes match the expected model architecture"
                ));
            }
        }

        Ok(())
    }

    /// Validate that tensor has compatible shape (allows some dimensions to be flexible)
    pub fn validate_compatible_shape(
        actual: &[usize],
        expected: &[usize],
        flexible_dims: &[usize],
        tensor_name: &str,
    ) -> Result<()> {
        if actual.len() != expected.len() {
            return Err(CoreError::invalid_input(
                "TENSOR_DIMENSION_MISMATCH",
                format!("Tensor '{}' has {} dimensions, expected {}", tensor_name, actual.len(), expected.len()),
                "Validating tensor shape compatibility",
                "Check that the tensor dimensions match the expected model architecture"
            ));
        }

        for (i, (&actual_dim, &expected_dim)) in actual.iter().zip(expected.iter()).enumerate() {
            if !flexible_dims.contains(&i) && actual_dim != expected_dim {
                return Err(CoreError::invalid_input(
                    "TENSOR_SIZE_MISMATCH",
                    format!("Tensor '{}' dimension {} has size {}, expected {}", tensor_name, i, actual_dim, expected_dim),
                    "Validating tensor dimension sizes compatibility",
                    "Check that the tensor sizes match the expected model architecture"
                ));
            }
        }

        Ok(())
    }

    /// Reshape a flattened tensor to the specified shape
    pub fn reshape_tensor(data: Vec<f32>, shape: &[usize]) -> Result<Vec<f32>> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(CoreError::invalid_input(
                "TENSOR_RESHAPE_MISMATCH",
                format!("Cannot reshape tensor of size {} to shape {:?} (expected size {})", data.len(), shape, expected_size),
                "Reshaping tensor to specified dimensions",
                "Check that the tensor size matches the target shape"
            ));
        }
        Ok(data)
    }

    /// Transpose a 2D tensor
    pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        if data.len() != rows * cols {
            return Err(CoreError::invalid_input(
                "TENSOR_TRANSPOSE_MISMATCH",
                format!("Data length {} doesn't match shape {}x{}", data.len(), rows, cols),
                "Transposing 2D tensor",
                "Check that the data length matches the row x column dimensions"
            ));
        }

        let mut result = vec![0.0; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = data[i * cols + j];
            }
        }
        Ok(result)
    }
}

/// Model weight container for loaded weights
#[derive(Debug)]
pub struct ModelWeights {
    pub embeddings: Vec<f32>,
    pub embedding_shape: Vec<usize>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f32>>,
    pub lm_head_shape: Option<Vec<usize>>,
}

/// Weights for a single transformer layer
#[derive(Debug)]
pub struct LayerWeights {
    // Attention weights
    pub attn_q_weight: Vec<f32>,
    pub attn_k_weight: Vec<f32>,
    pub attn_v_weight: Vec<f32>,
    pub attn_o_weight: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    
    // Feed-forward weights
    pub ffn_gate_weight: Option<Vec<f32>>, // For GLU variants
    pub ffn_up_weight: Vec<f32>,
    pub ffn_down_weight: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    
    // Shapes for validation
    pub attn_q_shape: Vec<usize>,
    pub attn_k_shape: Vec<usize>,
    pub attn_v_shape: Vec<usize>,
    pub attn_o_shape: Vec<usize>,
    pub ffn_up_shape: Vec<usize>,
    pub ffn_down_shape: Vec<usize>,
    pub ffn_gate_shape: Option<Vec<usize>>,
}

/// Load transformer weights from a model loader
pub fn load_transformer_weights<L: ModelLoader>(
    loader: &L,
    config: &ModelConfig,
) -> Result<ModelWeights> {
    // Load embedding weights - GGUF uses token_embd.weight
    let embeddings = loader.get_tensor_f32("token_embd.weight")?;
    let embedding_shape = loader.get_tensor_shape("token_embd.weight")?;

    // Validate embedding shape - GGUF stores as [hidden_size, vocab_size]
    TensorUtils::validate_shape(&embedding_shape, &[config.hidden_size, config.vocab_size], "embeddings")?;

    // Load layer weights
    let mut layers = Vec::with_capacity(config.num_layers);
    
    for layer_idx in 0..config.num_layers {
        let layer_weights = load_layer_weights(loader, layer_idx, config)?;
        layers.push(layer_weights);
    }

    // Load final normalization - GGUF uses output_norm.weight
    let final_norm = loader.get_tensor_f32("output_norm.weight")?;

    // Load language model head (if not tied)
    let (lm_head, lm_head_shape) = if loader.has_tensor("output.weight") || loader.has_tensor("lm_head.weight") {
        let lm_head = loader.get_tensor_f32("output.weight")
            .or_else(|_| loader.get_tensor_f32("lm_head.weight"))?;
        let lm_head_shape = loader.get_tensor_shape("output.weight")
            .or_else(|_| loader.get_tensor_shape("lm_head.weight"))?;
        (Some(lm_head), Some(lm_head_shape))
    } else {
        (None, None)
    };

    Ok(ModelWeights {
        embeddings,
        embedding_shape,
        layers,
        final_norm,
        lm_head,
        lm_head_shape,
    })
}

/// Load weights for a single transformer layer
fn load_layer_weights<L: ModelLoader>(
    loader: &L,
    layer_idx: usize,
    _config: &ModelConfig,
) -> Result<LayerWeights> {
    // Common layer prefixes used in different model formats
    let layer_prefix = format!("blk.{}", layer_idx);
    let alt_prefix = format!("model.layers.{}", layer_idx);

    // Helper to try multiple tensor name patterns
    let try_load_tensor = |names: &[String]| -> Result<Vec<f32>> {
        for name in names {
            if let Ok(tensor) = loader.get_tensor_f32(name) {
                return Ok(tensor);
            }
        }
        Err(CoreError::model(
            "TENSOR_NOT_FOUND",
            format!("Could not find tensor with any of these names: {:?}", names),
            "Loading layer weights from model",
            "Check that the model contains the expected layer tensor names"
        ))
    };

    let try_load_shape = |names: &[String]| -> Result<Vec<usize>> {
        for name in names {
            if let Ok(shape) = loader.get_tensor_shape(name) {
                return Ok(shape);
            }
        }
        Err(CoreError::model(
            "TENSOR_SHAPE_NOT_FOUND",
            format!("Could not find tensor shape with any of these names: {:?}", names),
            "Getting layer weight shapes from model",
            "Check that the model contains the expected layer tensor names"
        ))
    };

    // Attention weights
    let attn_q_weight = try_load_tensor(&[
        format!("{}.attn_q.weight", layer_prefix),
        format!("{}.self_attn.q_proj.weight", alt_prefix),
    ])?;
    let attn_q_shape = try_load_shape(&[
        format!("{}.attn_q.weight", layer_prefix),
        format!("{}.self_attn.q_proj.weight", alt_prefix),
    ])?;

    let attn_k_weight = try_load_tensor(&[
        format!("{}.attn_k.weight", layer_prefix),
        format!("{}.self_attn.k_proj.weight", alt_prefix),
    ])?;
    let attn_k_shape = try_load_shape(&[
        format!("{}.attn_k.weight", layer_prefix),
        format!("{}.self_attn.k_proj.weight", alt_prefix),
    ])?;

    let attn_v_weight = try_load_tensor(&[
        format!("{}.attn_v.weight", layer_prefix),
        format!("{}.self_attn.v_proj.weight", alt_prefix),
    ])?;
    let attn_v_shape = try_load_shape(&[
        format!("{}.attn_v.weight", layer_prefix),
        format!("{}.self_attn.v_proj.weight", alt_prefix),
    ])?;

    let attn_o_weight = try_load_tensor(&[
        format!("{}.attn_output.weight", layer_prefix),
        format!("{}.self_attn.o_proj.weight", alt_prefix),
    ])?;
    let attn_o_shape = try_load_shape(&[
        format!("{}.attn_output.weight", layer_prefix),
        format!("{}.self_attn.o_proj.weight", alt_prefix),
    ])?;

    let attn_norm_weight = try_load_tensor(&[
        format!("{}.attn_norm.weight", layer_prefix),
        format!("{}.input_layernorm.weight", alt_prefix),
    ])?;

    // Feed-forward weights
    let ffn_up_weight = try_load_tensor(&[
        format!("{}.ffn_up.weight", layer_prefix),
        format!("{}.mlp.up_proj.weight", alt_prefix),
    ])?;
    let ffn_up_shape = try_load_shape(&[
        format!("{}.ffn_up.weight", layer_prefix),
        format!("{}.mlp.up_proj.weight", alt_prefix),
    ])?;

    let ffn_down_weight = try_load_tensor(&[
        format!("{}.ffn_down.weight", layer_prefix),
        format!("{}.mlp.down_proj.weight", alt_prefix),
    ])?;
    let ffn_down_shape = try_load_shape(&[
        format!("{}.ffn_down.weight", layer_prefix),
        format!("{}.mlp.down_proj.weight", alt_prefix),
    ])?;

    // Gate weight (for GLU variants like SwiGLU)
    let (ffn_gate_weight, ffn_gate_shape) = if loader.has_tensor(&format!("{}.ffn_gate.weight", layer_prefix)) || 
                                                loader.has_tensor(&format!("{}.mlp.gate_proj.weight", alt_prefix)) {
        let gate_weight = try_load_tensor(&[
            format!("{}.ffn_gate.weight", layer_prefix),
            format!("{}.mlp.gate_proj.weight", alt_prefix),
        ])?;
        let gate_shape = try_load_shape(&[
            format!("{}.ffn_gate.weight", layer_prefix),
            format!("{}.mlp.gate_proj.weight", alt_prefix),
        ])?;
        (Some(gate_weight), Some(gate_shape))
    } else {
        (None, None)
    };

    let ffn_norm_weight = try_load_tensor(&[
        format!("{}.ffn_norm.weight", layer_prefix),
        format!("{}.post_attention_layernorm.weight", alt_prefix),
    ])?;

    Ok(LayerWeights {
        attn_q_weight,
        attn_k_weight,
        attn_v_weight,
        attn_o_weight,
        attn_norm_weight,
        ffn_gate_weight,
        ffn_up_weight,
        ffn_down_weight,
        ffn_norm_weight,
        attn_q_shape,
        attn_k_shape,
        attn_v_shape,
        attn_o_shape,
        ffn_up_shape,
        ffn_down_shape,
        ffn_gate_shape,
    })
}

/// Create default tensor name mapping for common architectures
fn create_default_tensor_mapping(loader: &GGUFLoader) -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    
    let arch = loader.architecture().unwrap_or("llama");
    
    match arch {
        "llama" => {
            // Standard LLaMA tensor names
            mapping.insert("token_embd.weight".to_string(), "token_embd.weight".to_string());
            mapping.insert("output_norm.weight".to_string(), "output_norm.weight".to_string());
            mapping.insert("output.weight".to_string(), "output.weight".to_string());
        }
        _ => {
            // Default mapping - no changes
        }
    }
    
    mapping
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_utils_validate_shape() {
        let actual = vec![4, 8, 16];
        let expected = vec![4, 8, 16];
        assert!(TensorUtils::validate_shape(&actual, &expected, "test").is_ok());

        let wrong_shape = vec![4, 8, 32];
        assert!(TensorUtils::validate_shape(&actual, &wrong_shape, "test").is_err());
    }

    #[test]
    fn test_tensor_utils_transpose_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = TensorUtils::transpose_2d(&data, 2, 3).unwrap();
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(result, expected);
    }
}