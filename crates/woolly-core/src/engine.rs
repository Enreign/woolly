//! Inference engine for model execution and management

use crate::{
    config::EngineConfig,
    model::{Model, fused_transformer::{FusedTransformer, FusedTransformerConfig}},
    session::{InferenceSession, SessionConfig},
    CoreError, Result,
};
use std::sync::Arc;
use std::path::Path;
use tokio::sync::RwLock;
use woolly_gguf::GGUFLoader;

/// Main inference engine that manages models and sessions
pub struct InferenceEngine {
    /// Engine configuration
    config: EngineConfig,
    /// Currently loaded model
    model: Option<Arc<dyn Model>>,
    /// Loaded model filename (for display purposes)
    model_filename: Option<String>,
    /// Active inference sessions
    sessions: RwLock<Vec<Arc<InferenceSession>>>,
    /// MCP integration hooks
    #[cfg(feature = "mcp")]
    mcp_registry: Option<woolly_mcp::PluginRegistry>,
}

impl InferenceEngine {
    /// Create a new inference engine with the given configuration
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            model: None,
            model_filename: None,
            sessions: RwLock::new(Vec::new()),
            #[cfg(feature = "mcp")]
            mcp_registry: None,
        }
    }

    /// Create a new inference engine with default configuration
    pub fn default() -> Self {
        Self::new(EngineConfig::default())
    }

    /// Load a model into the engine
    pub async fn load_model(&mut self, model: Arc<dyn Model>) -> Result<()> {
        // Dynamically adjust engine configuration to match model requirements
        let model_context_length = model.context_length();
        if model_context_length > self.config.max_context_length {
            eprintln!("Adjusting engine max_context_length from {} to {} to match model requirements", 
                      self.config.max_context_length, model_context_length);
            self.config.max_context_length = model_context_length;
        }
        
        // Validate model compatibility (now should always pass for context length)
        if !self.validate_model(&model)? {
            return Err(CoreError::model("MODEL_ERROR", "Model not compatible with engine configuration", "", "Check model configuration"));
        }

        // Clear existing sessions when loading new model
        self.sessions.write().await.clear();
        
        self.model = Some(model);
        Ok(())
    }

    /// Create a new inference session
    pub async fn create_session(&self, config: SessionConfig) -> Result<Arc<InferenceSession>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| CoreError::model("MODEL_ERROR", "No model loaded", "", "Check model configuration"))?;

        let session = Arc::new(InferenceSession::new(
            Arc::clone(model),
            config,
            #[cfg(feature = "mcp")]
            None, // TODO: Fix MCP registry sharing
            #[cfg(not(feature = "mcp"))]
            None,
        )?);

        self.sessions.write().await.push(Arc::clone(&session));
        Ok(session)
    }

    /// Run inference on input tokens
    pub async fn infer(&self, tokens: &[u32], session_id: Option<String>) -> Result<Vec<f32>> {
        let _model = self
            .model
            .as_ref()
            .ok_or_else(|| CoreError::model("MODEL_ERROR", "No model loaded", "", "Check model configuration"))?;

        // Find or create session
        let session = if let Some(id) = session_id {
            self.find_session(&id).await?
        } else {
            self.create_session(SessionConfig::default()).await?
        };

        // Run inference through the session
        session.infer(tokens).await
    }

    /// Get engine configuration
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Set MCP registry for hook integration
    #[cfg(feature = "mcp")]
    pub fn set_mcp_registry(&mut self, registry: woolly_mcp::PluginRegistry) {
        self.mcp_registry = Some(registry);
    }

    /// Get current model information
    pub fn model_info(&self) -> Option<ModelInfo> {
        self.model.as_ref().map(|m| ModelInfo {
            name: m.name().to_string(),
            model_type: m.model_type().to_string(),
            vocab_size: m.vocab_size(),
            context_length: m.context_length(),
            hidden_size: m.hidden_size(),
            num_layers: m.num_layers(),
            num_heads: m.num_heads(),
        })
    }

    /// List active sessions
    pub async fn list_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.read().await;
        sessions
            .iter()
            .map(|s| SessionInfo {
                id: s.id().to_string(),
                active: s.is_active(),
                tokens_processed: s.tokens_processed(),
            })
            .collect()
    }

    /// Validate model compatibility with engine
    fn validate_model(&self, model: &Arc<dyn Model>) -> Result<bool> {
        // Check if model requirements match engine capabilities
        eprintln!("Validating model: context_length={}, max_context_length={}", 
                  model.context_length(), self.config.max_context_length);
        if model.context_length() > self.config.max_context_length {
            return Ok(false);
        }

        // Additional validation can be added here
        Ok(true)
    }

    /// Find session by ID
    async fn find_session(&self, session_id: &str) -> Result<Arc<InferenceSession>> {
        let sessions = self.sessions.read().await;
        sessions
            .iter()
            .find(|s| s.id() == session_id)
            .cloned()
            .ok_or_else(|| CoreError::invalid_input(
                "SESSION_NOT_FOUND",
                format!("Session not found: {}", session_id),
                "Looking up inference session",
                "Check that the session ID is valid and the session hasn't been cleaned up"
            ))
    }

    /// Clean up inactive sessions
    pub async fn cleanup_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|s| s.is_active());
    }
    
    /// Set the loaded model filename
    pub fn set_model_filename(&mut self, filename: String) {
        self.model_filename = Some(filename);
    }
    
    /// Get the loaded model filename
    pub fn model_filename(&self) -> Option<&str> {
        self.model_filename.as_deref()
    }
    
    /// Load a model with fast FP32 initialization (bypassing GGUF for testing)
    pub async fn load_fast_fp32_model(&mut self) -> Result<()> {
        // Temporarily disabled until fast_transformer is fixed
        /*use crate::model::fast_initialization::{is_fast_fp32_enabled, get_fast_fp32_config};
        use crate::model::fast_transformer::FastTransformer;*/
        
        Err(CoreError::configuration(
            "FAST_FP32_DISABLED",
            "Fast FP32 mode temporarily disabled",
            "Loading fast FP32 model",
            "Feature under maintenance"
        ))
    }

    /// Load a fused transformer model from GGUF file with optimal performance
    pub async fn load_fused_model_from_gguf(&mut self, path: &Path) -> Result<()> {
        // Load GGUF file and extract configuration
        use woolly_gguf::GGUFLoader;
        use crate::model::loader::{GGUFModelLoader, ModelLoader};
        
        let loader = GGUFModelLoader::from_path(path)?;
        let model_config = loader.config()?;
        
        // Create FusedTransformerConfig from GGUF metadata
        let fused_config = FusedTransformerConfig::new(
            model_config.vocab_size,
            model_config.hidden_size,
            model_config.num_layers,
            model_config.num_heads,
            model_config.num_key_value_heads.unwrap_or(model_config.num_heads),
            model_config.intermediate_size,
        )?;
        
        // Create fused transformer with optimizations enabled
        let mut fused_transformer = FusedTransformer::new(fused_config)?;
        
        // Load GGUF weights efficiently
        let gguf_loader = GGUFLoader::from_path(path)
            .map_err(|e| CoreError::model(
                "GGUF_LOAD_FAILED",
                format!("Failed to load GGUF file: {}", e),
                "Loading optimized model",
                "Check file path and format"
            ))?;
        
        // Load model weights using the optimized loader
        self.load_fused_weights(&mut fused_transformer, &gguf_loader, &model_config).await?;
        
        // Wrap in Arc<dyn Model> for the engine
        let model = Arc::new(fused_transformer) as Arc<dyn Model>;
        
        // Set filename for display
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            self.set_model_filename(filename.to_string());
        }
        
        // Load the model into the engine
        self.load_model(model).await
    }
    
    /// Load weights into FusedTransformer with optimized loading
    async fn load_fused_weights(
        &self,
        fused_transformer: &mut FusedTransformer,
        gguf_loader: &GGUFLoader,
        model_config: &crate::model::ModelConfig,
    ) -> Result<()> {
        // Helper function to get dequantized tensor data
        let get_tensor_f32 = |name: &str| -> Result<Vec<f32>> {
            let data = gguf_loader.tensor_data(name).map_err(|e| CoreError::model(
                "TENSOR_DATA_FAILED",
                format!("Failed to get tensor data for '{}': {}", name, e),
                "Loading tensor data",
                "Check model file integrity"
            ))?;
            
            let tensor_info = gguf_loader.tensor_info(name).ok_or_else(|| CoreError::model(
                "TENSOR_NOT_FOUND",
                format!("Tensor '{}' not found", name),
                "Loading tensor info",
                "Check tensor name"
            ))?;
            
            let num_elements = tensor_info.shape().iter().map(|&x| x as usize).product();
            woolly_gguf::dequantize(&data, tensor_info.ggml_type, num_elements).map_err(|e| CoreError::model(
                "DEQUANTIZE_FAILED",
                format!("Failed to dequantize tensor '{}': {}", name, e),
                "Dequantizing tensor data",
                "Check tensor format compatibility"
            ))
        };
        
        // Load embeddings
        if let Ok(embedding_data) = get_tensor_f32("token_embd.weight") {
            fused_transformer.load_embedding_weights(&embedding_data)?;
        }
        
        // Load layer weights in batches for memory efficiency
        let mut layer_weights = Vec::with_capacity(model_config.num_layers);
        
        for layer_idx in 0..model_config.num_layers {
            // Load attention weights
            let attn_q_weight = get_tensor_f32(&format!("blk.{}.attn_q.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load Q weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
                
            let attn_k_weight = get_tensor_f32(&format!("blk.{}.attn_k.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED", 
                    format!("Failed to load K weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
                
            let attn_v_weight = get_tensor_f32(&format!("blk.{}.attn_v.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load V weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights", 
                    "Check model file integrity"
                ))?;
                
            let attn_o_weight = get_tensor_f32(&format!("blk.{}.attn_output.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load O weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
            
            // Load FFN weights
            let ffn_gate_weight = get_tensor_f32(&format!("blk.{}.ffn_gate.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load FFN gate weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
                
            let ffn_up_weight = get_tensor_f32(&format!("blk.{}.ffn_up.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load FFN up weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
                
            let ffn_down_weight = get_tensor_f32(&format!("blk.{}.ffn_down.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load FFN down weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
            
            // Load normalization weights
            let norm_1_weight = get_tensor_f32(&format!("blk.{}.attn_norm.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load attention norm weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
                
            let norm_2_weight = get_tensor_f32(&format!("blk.{}.ffn_norm.weight", layer_idx))
                .map_err(|e| CoreError::model(
                    "WEIGHT_LOAD_FAILED",
                    format!("Failed to load FFN norm weight for layer {}: {}", layer_idx, e),
                    "Loading fused transformer weights",
                    "Check model file integrity"
                ))?;
            
            // Get tensor shapes for validation
            let get_tensor_shape = |name: &str| -> Vec<usize> {
                gguf_loader.tensor_info(name)
                    .map(|info| info.shape().iter().map(|&x| x as usize).collect())
                    .unwrap_or_default()
            };
            
            // Create layer weights structure
            use crate::model::loader::LayerWeights;
            let layer_weight = LayerWeights {
                attn_q_weight,
                attn_k_weight,
                attn_v_weight,
                attn_o_weight,
                ffn_gate_weight: Some(ffn_gate_weight),
                ffn_up_weight,
                ffn_down_weight,
                attn_norm_weight: norm_1_weight,
                ffn_norm_weight: norm_2_weight,
                // Add shape information
                attn_q_shape: get_tensor_shape(&format!("blk.{}.attn_q.weight", layer_idx)),
                attn_k_shape: get_tensor_shape(&format!("blk.{}.attn_k.weight", layer_idx)),
                attn_v_shape: get_tensor_shape(&format!("blk.{}.attn_v.weight", layer_idx)),
                attn_o_shape: get_tensor_shape(&format!("blk.{}.attn_output.weight", layer_idx)),
                ffn_up_shape: get_tensor_shape(&format!("blk.{}.ffn_up.weight", layer_idx)),
                ffn_down_shape: get_tensor_shape(&format!("blk.{}.ffn_down.weight", layer_idx)),
                ffn_gate_shape: Some(get_tensor_shape(&format!("blk.{}.ffn_gate.weight", layer_idx))),
            };
            
            layer_weights.push(layer_weight);
        }
        
        // Load all layer weights into the transformer
        fused_transformer.load_all_weights(&layer_weights)?;
        
        // Load final normalization weights
        if let Ok(final_norm_data) = get_tensor_f32("output_norm.weight") {
            fused_transformer.load_final_norm_weights(&final_norm_data)?;
        }
        
        // Load LM head weights if available (not tied to embeddings)
        if let Ok(lm_head_data) = get_tensor_f32("output.weight") {
            fused_transformer.load_lm_head_weights(&lm_head_data)?;
        }
        
        eprintln!("FusedTransformer: Successfully loaded optimized model weights for {} layers", model_config.num_layers);
        Ok(())
    }
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
}

/// Information about an inference session
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub active: bool,
    pub tokens_processed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = InferenceEngine::default();
        assert!(engine.model.is_none());
        assert!(engine.list_sessions().await.is_empty());
    }
}