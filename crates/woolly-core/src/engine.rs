//! Inference engine for model execution and management

use crate::{
    config::EngineConfig,
    model::Model,
    session::{InferenceSession, SessionConfig},
    CoreError, Result,
};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main inference engine that manages models and sessions
pub struct InferenceEngine {
    /// Engine configuration
    config: EngineConfig,
    /// Currently loaded model
    model: Option<Arc<dyn Model>>,
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
        // Validate model compatibility
        if !self.validate_model(&model)? {
            return Err(CoreError::Model(
                "Model not compatible with engine configuration".to_string(),
            ));
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
            .ok_or_else(|| CoreError::Model("No model loaded".to_string()))?;

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
            .ok_or_else(|| CoreError::Model("No model loaded".to_string()))?;

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
            .ok_or_else(|| CoreError::Other(anyhow::anyhow!("Session not found: {}", session_id)))
    }

    /// Clean up inactive sessions
    pub async fn cleanup_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|s| s.is_active());
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