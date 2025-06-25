//! Integration hooks for the inference engine

use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value;

use crate::types::{McpMessage, McpError, ToolCallRequest, ToolCallResponse};

/// Result type for hook operations
pub type HookResult<T> = Result<T, HookError>;

/// Hook error type
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("Hook execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Invalid hook data: {0}")]
    InvalidData(String),
    
    #[error("Hook not found: {0}")]
    NotFound(String),
    
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Inference context passed to hooks
#[derive(Debug, Clone)]
pub struct InferenceContext {
    /// Current model being used
    pub model_id: String,
    
    /// Current prompt/input
    pub prompt: String,
    
    /// Generation parameters
    pub parameters: GenerationParameters,
    
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}

/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerationParameters {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_tokens: Option<usize>,
    pub stop_sequences: Vec<String>,
    pub repetition_penalty: f32,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            max_tokens: None,
            stop_sequences: Vec::new(),
            repetition_penalty: 1.0,
        }
    }
}

/// Token generated during inference
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    pub token_id: u32,
    pub token_text: String,
    pub logprob: f32,
    pub timestamp: std::time::Instant,
}

impl PartialEq for GeneratedToken {
    fn eq(&self, other: &Self) -> bool {
        self.token_id == other.token_id
            && self.token_text == other.token_text
            && self.logprob == other.logprob
        // Note: We exclude timestamp from equality comparison
    }
}

/// Hook that can be called before inference starts
#[async_trait]
pub trait PreInferenceHook: Send + Sync {
    /// Called before inference begins
    /// Can modify the context or abort inference by returning an error
    async fn execute(&self, context: &mut InferenceContext) -> HookResult<()>;
}

/// Hook that can be called after each token is generated
#[async_trait]
pub trait TokenHook: Send + Sync {
    /// Called after each token is generated
    /// Can modify or filter tokens
    async fn execute(
        &self,
        context: &InferenceContext,
        token: &mut GeneratedToken
    ) -> HookResult<TokenAction>;
}

/// Action to take after token hook execution
#[derive(Debug, Clone, PartialEq)]
pub enum TokenAction {
    /// Continue with the token as-is
    Continue,
    
    /// Skip this token and generate another
    Skip,
    
    /// Replace the token with a different one
    Replace(GeneratedToken),
    
    /// Stop generation
    Stop,
}

/// Hook that can be called after inference completes
#[async_trait]
pub trait PostInferenceHook: Send + Sync {
    /// Called after inference completes
    async fn execute(
        &self,
        context: &InferenceContext,
        output: &mut String
    ) -> HookResult<()>;
}

/// Hook for handling tool calls during inference
#[async_trait]
pub trait ToolCallHook: Send + Sync {
    /// Called when a tool call is detected in the output
    async fn execute(
        &self,
        context: &InferenceContext,
        tool_call: &ToolCallRequest
    ) -> HookResult<ToolCallResponse>;
}

/// Hook manager for registering and executing hooks
pub struct HookManager {
    pre_inference_hooks: Vec<Arc<dyn PreInferenceHook>>,
    token_hooks: Vec<Arc<dyn TokenHook>>,
    post_inference_hooks: Vec<Arc<dyn PostInferenceHook>>,
    tool_call_hooks: HashMap<String, Arc<dyn ToolCallHook>>,
}

impl HookManager {
    /// Create a new hook manager
    pub fn new() -> Self {
        Self {
            pre_inference_hooks: Vec::new(),
            token_hooks: Vec::new(),
            post_inference_hooks: Vec::new(),
            tool_call_hooks: HashMap::new(),
        }
    }
    
    /// Register a pre-inference hook
    pub fn register_pre_inference(&mut self, hook: Arc<dyn PreInferenceHook>) {
        self.pre_inference_hooks.push(hook);
    }
    
    /// Register a token hook
    pub fn register_token(&mut self, hook: Arc<dyn TokenHook>) {
        self.token_hooks.push(hook);
    }
    
    /// Register a post-inference hook
    pub fn register_post_inference(&mut self, hook: Arc<dyn PostInferenceHook>) {
        self.post_inference_hooks.push(hook);
    }
    
    /// Register a tool call hook
    pub fn register_tool_call(&mut self, tool_name: String, hook: Arc<dyn ToolCallHook>) {
        self.tool_call_hooks.insert(tool_name, hook);
    }
    
    /// Execute pre-inference hooks
    pub async fn execute_pre_inference(&self, context: &mut InferenceContext) -> HookResult<()> {
        for hook in &self.pre_inference_hooks {
            hook.execute(context).await?;
        }
        Ok(())
    }
    
    /// Execute token hooks
    pub async fn execute_token(
        &self,
        context: &InferenceContext,
        token: &mut GeneratedToken
    ) -> HookResult<TokenAction> {
        let mut action = TokenAction::Continue;
        
        for hook in &self.token_hooks {
            match hook.execute(context, token).await? {
                TokenAction::Continue => continue,
                TokenAction::Stop => return Ok(TokenAction::Stop),
                other => action = other,
            }
        }
        
        Ok(action)
    }
    
    /// Execute post-inference hooks
    pub async fn execute_post_inference(
        &self,
        context: &InferenceContext,
        output: &mut String
    ) -> HookResult<()> {
        for hook in &self.post_inference_hooks {
            hook.execute(context, output).await?;
        }
        Ok(())
    }
    
    /// Execute tool call hook
    pub async fn execute_tool_call(
        &self,
        context: &InferenceContext,
        tool_call: &ToolCallRequest
    ) -> HookResult<ToolCallResponse> {
        if let Some(hook) = self.tool_call_hooks.get(&tool_call.name) {
            hook.execute(context, tool_call).await
        } else {
            Err(HookError::NotFound(format!("Tool hook not found: {}", tool_call.name)))
        }
    }
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

/// MCP integration hook that bridges MCP protocol with inference engine
pub struct McpIntegrationHook {
    /// MCP protocol handler
    protocol: Arc<dyn crate::protocol::McpProtocol>,
}

impl McpIntegrationHook {
    /// Create a new MCP integration hook
    pub fn new(protocol: Arc<dyn crate::protocol::McpProtocol>) -> Self {
        Self { protocol }
    }
    
    /// Handle MCP message during inference
    pub async fn handle_message(&self, message: McpMessage) -> Result<Option<McpMessage>, McpError> {
        self.protocol.handle_message(message).await
    }
}

/// Example pre-inference hook that adds MCP context
pub struct McpContextHook {
    context_key: String,
}

impl McpContextHook {
    pub fn new(context_key: String) -> Self {
        Self { context_key }
    }
}

#[async_trait]
impl PreInferenceHook for McpContextHook {
    async fn execute(&self, context: &mut InferenceContext) -> HookResult<()> {
        // Add MCP-specific context
        context.metadata.insert(
            self.context_key.clone(),
            serde_json::json!({
                "mcp_version": crate::types::MCP_VERSION,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            })
        );
        Ok(())
    }
}

/// Example token hook that filters sensitive information
pub struct SensitiveInfoFilter {
    patterns: Vec<regex::Regex>,
}

impl SensitiveInfoFilter {
    pub fn new(patterns: Vec<String>) -> Result<Self, regex::Error> {
        let patterns = patterns
            .into_iter()
            .map(|p| regex::Regex::new(&p))
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(Self { patterns })
    }
}

#[async_trait]
impl TokenHook for SensitiveInfoFilter {
    async fn execute(
        &self,
        _context: &InferenceContext,
        token: &mut GeneratedToken
    ) -> HookResult<TokenAction> {
        // Check if token contains sensitive information
        for pattern in &self.patterns {
            if pattern.is_match(&token.token_text) {
                // Replace with placeholder
                token.token_text = "[FILTERED]".to_string();
                return Ok(TokenAction::Continue);
            }
        }
        
        Ok(TokenAction::Continue)
    }
}

/// Example post-inference hook that adds metadata
pub struct MetadataAppenderHook {
    metadata: HashMap<String, String>,
}

impl MetadataAppenderHook {
    pub fn new(metadata: HashMap<String, String>) -> Self {
        Self { metadata }
    }
}

#[async_trait]
impl PostInferenceHook for MetadataAppenderHook {
    async fn execute(
        &self,
        _context: &InferenceContext,
        output: &mut String
    ) -> HookResult<()> {
        // Append metadata as JSON comment
        let metadata_str = serde_json::to_string(&self.metadata)
            .map_err(|e| HookError::InvalidData(e.to_string()))?;
        
        output.push_str(&format!("\n<!-- metadata: {} -->", metadata_str));
        Ok(())
    }
}

/// Builder for creating hook managers with common configurations
pub struct HookManagerBuilder {
    manager: HookManager,
}

impl HookManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            manager: HookManager::new(),
        }
    }
    
    /// Add MCP context hook
    pub fn with_mcp_context(mut self, context_key: String) -> Self {
        self.manager.register_pre_inference(
            Arc::new(McpContextHook::new(context_key))
        );
        self
    }
    
    /// Add sensitive info filter
    pub fn with_sensitive_filter(mut self, patterns: Vec<String>) -> Result<Self, regex::Error> {
        let filter = SensitiveInfoFilter::new(patterns)?;
        self.manager.register_token(Arc::new(filter));
        Ok(self)
    }
    
    /// Add metadata appender
    pub fn with_metadata_appender(mut self, metadata: HashMap<String, String>) -> Self {
        self.manager.register_post_inference(
            Arc::new(MetadataAppenderHook::new(metadata))
        );
        self
    }
    
    /// Build the hook manager
    pub fn build(self) -> HookManager {
        self.manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hook_manager_creation() {
        let manager = HookManager::new();
        assert_eq!(manager.pre_inference_hooks.len(), 0);
        assert_eq!(manager.token_hooks.len(), 0);
        assert_eq!(manager.post_inference_hooks.len(), 0);
        assert_eq!(manager.tool_call_hooks.len(), 0);
    }
    
    #[tokio::test]
    async fn test_mcp_context_hook() {
        let hook = McpContextHook::new("mcp_context".to_string());
        let mut context = InferenceContext {
            model_id: "test-model".to_string(),
            prompt: "test prompt".to_string(),
            parameters: GenerationParameters::default(),
            metadata: HashMap::new(),
        };
        
        hook.execute(&mut context).await.unwrap();
        assert!(context.metadata.contains_key("mcp_context"));
    }
    
    #[tokio::test]
    async fn test_hook_manager_builder() {
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());
        
        let manager = HookManagerBuilder::new()
            .with_mcp_context("context".to_string())
            .with_metadata_appender(metadata)
            .build();
        
        assert_eq!(manager.pre_inference_hooks.len(), 1);
        assert_eq!(manager.post_inference_hooks.len(), 1);
    }
}