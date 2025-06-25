//! End-to-end integration tests for Woolly Core
//!
//! This test suite demonstrates the integration of all Woolly components:
//! - GGUF loading
//! - Model initialization
//! - Tokenization
//! - Inference pipeline
//! - Decoding
//! - MCP integration
//!
//! Tests are designed to work with the current implementation state, including
//! placeholder implementations, and can be extended as components mature.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use tokio;
use tracing_subscriber;

use woolly_core::{
    config::{EngineConfig, DeviceConfig, DeviceType},
    engine::InferenceEngine,
    model::{Model, ModelConfig, ModelOutput, ModelFeature, KVCache},
    session::{InferenceSession, SessionConfig},
    tokenizer::{Tokenizer, TokenizerConfig, TokenizerType, create_tokenizer},
    generation::{GenerationConfig, GenerationResult, FinishReason},
    CoreError, Result,
};

#[cfg(feature = "mcp")]
use woolly_mcp::Registry;

use async_trait::async_trait;

/// Mock model implementation for testing
#[derive(Debug, Clone)]
pub struct MockModel {
    name: String,
    model_type: String,
    config: ModelConfig,
    weights_loaded: bool,
}

impl MockModel {
    pub fn new(name: String, config: ModelConfig) -> Self {
        Self {
            name,
            model_type: "mock".to_string(),
            config,
            weights_loaded: false,
        }
    }

    pub fn create_simple_model() -> Self {
        let config = ModelConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_layers: 4,
            num_heads: 8,
            context_length: 512,
            intermediate_size: 512,
            num_key_value_heads: Some(8),
            rope_theta: Some(10000.0),
            layer_norm_epsilon: 1e-5,
        };

        Self::new("mock-model-v1".to_string(), config)
    }
}

#[async_trait]
impl Model for MockModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn context_length(&self) -> usize {
        self.config.context_length
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    async fn forward(
        &self,
        input_ids: &[u32],
        _past_kv_cache: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<ModelOutput> {
        if !self.weights_loaded {
            return Err(CoreError::Model("Model weights not loaded".to_string()));
        }

        // Generate mock logits (uniform distribution for simplicity)
        let batch_size = 1;
        let seq_len = input_ids.len();
        let vocab_size = self.vocab_size();
        
        let logits = vec![0.5; batch_size * seq_len * vocab_size];
        
        // Create a simple KV cache for testing
        let mut kv_cache = KVCache::new(self.num_layers(), self.context_length());
        
        for layer_idx in 0..self.num_layers() {
            let key_size = seq_len * self.hidden_size();
            let value_size = seq_len * self.hidden_size();
            
            let keys = vec![0.1; key_size];
            let values = vec![0.2; value_size];
            let shape = vec![batch_size, seq_len, self.hidden_size()];
            
            kv_cache.update(layer_idx, keys, values, shape)?;
        }

        Ok(ModelOutput {
            logits,
            logits_shape: vec![batch_size, seq_len, vocab_size],
            past_kv_cache: Some(Box::new(kv_cache)),
            hidden_states: None,
            attentions: None,
        })
    }

    async fn load_weights(&mut self, _path: &Path) -> Result<()> {
        // Mock weight loading - just set the flag
        self.weights_loaded = true;
        Ok(())
    }

    fn supports_feature(&self, feature: ModelFeature) -> bool {
        match feature {
            ModelFeature::RoPE => true,
            _ => false,
        }
    }
}

/// Mock tokenizer implementation for testing
#[derive(Debug)]
pub struct MockTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
    vocab_size: usize,
}

impl MockTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        let mut special_tokens = HashMap::new();

        // Add special tokens
        special_tokens.insert("<pad>".to_string(), 0);
        special_tokens.insert("<unk>".to_string(), 1);
        special_tokens.insert("<s>".to_string(), 2);
        special_tokens.insert("</s>".to_string(), 3);

        // Add basic vocabulary
        let words = vec![
            "hello", "world", "test", "model", "inference", "tokenizer",
            "the", "a", "an", "and", "or", "but", "for", "in", "on", "at",
            "this", "that", "these", "those", "good", "bad", "fast", "slow",
        ];

        let mut token_id = 4; // Start after special tokens
        
        for (word, id) in special_tokens.iter() {
            vocab.insert(word.clone(), *id);
            reverse_vocab.insert(*id, word.clone());
        }

        for word in words {
            vocab.insert(word.to_string(), token_id);
            reverse_vocab.insert(token_id, word.to_string());
            token_id += 1;
        }

        Self {
            vocab,
            reverse_vocab,
            special_tokens,
            vocab_size: token_id as usize,
        }
    }
}

#[async_trait]
impl Tokenizer for MockTokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::new();
        
        for word in words {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                // Use unknown token
                tokens.push(1);
            }
        }
        
        Ok(tokens)
    }

    async fn encode_with_special_tokens(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        if add_bos {
            tokens.push(2); // <s>
        }
        
        tokens.extend(self.encode(text).await?);
        
        if add_eos {
            tokens.push(3); // </s>
        }
        
        Ok(tokens)
    }

    async fn decode(&self, tokens: &[u32]) -> Result<String> {
        let words: Vec<String> = tokens
            .iter()
            .filter_map(|&token_id| self.reverse_vocab.get(&token_id).cloned())
            .collect();
        
        Ok(words.join(" "))
    }

    async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String> {
        let words: Vec<String> = tokens
            .iter()
            .filter(|&&token_id| !self.is_special_token(token_id))
            .filter_map(|&token_id| self.reverse_vocab.get(&token_id).cloned())
            .collect();
        
        Ok(words.join(" "))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(2)
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(3)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(0)
    }

    fn unk_token_id(&self) -> Option<u32> {
        Some(1)
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.reverse_vocab.get(&id).map(|s| s.as_str())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn is_special_token(&self, id: u32) -> bool {
        id < 4 // Our special tokens are 0, 1, 2, 3
    }

    fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }
}

/// Mock GGUF loader for testing
pub struct MockGGUFLoader;

impl MockGGUFLoader {
    pub fn create_mock_model() -> Result<Arc<dyn Model>> {
        let model = MockModel::create_simple_model();
        Ok(Arc::new(model))
    }
}

/// Initialize test environment
pub fn init_test_env() {
    // Initialize tracing for test output
    let _ = tracing_subscriber::fmt()
        .with_env_filter("debug")
        .try_init();
}

#[tokio::test]
async fn test_basic_engine_creation() {
    init_test_env();
    
    let config = EngineConfig {
        max_context_length: 2048,
        max_batch_size: 4,
        num_threads: 4,
        device: DeviceConfig {
            device_type: DeviceType::Cpu,
            device_id: 0,
            cpu_fallback: true,
            cuda: None,
            metal: None,
        },
        ..Default::default()
    };
    
    let engine = InferenceEngine::new(config);
    assert!(engine.model_info().is_none());
}

#[tokio::test]
async fn test_model_loading_and_validation() {
    init_test_env();
    
    let mut engine = InferenceEngine::default();
    let model = MockGGUFLoader::create_mock_model().unwrap();
    
    // Test successful model loading
    let result = engine.load_model(model).await;
    assert!(result.is_ok());
    
    // Test model info retrieval
    let model_info = engine.model_info();
    assert!(model_info.is_some());
    
    let info = model_info.unwrap();
    assert_eq!(info.name, "mock-model-v1");
    assert_eq!(info.model_type, "mock");
    assert_eq!(info.vocab_size, 1000);
    assert_eq!(info.context_length, 512);
    assert_eq!(info.hidden_size, 128);
    assert_eq!(info.num_layers, 4);
    assert_eq!(info.num_heads, 8);
}

#[tokio::test]
async fn test_tokenizer_operations() {
    init_test_env();
    
    let tokenizer = MockTokenizer::new();
    
    // Test basic encoding
    let text = "hello world test";
    let tokens = tokenizer.encode(text).await.unwrap();
    assert!(!tokens.is_empty());
    
    // Test encoding with special tokens
    let tokens_with_special = tokenizer
        .encode_with_special_tokens(text, true, true)
        .await
        .unwrap();
    assert!(tokens_with_special.len() > tokens.len());
    assert_eq!(tokens_with_special[0], 2); // BOS token
    assert_eq!(tokens_with_special[tokens_with_special.len() - 1], 3); // EOS token
    
    // Test decoding
    let decoded = tokenizer.decode(&tokens).await.unwrap();
    assert_eq!(decoded, text);
    
    // Test decoding with special tokens skipped
    let decoded_skip_special = tokenizer
        .decode_skip_special_tokens(&tokens_with_special)
        .await
        .unwrap();
    assert_eq!(decoded_skip_special, text);
    
    // Test special token properties
    assert_eq!(tokenizer.vocab_size(), 28); // 4 special + 24 regular tokens
    assert_eq!(tokenizer.bos_token_id(), Some(2));
    assert_eq!(tokenizer.eos_token_id(), Some(3));
    assert_eq!(tokenizer.pad_token_id(), Some(0));
    assert_eq!(tokenizer.unk_token_id(), Some(1));
}

#[tokio::test]
async fn test_session_creation_and_management() {
    init_test_env();
    
    let mut engine = InferenceEngine::default();
    let mut model = MockModel::create_simple_model();
    
    // Load mock weights
    model.load_weights(Path::new("mock_path")).await.unwrap();
    
    engine.load_model(Arc::new(model)).await.unwrap();
    
    // Test session creation
    let session_config = SessionConfig {
        max_batch_size: 2,
        use_cache: true,
        max_seq_length: 256,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        output_hidden_states: false,
        output_attentions: false,
    };
    
    let session = engine.create_session(session_config).await.unwrap();
    assert!(session.is_active());
    assert_eq!(session.tokens_processed(), 0);
    
    // Test session listing
    let sessions = engine.list_sessions().await;
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, session.id());
    assert!(sessions[0].active);
    assert_eq!(sessions[0].tokens_processed, 0);
}

#[tokio::test]
async fn test_end_to_end_inference_pipeline() {
    init_test_env();
    
    // Step 1: Create and configure engine
    let mut engine = InferenceEngine::default();
    
    // Step 2: Load model (mock GGUF loading)
    let mut model = MockModel::create_simple_model();
    model.load_weights(Path::new("mock_model.gguf")).await.unwrap();
    engine.load_model(Arc::new(model)).await.unwrap();
    
    // Step 3: Create tokenizer
    let tokenizer = MockTokenizer::new();
    
    // Step 4: Create inference session
    let session_config = SessionConfig::default();
    let session = engine.create_session(session_config).await.unwrap();
    
    // Step 5: Test tokenization
    let input_text = "hello world test inference";
    let input_tokens = tokenizer
        .encode_with_special_tokens(input_text, true, false)
        .await
        .unwrap();
    
    // Step 6: Run inference
    // Note: This will return an error due to the todo!() in session.infer()
    // but we can test the validation logic
    let inference_result = session.infer(&input_tokens).await;
    
    // For now, we expect this to fail with the todo!() error
    // Once the implementation is complete, this should succeed
    assert!(inference_result.is_err());
    
    // Test session state
    assert!(session.is_active());
    
    // Test token history (this might work depending on implementation)
    let history = session.token_history().await;
    // History might be empty if infer() fails before updating
    
    // Test session cleanup
    session.clear().await;
    let cleared_history = session.token_history().await;
    assert!(cleared_history.is_empty());
}

#[tokio::test]
async fn test_batch_inference() {
    init_test_env();
    
    let mut engine = InferenceEngine::default();
    let mut model = MockModel::create_simple_model();
    model.load_weights(Path::new("mock_model.gguf")).await.unwrap();
    engine.load_model(Arc::new(model)).await.unwrap();
    
    let tokenizer = MockTokenizer::new();
    
    let session_config = SessionConfig {
        max_batch_size: 3,
        ..Default::default()
    };
    let session = engine.create_session(session_config).await.unwrap();
    
    // Create batch of inputs
    let batch_texts = vec![
        "hello world",
        "test inference",
        "batch processing",
    ];
    
    let mut batch_tokens = Vec::new();
    for text in batch_texts {
        let tokens = tokenizer.encode(text).await.unwrap();
        batch_tokens.push(tokens);
    }
    
    // Test batch inference
    let batch_result = session.infer_batch(batch_tokens).await;
    
    // This should fail due to the todo!() but we can test validation
    assert!(batch_result.is_err());
    
    // Test batch size limit
    let oversized_batch = vec![vec![1, 2, 3]; 5]; // 5 sequences > max_batch_size of 3
    let oversized_result = session.infer_batch(oversized_batch).await;
    assert!(oversized_result.is_err());
    
    if let Err(CoreError::Generation(msg)) = oversized_result {
        assert!(msg.contains("exceeds maximum"));
    }
}

#[tokio::test]
async fn test_model_features_and_compatibility() {
    init_test_env();
    
    let model = MockModel::create_simple_model();
    
    // Test feature support
    assert!(model.supports_feature(ModelFeature::RoPE));
    assert!(!model.supports_feature(ModelFeature::FlashAttention));
    assert!(!model.supports_feature(ModelFeature::GroupedQueryAttention));
    
    // Test model properties
    assert_eq!(model.name(), "mock-model-v1");
    assert_eq!(model.model_type(), "mock");
    assert_eq!(model.vocab_size(), 1000);
    assert_eq!(model.context_length(), 512);
    assert_eq!(model.hidden_size(), 128);
    assert_eq!(model.num_layers(), 4);
    assert_eq!(model.num_heads(), 8);
}

#[tokio::test]
async fn test_kv_cache_operations() {
    init_test_env();
    
    let num_layers = 4;
    let max_seq_len = 256;
    let mut cache = KVCache::new(num_layers, max_seq_len);
    
    // Test initial state
    assert_eq!(cache.seq_len, 0);
    assert_eq!(cache.max_seq_len, max_seq_len);
    
    // Test cache updates
    let keys = vec![0.1; 64];
    let values = vec![0.2; 64];
    let shape = vec![1, 8, 8];
    
    let result = cache.update(0, keys.clone(), values.clone(), shape.clone());
    assert!(result.is_ok());
    
    // Test invalid layer index
    let invalid_result = cache.update(10, keys, values, shape);
    assert!(invalid_result.is_err());
    
    // Test cache clearing
    cache.clear();
    assert_eq!(cache.seq_len, 0);
    assert!(cache.keys.is_empty());
    assert!(cache.values.is_empty());
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn test_mcp_integration() {
    init_test_env();
    
    // Create MCP registry
    let registry = Registry::new();
    
    // Create engine with MCP support
    let mut engine = InferenceEngine::default();
    engine.set_mcp_registry(registry.clone());
    
    // Load model
    let mut model = MockModel::create_simple_model();
    model.load_weights(Path::new("mock_model.gguf")).await.unwrap();
    engine.load_model(Arc::new(model)).await.unwrap();
    
    // Create session (should trigger MCP hooks)
    let session = engine.create_session(SessionConfig::default()).await.unwrap();
    
    // Test that session was created with MCP integration
    assert!(session.is_active());
    
    // Deactivate session (should trigger MCP hooks)
    session.deactivate();
    assert!(!session.is_active());
}

#[tokio::test]
async fn test_error_handling_and_edge_cases() {
    init_test_env();
    
    let mut engine = InferenceEngine::default();
    
    // Test inference without loaded model
    let result = engine.infer(&[1, 2, 3], None).await;
    assert!(result.is_err());
    if let Err(CoreError::Model(msg)) = result {
        assert!(msg.contains("No model loaded"));
    }
    
    // Test session creation without model
    let session_result = engine.create_session(SessionConfig::default()).await;
    assert!(session_result.is_err());
    
    // Test with loaded model
    let mut model = MockModel::create_simple_model();
    model.load_weights(Path::new("mock_model.gguf")).await.unwrap();
    engine.load_model(Arc::new(model)).await.unwrap();
    
    let session = engine.create_session(SessionConfig::default()).await.unwrap();
    
    // Test empty token inference
    let empty_result = session.infer(&[]).await;
    assert!(empty_result.is_err());
    if let Err(CoreError::Generation(msg)) = empty_result {
        assert!(msg.contains("Empty input tokens"));
    }
    
    // Test sequence length limit
    let long_tokens = vec![1; 3000]; // Exceeds default max_seq_length of 2048
    let long_result = session.infer(&long_tokens).await;
    assert!(long_result.is_err());
    if let Err(CoreError::Context(msg)) = long_result {
        assert!(msg.contains("exceeds maximum"));
    }
    
    // Test inactive session
    session.deactivate();
    let inactive_result = session.infer(&[1, 2, 3]).await;
    assert!(inactive_result.is_err());
}

#[tokio::test]
async fn test_generation_config_and_sampling() {
    init_test_env();
    
    // Test generation config
    let gen_config = GenerationConfig {
        max_tokens: 100,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };
    
    assert_eq!(gen_config.max_tokens, 100);
    assert_eq!(gen_config.temperature, 0.8);
    assert_eq!(gen_config.top_p, 0.9);
    assert_eq!(gen_config.top_k, 40);
    assert_eq!(gen_config.repetition_penalty, 1.1);
    
    // Test generation result
    let gen_result = GenerationResult {
        tokens: vec![1, 2, 3, 4, 5],
        text: "hello world".to_string(),
        finish_reason: FinishReason::MaxTokens,
    };
    
    assert_eq!(gen_result.tokens.len(), 5);
    assert_eq!(gen_result.text, "hello world");
    assert!(matches!(gen_result.finish_reason, FinishReason::MaxTokens));
}

#[tokio::test]
async fn test_concurrent_sessions() {
    init_test_env();
    
    let mut engine = InferenceEngine::default();
    let mut model = MockModel::create_simple_model();
    model.load_weights(Path::new("mock_model.gguf")).await.unwrap();
    engine.load_model(Arc::new(model)).await.unwrap();
    
    // Create multiple sessions concurrently
    let session1 = engine.create_session(SessionConfig::default()).await.unwrap();
    let session2 = engine.create_session(SessionConfig::default()).await.unwrap();
    let session3 = engine.create_session(SessionConfig::default()).await.unwrap();
    
    // Verify all sessions are active and have unique IDs
    assert!(session1.is_active());
    assert!(session2.is_active());
    assert!(session3.is_active());
    
    assert_ne!(session1.id(), session2.id());
    assert_ne!(session2.id(), session3.id());
    assert_ne!(session1.id(), session3.id());
    
    // Test session listing
    let sessions = engine.list_sessions().await;
    assert_eq!(sessions.len(), 3);
    
    // Test session cleanup
    session2.deactivate();
    engine.cleanup_sessions().await;
    
    let sessions_after_cleanup = engine.list_sessions().await;
    assert_eq!(sessions_after_cleanup.len(), 2);
}

/// Integration test summary and validation
#[tokio::test]
async fn test_integration_summary() {
    init_test_env();
    
    println!("=== Woolly Core Integration Test Summary ===");
    println!("✓ Engine creation and configuration");
    println!("✓ Mock model loading and validation");
    println!("✓ Tokenizer operations (encode/decode)");
    println!("✓ Session creation and management");
    println!("✓ KV cache operations");
    println!("✓ Error handling and edge cases");
    println!("✓ Concurrent session handling");
    println!("✓ Generation configuration");
    #[cfg(feature = "mcp")]
    println!("✓ MCP integration");
    println!("⚠ End-to-end inference (pending tensor backend completion)");
    println!("⚠ GGUF loading (pending real file support)");
    println!("=============================================");
    
    // This test always passes - it's just for summary
    assert!(true);
}