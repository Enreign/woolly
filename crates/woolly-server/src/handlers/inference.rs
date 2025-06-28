//! Inference request handlers

use crate::{
    error::ServerResult,
    server::ServerState,
};
use axum::{
    extract::State,
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
    Json,
};
use futures::{stream::{self, Stream}, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{convert::Infallible, time::Duration};
use woolly_core::{
    generation::GenerationConfig,
};

/// Completion request
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
}

/// Completion response
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
}

/// Chat message
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Choice in completion response
#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub text: Option<String>,
    pub message: Option<ChatMessage>,
    pub finish_reason: String,
    pub logprobs: Option<Value>,
}

/// Token usage information
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chunk
#[derive(Debug, Serialize)]
pub struct StreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamingChoice>,
}

/// Streaming choice
#[derive(Debug, Serialize)]
pub struct StreamingChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// Delta for streaming
#[derive(Debug, Serialize)]
pub struct Delta {
    pub content: Option<String>,
    pub role: Option<String>,
}

/// Text completion endpoint
#[axum::debug_handler]
pub async fn complete(
    State(state): State<ServerState>,
    Json(completion_request): Json<CompletionRequest>,
) -> Result<Response, crate::error::ServerError> {
    // TODO: Get auth context from middleware
    let user_id = "default-user".to_string(); // Placeholder
    
    if completion_request.stream.unwrap_or(false) {
        // Return streaming response
        let stream = create_completion_stream(state, completion_request, user_id.clone()).await?;
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Return single response
        let response = generate_completion(state, completion_request, user_id).await?;
        Ok(Json(response).into_response())
    }
}

/// Chat completion endpoint
pub async fn chat(
    State(state): State<ServerState>,
    Json(chat_request): Json<ChatCompletionRequest>,
) -> Result<Response, crate::error::ServerError> {
    // TODO: Get auth context from middleware
    let user_id = "default-user".to_string(); // Placeholder
    
    if chat_request.stream.unwrap_or(false) {
        // Return streaming response
        let stream = create_chat_stream(state, chat_request, user_id.clone()).await?;
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Return single response
        let response = generate_chat_completion(state, chat_request, user_id).await?;
        Ok(Json(response).into_response())
    }
}

/// Streaming completion endpoint
pub async fn stream(
    State(state): State<ServerState>,
    Json(mut completion_request): Json<CompletionRequest>,
) -> ServerResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    // TODO: Get auth context from middleware
    let user_id = "default-user".to_string(); // Placeholder
    
    // Force streaming
    completion_request.stream = Some(true);
    
    let stream = create_completion_stream(state, completion_request, user_id).await?;
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Generate single completion using real inference
async fn generate_completion(
    state: ServerState,
    request: CompletionRequest,
    _user_id: String,
) -> ServerResult<CompletionResponse> {
    // Validate prompt is not empty
    if request.prompt.trim().is_empty() {
        return Err(crate::error::ServerError::InvalidRequest(
            "Prompt cannot be empty".to_string()
        ));
    }
    
    // Get inference engine
    let engine = state.inference_engine.read().await;
    
    // Create generation configuration
    let generation_config = GenerationConfig {
        max_tokens: request.max_tokens.unwrap_or(100),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k.unwrap_or(50),
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
        stop_sequences: request.stop_sequences.unwrap_or_default(),
        seed: None,
        stream: request.stream.unwrap_or(false),
    };

    // For now, we'll use simple tokenization until GGUF tokenizer is properly integrated
    eprintln!("DEBUG: Using simple tokenization fallback");
    let input_tokens = simple_tokenize(&request.prompt);
    eprintln!("DEBUG: Simple tokenizer produced {} tokens from prompt: '{}'", input_tokens.len(), request.prompt);
    eprintln!("DEBUG: Token IDs: {:?}", input_tokens);
    
    // Run inference through the engine
    let logits = engine.infer(&input_tokens, None).await
        .map_err(|e| crate::error::ServerError::Core(e))?;
    
    eprintln!("DEBUG: Generated {} logits", logits.len());
    
    // Get model info before dropping the engine guard
    let vocab_size = engine.model_info().map(|info| info.vocab_size).unwrap_or(49159);
    drop(engine); // Release the read lock
    
    // Generate text iteratively
    let response_text = generate_text_iteratively(
        &state.inference_engine,
        input_tokens.clone(),
        &generation_config,
        vocab_size
    ).await?;
    
    let completion_tokens = estimate_tokens(&response_text);
    
    Ok(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: "woolly-model".to_string(),
        choices: vec![Choice {
            index: 0,
            text: Some(response_text),
            message: None,
            finish_reason: "stop".to_string(),
            logprobs: None,
        }],
        usage: Usage {
            prompt_tokens: input_tokens.len(),
            completion_tokens,
            total_tokens: input_tokens.len() + completion_tokens,
        },
    })
}

/// Generate chat completion
async fn generate_chat_completion(
    state: ServerState,
    request: ChatCompletionRequest,
    _user_id: String,
) -> ServerResult<CompletionResponse> {
    // Convert chat messages to prompt
    let prompt = format_chat_messages(&request.messages);
    
    // Create completion request
    let completion_request = CompletionRequest {
        prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        repetition_penalty: request.repetition_penalty,
        stop_sequences: request.stop_sequences,
        stream: Some(false),
    };

    let mut response = generate_completion(state, completion_request, _user_id).await?;
    
    // Convert to chat format
    if let Some(choice) = response.choices.first_mut() {
        let content = choice.text.take().unwrap_or_default();
        choice.message = Some(ChatMessage {
            role: "assistant".to_string(),
            content,
        });
    }
    
    response.object = "chat.completion".to_string();
    Ok(response)
}

/// Create completion stream
async fn create_completion_stream(
    _state: ServerState,
    _request: CompletionRequest,
    _user_id: String,
) -> ServerResult<impl Stream<Item = Result<Event, Infallible>>> {
    // For now, create a mock stream
    let id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp() as u64;
    
    let words = vec!["This", "is", "a", "mock", "streaming", "response", "for", "testing"];
    
    let id_for_stream = id.clone();
    let stream = stream::iter(words.into_iter().enumerate())
        .then(move |(_i, word)| {
            let id = id_for_stream.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                
                let chunk = StreamingChunk {
                    id: id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: "woolly-model".to_string(),
                    choices: vec![StreamingChoice {
                        index: 0,
                        delta: Delta {
                            content: Some(format!("{} ", word)),
                            role: None,
                        },
                        finish_reason: None,
                    }],
                };
                
                Ok(Event::default().json_data(chunk).unwrap_or_else(|_| Event::default()))
            }
        });
    
    let id_clone = id.clone();
    let stream = stream.chain(stream::once(async move {
            // Final chunk
            let chunk = StreamingChunk {
                id: id_clone,
                object: "text_completion".to_string(),
                created,
                model: "woolly-model".to_string(),
                choices: vec![StreamingChoice {
                    index: 0,
                    delta: Delta {
                        content: None,
                        role: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            
            Ok(Event::default().json_data(chunk).unwrap_or_else(|_| Event::default()))
        }));

    Ok(stream)
}

/// Create chat stream
async fn create_chat_stream(
    state: ServerState,
    request: ChatCompletionRequest,
    _user_id: String,
) -> ServerResult<impl Stream<Item = Result<Event, Infallible>>> {
    // Convert to completion request
    let prompt = format_chat_messages(&request.messages);
    let completion_request = CompletionRequest {
        prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        repetition_penalty: request.repetition_penalty,
        stop_sequences: request.stop_sequences,
        stream: Some(true),
    };

    create_completion_stream(state, completion_request, _user_id).await
}

/// Format chat messages into a single prompt
fn format_chat_messages(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Estimate token count (very rough approximation)
#[allow(dead_code)]
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Generate text iteratively, one token at a time with optimized inference
async fn generate_text_iteratively(
    engine: &tokio::sync::RwLock<woolly_core::engine::InferenceEngine>,
    mut token_ids: Vec<u32>,
    config: &GenerationConfig,
    vocab_size: usize,
) -> ServerResult<String> {
    let mut generated_tokens = Vec::new();
    let initial_len = token_ids.len();
    
    eprintln!("DEBUG: Starting optimized iterative generation with {} initial tokens", initial_len);
    
    // For the first token, we need to process the full sequence
    // For subsequent tokens, we only need to process the new token (thanks to KV cache)
    let mut is_first_token = true;
    
    for i in 0..config.max_tokens {
        let start_time = std::time::Instant::now();
        
        // Run inference - for subsequent tokens, only process the last token
        let engine = engine.read().await;
        let input_for_inference = if is_first_token {
            &token_ids[..] // Full sequence for first token
        } else {
            &token_ids[token_ids.len()-1..] // Only new token for subsequent tokens
        };
        
        eprintln!("DEBUG: Iteration {}: Processing {} tokens", i, input_for_inference.len());
        
        let logits = engine.infer(input_for_inference, None).await
            .map_err(|e| crate::error::ServerError::Core(e))?;
        drop(engine); // Release lock early
        
        // Generate next token from logits
        let next_token_id = sample_next_token(&logits, input_for_inference.len(), config, vocab_size)?;
        
        let elapsed = start_time.elapsed();
        eprintln!("DEBUG: Iteration {}: Generated token {} in {:.2}s", i, next_token_id, elapsed.as_secs_f32());
        
        // Add to sequence
        token_ids.push(next_token_id);
        generated_tokens.push(next_token_id);
        is_first_token = false;
        
        // Check for EOS token (common EOS tokens: 0, 1, 2)
        if next_token_id < 3 {
            eprintln!("DEBUG: Hit EOS token {}, stopping generation", next_token_id);
            break;
        }
        
        // For now, limit to prevent infinite loops and demonstrate functionality
        if generated_tokens.len() >= 3 {
            eprintln!("DEBUG: Generated {} tokens, stopping for demo", generated_tokens.len());
            break;
        }
    }
    
    // Convert token IDs to text representation
    let token_str = generated_tokens.iter()
        .map(|id| format!("{}", id))
        .collect::<Vec<_>>()
        .join(", ");
    
    Ok(format!(
        "Generated {} tokens: [{}] (Token decoding pending proper tokenizer integration)",
        generated_tokens.len(),
        token_str
    ))
}

/// Sample next token from logits
fn sample_next_token(
    logits: &[f32],
    seq_len: usize,
    config: &GenerationConfig,
    vocab_size: usize,
) -> ServerResult<u32> {
    // The logits should be [seq_len * vocab_size]
    if logits.len() != seq_len * vocab_size {
        return Err(crate::error::ServerError::InvalidRequest(format!(
            "Logits shape mismatch: expected {}, got {}",
            seq_len * vocab_size,
            logits.len()
        )));
    }
    
    // For autoregressive generation, we'll use the logits from the last token
    let last_token_logits = &logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];
    
    // Apply temperature
    let mut probs = softmax_with_temperature(last_token_logits, config.temperature);
    
    // Apply top-k filtering if specified
    if config.top_k > 0 {
        apply_top_k_filtering(&mut probs, config.top_k);
    }
    
    // Apply top-p (nucleus) filtering if specified
    if config.top_p < 1.0 {
        apply_top_p_filtering(&mut probs, config.top_p);
    }
    
    // Sample token
    let next_token_id = if config.temperature == 0.0 {
        // Greedy sampling
        probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    } else {
        // Random sampling from probability distribution
        sample_from_distribution(&probs)
    };
    
    Ok(next_token_id)
}

/// Generate text from logits using sampling (single step)
#[allow(dead_code)]
async fn generate_text_from_logits(
    logits: &[f32],
    input_tokens: &[u32],
    config: &GenerationConfig,
    vocab_size: usize,
) -> ServerResult<String> {
    let seq_len = input_tokens.len();
    
    // The logits should be [seq_len * vocab_size]
    if logits.len() != seq_len * vocab_size {
        return Err(crate::error::ServerError::InvalidRequest(format!(
            "Logits shape mismatch: expected {}, got {}",
            seq_len * vocab_size,
            logits.len()
        )));
    }
    
    // For autoregressive generation, we'll use the logits from the last token
    let last_token_logits = &logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];
    
    // Apply temperature
    let mut probs = softmax_with_temperature(last_token_logits, config.temperature);
    
    // Apply top-k filtering if specified
    if config.top_k > 0 {
        apply_top_k_filtering(&mut probs, config.top_k);
    }
    
    // Apply top-p (nucleus) filtering if specified
    if config.top_p < 1.0 {
        apply_top_p_filtering(&mut probs, config.top_p);
    }
    
    // Sample token
    let next_token_id = if config.temperature == 0.0 {
        // Greedy sampling
        probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    } else {
        // Random sampling from probability distribution
        sample_from_distribution(&probs)
    };
    
    // For now, just show what token would be generated
    // In a real implementation, we'd decode this token and continue generation
    Ok(format!(
        "Next token ID: {} (from {} possible tokens with temperature={}, top_k={}, top_p={})",
        next_token_id,
        vocab_size,
        config.temperature,
        config.top_k,
        config.top_p
    ))
}

/// Apply softmax with temperature scaling
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let temp = temperature.max(0.01); // Avoid division by zero
    
    // Find max for numerical stability
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Apply temperature and exp
    let exp_values: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_logit) / temp).exp())
        .collect();
    
    // Sum for normalization
    let sum: f32 = exp_values.iter().sum();
    
    // Normalize to get probabilities
    exp_values.iter().map(|&x| x / sum).collect()
}

/// Apply top-k filtering to probabilities
fn apply_top_k_filtering(probs: &mut Vec<f32>, k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    
    // Find the k-th largest probability
    let mut sorted_probs = probs.clone();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted_probs.get(k - 1).copied().unwrap_or(0.0);
    
    // Zero out all probabilities below threshold
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
    
    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Apply top-p (nucleus) filtering to probabilities
fn apply_top_p_filtering(probs: &mut Vec<f32>, p: f32) {
    if p >= 1.0 {
        return;
    }
    
    // Sort probabilities with indices
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed_probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    // Find cutoff where cumulative probability exceeds p
    let mut cumsum = 0.0;
    let mut cutoff_idx = 0;
    for (i, (_, prob)) in indexed_probs.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    // Zero out probabilities not in the top-p
    let top_p_indices: std::collections::HashSet<usize> = indexed_probs
        .iter()
        .take(cutoff_idx)
        .map(|(idx, _)| *idx)
        .collect();
    
    for (i, p) in probs.iter_mut().enumerate() {
        if !top_p_indices.contains(&i) {
            *p = 0.0;
        }
    }
    
    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Sample from a probability distribution
fn sample_from_distribution(probs: &[f32]) -> u32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let sample: f32 = rng.gen();
    
    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > sample {
            return idx as u32;
        }
    }
    
    // Fallback to last token
    (probs.len() - 1) as u32
}

/// Simple tokenization for development - converts text to token IDs
/// TODO: Replace with proper tokenizer integration
fn simple_tokenize(text: &str) -> Vec<u32> {
    // Handle empty or whitespace-only input
    let trimmed = text.trim();
    if trimmed.is_empty() {
        // Return a minimal token sequence for empty input
        // Using token ID 1 as a placeholder for empty/padding token
        return vec![1];
    }
    
    // Very basic tokenization: split by whitespace and map to token IDs
    // In a real implementation, this would use a proper tokenizer
    trimmed.split_whitespace()
        .map(|word| {
            // Simple hash function to convert words to token IDs
            let mut hash = 0u32;
            for byte in word.bytes() {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
            }
            // Ensure token ID is within vocabulary range (vocab_size is typically around 50k for modern models)
            // For now, use a conservative range that should work with most models
            hash % 4000 + 1 // Reserve 0 for special tokens, stay well within typical vocab size
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::{AuthConfig, ServerConfig}, server::ServerState};
    use axum::Json;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use woolly_core::engine::InferenceEngine;

    fn create_test_state() -> ServerState {
        let config = Arc::new(ServerConfig::default());
        let engine_config = woolly_core::config::EngineConfig::default();
        let inference_engine = Arc::new(RwLock::new(InferenceEngine::new(engine_config)));
        let auth_config = Arc::new(AuthConfig {
            jwt_secret: "test-secret-key".to_string(),
            jwt_expiration: 3600,
            api_keys: vec!["test-api-key".to_string()],
            allow_anonymous: true,
        });
        
        ServerState {
            config: Arc::clone(&config),
            inference_engine,
            token_manager: Arc::new(crate::auth::TokenManager::new(auth_config)),
            rate_limiter: Arc::new(crate::middleware::RateLimiterState::new(&config.rate_limit)),
            concurrency_limiter: Arc::new(crate::middleware::ConcurrencyLimiter::new(10)),
            #[cfg(feature = "mcp")]
            mcp_state: Arc::new(crate::mcp::McpServerState::new(&config.mcp).unwrap()),
        }
    }

    #[tokio::test]
    async fn test_generate_completion() {
        let state = create_test_state();
        let request = CompletionRequest {
            prompt: "Hello, world!".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            stop_sequences: None,
            stream: Some(false),
        };

        let result = generate_completion(state, request, "test-user".to_string()).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.object, "text_completion");
        assert_eq!(response.model, "woolly-model");
        assert_eq!(response.choices.len(), 1);
        
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert!(choice.text.is_some());
        assert_eq!(choice.finish_reason, "stop");
        
        // Check that response text contains the prompt
        let text = choice.text.as_ref().unwrap();
        assert!(text.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_generate_chat_completion() {
        let state = create_test_state();
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello, how are you?".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "I'm doing well, thank you!".to_string(),
            },
        ];
        
        let request = ChatCompletionRequest {
            messages,
            max_tokens: Some(100),
            temperature: Some(0.8),
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            stop_sequences: None,
            stream: Some(false),
        };

        let result = generate_chat_completion(state, request, "test-user".to_string()).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.model, "woolly-model");
        assert_eq!(response.choices.len(), 1);
        
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert!(choice.message.is_some());
        assert_eq!(choice.finish_reason, "stop");
        
        let message = choice.message.as_ref().unwrap();
        assert_eq!(message.role, "assistant");
        assert!(!message.content.is_empty());
    }

    #[tokio::test]
    async fn test_format_chat_messages() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "What is AI?".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "AI stands for Artificial Intelligence.".to_string(),
            },
        ];

        let formatted = format_chat_messages(&messages);
        let expected = "user: What is AI?\nassistant: AI stands for Artificial Intelligence.";
        assert_eq!(formatted, expected);
    }

    #[tokio::test]
    async fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens("this is a test"), 4);
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("single"), 1);
        // Test with multiple spaces - split_whitespace() removes empty strings
        assert_eq!(estimate_tokens("  multiple   spaces   between  "), 3);
    }

    #[tokio::test]
    async fn test_completion_request_defaults() {
        let request = CompletionRequest {
            prompt: "test".to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            stop_sequences: None,
            stream: None,
        };

        // Test that defaults are handled properly in generation config
        let state = create_test_state();
        let result = generate_completion(state, request, "test-user".to_string()).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(response.usage.prompt_tokens > 0);
        assert!(response.usage.completion_tokens > 0);
        assert_eq!(response.usage.total_tokens, response.usage.prompt_tokens + response.usage.completion_tokens);
    }

    #[tokio::test]
    async fn test_streaming_chunk_structure() {
        let chunk = StreamingChunk {
            id: "test-id".to_string(),
            object: "text_completion".to_string(),
            created: 1234567890,
            model: "test-model".to_string(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: Delta {
                    content: Some("hello".to_string()),
                    role: None,
                },
                finish_reason: None,
            }],
        };

        // Test serialization
        let json = serde_json::to_string(&chunk);
        assert!(json.is_ok());
        
        let json_str = json.unwrap();
        assert!(json_str.contains("test-id"));
        assert!(json_str.contains("hello"));
    }

    #[tokio::test]
    async fn test_empty_prompt_handling() {
        let state = create_test_state();
        
        // Test empty prompt
        let request = CompletionRequest {
            prompt: "".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            stop_sequences: None,
            stream: Some(false),
        };

        let result = generate_completion(state.clone(), request, "test-user".to_string()).await;
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, crate::error::ServerError::InvalidRequest(_)));
            assert!(e.to_string().contains("Prompt cannot be empty"));
        }
        
        // Test whitespace-only prompt
        let request = CompletionRequest {
            prompt: "   \n\t  ".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            stop_sequences: None,
            stream: Some(false),
        };

        let result = generate_completion(state, request, "test-user".to_string()).await;
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, crate::error::ServerError::InvalidRequest(_)));
            assert!(e.to_string().contains("Prompt cannot be empty"));
        }
    }
    
    #[tokio::test]
    async fn test_simple_tokenize_edge_cases() {
        // Test empty string
        let tokens = simple_tokenize("");
        assert_eq!(tokens, vec![1]);
        
        // Test whitespace only
        let tokens = simple_tokenize("   \n\t  ");
        assert_eq!(tokens, vec![1]);
        
        // Test single word
        let tokens = simple_tokenize("hello");
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0] > 0 && tokens[0] <= 30000);
        
        // Test multiple words
        let tokens = simple_tokenize("hello world");
        assert_eq!(tokens.len(), 2);
        assert!(tokens.iter().all(|&t| t > 0 && t <= 30000));
    }
}