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
use woolly_core::generation::GenerationConfig;

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
pub async fn complete(
    State(state): State<ServerState>,
    Json(completion_request): Json<CompletionRequest>,
) -> ServerResult<Response> {
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
) -> ServerResult<Response> {
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

/// Generate single completion
async fn generate_completion(
    state: ServerState,
    request: CompletionRequest,
    _user_id: String,
) -> ServerResult<CompletionResponse> {
    // Get inference engine
    let _engine = state.inference_engine.read().await;
    
    // TODO: Implement actual tokenization and inference
    // This is a placeholder implementation
    let _generation_config = GenerationConfig {
        max_tokens: request.max_tokens.unwrap_or(100),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k.unwrap_or(50),
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
    };

    // For now, return a mock response
    let response_text = format!("Generated response for: {}", request.prompt);
    
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
            prompt_tokens: estimate_tokens(&request.prompt),
            completion_tokens: 10, // Mock value
            total_tokens: estimate_tokens(&request.prompt) + 10,
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
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
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
}