//! Generation configuration and utilities

pub mod pipeline;

use serde::{Deserialize, Serialize};

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    
    /// Temperature for sampling (0.0 = greedy, 1.0 = standard)
    pub temperature: f32,
    
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    
    /// Top-k sampling parameter
    pub top_k: usize,
    
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    
    /// Stop sequences that end generation
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    
    /// Whether to stream tokens as they're generated
    #[serde(default)]
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            seed: None,
            stream: false,
        }
    }
}

/// Result of a generation operation
#[derive(Debug, Clone, Serialize)]
pub struct GenerationResult {
    /// Generated tokens
    pub tokens: Vec<u32>,
    
    /// Generated text (if decoded)
    pub text: Option<String>,
    
    /// Reason generation stopped
    pub finish_reason: FinishReason,
    
    /// Number of tokens generated
    pub tokens_generated: usize,
    
    /// Generation statistics
    pub stats: GenerationStats,
}

/// Reason why generation stopped
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Reached max_tokens limit
    MaxTokens,
    
    /// Hit a stop sequence
    StopSequence,
    
    /// End of sequence token generated
    EndOfSequence,
    
    /// Error during generation
    Error,
}

/// Statistics about the generation process
#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    /// Time to first token (ms)
    pub time_to_first_token_ms: f64,
    
    /// Total generation time (ms)
    pub total_time_ms: f64,
    
    /// Tokens per second
    pub tokens_per_second: f64,
    
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: u64,
}