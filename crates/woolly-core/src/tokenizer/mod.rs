//! Tokenizer module for text tokenization
//!
//! This module provides tokenizer implementations for converting text to tokens
//! and vice versa, supporting various tokenization algorithms like SentencePiece and BPE.

pub mod bpe;
pub mod gguf_tokenizer;
pub mod sentencepiece;
pub mod vocab;

use crate::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;

/// Main tokenizer trait that all tokenizer implementations must implement
#[async_trait]
pub trait Tokenizer: Send + Sync {
    /// Encode text into a sequence of token IDs
    async fn encode(&self, text: &str) -> Result<Vec<u32>>;
    
    /// Encode text with special tokens handling
    async fn encode_with_special_tokens(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>>;
    
    /// Decode a sequence of token IDs back into text
    async fn decode(&self, tokens: &[u32]) -> Result<String>;
    
    /// Decode a sequence of token IDs, skipping special tokens
    async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String>;
    
    /// Get the vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get the beginning of sequence token ID
    fn bos_token_id(&self) -> Option<u32>;
    
    /// Get the end of sequence token ID
    fn eos_token_id(&self) -> Option<u32>;
    
    /// Get the padding token ID
    fn pad_token_id(&self) -> Option<u32>;
    
    /// Get the unknown token ID
    fn unk_token_id(&self) -> Option<u32>;
    
    /// Get token string by ID
    fn id_to_token(&self, id: u32) -> Option<&str>;
    
    /// Get token ID by string
    fn token_to_id(&self, token: &str) -> Option<u32>;
    
    /// Check if a token ID is a special token
    fn is_special_token(&self, id: u32) -> bool;
    
    /// Get all special tokens
    fn special_tokens(&self) -> &HashMap<String, u32>;
}

/// Configuration for tokenizer initialization
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Path to vocabulary file
    pub vocab_path: Option<String>,
    
    /// Path to merges file (for BPE)
    pub merges_path: Option<String>,
    
    /// Path to model file (for SentencePiece)
    pub model_path: Option<String>,
    
    /// Add prefix space for first word
    pub add_prefix_space: bool,
    
    /// Treat whitespace as part of token
    pub continuing_subword_prefix: Option<String>,
    
    /// End of word suffix
    pub end_of_word_suffix: Option<String>,
    
    /// Unknown token
    pub unk_token: Option<String>,
    
    /// Beginning of sequence token
    pub bos_token: Option<String>,
    
    /// End of sequence token
    pub eos_token: Option<String>,
    
    /// Padding token
    pub pad_token: Option<String>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_path: None,
            merges_path: None,
            model_path: None,
            add_prefix_space: true,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            unk_token: Some("<unk>".to_string()),
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            pad_token: Some("<pad>".to_string()),
        }
    }
}

/// Tokenizer type enumeration
#[derive(Debug, Clone, Copy)]
pub enum TokenizerType {
    /// Byte Pair Encoding tokenizer
    BPE,
    /// SentencePiece tokenizer
    SentencePiece,
    /// WordPiece tokenizer
    WordPiece,
    /// GGUF-based tokenizer
    GGUF,
}

/// Factory function to create a tokenizer based on type
pub async fn create_tokenizer(
    tokenizer_type: TokenizerType,
    config: TokenizerConfig,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    match tokenizer_type {
        TokenizerType::BPE => {
            let tokenizer = bpe::BPETokenizer::new(config).await?;
            Ok(Box::new(tokenizer))
        }
        TokenizerType::SentencePiece => {
            let tokenizer = sentencepiece::SentencePieceTokenizer::new(config).await?;
            Ok(Box::new(tokenizer))
        }
        TokenizerType::WordPiece => {
            todo!("WordPiece tokenizer not yet implemented")
        }
        TokenizerType::GGUF => {
            if let Some(model_path) = &config.model_path {
                let tokenizer = gguf_tokenizer::GGUFTokenizer::from_gguf_file(model_path, config.clone()).await?;
                Ok(Box::new(tokenizer))
            } else {
                return Err(crate::CoreError::tokenizer(
                    "GGUF_PATH_REQUIRED",
                    "GGUF tokenizer requires model_path in config",
                    "Creating GGUF tokenizer",
                    "Provide the path to a GGUF model file in the tokenizer config"
                ));
            }
        }
    }
}

/// Load tokenizer from model metadata
pub async fn load_tokenizer_from_model(_model_path: &Path) -> Result<Box<dyn Tokenizer>> {
    // This would inspect the model file to determine the tokenizer type
    // and load the appropriate tokenizer
    todo!("Auto-detection of tokenizer type from model not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tokenizer_trait() {
        // Basic test to ensure the trait is properly defined
        // Actual tests will be in the implementation modules
    }
}