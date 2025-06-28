//! Vocabulary management for tokenizers
//!
//! This module provides vocabulary loading, management, and lookup functionality
//! that can be shared across different tokenizer implementations.

use crate::{CoreError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Vocabulary container for managing token-to-id and id-to-token mappings
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    
    /// Special tokens
    special_tokens: HashMap<String, u32>,
    
    /// Unknown token ID
    unk_token_id: Option<u32>,
    
    /// Beginning of sequence token ID
    bos_token_id: Option<u32>,
    
    /// End of sequence token ID
    eos_token_id: Option<u32>,
    
    /// Padding token ID
    pad_token_id: Option<u32>,
}

impl Vocabulary {
    /// Create a new empty vocabulary
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
        }
    }

    /// Create vocabulary with special tokens
    pub fn with_special_tokens(
        unk_token: Option<&str>,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        pad_token: Option<&str>,
    ) -> Self {
        let mut vocab = Self::new();
        let mut current_id = 0u32;

        // Add special tokens
        if let Some(token) = unk_token {
            vocab.add_special_token(token, current_id);
            vocab.unk_token_id = Some(current_id);
            current_id += 1;
        }

        if let Some(token) = bos_token {
            vocab.add_special_token(token, current_id);
            vocab.bos_token_id = Some(current_id);
            current_id += 1;
        }

        if let Some(token) = eos_token {
            vocab.add_special_token(token, current_id);
            vocab.eos_token_id = Some(current_id);
            current_id += 1;
        }

        if let Some(token) = pad_token {
            vocab.add_special_token(token, current_id);
            vocab.pad_token_id = Some(current_id);
        }

        vocab
    }

    /// Load vocabulary from a text file
    /// Format: one token per line, ID is the line number
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            CoreError::tokenizer(
                "TOKENIZER_VOCAB_FILE_OPEN_ERROR",
                format!("Failed to open vocabulary file: {}", e),
                "vocabulary file loading",
                "Check file path and permissions"
            )
        })?;

        let reader = BufReader::new(file);
        let mut vocab = Self::new();

        for (idx, line) in reader.lines().enumerate() {
            let token = line.map_err(|e| {
                CoreError::tokenizer(
                    "TOKENIZER_VOCAB_LINE_READ_ERROR",
                    format!("Failed to read vocabulary line: {}", e),
                    "vocabulary file parsing",
                    "Check file format and encoding"
                )
            })?;

            let id = idx as u32;
            vocab.add_token(token, id);
        }

        Ok(vocab)
    }

    /// Load vocabulary from JSON format
    pub fn from_json(json_str: &str) -> Result<Self> {
        let token_map: HashMap<String, u32> = serde_json::from_str(json_str)
            .map_err(|e| CoreError::tokenizer(
                "TOKENIZER_VOCAB_JSON_PARSE_ERROR",
                format!("Failed to parse JSON vocabulary: {}", e),
                "JSON vocabulary file parsing",
                "Check JSON format and syntax"
            ))?;

        let mut vocab = Self::new();
        for (token, id) in token_map {
            vocab.add_token(token, id);
        }

        Ok(vocab)
    }

    /// Add a token to the vocabulary
    pub fn add_token(&mut self, token: String, id: u32) {
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
    }

    /// Add a special token
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_string(), id);
        self.add_token(token.to_string(), id);
    }

    /// Get token ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token by ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Check if a token is special
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    /// Check if an ID corresponds to a special token
    pub fn is_special_id(&self, id: u32) -> bool {
        if let Some(token) = self.id_to_token(id) {
            self.is_special_token(token)
        } else {
            false
        }
    }

    /// Get all special tokens
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// Get unknown token ID
    pub fn unk_token_id(&self) -> Option<u32> {
        self.unk_token_id
    }

    /// Get beginning of sequence token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get end of sequence token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Get padding token ID
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    /// Convert tokens to IDs, using unknown token for OOV
    pub fn tokens_to_ids(&self, tokens: &[&str]) -> Vec<u32> {
        tokens
            .iter()
            .map(|token| {
                self.token_to_id(token)
                    .or(self.unk_token_id)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Convert IDs to tokens
    pub fn ids_to_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .map(|&id| {
                self.id_to_token(id)
                    .unwrap_or("<unk>")
                    .to_string()
            })
            .collect()
    }

    /// Merge with another vocabulary
    pub fn merge(&mut self, other: &Vocabulary) {
        for (token, &id) in &other.token_to_id {
            if !self.token_to_id.contains_key(token) {
                self.add_token(token.clone(), id);
            }
        }

        for (token, &id) in &other.special_tokens {
            if !self.special_tokens.contains_key(token) {
                self.add_special_token(token, id);
            }
        }
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating vocabularies
#[allow(dead_code)]
pub struct VocabularyBuilder {
    tokens: Vec<String>,
    special_tokens: HashMap<String, u32>,
    unk_token: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    pad_token: Option<String>,
}

impl VocabularyBuilder {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            special_tokens: HashMap::new(),
            unk_token: None,
            bos_token: None,
            eos_token: None,
            pad_token: None,
        }
    }

    pub fn add_token(mut self, token: String) -> Self {
        self.tokens.push(token);
        self
    }

    pub fn add_tokens(mut self, tokens: Vec<String>) -> Self {
        self.tokens.extend(tokens);
        self
    }

    pub fn set_unk_token(mut self, token: String) -> Self {
        self.unk_token = Some(token);
        self
    }

    pub fn set_bos_token(mut self, token: String) -> Self {
        self.bos_token = Some(token);
        self
    }

    pub fn set_eos_token(mut self, token: String) -> Self {
        self.eos_token = Some(token);
        self
    }

    pub fn set_pad_token(mut self, token: String) -> Self {
        self.pad_token = Some(token);
        self
    }

    pub fn build(self) -> Vocabulary {
        let mut vocab = Vocabulary::with_special_tokens(
            self.unk_token.as_deref(),
            self.bos_token.as_deref(),
            self.eos_token.as_deref(),
            self.pad_token.as_deref(),
        );

        let mut current_id = vocab.size() as u32;

        for token in self.tokens {
            if !vocab.token_to_id.contains_key(&token) {
                vocab.add_token(token, current_id);
                current_id += 1;
            }
        }

        vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::with_special_tokens(
            Some("<unk>"),
            Some("<s>"),
            Some("</s>"),
            Some("<pad>"),
        );

        assert_eq!(vocab.size(), 4);
        assert_eq!(vocab.token_to_id("<unk>"), Some(0));
        assert_eq!(vocab.token_to_id("<s>"), Some(1));
        assert_eq!(vocab.token_to_id("</s>"), Some(2));
        assert_eq!(vocab.token_to_id("<pad>"), Some(3));
    }

    #[test]
    fn test_vocabulary_builder() {
        let vocab = VocabularyBuilder::new()
            .set_unk_token("<unk>".to_string())
            .set_bos_token("<s>".to_string())
            .set_eos_token("</s>".to_string())
            .add_tokens(vec![
                "hello".to_string(),
                "world".to_string(),
                "test".to_string(),
            ])
            .build();

        assert_eq!(vocab.size(), 6); // 3 special + 3 regular
        assert!(vocab.is_special_token("<unk>"));
        assert!(!vocab.is_special_token("hello"));
    }

    #[test]
    fn test_token_conversion() {
        let vocab = VocabularyBuilder::new()
            .set_unk_token("<unk>".to_string())
            .add_tokens(vec!["hello".to_string(), "world".to_string()])
            .build();

        let tokens = vec!["hello", "world", "unknown"];
        let ids = vocab.tokens_to_ids(&tokens);
        assert_eq!(ids, vec![1, 2, 0]); // unknown maps to unk_token_id

        let decoded = vocab.ids_to_tokens(&ids);
        assert_eq!(decoded, vec!["hello", "world", "<unk>"]);
    }
}