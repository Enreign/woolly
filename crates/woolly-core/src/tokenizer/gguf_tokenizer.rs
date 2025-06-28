//! GGUF tokenizer implementation
//!
//! This module provides a tokenizer that can load tokenizer data from GGUF model files

use crate::tokenizer::{Tokenizer, TokenizerConfig};
use crate::{CoreError, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use woolly_gguf::{GGUFLoader, GGMLType};
use unicode_normalization::{UnicodeNormalization, is_nfc};

use super::vocab::Vocabulary;

/// GGUF-based tokenizer that loads tokenizer data from GGUF model files
pub struct GGUFTokenizer {
    /// Vocabulary
    vocab: Vocabulary,
    
    /// Configuration
    config: TokenizerConfig,
    
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    
    /// Special token IDs
    special_token_ids: HashMap<String, u32>,
}

impl GGUFTokenizer {
    /// Normalize Unicode text for consistent tokenization
    fn normalize_unicode(&self, text: &str) -> String {
        if is_nfc(text) {
            text.to_string()
        } else {
            text.nfc().collect()
        }
    }
    
    /// Convert text to byte-level representation for BPE-style tokenization
    fn text_to_bytes(&self, text: &str) -> Vec<u8> {
        text.as_bytes().to_vec()
    }
    
    /// Convert bytes back to UTF-8 string, handling invalid sequences gracefully
    fn bytes_to_text(&self, bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).to_string()
    }
    
    /// Check if a token represents a byte-level encoding
    fn is_byte_token(&self, token: &str) -> bool {
        token.starts_with("<0x") && token.ends_with(">") && token.len() == 6
    }
    
    /// Decode a byte token to its byte value
    fn decode_byte_token(&self, token: &str) -> Option<u8> {
        if self.is_byte_token(token) {
            let hex_part = &token[3..5]; // Extract hex part from "<0xXX>"
            u8::from_str_radix(hex_part, 16).ok()
        } else {
            None
        }
    }
    /// Create a new GGUF tokenizer by loading from a GGUF file
    pub async fn from_gguf_file<P: AsRef<Path>>(path: P, config: TokenizerConfig) -> Result<Self> {
        let loader = GGUFLoader::from_path(&path)
            .map_err(|e| CoreError::tokenizer(
                "GGUF_LOAD_FAILED",
                format!("Failed to load GGUF file: {}", e),
                "Loading GGUF tokenizer",
                "Check that the file exists and is a valid GGUF format"
            ))?;

        Self::from_gguf_loader(loader, config).await
    }

    /// Create a new GGUF tokenizer from an existing GGUF loader
    pub async fn from_gguf_loader(loader: GGUFLoader, config: TokenizerConfig) -> Result<Self> {
        let mut vocab = Vocabulary::with_special_tokens(
            config.unk_token.as_deref(),
            config.bos_token.as_deref(),
            config.eos_token.as_deref(),
            config.pad_token.as_deref(),
        );

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut special_token_ids = HashMap::new();

        // Load tokens from GGUF metadata
        if let Ok(tokens_data) = loader.tensor_data("tokenizer.ggml.tokens") {
            let tokens_info = loader.tensor_info("tokenizer.ggml.tokens")
                .ok_or_else(|| CoreError::tokenizer(
                    "TOKENIZER_TOKENS_INFO_MISSING",
                    "Tokenizer tokens tensor info not found",
                    "Loading GGUF tokenizer tokens",
                    "Check that the GGUF file contains tokenizer data"
                ))?;

            // Parse tokens based on tensor type
            let tokens = match tokens_info.ggml_type {
                GGMLType::I32 => {
                    // Tokens stored as string indices - need to load strings separately
                    Self::load_string_tokens(&loader)?
                }
                _ => {
                    return Err(CoreError::tokenizer(
                        "UNSUPPORTED_TOKEN_TYPE",
                        format!("Unsupported token type: {:?}", tokens_info.ggml_type),
                        "Loading GGUF tokenizer tokens",
                        "Only string-based tokens are currently supported"
                    ));
                }
            };

            // Add tokens to vocabulary
            for (id, token) in tokens.into_iter().enumerate() {
                let id = id as u32;
                vocab.add_token(token.clone(), id);
                token_to_id.insert(token.clone(), id);
                id_to_token.insert(id, token);
            }
        } else {
            return Err(CoreError::tokenizer(
                "TOKENIZER_TOKENS_MISSING",
                "Tokenizer tokens not found in GGUF file",
                "Loading GGUF tokenizer",
                "Check that the GGUF file contains tokenizer data"
            ));
        }

        // Load special tokens from metadata with multiple possible key formats
        let metadata = loader.metadata();
        
        // Helper function to try multiple metadata key variants
        let try_get_special_token = |keys: &[&str]| -> Option<u32> {
            for key in keys {
                if let Some(id) = metadata.get_u32(key) {
                    return Some(id);
                }
            }
            None
        };
        
        // Try to get BOS token ID
        let bos_keys = ["tokenizer.ggml.bos_token_id", "bos_token_id", "tokenizer.bos_token_id"];
        if let Some(bos_id) = try_get_special_token(&bos_keys) {
            if let Some(token) = id_to_token.get(&bos_id) {
                special_token_ids.insert("bos".to_string(), bos_id);
                vocab.add_special_token(token, bos_id);
            } else if bos_id < id_to_token.len() as u32 {
                // Token ID exists but not in our loaded tokens - create default
                let bos_token = "<s>".to_string();
                special_token_ids.insert("bos".to_string(), bos_id);
                vocab.add_special_token(&bos_token, bos_id);
                token_to_id.insert(bos_token.clone(), bos_id);
                id_to_token.insert(bos_id, bos_token);
            }
        }
        
        // Try to get EOS token ID
        let eos_keys = ["tokenizer.ggml.eos_token_id", "eos_token_id", "tokenizer.eos_token_id"];
        if let Some(eos_id) = try_get_special_token(&eos_keys) {
            if let Some(token) = id_to_token.get(&eos_id) {
                special_token_ids.insert("eos".to_string(), eos_id);
                vocab.add_special_token(token, eos_id);
            } else if eos_id < id_to_token.len() as u32 {
                let eos_token = "</s>".to_string();
                special_token_ids.insert("eos".to_string(), eos_id);
                vocab.add_special_token(&eos_token, eos_id);
                token_to_id.insert(eos_token.clone(), eos_id);
                id_to_token.insert(eos_id, eos_token);
            }
        }
        
        // Try to get UNK token ID
        let unk_keys = ["tokenizer.ggml.unk_token_id", "unk_token_id", "tokenizer.unk_token_id"];
        if let Some(unk_id) = try_get_special_token(&unk_keys) {
            if let Some(token) = id_to_token.get(&unk_id) {
                special_token_ids.insert("unk".to_string(), unk_id);
                vocab.add_special_token(token, unk_id);
            } else if unk_id < id_to_token.len() as u32 {
                let unk_token = "<unk>".to_string();
                special_token_ids.insert("unk".to_string(), unk_id);
                vocab.add_special_token(&unk_token, unk_id);
                token_to_id.insert(unk_token.clone(), unk_id);
                id_to_token.insert(unk_id, unk_token);
            }
        }
        
        // Try to get PAD token ID
        let pad_keys = ["tokenizer.ggml.pad_token_id", "pad_token_id", "tokenizer.pad_token_id"];
        if let Some(pad_id) = try_get_special_token(&pad_keys) {
            if let Some(token) = id_to_token.get(&pad_id) {
                special_token_ids.insert("pad".to_string(), pad_id);
                vocab.add_special_token(token, pad_id);
            } else if pad_id < id_to_token.len() as u32 {
                let pad_token = "<pad>".to_string();
                special_token_ids.insert("pad".to_string(), pad_id);
                vocab.add_special_token(&pad_token, pad_id);
                token_to_id.insert(pad_token.clone(), pad_id);
                id_to_token.insert(pad_id, pad_token);
            }
        }
        
        // Try to detect special tokens by pattern matching if IDs not found in metadata
        if special_token_ids.is_empty() {
            for (id, token) in &id_to_token {
                match token.as_str() {
                    "<s>" | "<bos>" | "[BOS]" => {
                        special_token_ids.insert("bos".to_string(), *id);
                        vocab.add_special_token(token, *id);
                    }
                    "</s>" | "<eos>" | "[EOS]" => {
                        special_token_ids.insert("eos".to_string(), *id);
                        vocab.add_special_token(token, *id);
                    }
                    "<unk>" | "[UNK]" | "<unknown>" => {
                        special_token_ids.insert("unk".to_string(), *id);
                        vocab.add_special_token(token, *id);
                    }
                    "<pad>" | "[PAD]" | "<padding>" => {
                        special_token_ids.insert("pad".to_string(), *id);
                        vocab.add_special_token(token, *id);
                    }
                    _ => {}
                }
            }
        }

        Ok(Self {
            vocab,
            config,
            token_to_id,
            id_to_token,
            special_token_ids,
        })
    }

    /// Load string tokens from GGUF file
    fn load_string_tokens(loader: &GGUFLoader) -> Result<Vec<String>> {
        let metadata = loader.metadata();
        
        // Try to get token count from vocab size metadata
        let vocab_size = metadata.get_u32("tokenizer.ggml.tokens_length")
            .or_else(|| metadata.get_u32("tokenizer.ggml.vocab_size"))
            .unwrap_or(0) as usize;
        
        if vocab_size == 0 {
            return Err(CoreError::tokenizer(
                "TOKENIZER_EMPTY",
                "No tokens found in GGUF file - vocab_size is 0",
                "Loading GGUF tokenizer tokens",
                "Check that the GGUF file contains a valid tokenizer"
            ));
        }

        // Check if tokens are stored as string array in metadata
        if let Some(tokens_array) = metadata.get("tokenizer.ggml.tokens") {
            match tokens_array {
                woolly_gguf::metadata::MetadataValue::Array(array) => {
                    let mut tokens = Vec::with_capacity(array.len());
                    for value in array {
                        if let woolly_gguf::metadata::MetadataValue::String(token_str) = value {
                            tokens.push(token_str.clone());
                        } else {
                            return Err(CoreError::tokenizer(
                                "TOKENIZER_INVALID_TOKEN_TYPE",
                                "Expected string tokens in tokenizer.ggml.tokens array",
                                "Loading GGUF tokenizer tokens",
                                "Check that the GGUF file format is correct"
                            ));
                        }
                    }
                    return Ok(tokens);
                }
                _ => {
                    // Not an array, fall through to tensor-based loading
                }
            }
        }

        // Try to load tokens from tensor data
        if let Ok(tokens_data) = loader.tensor_data("tokenizer.ggml.tokens") {
            let tokens_info = loader.tensor_info("tokenizer.ggml.tokens")
                .ok_or_else(|| CoreError::tokenizer(
                    "TOKENIZER_TOKENS_INFO_MISSING",
                    "Tokenizer tokens tensor info not found",
                    "Loading GGUF tokenizer tokens",
                    "Check that the GGUF file contains tokenizer data"
                ))?;

            match tokens_info.ggml_type {
                GGMLType::I32 => {
                    // Tokens are stored as string indices - this is complex and needs proper implementation
                    // For now, create basic tokens based on vocab size
                    Self::create_basic_vocab(vocab_size)
                }
                _ => {
                    // Unsupported token storage format
                    Self::create_basic_vocab(vocab_size)
                }
            }
        } else {
            // No tokens tensor found, create basic vocabulary
            Self::create_basic_vocab(vocab_size)
        }
    }
    
    /// Create a basic vocabulary for testing purposes
    fn create_basic_vocab(vocab_size: usize) -> Result<Vec<String>> {
        let mut tokens = Vec::with_capacity(vocab_size);
        
        // Add special tokens first
        tokens.push("<unk>".to_string());     // 0
        tokens.push("<s>".to_string());       // 1 (BOS)
        tokens.push("</s>".to_string());      // 2 (EOS)
        tokens.push("<pad>".to_string());     // 3
        
        // Add some basic ASCII characters and common tokens
        for i in 0..256 {
            if tokens.len() >= vocab_size { break; }
            let ch = char::from(i as u8);
            if ch.is_ascii() && !ch.is_control() {
                tokens.push(ch.to_string());
            } else {
                tokens.push(format!("<0x{:02X}>", i));
            }
        }
        
        // Add word pieces for common patterns
        let common_tokens = [
            " the", " and", " a", " to", " of", " in", " that", " have", 
            " it", " for", " not", " on", " with", " he", " as", " you",
            " do", " at", " this", " but", " his", " by", " from", " they",
            " we", " say", " her", " she", " or", " an", " will", " my",
            " one", " all", " would", " there", " their", " what", " so",
            " up", " out", " if", " about", " who", " get", " which", " go",
            " me", " when", " make", " can", " like", " time", " no", " just",
            " him", " know", " take", " people", " into", " year", " your",
            " good", " some", " could", " them", " see", " other", " than",
            " then", " now", " look", " only", " come", " its", " over",
            " think", " also", " back", " after", " use", " two", " how",
            " our", " work", " first", " well", " way", " even", " new",
            " want", " because", " any", " these", " give", " day", " most",
            " us", "ing", "ed", "er", "est", "ly", "tion", "ness", "ment",
        ];
        
        for token in &common_tokens {
            if tokens.len() >= vocab_size { break; }
            tokens.push(token.to_string());
        }
        
        // Fill remaining slots with numbered tokens
        while tokens.len() < vocab_size {
            tokens.push(format!("<token_{}>", tokens.len()));
        }
        
        Ok(tokens)
    }

    /// Improved tokenization that handles BPE-style patterns
    fn improved_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_pos = 0;
        let chars: Vec<char> = text.chars().collect();
        
        while current_pos < chars.len() {
            let mut best_token = None;
            let mut best_len = 0;
            
            // Try to find the longest matching token starting from current position
            for len in (1..=std::cmp::min(20, chars.len() - current_pos)).rev() {
                let substr: String = chars[current_pos..current_pos + len].iter().collect();
                if self.token_to_id.contains_key(&substr) {
                    best_token = Some(substr);
                    best_len = len;
                    break;
                }
            }
            
            if let Some(token) = best_token {
                tokens.push(token);
                current_pos += best_len;
            } else {
                // Fallback: single character or byte-level encoding
                let ch = chars[current_pos];
                let char_str = ch.to_string();
                
                if self.token_to_id.contains_key(&char_str) {
                    tokens.push(char_str);
                } else {
                    // Try byte-level encoding
                    let bytes = char_str.as_bytes();
                    for &byte in bytes {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if self.token_to_id.contains_key(&byte_token) {
                            tokens.push(byte_token);
                        } else {
                            // Last resort: use unknown token representation
                            tokens.push(format!("<unk:{}>", ch));
                        }
                    }
                }
                current_pos += 1;
            }
        }
        
        tokens
    }
    
    /// Simple space-based tokenization for better text handling
    fn space_aware_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        
        // Split by whitespace but preserve the space information
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            // Add space prefix for words after the first (common BPE pattern)
            let word_with_space = if i > 0 {
                format!(" {}", word)
            } else {
                word.to_string()
            };
            
            // Try the word with space first
            if self.token_to_id.contains_key(&word_with_space) {
                tokens.push(word_with_space);
            } else if self.token_to_id.contains_key(&word.to_string()) {
                // Try without space
                if i > 0 {
                    // Add separate space token if available
                    if self.token_to_id.contains_key(" ") {
                        tokens.push(" ".to_string());
                    }
                }
                tokens.push(word.to_string());
            } else {
                // Split the word into subwords using improved tokenization
                if i > 0 && self.token_to_id.contains_key(" ") {
                    tokens.push(" ".to_string());
                }
                
                let subword_tokens = self.improved_tokenize(word);
                tokens.extend(subword_tokens);
            }
        }
        
        tokens
    }
}

#[async_trait]
impl Tokenizer for GGUFTokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Normalize Unicode first for consistent tokenization
        let normalized_text = self.normalize_unicode(text);
        let tokens = self.space_aware_tokenize(&normalized_text);
        let mut token_ids = Vec::new();
        
        for token in tokens {
            if let Some(&id) = self.token_to_id.get(&token) {
                token_ids.push(id);
            } else if let Some(unk_id) = self.unk_token_id() {
                token_ids.push(unk_id);
            } else {
                // If no unknown token, use token 0 as fallback
                token_ids.push(0);
            }
        }
        
        Ok(token_ids)
    }

    async fn encode_with_special_tokens(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<u32>> {
        let mut token_ids = Vec::new();

        if add_bos {
            if let Some(bos_id) = self.bos_token_id() {
                token_ids.push(bos_id);
            }
        }

        token_ids.extend(self.encode(text).await?);

        if add_eos {
            if let Some(eos_id) = self.eos_token_id() {
                token_ids.push(eos_id);
            }
        }

        Ok(token_ids)
    }

    async fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        let mut byte_buffer = Vec::new();
        
        for &token_id in tokens {
            if let Some(token_str) = self.id_to_token.get(&token_id) {
                // Check if this is a byte-level token
                if let Some(byte_val) = self.decode_byte_token(token_str) {
                    // Accumulate bytes for proper UTF-8 reconstruction
                    byte_buffer.push(byte_val);
                } else {
                    // Flush any accumulated bytes first
                    if !byte_buffer.is_empty() {
                        let decoded_text = self.bytes_to_text(&byte_buffer);
                        text.push_str(&decoded_text);
                        byte_buffer.clear();
                    }
                    
                    // Handle regular tokens
                    if token_str.starts_with(" ") {
                        // Token already contains space prefix, add directly
                        text.push_str(token_str);
                    } else if token_str.starts_with("<") && token_str.ends_with(">") {
                        // Special token, usually skip in normal decoding
                        // But we'll include it for now for debugging
                        text.push_str(token_str);
                    } else {
                        // Regular token
                        if !text.is_empty() && !text.ends_with(' ') && !token_str.chars().next().map_or(false, |c| c.is_ascii_punctuation()) {
                            // Add space before token if needed (basic heuristic)
                            if text.chars().last().unwrap_or(' ').is_alphanumeric() && 
                               token_str.chars().next().unwrap_or(' ').is_alphanumeric() {
                                text.push(' ');
                            }
                        }
                        text.push_str(token_str);
                    }
                }
            } else {
                // Flush any accumulated bytes first
                if !byte_buffer.is_empty() {
                    let decoded_text = self.bytes_to_text(&byte_buffer);
                    text.push_str(&decoded_text);
                    byte_buffer.clear();
                }
                
                // Unknown token ID, add placeholder
                text.push_str(&format!("<UNK:{}>", token_id));
            }
        }
        
        // Flush any remaining bytes
        if !byte_buffer.is_empty() {
            let decoded_text = self.bytes_to_text(&byte_buffer);
            text.push_str(&decoded_text);
        }
        
        Ok(text)
    }

    async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String> {
        // Filter out special tokens first
        let filtered_tokens: Vec<u32> = tokens
            .iter()
            .filter(|&&id| !self.is_special_token(id))
            .copied()
            .collect();
        
        // Use the same improved decoding logic but without special tokens
        self.decode(&filtered_tokens).await
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.special_token_ids.get("bos").copied()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.special_token_ids.get("eos").copied()
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.special_token_ids.get("pad").copied()
    }

    fn unk_token_id(&self) -> Option<u32> {
        self.special_token_ids.get("unk").copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn is_special_token(&self, id: u32) -> bool {
        self.special_token_ids.values().any(|&special_id| special_id == id)
    }

    fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_token_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gguf_tokenizer_basic() {
        // This test would need a real GGUF file to work properly
        // For now, just test the structure
        let config = TokenizerConfig::default();
        // Note: This will fail without a real GGUF file
        // let tokenizer = GGUFTokenizer::from_gguf_file("test.gguf", config).await;
        // assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_space_aware_tokenize() {
        let config = TokenizerConfig::default();
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 1);
        token_to_id.insert(" world".to_string(), 2);
        token_to_id.insert(" ".to_string(), 3);
        
        let tokenizer = GGUFTokenizer {
            vocab: Vocabulary::with_special_tokens(None, None, None, None),
            config,
            token_to_id,
            id_to_token: HashMap::new(),
            special_token_ids: HashMap::new(),
        };

        let tokens = tokenizer.space_aware_tokenize("hello world");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], "hello");
        assert_eq!(tokens[1], " world");
    }

    #[tokio::test]
    async fn test_encode_decode_roundtrip() {
        let config = TokenizerConfig::default();
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut special_token_ids = HashMap::new();
        
        // Set up a small vocabulary
        token_to_id.insert("<unk>".to_string(), 0);
        token_to_id.insert("<s>".to_string(), 1);
        token_to_id.insert("</s>".to_string(), 2);
        token_to_id.insert("hello".to_string(), 3);
        token_to_id.insert(" world".to_string(), 4);
        token_to_id.insert("!".to_string(), 5);
        
        id_to_token.insert(0, "<unk>".to_string());
        id_to_token.insert(1, "<s>".to_string());
        id_to_token.insert(2, "</s>".to_string());
        id_to_token.insert(3, "hello".to_string());
        id_to_token.insert(4, " world".to_string());
        id_to_token.insert(5, "!".to_string());
        
        special_token_ids.insert("unk".to_string(), 0);
        special_token_ids.insert("bos".to_string(), 1);
        special_token_ids.insert("eos".to_string(), 2);
        
        let tokenizer = GGUFTokenizer {
            vocab: Vocabulary::with_special_tokens(
                Some("<unk>"), 
                Some("<s>"), 
                Some("</s>"), 
                None
            ),
            config,
            token_to_id,
            id_to_token,
            special_token_ids,
        };

        let text = "hello world!";
        let token_ids = tokenizer.encode(text).await.unwrap();
        let decoded_text = tokenizer.decode(&token_ids).await.unwrap();
        
        // The decoded text should be close to the original
        assert!(decoded_text.contains("hello"));
        assert!(decoded_text.contains("world"));
    }

    #[test]
    fn test_byte_token_handling() {
        let config = TokenizerConfig::default();
        let tokenizer = GGUFTokenizer {
            vocab: Vocabulary::with_special_tokens(None, None, None, None),
            config,
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_token_ids: HashMap::new(),
        };

        // Test byte token detection
        assert!(tokenizer.is_byte_token("<0x41>"));
        assert!(tokenizer.is_byte_token("<0xFF>"));
        assert!(!tokenizer.is_byte_token("hello"));
        assert!(!tokenizer.is_byte_token("<0x41"));
        assert!(!tokenizer.is_byte_token("0x41>"));

        // Test byte token decoding
        assert_eq!(tokenizer.decode_byte_token("<0x41>"), Some(0x41));
        assert_eq!(tokenizer.decode_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(tokenizer.decode_byte_token("hello"), None);
    }

    #[test]
    fn test_unicode_normalization() {
        let config = TokenizerConfig::default();
        let tokenizer = GGUFTokenizer {
            vocab: Vocabulary::with_special_tokens(None, None, None, None),
            config,
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_token_ids: HashMap::new(),
        };

        let text = "café"; // Contains a composed character
        let normalized = tokenizer.normalize_unicode(text);
        
        // Should remain the same if already NFC
        assert_eq!(text, normalized);
    }

    #[test]
    fn test_special_token_detection() {
        let config = TokenizerConfig::default();
        let mut special_token_ids = HashMap::new();
        special_token_ids.insert("bos".to_string(), 1);
        special_token_ids.insert("eos".to_string(), 2);
        
        let tokenizer = GGUFTokenizer {
            vocab: Vocabulary::with_special_tokens(None, None, None, None),
            config,
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_token_ids,
        };

        assert!(tokenizer.is_special_token(1)); // BOS
        assert!(tokenizer.is_special_token(2)); // EOS
        assert!(!tokenizer.is_special_token(3)); // Regular token
    }

    #[test]
    fn test_create_basic_vocab() {
        let vocab = GGUFTokenizer::create_basic_vocab(1000).unwrap();
        
        assert_eq!(vocab.len(), 1000);
        
        // Check that special tokens are at the beginning
        assert_eq!(vocab[0], "<unk>");
        assert_eq!(vocab[1], "<s>");
        assert_eq!(vocab[2], "</s>");
        assert_eq!(vocab[3], "<pad>");
        
        // Check that some ASCII characters are included
        assert!(vocab.iter().any(|t| t == " "));
        assert!(vocab.iter().any(|t| t == "a"));
        assert!(vocab.iter().any(|t| t == "A"));
        
        // Check that some common tokens are included
        assert!(vocab.iter().any(|t| t == " the"));
        assert!(vocab.iter().any(|t| t == " and"));
    }
}