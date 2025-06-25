//! Byte Pair Encoding (BPE) tokenizer implementation
//!
//! This module implements the BPE algorithm commonly used in models like GPT.

use crate::tokenizer::{Tokenizer, TokenizerConfig};
use crate::{CoreError, Result};
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::vocab::Vocabulary;

/// BPE tokenizer implementation
#[allow(dead_code)]
pub struct BPETokenizer {
    /// Vocabulary
    vocab: Vocabulary,
    
    /// Merge rules (pair -> merged token)
    merges: HashMap<(String, String), String>,
    
    /// Byte to unicode mapping for encoding
    byte_encoder: HashMap<u8, char>,
    
    /// Unicode to byte mapping for decoding
    byte_decoder: HashMap<char, u8>,
    
    /// Pattern for tokenization
    pattern: Regex,
    
    /// Configuration
    config: TokenizerConfig,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    pub async fn new(config: TokenizerConfig) -> Result<Self> {
        // Initialize byte encoder/decoder
        let (byte_encoder, byte_decoder) = Self::create_byte_mappings();
        
        // Default pattern for GPT-style tokenization
        // Simplified pattern without lookahead assertions which aren't supported
        let pattern = Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"
        ).unwrap();

        let mut tokenizer = Self {
            vocab: Vocabulary::new(),
            merges: HashMap::new(),
            byte_encoder,
            byte_decoder,
            pattern,
            config: config.clone(),
        };

        // Load vocabulary if provided
        if let Some(vocab_path) = &config.vocab_path {
            tokenizer.load_vocab(vocab_path).await?;
        }

        // Load merges if provided
        if let Some(merges_path) = &config.merges_path {
            tokenizer.load_merges(merges_path).await?;
        }

        Ok(tokenizer)
    }

    /// Create byte to unicode mappings
    fn create_byte_mappings() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut byte_encoder = HashMap::new();
        let mut byte_decoder = HashMap::new();

        // Create a mapping that avoids control characters
        let mut n = 0;
        for b in 0u8..=255 {
            let c = if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) {
                b as char
            } else {
                // Map to unicode private use area
                char::from_u32(256 + n).unwrap()
            };
            
            byte_encoder.insert(b, c);
            byte_decoder.insert(c, b);
            
            if !(b >= 33 && b <= 126) && !(b >= 161 && b <= 172) && !(b >= 174) {
                n += 1;
            }
        }

        (byte_encoder, byte_decoder)
    }

    /// Load vocabulary from file
    async fn load_vocab(&mut self, path: &str) -> Result<()> {
        let vocab = Vocabulary::from_file(Path::new(path))?;
        self.vocab = vocab;
        Ok(())
    }

    /// Load merge rules from file
    async fn load_merges(&mut self, path: &str) -> Result<()> {
        let file = File::open(path)
            .map_err(|e| CoreError::Tokenizer(format!("Failed to open merges file: {}", e)))?;
        
        let reader = BufReader::new(file);
        
        for (idx, line) in reader.lines().enumerate() {
            // Skip first line if it's a header
            if idx == 0 {
                continue;
            }
            
            let line = line.map_err(|e| {
                CoreError::Tokenizer(format!("Failed to read merge line: {}", e))
            })?;
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let pair = (parts[0].to_string(), parts[1].to_string());
                let merged = format!("{}{}", parts[0], parts[1]);
                self.merges.insert(pair, merged);
            }
        }
        
        Ok(())
    }

    /// Apply BPE merges to a word
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        
        if word.len() == 1 {
            return word;
        }

        loop {
            let mut pairs = Vec::new();
            for i in 0..word.len() - 1 {
                pairs.push((word[i].clone(), word[i + 1].clone()));
            }

            // Find the pair with the lowest merge priority
            let mut min_pair = None;
            for pair in &pairs {
                if self.merges.contains_key(pair) {
                    min_pair = Some(pair.clone());
                    break;
                }
            }

            if min_pair.is_none() {
                break;
            }

            let pair = min_pair.unwrap();
            let merged = self.merges.get(&pair).unwrap().clone();

            // Merge the pair in the word
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                    new_word.push(merged.clone());
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            word = new_word;
        }

        word
    }

    /// Encode text to bytes then to unicode characters
    fn text_to_unicode(&self, text: &str) -> String {
        text.bytes()
            .map(|b| self.byte_encoder[&b])
            .collect()
    }

    /// Decode unicode characters back to bytes then to text
    fn unicode_to_text(&self, unicode: &str) -> Result<String> {
        let bytes: Result<Vec<u8>> = unicode
            .chars()
            .map(|c| {
                self.byte_decoder
                    .get(&c)
                    .copied()
                    .ok_or_else(|| CoreError::Tokenizer {
                        code: "TOKENIZER_INVALID_UNICODE",
                        message: format!("Invalid unicode character: {}", c),
                        context: "BPE tokenizer decoding".to_string(),
                        suggestion: "Check input encoding and data integrity".to_string(),
                        tokenizer_type: Some("BPE".to_string()),
                    })
            })
            .collect();
        
        let bytes = bytes?;
        String::from_utf8(bytes)
            .map_err(|e| CoreError::Tokenizer {
                code: "TOKENIZER_UTF8_DECODE_ERROR",
                message: format!("Failed to decode bytes to UTF-8: {}", e),
                context: "BPE tokenizer decoding".to_string(),
                suggestion: "Ensure input data is valid UTF-8 encoded text".to_string(),
                tokenizer_type: Some("BPE".to_string()),
            })
    }
}

#[async_trait]
impl Tokenizer for BPETokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut token_ids = Vec::new();

        // Split text into pre-tokens using regex
        for mat in self.pattern.find_iter(text) {
            let token = mat.as_str();
            
            // Convert to unicode representation
            let unicode_token = self.text_to_unicode(token);
            
            // Apply BPE
            let bpe_tokens = self.bpe(&unicode_token);
            
            // Convert to IDs
            for bpe_token in bpe_tokens {
                if let Some(id) = self.vocab.token_to_id(&bpe_token) {
                    token_ids.push(id);
                } else if let Some(unk_id) = self.vocab.unk_token_id() {
                    token_ids.push(unk_id);
                }
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
            if let Some(bos_id) = self.vocab.bos_token_id() {
                token_ids.push(bos_id);
            }
        }

        token_ids.extend(self.encode(text).await?);

        if add_eos {
            if let Some(eos_id) = self.vocab.eos_token_id() {
                token_ids.push(eos_id);
            }
        }

        Ok(token_ids)
    }

    async fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();

        for &token_id in tokens {
            if let Some(token) = self.vocab.id_to_token(token_id) {
                text.push_str(token);
            }
        }

        // Convert from unicode representation back to text
        self.unicode_to_text(&text)
    }

    async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();

        for &token_id in tokens {
            if !self.vocab.is_special_id(token_id) {
                if let Some(token) = self.vocab.id_to_token(token_id) {
                    text.push_str(token);
                }
            }
        }

        // Convert from unicode representation back to text
        self.unicode_to_text(&text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.vocab.bos_token_id()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.vocab.eos_token_id()
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.vocab.pad_token_id()
    }

    fn unk_token_id(&self) -> Option<u32> {
        self.vocab.unk_token_id()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.id_to_token(id)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.token_to_id(token)
    }

    fn is_special_token(&self, id: u32) -> bool {
        self.vocab.is_special_id(id)
    }

    fn special_tokens(&self) -> &HashMap<String, u32> {
        self.vocab.special_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bpe_tokenizer_creation() {
        let config = TokenizerConfig::default();
        let tokenizer = BPETokenizer::new(config).await.unwrap();
        
        assert_eq!(tokenizer.byte_encoder.len(), 256);
        assert_eq!(tokenizer.byte_decoder.len(), 256);
    }

    #[test]
    fn test_byte_mappings() {
        let (encoder, decoder) = BPETokenizer::create_byte_mappings();
        
        // Test that all bytes can be encoded and decoded
        for b in 0u8..=255 {
            let c = encoder[&b];
            assert_eq!(decoder[&c], b);
        }
    }

    #[tokio::test]
    async fn test_text_to_unicode_conversion() {
        let config = TokenizerConfig::default();
        let tokenizer = BPETokenizer::new(config).await.unwrap();
        
        let text = "Hello, world!";
        let unicode = tokenizer.text_to_unicode(text);
        let decoded = tokenizer.unicode_to_text(&unicode).unwrap();
        
        assert_eq!(text, decoded);
    }
}