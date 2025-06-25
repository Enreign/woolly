//! Custom Tokenizer Example
//!
//! This example demonstrates how to implement and use custom tokenizers with Woolly:
//! 1. Creating a custom BPE tokenizer from scratch
//! 2. Using SentencePiece tokenizer integration
//! 3. Implementing vocabulary management
//! 4. Adding special token handling
//! 5. Comparing different tokenization strategies
//! 6. Integrating with the Woolly inference engine
//!
//! Usage:
//!   cargo run --example custom_tokenizer -- --vocab-file vocab.json --text "Hello, world!"

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json;

use woolly_core::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (vocab_file, text, tokenizer_type) = parse_args(&args)?;
    
    println!("🔤 Woolly Custom Tokenizer Example");
    println!("Tokenizer Type: {:?}", tokenizer_type);
    println!("Text to tokenize: '{}'", text);
    if let Some(ref vocab) = vocab_file {
        println!("Vocabulary file: {}", vocab);
    }
    println!();
    
    // Step 1: Demonstrate different tokenizer implementations
    println!("🔧 Creating tokenizers...");
    
    // Create custom BPE tokenizer
    let bpe_tokenizer = create_custom_bpe_tokenizer().await?;
    println!("✅ Custom BPE tokenizer created");
    
    // Create SentencePiece tokenizer (if vocab file provided)
    let sentencepiece_tokenizer = if let Some(vocab_path) = &vocab_file {
        Some(create_sentencepiece_tokenizer(vocab_path).await?)
    } else {
        None
    };
    if sentencepiece_tokenizer.is_some() {
        println!("✅ SentencePiece tokenizer created");
    }
    
    // Create simple word tokenizer for comparison
    let word_tokenizer = create_word_tokenizer().await?;
    println!("✅ Word tokenizer created");
    
    // Step 2: Compare tokenization results
    println!("\n📊 Tokenization Comparison:");
    
    // BPE tokenization
    println!("\n1️⃣ BPE Tokenizer:");
    demonstrate_tokenizer(&*bpe_tokenizer, &text, "BPE").await?;
    
    // SentencePiece tokenization
    if let Some(ref sp_tokenizer) = sentencepiece_tokenizer {
        println!("\n2️⃣ SentencePiece Tokenizer:");
        demonstrate_tokenizer(&**sp_tokenizer, &text, "SentencePiece").await?;
    }
    
    // Word tokenization
    println!("\n3️⃣ Word Tokenizer:");
    demonstrate_tokenizer(&*word_tokenizer, &text, "Word").await?;
    
    // Step 3: Demonstrate special token handling
    println!("\n🎯 Special Token Handling:");
    demonstrate_special_tokens(&*bpe_tokenizer).await?;
    
    // Step 4: Demonstrate batch processing
    println!("\n📦 Batch Processing:");
    let batch_texts = vec![
        "Hello, world!",
        "This is a test sentence.",
        "Tokenization is fun!",
        "🦙 Woolly rocks!",
    ];
    demonstrate_batch_processing(&*bpe_tokenizer, &batch_texts).await?;
    
    // Step 5: Demonstrate tokenizer statistics
    println!("\n📈 Tokenizer Statistics:");
    demonstrate_tokenizer_stats(&*bpe_tokenizer, &text).await?;
    
    // Step 6: Demonstrate integration with inference engine
    println!("\n🧠 Integration with Inference Engine:");
    demonstrate_inference_integration(&*bpe_tokenizer, &text).await?;
    
    println!("\n✨ Custom tokenizer example completed!");
    Ok(())
}

fn parse_args(args: &[String]) -> Result<(Option<String>, String, TokenizerType), Box<dyn std::error::Error>> {
    let mut vocab_file = None;
    let mut text = "Hello, world! This is a test of the Woolly tokenizer system.".to_string();
    let mut tokenizer_type = TokenizerType::BPE;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab-file" => {
                if i + 1 < args.len() {
                    vocab_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing vocab file path".into());
                }
            }
            "--text" => {
                if i + 1 < args.len() {
                    text = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing text".into());
                }
            }
            "--type" => {
                if i + 1 < args.len() {
                    tokenizer_type = match args[i + 1].as_str() {
                        "bpe" => TokenizerType::BPE,
                        "sentencepiece" => TokenizerType::SentencePiece,
                        "word" => TokenizerType::WordPiece, // Using WordPiece for word tokenizer
                        _ => return Err("Invalid tokenizer type. Use: bpe, sentencepiece, word".into()),
                    };
                    i += 2;
                } else {
                    return Err("Missing tokenizer type".into());
                }
            }
            _ => i += 1,
        }
    }
    
    Ok((vocab_file, text, tokenizer_type))
}

async fn create_custom_bpe_tokenizer() -> Result<Box<dyn Tokenizer>, Box<dyn std::error::Error>> {
    println!("  📝 Building BPE vocabulary...");
    
    // Create a custom BPE tokenizer with a predefined vocabulary
    let config = TokenizerConfig {
        vocab_path: None, // We'll build the vocab programmatically
        merges_path: None,
        model_path: None,
        add_prefix_space: true,
        continuing_subword_prefix: Some("##".to_string()),
        end_of_word_suffix: Some("</w>".to_string()),
        unk_token: Some("<unk>".to_string()),
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        pad_token: Some("<pad>".to_string()),
    };
    
    let tokenizer = CustomBPETokenizer::new(config).await?;
    Ok(Box::new(tokenizer))
}

async fn create_sentencepiece_tokenizer(vocab_path: &str) -> Result<Box<dyn Tokenizer>, Box<dyn std::error::Error>> {
    println!("  📄 Loading SentencePiece model from {}...", vocab_path);
    
    let config = TokenizerConfig {
        model_path: Some(vocab_path.to_string()),
        ..Default::default()
    };
    
    // Note: This would use the actual SentencePiece implementation
    // For now, we'll use a mock implementation
    let tokenizer = MockSentencePieceTokenizer::new(config).await?;
    Ok(Box::new(tokenizer))
}

async fn create_word_tokenizer() -> Result<Box<dyn Tokenizer>, Box<dyn std::error::Error>> {
    println!("  🔤 Creating word-based tokenizer...");
    
    let config = TokenizerConfig::default();
    let tokenizer = WordTokenizer::new(config).await?;
    Ok(Box::new(tokenizer))
}

async fn demonstrate_tokenizer(
    tokenizer: &dyn Tokenizer,
    text: &str,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Encode text
    let tokens = tokenizer.encode(text).await?;
    println!("  🔢 Tokens ({}): {:?}", tokens.len(), tokens);
    
    // Show token strings
    let token_strings: Vec<String> = tokens.iter()
        .map(|&id| tokenizer.id_to_token(id).unwrap_or("<unk>").to_string())
        .collect();
    println!("  📝 Token strings: {:?}", token_strings);
    
    // Decode back to text
    let decoded = tokenizer.decode(&tokens).await?;
    println!("  🔄 Decoded text: '{}'", decoded);
    
    // Check if decode is lossless
    let is_lossless = text.trim() == decoded.trim();
    println!("  ✅ Lossless: {}", if is_lossless { "Yes" } else { "No" });
    
    // Show compression ratio
    let compression_ratio = tokens.len() as f32 / text.len() as f32;
    println!("  📊 Compression ratio: {:.3} tokens/char", compression_ratio);
    
    Ok(())
}

async fn demonstrate_special_tokens(
    tokenizer: &dyn Tokenizer,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Special token IDs:");
    
    if let Some(bos_id) = tokenizer.bos_token_id() {
        println!("    BOS: {} ('{}')", bos_id, tokenizer.id_to_token(bos_id).unwrap_or(""));
    }
    
    if let Some(eos_id) = tokenizer.eos_token_id() {
        println!("    EOS: {} ('{}')", eos_id, tokenizer.id_to_token(eos_id).unwrap_or(""));
    }
    
    if let Some(pad_id) = tokenizer.pad_token_id() {
        println!("    PAD: {} ('{}')", pad_id, tokenizer.id_to_token(pad_id).unwrap_or(""));
    }
    
    if let Some(unk_id) = tokenizer.unk_token_id() {
        println!("    UNK: {} ('{}')", unk_id, tokenizer.id_to_token(unk_id).unwrap_or(""));
    }
    
    // Test encoding with special tokens
    let text_with_special = "Hello, world!";
    let tokens_no_special = tokenizer.encode(text_with_special).await?;
    let tokens_with_special = tokenizer.encode_with_special_tokens(text_with_special, true, true).await?;
    
    println!("  Without special tokens: {:?}", tokens_no_special);
    println!("  With BOS/EOS: {:?}", tokens_with_special);
    
    Ok(())
}

async fn demonstrate_batch_processing(
    tokenizer: &dyn Tokenizer,
    texts: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Processing {} texts...", texts.len());
    
    let mut all_tokens = Vec::new();
    let mut total_tokens = 0;
    
    for (i, text) in texts.iter().enumerate() {
        let tokens = tokenizer.encode(text).await?;
        total_tokens += tokens.len();
        all_tokens.push(tokens);
        println!("    Text {}: '{}' -> {} tokens", i + 1, text, all_tokens[i].len());
    }
    
    println!("  📊 Batch statistics:");
    println!("    Total tokens: {}", total_tokens);
    println!("    Average tokens per text: {:.1}", total_tokens as f32 / texts.len() as f32);
    
    // Find the longest sequence
    let max_length = all_tokens.iter().map(|t| t.len()).max().unwrap_or(0);
    println!("    Max sequence length: {}", max_length);
    
    Ok(())
}

async fn demonstrate_tokenizer_stats(
    tokenizer: &dyn Tokenizer,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = tokenizer.encode(text).await?;
    
    println!("  📈 Vocabulary statistics:");
    println!("    Vocabulary size: {}", tokenizer.vocab_size());
    println!("    Tokens in text: {}", tokens.len());
    println!("    Characters in text: {}", text.len());
    println!("    Average chars per token: {:.2}", text.len() as f32 / tokens.len() as f32);
    
    // Analyze token distribution
    let mut token_counts = HashMap::new();
    for &token_id in &tokens {
        *token_counts.entry(token_id).or_insert(0) += 1;
    }
    
    println!("    Unique tokens used: {}", token_counts.len());
    println!("    Vocabulary coverage: {:.2}%", 
             token_counts.len() as f32 / tokenizer.vocab_size() as f32 * 100.0);
    
    // Show most frequent tokens
    let mut sorted_tokens: Vec<_> = token_counts.into_iter().collect();
    sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1));
    
    println!("    Most frequent tokens:");
    for (token_id, count) in sorted_tokens.iter().take(5) {
        let token_str = tokenizer.id_to_token(*token_id).unwrap_or("<unk>");
        println!("      '{}' ({}): {} times", token_str, token_id, count);
    }
    
    Ok(())
}

async fn demonstrate_inference_integration(
    tokenizer: &dyn Tokenizer,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  🧠 Tokenizer integration with inference:");
    
    // Tokenize input
    let input_tokens = tokenizer.encode_with_special_tokens(text, true, false).await?;
    println!("    Input tokens: {:?}", input_tokens);
    
    // Simulate inference (would use actual model in practice)
    let mut output_tokens = input_tokens.clone();
    // Add some "generated" tokens
    output_tokens.extend_from_slice(&[15, 25, 35, 45]); // Mock generated token IDs
    
    // Add EOS token
    if let Some(eos_id) = tokenizer.eos_token_id() {
        output_tokens.push(eos_id);
    }
    
    println!("    Output tokens (with generated): {:?}", output_tokens);
    
    // Decode the full response
    let full_response = tokenizer.decode(&output_tokens).await?;
    println!("    Full response: '{}'", full_response);
    
    // Decode only the generated part (excluding input and special tokens)
    let generated_start = input_tokens.len();
    let generated_end = if tokenizer.eos_token_id().is_some() {
        output_tokens.len() - 1
    } else {
        output_tokens.len()
    };
    
    if generated_end > generated_start {
        let generated_tokens = &output_tokens[generated_start..generated_end];
        let generated_text = tokenizer.decode(generated_tokens).await?;
        println!("    Generated text only: '{}'", generated_text);
    }
    
    Ok(())
}

// Custom BPE Tokenizer Implementation

#[derive(Clone)]
struct CustomBPETokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    merges: Vec<(String, String, u32)>, // (token1, token2, merge_priority)
    special_tokens: HashMap<String, u32>,
    config: TokenizerConfig,
}

impl CustomBPETokenizer {
    async fn new(config: TokenizerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            special_tokens: HashMap::new(),
            config,
        };
        
        tokenizer.build_vocabulary().await?;
        Ok(tokenizer)
    }
    
    async fn build_vocabulary(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize with special tokens
        let mut next_id = 0;
        
        // Add special tokens first
        let special_token_list = vec![
            ("<pad>", self.config.pad_token.as_deref()),
            ("<unk>", self.config.unk_token.as_deref()),
            ("<s>", self.config.bos_token.as_deref()),
            ("</s>", self.config.eos_token.as_deref()),
        ];
        
        for (default_token, config_token) in special_token_list {
            let token = config_token.unwrap_or(default_token);
            self.add_token(token, next_id);
            self.special_tokens.insert(token.to_string(), next_id);
            next_id += 1;
        }
        
        // Add basic character vocabulary
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        for ch in chars.chars() {
            let token = ch.to_string();
            if !self.vocab.contains_key(&token) {
                self.add_token(&token, next_id);
                next_id += 1;
            }
        }
        
        // Add some common subword patterns for demonstration
        let common_subwords = vec![
            "the", "and", "ing", "ion", "tion", "er", "ed", "ly", "al", "en",
            "##s", "##ed", "##ing", "##er", "##est", "##ly", "##tion", "##al",
            "hello", "world", "test", "example", "woolly", "rust", "token",
        ];
        
        for subword in common_subwords {
            if !self.vocab.contains_key(subword) {
                self.add_token(subword, next_id);
                next_id += 1;
            }
        }
        
        // Build some simple merge rules
        self.build_merges();
        
        Ok(())
    }
    
    fn add_token(&mut self, token: &str, id: u32) {
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }
    
    fn build_merges(&mut self) {
        // Simple merge rules for demonstration
        let merge_pairs = vec![
            ("t", "h", 1000),
            ("e", "r", 1001),
            ("i", "n", 1002),
            ("o", "n", 1003),
            ("a", "l", 1004),
        ];
        
        for (token1, token2, priority) in merge_pairs {
            self.merges.push((token1.to_string(), token2.to_string(), priority));
        }
    }
    
    fn encode_word(&self, word: &str) -> Vec<u32> {
        // Simple character-level fallback for demonstration
        let mut tokens = Vec::new();
        
        for ch in word.chars() {
            let char_str = ch.to_string();
            if let Some(&token_id) = self.vocab.get(&char_str) {
                tokens.push(token_id);
            } else if let Some(&unk_id) = self.special_tokens.get("<unk>") {
                tokens.push(unk_id);
            }
        }
        
        // Apply simple merge rules (very basic implementation)
        self.apply_merges(tokens)
    }
    
    fn apply_merges(&self, mut tokens: Vec<u32>) -> Vec<u32> {
        // This is a simplified merge implementation
        // In a real BPE tokenizer, this would be much more sophisticated
        tokens
    }
    
    fn simple_word_split(&self, text: &str) -> Vec<&str> {
        // Simple whitespace splitting for demonstration
        text.split_whitespace().collect()
    }
}

#[async_trait::async_trait]
impl Tokenizer for CustomBPETokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let words = self.simple_word_split(text);
        let mut tokens = Vec::new();
        
        for word in words {
            let word_tokens = self.encode_word(word);
            tokens.extend(word_tokens);
        }
        
        Ok(tokens)
    }
    
    async fn encode_with_special_tokens(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        if add_bos {
            if let Some(&bos_id) = self.special_tokens.get("<s>") {
                tokens.push(bos_id);
            }
        }
        
        let text_tokens = self.encode(text).await?;
        tokens.extend(text_tokens);
        
        if add_eos {
            if let Some(&eos_id) = self.special_tokens.get("</s>") {
                tokens.push(eos_id);
            }
        }
        
        Ok(tokens)
    }
    
    async fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut result = String::new();
        
        for &token_id in tokens {
            if let Some(token_str) = self.id_to_token.get(&token_id) {
                // Skip special tokens in basic decode
                if !self.is_special_token(token_id) {
                    result.push_str(token_str);
                }
            }
        }
        
        Ok(result)
    }
    
    async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String> {
        let filtered_tokens: Vec<u32> = tokens.iter()
            .copied()
            .filter(|&id| !self.is_special_token(id))
            .collect();
        
        self.decode(&filtered_tokens).await
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<s>").copied()
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("</s>").copied()
    }
    
    fn pad_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<pad>").copied()
    }
    
    fn unk_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<unk>").copied()
    }
    
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    
    fn is_special_token(&self, id: u32) -> bool {
        self.special_tokens.values().any(|&special_id| special_id == id)
    }
    
    fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }
}

// Mock SentencePiece Tokenizer

#[derive(Clone)]
struct MockSentencePieceTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
    config: TokenizerConfig,
}

impl MockSentencePieceTokenizer {
    async fn new(config: TokenizerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
            config,
        };
        
        tokenizer.build_vocabulary().await?;
        Ok(tokenizer)
    }
    
    async fn build_vocabulary(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Mock SentencePiece vocabulary
        let mut next_id = 0;
        
        // Special tokens
        self.add_token("<pad>", next_id); self.special_tokens.insert("<pad>".to_string(), next_id); next_id += 1;
        self.add_token("<unk>", next_id); self.special_tokens.insert("<unk>".to_string(), next_id); next_id += 1;
        self.add_token("<s>", next_id); self.special_tokens.insert("<s>".to_string(), next_id); next_id += 1;
        self.add_token("</s>", next_id); self.special_tokens.insert("</s>".to_string(), next_id); next_id += 1;
        
        // SentencePiece-style subword pieces (with ▁ for spaces)
        let pieces = vec![
            "▁hello", "▁world", "▁this", "▁is", "▁a", "▁test", "▁of", "▁the",
            "▁token", "izer", "▁system", "▁wool", "ly", "▁rock", "s",
            "▁fun", "!", "?", ".", ",", "▁and", "▁or", "▁but",
            "ing", "ed", "er", "est", "ly", "tion", "ness", "ment",
            "a", "e", "i", "o", "u", "n", "r", "t", "l", "s", "d",
        ];
        
        for piece in pieces {
            self.add_token(piece, next_id);
            next_id += 1;
        }
        
        Ok(())
    }
    
    fn add_token(&mut self, token: &str, id: u32) {
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }
}

#[async_trait::async_trait]
impl Tokenizer for MockSentencePieceTokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Mock SentencePiece encoding - convert spaces and split into subwords
        let text_with_spaces = format!("▁{}", text.replace(' ', "▁"));
        let mut tokens = Vec::new();
        
        // Very basic subword segmentation
        let mut i = 0;
        let chars: Vec<char> = text_with_spaces.chars().collect();
        
        while i < chars.len() {
            let mut found = false;
            
            // Try to find the longest matching piece
            for len in (1..=std::cmp::min(10, chars.len() - i)).rev() {
                let piece: String = chars[i..i + len].iter().collect();
                if let Some(&token_id) = self.vocab.get(&piece) {
                    tokens.push(token_id);
                    i += len;
                    found = true;
                    break;
                }
            }
            
            if !found {
                // Use unknown token
                if let Some(&unk_id) = self.special_tokens.get("<unk>") {
                    tokens.push(unk_id);
                }
                i += 1;
            }
        }
        
        Ok(tokens)
    }
    
    async fn encode_with_special_tokens(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        if add_bos {
            if let Some(&bos_id) = self.special_tokens.get("<s>") {
                tokens.push(bos_id);
            }
        }
        
        let text_tokens = self.encode(text).await?;
        tokens.extend(text_tokens);
        
        if add_eos {
            if let Some(&eos_id) = self.special_tokens.get("</s>") {
                tokens.push(eos_id);
            }
        }
        
        Ok(tokens)
    }
    
    async fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut pieces = Vec::new();
        
        for &token_id in tokens {
            if let Some(piece) = self.id_to_token.get(&token_id) {
                if !self.is_special_token(token_id) {
                    pieces.push(piece.clone());
                }
            }
        }
        
        // Join pieces and replace ▁ with spaces
        let result = pieces.join("")
            .replace("▁", " ")
            .trim()
            .to_string();
        
        Ok(result)
    }
    
    async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.decode(tokens).await // Already skips special tokens
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<s>").copied()
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("</s>").copied()
    }
    
    fn pad_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<pad>").copied()
    }
    
    fn unk_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<unk>").copied()
    }
    
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    
    fn is_special_token(&self, id: u32) -> bool {
        self.special_tokens.values().any(|&special_id| special_id == id)
    }
    
    fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }
}

// Simple Word Tokenizer

#[derive(Clone)]
struct WordTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
}

impl WordTokenizer {
    async fn new(_config: TokenizerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
        };
        
        tokenizer.build_vocabulary().await?;
        Ok(tokenizer)
    }
    
    async fn build_vocabulary(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut next_id = 0;
        
        // Special tokens
        self.add_token("<pad>", next_id); self.special_tokens.insert("<pad>".to_string(), next_id); next_id += 1;
        self.add_token("<unk>", next_id); self.special_tokens.insert("<unk>".to_string(), next_id); next_id += 1;
        self.add_token("<s>", next_id); self.special_tokens.insert("<s>".to_string(), next_id); next_id += 1;
        self.add_token("</s>", next_id); self.special_tokens.insert("</s>".to_string(), next_id); next_id += 1;
        
        // Common words
        let words = vec![
            "hello", "world", "this", "is", "a", "test", "of", "the",
            "tokenizer", "system", "woolly", "rocks", "and", "fun",
            "example", "custom", "implementation", "rust", "language",
            "programming", "machine", "learning", "artificial", "intelligence",
        ];
        
        for word in words {
            self.add_token(word, next_id);
            next_id += 1;
        }
        
        // Punctuation
        let punctuation = vec![".", ",", "!", "?", ":", ";", "'", "\"", "-", "_"];
        for punct in punctuation {
            self.add_token(punct, next_id);
            next_id += 1;
        }
        
        Ok(())
    }
    
    fn add_token(&mut self, token: &str, id: u32) {
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }
}

#[async_trait::async_trait]
impl Tokenizer for WordTokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let words: Vec<&str> = text
            .split_whitespace()
            .flat_map(|word| {
                // Simple punctuation splitting
                let mut parts = Vec::new();
                let mut current = String::new();
                
                for ch in word.chars() {
                    if ch.is_alphanumeric() {
                        current.push(ch);
                    } else {
                        if !current.is_empty() {
                            parts.push(current.clone());
                            current.clear();
                        }
                        parts.push(ch.to_string());
                    }
                }
                
                if !current.is_empty() {
                    parts.push(current);
                }
                
                parts
            })
            .collect::<Vec<String>>()
            .iter()
            .map(|s| s.as_str())
            .collect();
        
        let mut tokens = Vec::new();
        
        for word in words {
            let word_lower = word.to_lowercase();
            if let Some(&token_id) = self.vocab.get(&word_lower) {
                tokens.push(token_id);
            } else if let Some(&unk_id) = self.special_tokens.get("<unk>") {
                tokens.push(unk_id);
            }
        }
        
        Ok(tokens)
    }
    
    async fn encode_with_special_tokens(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        if add_bos {
            if let Some(&bos_id) = self.special_tokens.get("<s>") {
                tokens.push(bos_id);
            }
        }
        
        let text_tokens = self.encode(text).await?;
        tokens.extend(text_tokens);
        
        if add_eos {
            if let Some(&eos_id) = self.special_tokens.get("</s>") {
                tokens.push(eos_id);
            }
        }
        
        Ok(tokens)
    }
    
    async fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut words = Vec::new();
        
        for &token_id in tokens {
            if let Some(token_str) = self.id_to_token.get(&token_id) {
                if !self.is_special_token(token_id) {
                    words.push(token_str.clone());
                }
            }
        }
        
        Ok(words.join(" "))
    }
    
    async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.decode(tokens).await
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<s>").copied()
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("</s>").copied()
    }
    
    fn pad_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<pad>").copied()
    }
    
    fn unk_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<unk>").copied()
    }
    
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    
    fn is_special_token(&self, id: u32) -> bool {
        self.special_tokens.values().any(|&special_id| special_id == id)
    }
    
    fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_custom_bpe_tokenizer() {
        let config = TokenizerConfig::default();
        let tokenizer = CustomBPETokenizer::new(config).await.unwrap();
        
        let text = "hello world";
        let tokens = tokenizer.encode(text).await.unwrap();
        assert!(!tokens.is_empty());
        
        let decoded = tokenizer.decode(&tokens).await.unwrap();
        assert!(!decoded.is_empty());
    }

    #[tokio::test]
    async fn test_word_tokenizer() {
        let config = TokenizerConfig::default();
        let tokenizer = WordTokenizer::new(config).await.unwrap();
        
        let text = "hello world test";
        let tokens = tokenizer.encode(text).await.unwrap();
        assert_eq!(tokens.len(), 3);
        
        let decoded = tokenizer.decode(&tokens).await.unwrap();
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
        assert!(decoded.contains("test"));
    }

    #[test]
    fn test_parse_args() {
        let args = vec![
            "program".to_string(),
            "--text".to_string(),
            "test text".to_string(),
            "--type".to_string(),
            "bpe".to_string(),
        ];
        
        let (vocab_file, text, tokenizer_type) = parse_args(&args).unwrap();
        assert_eq!(vocab_file, None);
        assert_eq!(text, "test text");
        assert!(matches!(tokenizer_type, TokenizerType::BPE));
    }
}