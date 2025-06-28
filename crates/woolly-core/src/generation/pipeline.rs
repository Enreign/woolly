//! Text generation pipeline implementing various sampling strategies

use crate::{
    generation::{GenerationConfig, GenerationResult, FinishReason, GenerationStats},
    tokenizer::Tokenizer,
    CoreError, Result,
};
use std::time::Instant;

/// Text generation pipeline that handles sampling and decoding
pub struct GenerationPipeline {
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl GenerationPipeline {
    /// Create a new generation pipeline
    pub fn new(tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        Self { tokenizer }
    }

    /// Generate text from logits using the specified configuration
    pub async fn generate_from_logits(
        &self,
        logits: &[f32],
        _prompt_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        let start_time = Instant::now();
        let mut generated_tokens = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        
        // Current implementation: simplified greedy decoding
        // TODO: Implement proper sampling strategies (temperature, top-k, top-p)
        
        for _step in 0..config.max_tokens {
            // Find the token with highest logit (greedy decoding)
            let next_token_id = self.sample_token(logits, config)?;
            
            // Check for stop sequences
            generated_tokens.push(next_token_id);
            
            // Decode current sequence to check for stop sequences
            if let Ok(text) = self.tokenizer.decode(&generated_tokens).await {
                if self.contains_stop_sequence(&text, &config.stop_sequences) {
                    finish_reason = FinishReason::StopSequence;
                    break;
                }
            }
            
            // Check for EOS token
            if Some(next_token_id) == self.tokenizer.eos_token_id() {
                finish_reason = FinishReason::EndOfSequence;
                break;
            }
        }
        
        let generation_time = start_time.elapsed();
        let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        };
        
        // Decode generated tokens to text
        let generated_text = self.tokenizer.decode(&generated_tokens).await
            .unwrap_or_else(|_| "[decode_error]".to_string());
        
        let tokens_generated = generated_tokens.len();
        
        Ok(GenerationResult {
            tokens: generated_tokens,
            text: Some(generated_text),
            finish_reason,
            tokens_generated,
            stats: GenerationStats {
                time_to_first_token_ms: 50.0, // Placeholder
                total_time_ms: generation_time.as_millis() as f64,
                tokens_per_second,
                peak_memory_bytes: 0, // TODO: Track memory usage
            },
        })
    }

    /// Sample a token from logits using the configured strategy
    fn sample_token(&self, logits: &[f32], config: &GenerationConfig) -> Result<u32> {
        if logits.is_empty() {
            return Err(CoreError::generation(
                "EMPTY_LOGITS",
                "Cannot sample from empty logits",
                "Token sampling",
                "Check that the model forward pass produces valid logits"
            ));
        }

        if config.temperature == 0.0 {
            // Greedy decoding - return the token with highest logit
            self.greedy_sample(logits)
        } else {
            // Temperature sampling
            self.temperature_sample(logits, config)
        }
    }

    /// Greedy sampling - select token with highest probability
    fn greedy_sample(&self, logits: &[f32]) -> Result<u32> {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| CoreError::generation(
                "SAMPLING_FAILED",
                "Failed to find maximum logit",
                "Greedy sampling",
                "Check that logits contain valid float values"
            ))?;
        
        Ok(max_idx as u32)
    }

    /// Temperature sampling with optional top-k and top-p filtering
    fn temperature_sample(&self, logits: &[f32], config: &GenerationConfig) -> Result<u32> {
        let mut logits = logits.to_vec();
        
        // Apply temperature scaling
        for logit in &mut logits {
            *logit /= config.temperature;
        }
        
        // Apply top-k filtering if specified
        if config.top_k > 0 && config.top_k < logits.len() {
            self.apply_top_k_filtering(&mut logits, config.top_k);
        }
        
        // Apply top-p (nucleus) filtering if specified
        if config.top_p < 1.0 {
            self.apply_top_p_filtering(&mut logits, config.top_p)?;
        }
        
        // Convert logits to probabilities using softmax
        let probabilities = self.softmax(&logits);
        
        // Sample from the probability distribution
        self.sample_from_distribution(&probabilities, config.seed)
    }

    /// Apply top-k filtering to logits
    fn apply_top_k_filtering(&self, logits: &mut [f32], top_k: usize) {
        // Find the k-th largest value
        let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
        sorted_indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        
        // Set all values beyond top-k to negative infinity
        for &idx in sorted_indices.iter().skip(top_k) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    /// Apply top-p (nucleus) filtering to logits
    fn apply_top_p_filtering(&self, logits: &mut [f32], top_p: f32) -> Result<()> {
        // Sort indices by logit values (descending)
        let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
        sorted_indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate softmax for sorted logits
        let sorted_logits: Vec<f32> = sorted_indices.iter().map(|&i| logits[i]).collect();
        let sorted_probs = self.softmax(&sorted_logits);
        
        // Find cumulative probability cutoff
        let mut cumulative_prob = 0.0;
        let mut cutoff_index = sorted_probs.len();
        
        for (i, &prob) in sorted_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= top_p {
                cutoff_index = i + 1;
                break;
            }
        }
        
        // Set logits beyond cutoff to negative infinity
        for &idx in sorted_indices.iter().skip(cutoff_index) {
            logits[idx] = f32::NEG_INFINITY;
        }
        
        Ok(())
    }

    /// Compute softmax of logits
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        
        // Compute exp(logit - max) for numerical stability
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        
        // Compute sum of exponentials
        let sum_exp: f32 = exp_logits.iter().sum();
        
        // Normalize to get probabilities
        if sum_exp > 0.0 {
            exp_logits.iter().map(|&x| x / sum_exp).collect()
        } else {
            // Fallback to uniform distribution
            vec![1.0 / logits.len() as f32; logits.len()]
        }
    }

    /// Sample from a probability distribution
    fn sample_from_distribution(&self, probabilities: &[f32], seed: Option<u64>) -> Result<u32> {
        // Simple sampling implementation
        // TODO: Use proper random number generator with seed support
        
        if probabilities.is_empty() {
            return Err(CoreError::generation(
                "EMPTY_DISTRIBUTION",
                "Cannot sample from empty probability distribution",
                "Probability sampling",
                "Check that probabilities are properly computed"
            ));
        }
        
        // For now, use a simple deterministic approach based on seed
        let random_value = if let Some(seed) = seed {
            // Use seed to generate deterministic "random" value
            ((seed % 1000) as f32) / 1000.0
        } else {
            // Use timestamp for pseudo-randomness
            use std::time::{SystemTime, UNIX_EPOCH};
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            ((timestamp % 1000) as f32) / 1000.0
        };
        
        // Find the token to sample using cumulative distribution
        let mut cumulative_prob = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                return Ok(i as u32);
            }
        }
        
        // Fallback: return last token
        Ok((probabilities.len() - 1) as u32)
    }

    /// Check if text contains any stop sequence
    fn contains_stop_sequence(&self, text: &str, stop_sequences: &[String]) -> bool {
        for stop_seq in stop_sequences {
            if text.contains(stop_seq) {
                return true;
            }
        }
        false
    }

    /// Apply repetition penalty to logits
    pub fn apply_repetition_penalty(
        &self,
        logits: &mut [f32],
        previous_tokens: &[u32],
        penalty: f32,
    ) {
        if penalty == 1.0 {
            return; // No penalty to apply
        }
        
        for &token in previous_tokens {
            let token_idx = token as usize;
            if token_idx < logits.len() {
                if logits[token_idx] > 0.0 {
                    logits[token_idx] /= penalty;
                } else {
                    logits[token_idx] *= penalty;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{TokenizerConfig, vocab::Vocabulary};
    use std::collections::HashMap;

    // Mock tokenizer for testing
    struct MockTokenizer {
        vocab: HashMap<u32, String>,
    }

    #[async_trait::async_trait]
    impl Tokenizer for MockTokenizer {
        async fn encode(&self, _text: &str) -> Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }
        
        async fn encode_with_special_tokens(&self, _text: &str, _add_bos: bool, _add_eos: bool) -> Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }
        
        async fn decode(&self, tokens: &[u32]) -> Result<String> {
            let words: Vec<String> = tokens.iter()
                .filter_map(|&id| self.vocab.get(&id))
                .cloned()
                .collect();
            Ok(words.join(" "))
        }
        
        async fn decode_skip_special_tokens(&self, tokens: &[u32]) -> Result<String> {
            self.decode(tokens).await
        }
        
        fn vocab_size(&self) -> usize { 100 }
        fn bos_token_id(&self) -> Option<u32> { Some(0) }
        fn eos_token_id(&self) -> Option<u32> { Some(1) }
        fn pad_token_id(&self) -> Option<u32> { Some(2) }
        fn unk_token_id(&self) -> Option<u32> { Some(3) }
        fn id_to_token(&self, id: u32) -> Option<&str> { 
            self.vocab.get(&id).map(|s| s.as_str())
        }
        fn token_to_id(&self, _token: &str) -> Option<u32> { None }
        fn is_special_token(&self, id: u32) -> bool { id < 4 }
        fn special_tokens(&self) -> &HashMap<String, u32> { 
            static EMPTY: HashMap<String, u32> = HashMap::new();
            &EMPTY
        }
    }

    #[tokio::test]
    async fn test_greedy_sampling() {
        let mut vocab = HashMap::new();
        vocab.insert(0, "hello".to_string());
        vocab.insert(1, "</s>".to_string());
        vocab.insert(5, "world".to_string());
        
        let tokenizer = Box::new(MockTokenizer { vocab });
        let pipeline = GenerationPipeline::new(tokenizer);
        
        let logits = vec![0.1, 0.2, 0.3, 0.15, 0.1, 0.8]; // Token 5 has highest logit
        let token = pipeline.greedy_sample(&logits).unwrap();
        
        assert_eq!(token, 5);
    }

    #[test]
    fn test_softmax() {
        let mut vocab = HashMap::new();
        vocab.insert(0, "test".to_string());
        
        let tokenizer = Box::new(MockTokenizer { vocab });
        let pipeline = GenerationPipeline::new(tokenizer);
        
        let logits = vec![1.0, 2.0, 3.0];
        let probs = pipeline.softmax(&logits);
        
        // Check that probabilities sum to 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that highest logit gives highest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_top_k_filtering() {
        let mut vocab = HashMap::new();
        vocab.insert(0, "test".to_string());
        
        let tokenizer = Box::new(MockTokenizer { vocab });
        let pipeline = GenerationPipeline::new(tokenizer);
        
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        pipeline.apply_top_k_filtering(&mut logits, 3);
        
        // Only top 3 should remain, others should be -inf
        assert!(logits[4] > logits[3]); // 5.0 > 4.0
        assert!(logits[3] > logits[2]); // 4.0 > 3.0
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[0], f32::NEG_INFINITY);
    }

    #[tokio::test]
    async fn test_generation_with_eos() {
        let mut vocab = HashMap::new();
        vocab.insert(0, "hello".to_string());
        vocab.insert(1, "</s>".to_string()); // EOS token
        vocab.insert(2, "world".to_string());
        
        let tokenizer = Box::new(MockTokenizer { vocab });
        let pipeline = GenerationPipeline::new(tokenizer);
        
        // Logits that would select EOS token (index 1)
        let logits = vec![0.1, 0.9, 0.1]; 
        let config = GenerationConfig {
            max_tokens: 10,
            temperature: 0.0, // Greedy
            ..Default::default()
        };
        
        let result = pipeline.generate_from_logits(&logits, &[], &config).await.unwrap();
        
        assert_eq!(result.finish_reason, FinishReason::EndOfSequence);
        assert_eq!(result.tokens, vec![1]); // Should generate EOS token
    }
}