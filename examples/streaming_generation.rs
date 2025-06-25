//! Streaming Text Generation Example
//! 
//! This example demonstrates real-time token-by-token text generation
//! with progress indicators, cancellation support, and different streaming strategies.

use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::oneshot;
use woolly_core::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    println!("Loading model...");
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/llama-2-7b-q4_k_m.gguf".to_string());
    
    let model = Model::load(&model_path)?;
    println!("✓ Model loaded: {}", model.name());

    // Create engine with default configuration
    let engine = InferenceEngine::new(InferenceConfig::default())?;
    engine.load_model(model).await?;

    // Create session
    let session_config = SessionConfig::default()
        .max_tokens(500)
        .temperature(0.8);
    
    let session = engine.create_session(session_config).await?;

    // Example 1: Basic streaming with progress
    println!("\n=== Example 1: Basic Streaming ===");
    basic_streaming(&session, "Tell me a short story about a robot:").await?;

    // Example 2: Word-level streaming (groups tokens into words)
    println!("\n\n=== Example 2: Word-Level Streaming ===");
    word_level_streaming(&session, "Explain quantum computing in simple terms:").await?;

    // Example 3: Streaming with cancellation
    println!("\n\n=== Example 3: Cancellable Streaming ===");
    cancellable_streaming(&session, "Count from 1 to 100 slowly:").await?;

    // Example 4: Streaming with metadata (tokens/sec, etc.)
    println!("\n\n=== Example 4: Streaming with Metadata ===");
    streaming_with_metadata(&session, "Write a haiku about programming:").await?;

    Ok(())
}

/// Basic token-by-token streaming
async fn basic_streaming(session: &InferenceSession, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Prompt: {}", prompt);
    println!("Response: ");
    
    let stream = session.stream_text(prompt).await?;
    tokio::pin!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{}", token.text);
                io::stdout().flush()?;
            }
            Err(e) => {
                eprintln!("\nStreaming error: {}", e);
                break;
            }
        }
    }
    println!();
    Ok(())
}

/// Word-level streaming - groups tokens into complete words
async fn word_level_streaming(session: &InferenceSession, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Prompt: {}", prompt);
    println!("Response: ");
    
    let stream = session.stream_text(prompt).await?;
    tokio::pin!(stream);

    let mut word_buffer = String::new();
    
    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                word_buffer.push_str(&token.text);
                
                // Check if we have a complete word (ends with space or punctuation)
                if token.text.chars().any(|c| c.is_whitespace() || c.is_ascii_punctuation()) {
                    print!("{}", word_buffer);
                    io::stdout().flush()?;
                    word_buffer.clear();
                }
            }
            Err(e) => {
                eprintln!("\nStreaming error: {}", e);
                break;
            }
        }
    }
    
    // Print any remaining text
    if !word_buffer.is_empty() {
        print!("{}", word_buffer);
        io::stdout().flush()?;
    }
    println!();
    Ok(())
}

/// Streaming with cancellation support
async fn cancellable_streaming(session: &InferenceSession, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Prompt: {}", prompt);
    println!("Response (will cancel after 2 seconds): ");
    
    // Create cancellation channel
    let (cancel_tx, cancel_rx) = oneshot::channel();
    
    // Create cancellable stream
    let stream = session.stream_text_cancellable(prompt, cancel_rx).await?;
    tokio::pin!(stream);

    // Spawn task to cancel after 2 seconds
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        let _ = cancel_tx.send(());
        println!("\n[Cancelled by timer]");
    });

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{}", token.text);
                io::stdout().flush()?;
            }
            Err(CoreError::Cancelled) => {
                println!("\n[Generation cancelled]");
                break;
            }
            Err(e) => {
                eprintln!("\nStreaming error: {}", e);
                break;
            }
        }
    }
    println!();
    Ok(())
}

/// Streaming with detailed metadata
async fn streaming_with_metadata(session: &InferenceSession, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Prompt: {}", prompt);
    println!("Response: ");
    
    let start_time = Instant::now();
    let mut token_count = 0;
    let mut total_chars = 0;
    
    let stream = session.stream_text_with_metadata(prompt).await?;
    tokio::pin!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(token_data) => {
                print!("{}", token_data.text);
                io::stdout().flush()?;
                
                token_count += 1;
                total_chars += token_data.text.len();
                
                // Print metadata every 10 tokens
                if token_count % 10 == 0 {
                    let elapsed = start_time.elapsed();
                    let tokens_per_sec = token_count as f64 / elapsed.as_secs_f64();
                    eprintln!("\n[Metadata: {} tokens, {:.1} tok/s, logprob: {:.3}]", 
                             token_count, tokens_per_sec, token_data.logprob);
                }
            }
            Err(e) => {
                eprintln!("\nStreaming error: {}", e);
                break;
            }
        }
    }
    
    // Final statistics
    let elapsed = start_time.elapsed();
    let tokens_per_sec = token_count as f64 / elapsed.as_secs_f64();
    
    println!("\n\nGeneration Statistics:");
    println!("  Total tokens: {}", token_count);
    println!("  Total characters: {}", total_chars);
    println!("  Time elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("  Tokens/second: {:.1}", tokens_per_sec);
    
    Ok(())
}

// Custom streaming callback example
#[allow(dead_code)]
async fn custom_callback_streaming(session: &InferenceSession, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a custom callback for token processing
    let callback = Arc::new(move |token: &StreamToken| {
        // Custom processing - e.g., filter profanity, log to file, etc.
        if token.text.to_lowercase().contains("robot") {
            println!("[Found keyword 'robot' with confidence: {:.2}]", token.logprob.exp());
        }
        // Return true to continue, false to stop generation
        true
    });
    
    let stream = session.stream_text_with_callback(prompt, callback).await?;
    tokio::pin!(stream);
    
    while let Some(result) = stream.next().await {
        if let Ok(token) = result {
            print!("{}", token.text);
            io::stdout().flush()?;
        }
    }
    
    Ok(())
}