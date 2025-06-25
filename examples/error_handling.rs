//! Error Handling Patterns
//!
//! This example demonstrates comprehensive error handling strategies
//! for common scenarios in woolly applications.

use std::fs;
use std::path::Path;
use woolly_core::prelude::*;
use woolly_core::error::{CoreError, ErrorContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Woolly Error Handling Patterns ===\n");

    // Example 1: Basic error handling
    basic_error_handling().await;
    
    // Example 2: Model loading errors
    model_loading_errors().await;
    
    // Example 3: Inference errors
    inference_errors().await;
    
    // Example 4: Resource errors
    resource_errors().await;
    
    // Example 5: Custom error handling with context
    custom_error_context().await;
    
    // Example 6: Error recovery strategies
    error_recovery_strategies().await;
    
    // Example 7: Production error handling
    production_error_handling().await;

    Ok(())
}

/// Example 1: Basic error handling patterns
async fn basic_error_handling() {
    println!("### Example 1: Basic Error Handling ###");
    
    // Pattern 1: Using Result<T, E> with ? operator
    fn load_model_basic(path: &str) -> Result<Model, CoreError> {
        // This will propagate errors automatically
        let model = Model::load(path)?;
        Ok(model)
    }
    
    // Pattern 2: Match on specific error types
    match load_model_basic("nonexistent.gguf") {
        Ok(_) => println!("Model loaded successfully"),
        Err(CoreError::ModelError { code, message, .. }) => {
            println!("Model error ({}): {}", code, message);
        }
        Err(e) => println!("Other error: {}", e),
    }
    
    // Pattern 3: Using map_err to add context
    let result = Model::load("test.gguf")
        .map_err(|e| e.with_context("Failed to load test model"));
    
    if let Err(e) = result {
        println!("Error with context: {}", e);
    }
    
    println!();
}

/// Example 2: Handling model loading errors
async fn model_loading_errors() {
    println!("### Example 2: Model Loading Errors ###");
    
    // Different scenarios that can fail
    let test_cases = vec![
        ("missing_model.gguf", "File not found"),
        ("corrupted.gguf", "Invalid format"),
        ("wrong_version.gguf", "Unsupported version"),
        ("/invalid/path/model.gguf", "Invalid path"),
    ];
    
    for (path, scenario) in test_cases {
        println!("\nTesting {}: ", scenario);
        
        match Model::load(path) {
            Ok(_) => println!("  ✓ Loaded successfully"),
            Err(e) => {
                // Extract detailed error information
                match &e {
                    CoreError::ModelError { code, message, context, suggestion } => {
                        println!("  ✗ Error Code: {}", code);
                        println!("    Message: {}", message);
                        println!("    Context: {}", context);
                        println!("    Suggestion: {}", suggestion);
                    }
                    CoreError::IoError(io_err) => {
                        println!("  ✗ IO Error: {}", io_err);
                        // Handle specific IO errors
                        match io_err.kind() {
                            std::io::ErrorKind::NotFound => {
                                println!("    → Check if file exists at: {}", path);
                            }
                            std::io::ErrorKind::PermissionDenied => {
                                println!("    → Check file permissions");
                            }
                            _ => {}
                        }
                    }
                    _ => println!("  ✗ Unexpected error: {}", e),
                }
            }
        }
    }
    
    // Robust model loading with fallback
    async fn load_model_with_fallback(primary: &str, fallback: &str) -> Result<Model, CoreError> {
        match Model::load(primary) {
            Ok(model) => {
                println!("Loaded primary model: {}", primary);
                Ok(model)
            }
            Err(primary_err) => {
                println!("Primary model failed: {}, trying fallback...", primary_err);
                Model::load(fallback)
                    .map_err(|e| e.with_context(format!("Both models failed. Primary: {}", primary_err)))
            }
        }
    }
    
    let _ = load_model_with_fallback("primary.gguf", "fallback.gguf").await;
    println!();
}

/// Example 3: Handling inference errors
async fn inference_errors() {
    println!("### Example 3: Inference Errors ###");
    
    // Mock model for demonstration
    async fn mock_inference(prompt: &str, max_tokens: usize) -> Result<String, CoreError> {
        // Simulate various error conditions
        if prompt.is_empty() {
            return Err(CoreError::inference(
                "EMPTY_PROMPT",
                "Prompt cannot be empty",
                "Inference request validation",
                "Provide a non-empty prompt"
            ));
        }
        
        if max_tokens > 2048 {
            return Err(CoreError::inference(
                "TOKENS_EXCEEDED",
                "Maximum tokens exceeded",
                format!("Requested {} tokens, max is 2048", max_tokens),
                "Reduce max_tokens to 2048 or less"
            ));
        }
        
        if prompt.len() > 1000 {
            return Err(CoreError::inference(
                "PROMPT_TOO_LONG",
                "Prompt exceeds maximum length",
                format!("Prompt length: {}", prompt.len()),
                "Shorten your prompt to under 1000 characters"
            ));
        }
        
        Ok(format!("Generated response for: {}", prompt))
    }
    
    // Test various error conditions
    let test_cases = vec![
        ("", 100, "Empty prompt"),
        ("Hello", 5000, "Too many tokens"),
        (&"x".repeat(1500), 100, "Prompt too long"),
        ("Valid prompt", 100, "Should succeed"),
    ];
    
    for (prompt, tokens, description) in test_cases {
        println!("\nTest: {}", description);
        match mock_inference(prompt, tokens).await {
            Ok(response) => println!("  ✓ Success: {}", response),
            Err(e) => println!("  ✗ Error: {}", e),
        }
    }
    
    println!();
}

/// Example 4: Handling resource errors
async fn resource_errors() {
    println!("### Example 4: Resource Errors ###");
    
    // Simulate resource constraints
    fn check_resources(required_memory_gb: f32) -> Result<(), CoreError> {
        let available_memory_gb = 8.0; // Mock value
        
        if required_memory_gb > available_memory_gb {
            return Err(CoreError::resource(
                "INSUFFICIENT_MEMORY",
                "Not enough memory available",
                format!("Required: {:.1}GB, Available: {:.1}GB", required_memory_gb, available_memory_gb),
                "Try using a smaller model or quantized version"
            ));
        }
        
        Ok(())
    }
    
    // Test different model sizes
    let model_sizes = vec![
        ("llama-7b-f16", 14.0),
        ("llama-7b-q4", 4.0),
        ("llama-13b-q4", 7.5),
        ("llama-70b-q4", 35.0),
    ];
    
    for (model_name, size_gb) in model_sizes {
        print!("Checking resources for {}: ", model_name);
        match check_resources(size_gb) {
            Ok(_) => println!("✓ Sufficient resources"),
            Err(e) => println!("✗ {}", e),
        }
    }
    
    // Memory pressure handling
    async fn handle_memory_pressure() -> Result<(), CoreError> {
        // Simulate memory pressure detection
        let memory_usage_percent = 95.0;
        
        if memory_usage_percent > 90.0 {
            // Try to free memory
            println!("\nMemory pressure detected ({}%), attempting recovery...", memory_usage_percent);
            
            // Clear caches
            if let Ok(pool) = MemoryPool::try_global() {
                pool.clear();
                println!("  → Cleared memory pool");
            }
            
            // Suggest garbage collection
            println!("  → Suggested: Reduce batch size or use smaller model");
            
            // If still not enough, return error
            if memory_usage_percent > 95.0 {
                return Err(CoreError::resource(
                    "MEMORY_CRITICAL",
                    "Critical memory pressure",
                    format!("Memory usage at {}%", memory_usage_percent),
                    "Restart application or reduce workload"
                ));
            }
        }
        
        Ok(())
    }
    
    let _ = handle_memory_pressure().await;
    println!();
}

/// Example 5: Custom error context
async fn custom_error_context() {
    println!("### Example 5: Custom Error Context ###");
    
    // Chain of operations with context
    async fn process_request(request_id: &str, model_path: &str, prompt: &str) -> Result<String, CoreError> {
        // Add request context to all errors
        let context = ErrorContext::new()
            .with("request_id", request_id)
            .with("model", model_path);
        
        // Load model with context
        let model = Model::load(model_path)
            .map_err(|e| e.with_context_map(context.clone()))?;
        
        // Validate prompt with context
        if prompt.is_empty() {
            return Err(CoreError::validation(
                "EMPTY_PROMPT",
                "Prompt validation failed",
                "No prompt provided",
                "Provide a valid prompt"
            ).with_context_map(context));
        }
        
        // Simulate inference with context
        println!("Processing request {} with model {}", request_id, model_path);
        Ok(format!("Response for request {}", request_id))
    }
    
    match process_request("req-123", "model.gguf", "").await {
        Ok(response) => println!("Success: {}", response),
        Err(e) => {
            println!("Error with full context:");
            println!("{:#}", e); // Pretty print with context
        }
    }
    
    println!();
}

/// Example 6: Error recovery strategies
async fn error_recovery_strategies() {
    println!("### Example 6: Error Recovery Strategies ###");
    
    // Retry with exponential backoff
    async fn retry_with_backoff<F, T>(
        operation: F,
        max_retries: u32,
    ) -> Result<T, CoreError>
    where
        F: Fn() -> Result<T, CoreError>,
    {
        let mut retries = 0;
        let mut delay_ms = 100;
        
        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) if retries < max_retries => {
                    // Check if error is retryable
                    let is_retryable = matches!(e, 
                        CoreError::NetworkError { .. } | 
                        CoreError::ResourceError { code, .. } if code == "TEMPORARY_UNAVAILABLE"
                    );
                    
                    if is_retryable {
                        println!("  Retry {}/{} after {}ms: {}", retries + 1, max_retries, delay_ms, e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                        delay_ms *= 2; // Exponential backoff
                        retries += 1;
                    } else {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }
    
    // Circuit breaker pattern
    struct CircuitBreaker {
        failure_count: std::sync::atomic::AtomicU32,
        threshold: u32,
        is_open: std::sync::atomic::AtomicBool,
    }
    
    impl CircuitBreaker {
        fn new(threshold: u32) -> Self {
            Self {
                failure_count: std::sync::atomic::AtomicU32::new(0),
                threshold,
                is_open: std::sync::atomic::AtomicBool::new(false),
            }
        }
        
        async fn call<F, T>(&self, operation: F) -> Result<T, CoreError>
        where
            F: Fn() -> Result<T, CoreError>,
        {
            // Check if circuit is open
            if self.is_open.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(CoreError::service(
                    "CIRCUIT_OPEN",
                    "Service temporarily unavailable",
                    "Circuit breaker is open due to repeated failures",
                    "Wait for service to recover"
                ));
            }
            
            match operation() {
                Ok(result) => {
                    // Reset failure count on success
                    self.failure_count.store(0, std::sync::atomic::Ordering::Relaxed);
                    Ok(result)
                }
                Err(e) => {
                    let failures = self.failure_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    
                    if failures >= self.threshold {
                        self.is_open.store(true, std::sync::atomic::Ordering::Relaxed);
                        println!("  Circuit breaker opened after {} failures", failures);
                        
                        // Schedule circuit reset
                        let is_open = self.is_open.clone();
                        tokio::spawn(async move {
                            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                            is_open.store(false, std::sync::atomic::Ordering::Relaxed);
                            println!("  Circuit breaker reset");
                        });
                    }
                    
                    Err(e)
                }
            }
        }
    }
    
    // Example usage
    let circuit = CircuitBreaker::new(3);
    
    // Simulate failing operation
    let failing_op = || -> Result<String, CoreError> {
        Err(CoreError::network(
            "CONNECTION_FAILED",
            "Failed to connect to service",
            "Connection timeout",
            "Check network connectivity"
        ))
    };
    
    println!("Testing circuit breaker:");
    for i in 0..5 {
        println!("  Attempt {}: ", i + 1);
        match circuit.call(failing_op).await {
            Ok(_) => println!("    Success"),
            Err(e) => println!("    Failed: {}", e),
        }
    }
    
    println!();
}

/// Example 7: Production error handling
async fn production_error_handling() {
    println!("### Example 7: Production Error Handling ###");
    
    // Structured logging for errors
    fn log_error(error: &CoreError, request_id: &str) {
        // In production, use proper logging framework
        println!("ERROR [{}]: {}", request_id, error);
        
        // Log additional context
        if let Some(source) = error.source() {
            println!("  Caused by: {}", source);
        }
        
        // Log stack trace in debug mode
        #[cfg(debug_assertions)]
        {
            println!("  Stack trace: {:?}", error);
        }
    }
    
    // User-friendly error responses
    fn user_friendly_error(error: &CoreError) -> String {
        match error {
            CoreError::ModelError { suggestion, .. } => {
                format!("Model loading failed. {}", suggestion)
            }
            CoreError::InferenceError { message, .. } => {
                format!("Generation failed: {}", message)
            }
            CoreError::ResourceError { .. } => {
                "Server is temporarily at capacity. Please try again later.".to_string()
            }
            CoreError::ValidationError { message, .. } => {
                format!("Invalid request: {}", message)
            }
            _ => "An unexpected error occurred. Please try again.".to_string()
        }
    }
    
    // Error metrics collection
    struct ErrorMetrics {
        total_errors: std::sync::atomic::AtomicU64,
        errors_by_type: std::sync::Mutex<std::collections::HashMap<String, u64>>,
    }
    
    impl ErrorMetrics {
        fn new() -> Self {
            Self {
                total_errors: std::sync::atomic::AtomicU64::new(0),
                errors_by_type: std::sync::Mutex::new(std::collections::HashMap::new()),
            }
        }
        
        fn record_error(&self, error: &CoreError) {
            self.total_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            let error_type = match error {
                CoreError::ModelError { .. } => "model",
                CoreError::InferenceError { .. } => "inference",
                CoreError::ResourceError { .. } => "resource",
                CoreError::ValidationError { .. } => "validation",
                _ => "other",
            };
            
            if let Ok(mut map) = self.errors_by_type.lock() {
                *map.entry(error_type.to_string()).or_insert(0) += 1;
            }
        }
        
        fn report(&self) {
            let total = self.total_errors.load(std::sync::atomic::Ordering::Relaxed);
            println!("Error Metrics:");
            println!("  Total errors: {}", total);
            
            if let Ok(map) = self.errors_by_type.lock() {
                println!("  Errors by type:");
                for (error_type, count) in map.iter() {
                    println!("    {}: {}", error_type, count);
                }
            }
        }
    }
    
    // Simulate production error handling
    let metrics = ErrorMetrics::new();
    let request_id = "prod-req-456";
    
    // Simulate various errors
    let errors = vec![
        CoreError::model("NOT_FOUND", "Model not found", "Loading model.gguf", "Check model path"),
        CoreError::inference("TIMEOUT", "Inference timeout", "Processing request", "Reduce max tokens"),
        CoreError::resource("OOM", "Out of memory", "Allocating tensors", "Use smaller model"),
    ];
    
    for error in errors {
        log_error(&error, request_id);
        println!("  User message: {}", user_friendly_error(&error));
        metrics.record_error(&error);
        println!();
    }
    
    metrics.report();
}