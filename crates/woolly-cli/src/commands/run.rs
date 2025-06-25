//! Simplified run command implementation
//!
//! This is a simplified version that works with the current state of the codebase

use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Args;
use console::style;
use serde_json::json;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info};

use crate::commands::Command;
use crate::config::Config;
use crate::utils::{create_spinner, format_duration, print_output, print_success};

use woolly_gguf::GGUFLoader;

#[derive(Args, Debug)]
pub struct RunCommand {
    /// Path to the model file (GGUF format)
    #[arg(short, long)]
    pub model: Option<PathBuf>,

    /// Input text or prompt
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Interactive mode
    #[arg(short, long)]
    pub interactive: bool,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "100")]
    pub max_tokens: usize,

    /// Temperature for sampling (0.0 to 2.0)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Show timing information
    #[arg(long)]
    pub timing: bool,

    /// Dry run - only load and validate model
    #[arg(long)]
    pub dry_run: bool,
}

#[async_trait]
impl Command for RunCommand {
    async fn execute(&self, config: &Config, json_output: bool) -> Result<()> {
        debug!("Executing simplified run command: {:?}", self);

        // Validate arguments first
        self.validate_arguments()
            .context("Command validation failed")?;

        // Determine model path
        let model_path = self.resolve_model_path(config)
            .context("Model path resolution failed")?;
        info!("Using model: {}", model_path.display());

        // Load and validate model
        let start_time = Instant::now();
        let model_info = self.load_and_validate_model(&model_path).await?;
        let load_time = start_time.elapsed();

        if json_output {
            let output = json!({
                "model_path": model_path.to_string_lossy(),
                "model_info": model_info,
                "load_time_ms": load_time.as_millis(),
                "status": "loaded"
            });
            print_output(&output, true)?;
        } else {
            print_success(&format!("Model loaded successfully in {}", format_duration(load_time)));
            self.print_model_info(&model_info)?;
        }

        // If dry run, just exit here
        if self.dry_run {
            return Ok(());
        }

        // Handle interactive mode
        if self.interactive {
            return self.run_interactive_mode().await;
        }

        // Handle single prompt
        if let Some(prompt) = &self.prompt {
            return self.run_single_prompt(prompt).await;
        }

        // No prompt provided, show help
        println!("No prompt provided. Use --prompt for single inference or --interactive for interactive mode.");
        Ok(())
    }
}

impl RunCommand {
    /// Validate command arguments and parameters
    fn validate_arguments(&self) -> Result<()> {
        // Validate temperature range
        if self.temperature < 0.0 || self.temperature > 2.0 {
            anyhow::bail!(
                "Temperature must be between 0.0 and 2.0 (got: {})\n\
                Suggestion: Use 0.0 for deterministic output, 0.7 for balanced creativity, 1.0+ for more randomness",
                self.temperature
            );
        }

        // Validate max_tokens
        if self.max_tokens == 0 {
            anyhow::bail!(
                "max_tokens must be greater than 0 (got: {})\n\
                Suggestion: Use a positive value like 100 for short responses or 1000+ for longer text",
                self.max_tokens
            );
        }

        if self.max_tokens > 10_000 {
            anyhow::bail!(
                "max_tokens is very large ({}) and may cause long generation times\n\
                Suggestion: Use smaller values (100-1000) for better performance",
                self.max_tokens
            );
        }

        // Validate prompt if provided
        if let Some(prompt) = &self.prompt {
            if prompt.trim().is_empty() {
                anyhow::bail!(
                    "Prompt cannot be empty or whitespace only\n\
                    Suggestion: Provide meaningful text for the model to process"
                );
            }

            if prompt.len() > 10_000 {
                anyhow::bail!(
                    "Prompt is very long ({} characters) and may exceed context limits\n\
                    Suggestion: Use shorter prompts (under 2000 characters) for better results",
                    prompt.len()
                );
            }

            // Check for potential problematic characters
            let control_chars: Vec<char> = prompt.chars()
                .filter(|c| c.is_control() && *c != '\n' && *c != '\t' && *c != '\r')
                .collect();
            
            if !control_chars.is_empty() {
                anyhow::bail!(
                    "Prompt contains {} control characters that may cause issues\n\
                    Suggestion: Remove or replace control characters in your prompt",
                    control_chars.len()
                );
            }
        }

        // Validate combination of arguments
        if !self.interactive && self.prompt.is_none() && !self.dry_run {
            anyhow::bail!(
                "No operation specified. Choose one of:\n\
                • Use --prompt 'text' for single inference\n\
                • Use --interactive for interactive mode\n\
                • Use --dry-run to only validate the model"
            );
        }

        if self.interactive && self.prompt.is_some() {
            anyhow::bail!(
                "Cannot use both --interactive and --prompt together\n\
                Suggestion: Choose either interactive mode or provide a single prompt"
            );
        }

        Ok(())
    }

    /// Resolve the model path from arguments or configuration
    fn resolve_model_path(&self, config: &Config) -> Result<PathBuf> {
        let model_path = match &self.model {
            Some(path) => path.clone(),
            None => {
                if let Some(default_model) = &config.default_model {
                    config.find_model(&default_model.to_string_lossy())?
                } else {
                    anyhow::bail!(
                        "No model specified and no default model configured\n\
                        Suggestion: Use --model /path/to/model.gguf or configure a default model"
                    );
                }
            }
        };

        // Validate model path using Woolly's validation
        match woolly_core::validation::Validator::validate_model_path(&model_path) {
            Ok(validated_path) => Ok(validated_path),
            Err(core_err) => {
                // Convert core error to anyhow error with better CLI context
                match core_err.code() {
                    "MODEL_FILE_NOT_FOUND" => {
                        anyhow::bail!(
                            "Model file not found: {}\n\
                            Suggestion: Check the file path and ensure the model file exists",
                            model_path.display()
                        )
                    }
                    "MODEL_UNSUPPORTED_FORMAT" => {
                        anyhow::bail!(
                            "Unsupported model format: {}\n\
                            Suggestion: Use a GGUF (.gguf) model file or convert your model to GGUF format",
                            model_path.display()
                        )
                    }
                    "MODEL_FILE_EMPTY" => {
                        anyhow::bail!(
                            "Model file is empty: {}\n\
                            Suggestion: Re-download the model file or check if the download completed",
                            model_path.display()
                        )
                    }
                    _ => {
                        anyhow::bail!(
                            "Model validation failed: {}\n\
                            Path: {}\n\
                            Suggestion: Ensure the model file is valid and accessible",
                            core_err,
                            model_path.display()
                        )
                    }
                }
            }
        }
    }

    /// Check system resources before loading model
    fn check_system_resources(&self, model_path: &PathBuf) -> Result<()> {
        // Get model file size
        let model_size = std::fs::metadata(model_path)
            .with_context(|| format!("Failed to get model file info: {}", model_path.display()))?
            .len();

        // Estimate memory requirements (model size + overhead)
        let estimated_memory_mb = (model_size as f64 / (1024.0 * 1024.0)) * 1.5; // 50% overhead estimate
        let required_memory_bytes = (estimated_memory_mb * 1024.0 * 1024.0) as u64;

        println!("📊 Resource Check:");
        println!("   Model size: {:.2} MB", model_size as f64 / (1024.0 * 1024.0));
        println!("   Estimated memory needed: {:.2} MB", estimated_memory_mb);

        // Basic memory check using available system calls
        match Self::get_available_memory() {
            Ok(available_mb) => {
                println!("   Available memory: {:.2} MB", available_mb);
                
                if estimated_memory_mb > available_mb {
                    anyhow::bail!(
                        "Insufficient memory: need {:.2} MB but only {:.2} MB available\n\
                        Suggestion: Close other applications or use a smaller model",
                        estimated_memory_mb,
                        available_mb
                    );
                }
                
                if estimated_memory_mb > available_mb * 0.8 {
                    println!("   ⚠️  Warning: High memory usage ({:.1}% of available)", 
                        (estimated_memory_mb / available_mb) * 100.0);
                }
            }
            Err(e) => {
                println!("   ⚠️  Could not check available memory: {}", e);
                println!("   Proceeding without memory validation...");
            }
        }

        // Check disk space for temporary files
        match Self::get_available_disk_space(model_path.parent().unwrap_or(std::path::Path::new("."))) {
            Ok(available_mb) => {
                println!("   Available disk space: {:.2} MB", available_mb);
                
                // Need some space for temporary files during loading
                let required_disk_mb = estimated_memory_mb * 0.1; // 10% of model size
                if available_mb < required_disk_mb {
                    anyhow::bail!(
                        "Insufficient disk space: need {:.2} MB for temporary files but only {:.2} MB available\n\
                        Suggestion: Free up disk space or move model to a different location",
                        required_disk_mb,
                        available_mb
                    );
                }
            }
            Err(e) => {
                println!("   ⚠️  Could not check available disk space: {}", e);
            }
        }

        println!("   ✅ Resource check passed\n");
        Ok(())
    }

    /// Get available system memory in MB (simplified cross-platform approach)
    fn get_available_memory() -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            let meminfo = std::fs::read_to_string("/proc/meminfo")
                .context("Failed to read /proc/meminfo")?;
            
            let available_kb = meminfo
                .lines()
                .find(|line| line.starts_with("MemAvailable:"))
                .and_then(|line| {
                    line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .context("Could not parse MemAvailable from /proc/meminfo")?;
            
            Ok(available_kb as f64 / 1024.0) // Convert KB to MB
        }

        #[cfg(target_os = "macos")]
        {
            // For macOS, use a simplified approach with vm_stat
            let output = std::process::Command::new("vm_stat")
                .output()
                .context("Failed to run vm_stat command")?;
            
            let output_str = String::from_utf8_lossy(&output.stdout);
            
            // Parse page size and free pages (simplified)
            let page_size = 4096; // Typical page size on macOS
            let free_pages = output_str
                .lines()
                .find(|line| line.contains("Pages free:"))
                .and_then(|line| {
                    line.split_whitespace()
                        .last()
                        .and_then(|s| s.trim_end_matches('.').parse::<u64>().ok())
                })
                .context("Could not parse free pages from vm_stat")?;
            
            let free_bytes = free_pages * page_size;
            Ok(free_bytes as f64 / (1024.0 * 1024.0)) // Convert to MB
        }

        #[cfg(target_os = "windows")]
        {
            // For Windows, this would require Windows API calls
            // For now, just return a large value
            anyhow::bail!("Memory checking not implemented for Windows yet");
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            anyhow::bail!("Memory checking not supported on this platform");
        }
    }

    /// Get available disk space in MB
    fn get_available_disk_space(path: &std::path::Path) -> Result<f64> {
        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::mem;
            
            let path_cstr = CString::new(path.to_string_lossy().as_bytes())
                .context("Failed to convert path to CString")?;
            
            let mut statvfs: libc::statvfs = unsafe { mem::zeroed() };
            let result = unsafe { libc::statvfs(path_cstr.as_ptr(), &mut statvfs) };
            
            if result != 0 {
                anyhow::bail!("statvfs call failed");
            }
            
            let available_bytes = statvfs.f_bavail * statvfs.f_frsize;
            Ok(available_bytes as f64 / (1024.0 * 1024.0)) // Convert to MB
        }

        #[cfg(not(unix))]
        {
            anyhow::bail!("Disk space checking not implemented for this platform");
        }
    }

    /// Load and validate model, returning basic info
    async fn load_and_validate_model(&self, model_path: &PathBuf) -> Result<serde_json::Value> {
        // Check system resources first
        self.check_system_resources(model_path)?;
        
        let spinner = create_spinner("Loading and validating model...");

        // Load GGUF file
        let gguf_file = GGUFLoader::from_path(model_path)
            .context("Failed to load GGUF file")?;

        // Extract basic information
        let metadata = gguf_file.metadata();
        let tensors: Vec<_> = gguf_file.tensors().values().collect();

        let model_info = json!({
            "format": "GGUF",
            "version": gguf_file.header().version.0,
            "tensor_count": tensors.len(),
            "total_size_mb": tensors.iter().map(|t| t.data_size()).sum::<u64>() as f64 / (1024.0 * 1024.0),
            "architecture": metadata.get("general.architecture").map(|v| self.metadata_value_to_string(v)).unwrap_or_else(|| "unknown".to_string()),
            "name": metadata.get("general.name").map(|v| self.metadata_value_to_string(v)).unwrap_or_else(|| "unnamed".to_string()),
            "context_length": metadata.get("llama.context_length")
                .or_else(|| metadata.get("context_length"))
                .map(|v| self.metadata_value_to_string(v))
                .unwrap_or_else(|| "unknown".to_string()),
        });

        spinner.finish_with_message("Model validated successfully");
        Ok(model_info)
    }

    /// Print model information in human-readable format
    fn print_model_info(&self, model_info: &serde_json::Value) -> Result<()> {
        println!();
        println!("{}", style("Model Information").bold().cyan());
        println!("─────────────────────────────────────────");
        
        if let Some(name) = model_info.get("name") {
            println!("Name: {}", name.as_str().unwrap_or("unknown"));
        }
        
        if let Some(arch) = model_info.get("architecture") {
            println!("Architecture: {}", arch.as_str().unwrap_or("unknown"));
        }
        
        if let Some(context_length) = model_info.get("context_length") {
            println!("Context Length: {}", context_length.as_str().unwrap_or("unknown"));
        }
        
        println!("Format: {}", model_info.get("format").and_then(|v| v.as_str()).unwrap_or("unknown"));
        println!("Tensors: {}", model_info.get("tensor_count").and_then(|v| v.as_u64()).unwrap_or(0));
        println!("Size: {:.2} MB", model_info.get("total_size_mb").and_then(|v| v.as_f64()).unwrap_or(0.0));
        
        Ok(())
    }

    /// Run in interactive mode
    async fn run_interactive_mode(&self) -> Result<()> {
        println!("{}", style("Woolly Interactive Mode (Simplified)").bold().cyan());
        println!("This is a simplified implementation for demonstration.");
        println!("Type 'exit' to quit, or any text to see a simulated response.");
        println!();

        let mut input = String::new();
        loop {
            print!("{} ", style("User:").bold().green());
            io::stdout().flush()?;
            
            input.clear();
            io::stdin().read_line(&mut input)?;
            let input_text = input.trim();

            if input_text.is_empty() {
                continue;
            }

            if input_text.to_lowercase() == "exit" {
                println!("Goodbye!");
                break;
            }

            // Simulate inference
            print!("{} ", style("Assistant:").bold().blue());
            self.simulate_inference(input_text).await?;
            println!();
        }

        Ok(())
    }

    /// Run single prompt inference
    async fn run_single_prompt(&self, prompt: &str) -> Result<()> {
        println!("Processing prompt: {}", style(prompt).dim());
        self.simulate_inference(prompt).await?;
        Ok(())
    }

    /// Simulate inference (placeholder implementation)
    async fn simulate_inference(&self, prompt: &str) -> Result<()> {
        let spinner = create_spinner("Generating response...");
        
        // Simulate processing time
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        
        spinner.finish_and_clear();
        
        // Generate a simple response
        let responses = [
            "I understand you said: {}. This is a simulated response from the Woolly CLI.",
            "Thank you for the input: {}. The full inference engine is still in development.",
            "Processing: {}. This demonstrates the CLI interface working with GGUF model loading.",
            "Your message: {}. Real inference will be available when the tensor backend is complete.",
        ];
        
        let response_template = responses[prompt.len() % responses.len()];
        let response = response_template.replace("{}", prompt);
        
        // Simulate streaming output
        for word in response.split(' ') {
            print!("{} ", word);
            io::stdout().flush()?;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        
        if self.timing {
            println!();
            println!("{} Simulated inference - real timing will be available when the engine is complete", 
                style("Note:").yellow().bold());
        }
        
        Ok(())
    }

    /// Convert MetadataValue to string
    fn metadata_value_to_string(&self, value: &woolly_gguf::MetadataValue) -> String {
        match value {
            woolly_gguf::MetadataValue::UInt8(v) => v.to_string(),
            woolly_gguf::MetadataValue::Int8(v) => v.to_string(),
            woolly_gguf::MetadataValue::UInt16(v) => v.to_string(),
            woolly_gguf::MetadataValue::Int16(v) => v.to_string(),
            woolly_gguf::MetadataValue::UInt32(v) => v.to_string(),
            woolly_gguf::MetadataValue::Int32(v) => v.to_string(),
            woolly_gguf::MetadataValue::Float32(v) => v.to_string(),
            woolly_gguf::MetadataValue::Bool(v) => v.to_string(),
            woolly_gguf::MetadataValue::String(v) => v.clone(),
            woolly_gguf::MetadataValue::Array(v) => format!("[array of {} items]", v.len()),
            woolly_gguf::MetadataValue::UInt64(v) => v.to_string(),
            woolly_gguf::MetadataValue::Int64(v) => v.to_string(),
            woolly_gguf::MetadataValue::Float64(v) => v.to_string(),
        }
    }
}

impl Default for RunCommand {
    fn default() -> Self {
        Self {
            model: None,
            prompt: None,
            interactive: false,
            max_tokens: 100,
            temperature: 0.7,
            timing: false,
            dry_run: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_run_command() {
        let cmd = RunCommand::default();
        assert_eq!(cmd.max_tokens, 100);
        assert_eq!(cmd.temperature, 0.7);
        assert!(!cmd.interactive);
        assert!(!cmd.timing);
        assert!(!cmd.dry_run);
    }
}