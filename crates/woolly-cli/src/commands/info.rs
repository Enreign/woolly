//! Simplified info command implementation
//!
//! This module implements the info command for displaying model information from GGUF files.

use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Args;
use console::style;
use serde_json::json;
use std::path::PathBuf;
use tabled::{settings::Style, Table, Tabled};
use tracing::{debug, info};

use crate::commands::Command;
use crate::config::Config;
use crate::utils::{create_spinner, format_bytes, print_output, print_success};

use woolly_gguf::{GGUFLoader, MetadataValue, TensorInfo};

#[derive(Args, Debug)]
pub struct InfoCommand {
    /// Path to the model file (GGUF format)
    #[arg(short, long)]
    pub model: Option<PathBuf>,

    /// Show detailed tensor information
    #[arg(long)]
    pub tensors: bool,

    /// Show metadata only
    #[arg(long)]
    pub metadata_only: bool,

    /// Show architecture details
    #[arg(long)]
    pub architecture: bool,

    /// Show tokenizer information
    #[arg(long)]
    pub tokenizer: bool,

    /// Validate model integrity
    #[arg(long)]
    pub validate: bool,

    /// Show raw metadata (all key-value pairs)
    #[arg(long)]
    pub raw: bool,
}

#[derive(Tabled)]
struct TensorSummary {
    name: String,
    shape: String,
    dtype: String,
    size_mb: String,
}

#[derive(Tabled)]
struct MetadataEntry {
    key: String,
    value: String,
}

#[async_trait]
impl Command for InfoCommand {
    async fn execute(&self, config: &Config, json_output: bool) -> Result<()> {
        debug!("Executing info command with config: {:?}", self);

        // Determine model path
        let model_path = self.resolve_model_path(config)?;
        info!("Analyzing model: {}", model_path.display());

        // Load GGUF file
        let spinner = create_spinner("Loading model information...");
        let gguf_file = GGUFLoader::from_path(&model_path)
            .context("Failed to load GGUF file")?;
        spinner.finish_with_message("Model information loaded");

        // Validate if requested
        if self.validate {
            self.validate_model(&gguf_file, json_output)?;
        }

        // Extract information
        let model_info = self.extract_model_info(&gguf_file)?;

        // Output information
        if json_output {
            print_output(&model_info, true)?;
        } else {
            self.print_human_readable(&model_info, &gguf_file)?;
        }

        Ok(())
    }
}

impl InfoCommand {
    /// Resolve the model path from arguments or configuration
    fn resolve_model_path(&self, config: &Config) -> Result<PathBuf> {
        match &self.model {
            Some(path) => Ok(path.clone()),
            None => {
                if let Some(default_model) = &config.default_model {
                    config.find_model(&default_model.to_string_lossy())
                } else {
                    anyhow::bail!("No model specified. Use --model or set a default model in config.");
                }
            }
        }
    }

    /// Extract comprehensive model information
    fn extract_model_info(&self, gguf_file: &GGUFLoader) -> Result<serde_json::Value> {
        let metadata = &gguf_file.metadata().kv_pairs;
        let tensors: Vec<_> = gguf_file.tensors().values().collect();
        
        let mut info = json!({
            "file_info": {
                "format": "GGUF",
                "version": gguf_file.header().version.0,
                "tensor_count": tensors.len(),
                "total_size": self.calculate_total_size(&tensors),
            }
        });

        // Basic model information
        if let Ok(model_info) = self.extract_basic_info(metadata) {
            info["model"] = model_info;
        }

        // Architecture information
        if self.architecture || !self.metadata_only {
            if let Ok(arch_info) = self.extract_architecture_info(metadata) {
                info["architecture"] = arch_info;
            }
        }

        // Tokenizer information
        if self.tokenizer || !self.metadata_only {
            if let Ok(tokenizer_info) = self.extract_tokenizer_info(metadata) {
                info["tokenizer"] = tokenizer_info;
            }
        }

        // Tensor information
        if self.tensors || !self.metadata_only {
            info["tensors"] = self.extract_tensor_info(&tensors)?;
        }

        // Raw metadata
        if self.raw {
            info["raw_metadata"] = self.extract_raw_metadata(metadata)?;
        }

        // Metadata summary
        if !self.raw {
            info["metadata"] = self.extract_metadata_summary(metadata)?;
        }

        Ok(info)
    }

    /// Extract basic model information
    fn extract_basic_info(&self, metadata: &std::collections::HashMap<String, MetadataValue>) -> Result<serde_json::Value> {
        let mut info = json!({});

        // Model name and description
        if let Some(name) = metadata.get("general.name") {
            info["name"] = json!(self.metadata_value_to_string(name));
        }
        
        if let Some(description) = metadata.get("general.description") {
            info["description"] = json!(self.metadata_value_to_string(description));
        }

        if let Some(author) = metadata.get("general.author") {
            info["author"] = json!(self.metadata_value_to_string(author));
        }

        if let Some(version) = metadata.get("general.version") {
            info["version"] = json!(self.metadata_value_to_string(version));
        }

        if let Some(license) = metadata.get("general.license") {
            info["license"] = json!(self.metadata_value_to_string(license));
        }

        if let Some(url) = metadata.get("general.url") {
            info["url"] = json!(self.metadata_value_to_string(url));
        }

        Ok(info)
    }

    /// Extract architecture information
    fn extract_architecture_info(&self, metadata: &std::collections::HashMap<String, MetadataValue>) -> Result<serde_json::Value> {
        let mut info = json!({});

        // Get architecture name
        if let Some(arch) = metadata.get("general.architecture") {
            info["type"] = json!(self.metadata_value_to_string(arch));
        }

        // Model parameters
        if let Some(param_count) = metadata.get("general.parameter_count") {
            info["parameter_count"] = json!(self.metadata_value_to_string(param_count));
        }

        // Context length
        if let Some(context_length) = metadata.get("llama.context_length") 
            .or_else(|| metadata.get("gpt2.context_length"))
            .or_else(|| metadata.get("context_length")) {
            info["context_length"] = json!(self.metadata_value_to_string(context_length));
        }

        // Model dimensions
        if let Some(embedding_length) = metadata.get("llama.embedding_length")
            .or_else(|| metadata.get("embedding_length")) {
            info["embedding_dimension"] = json!(self.metadata_value_to_string(embedding_length));
        }

        if let Some(head_count) = metadata.get("llama.attention.head_count")
            .or_else(|| metadata.get("attention.head_count")) {
            info["attention_heads"] = json!(self.metadata_value_to_string(head_count));
        }

        if let Some(layer_count) = metadata.get("llama.block_count")
            .or_else(|| metadata.get("block_count")) {
            info["layers"] = json!(self.metadata_value_to_string(layer_count));
        }

        if let Some(feed_forward_length) = metadata.get("llama.feed_forward_length")
            .or_else(|| metadata.get("feed_forward_length")) {
            info["feed_forward_dimension"] = json!(self.metadata_value_to_string(feed_forward_length));
        }

        // Quantization info
        if let Some(file_type) = metadata.get("general.file_type") {
            info["quantization"] = json!(self.format_quantization_type(self.metadata_value_to_string(file_type)));
        }

        Ok(info)
    }

    /// Extract tokenizer information
    fn extract_tokenizer_info(&self, metadata: &std::collections::HashMap<String, MetadataValue>) -> Result<serde_json::Value> {
        let mut info = json!({});

        if let Some(model) = metadata.get("tokenizer.ggml.model") {
            info["model"] = json!(self.metadata_value_to_string(model));
        }

        if let Some(vocab_size) = metadata.get("tokenizer.ggml.token_count") {
            info["vocab_size"] = json!(self.metadata_value_to_string(vocab_size));
        }

        // Special tokens
        let mut special_tokens = json!({});
        
        if let Some(bos_token) = metadata.get("tokenizer.ggml.bos_token_id") {
            special_tokens["bos_token_id"] = json!(self.metadata_value_to_string(bos_token));
        }
        
        if let Some(eos_token) = metadata.get("tokenizer.ggml.eos_token_id") {
            special_tokens["eos_token_id"] = json!(self.metadata_value_to_string(eos_token));
        }
        
        if let Some(unk_token) = metadata.get("tokenizer.ggml.unknown_token_id") {
            special_tokens["unknown_token_id"] = json!(self.metadata_value_to_string(unk_token));
        }
        
        if let Some(sep_token) = metadata.get("tokenizer.ggml.separator_token_id") {
            special_tokens["separator_token_id"] = json!(self.metadata_value_to_string(sep_token));
        }
        
        if let Some(pad_token) = metadata.get("tokenizer.ggml.padding_token_id") {
            special_tokens["padding_token_id"] = json!(self.metadata_value_to_string(pad_token));
        }

        if !special_tokens.as_object().unwrap().is_empty() {
            info["special_tokens"] = special_tokens;
        }

        Ok(info)
    }

    /// Extract tensor information
    fn extract_tensor_info(&self, tensors: &[&TensorInfo]) -> Result<serde_json::Value> {
        let mut tensor_list = Vec::new();
        let mut total_size = 0u64;
        let mut type_counts = std::collections::HashMap::new();

        for tensor in tensors {
            let size = tensor.data_size();
            total_size += size;
            
            let dtype_str = format!("{:?}", tensor.ggml_type);
            *type_counts.entry(dtype_str.clone()).or_insert(0) += 1;

            tensor_list.push(json!({
                "name": tensor.name,
                "shape": tensor.dims,
                "data_type": dtype_str,
                "size_bytes": size,
                "size_mb": size as f64 / (1024.0 * 1024.0),
            }));
        }

        Ok(json!({
            "count": tensors.len(),
            "total_size_bytes": total_size,
            "total_size_mb": total_size as f64 / (1024.0 * 1024.0),
            "type_distribution": type_counts,
            "tensors": if self.tensors { 
                serde_json::Value::Array(tensor_list) 
            } else { 
                json!([]) 
            }
        }))
    }

    /// Extract raw metadata
    fn extract_raw_metadata(&self, metadata: &std::collections::HashMap<String, MetadataValue>) -> Result<serde_json::Value> {
        let mut raw_metadata = json!({});
        
        for (key, value) in metadata {
            raw_metadata[key] = json!(self.metadata_value_to_string(value));
        }
        
        Ok(raw_metadata)
    }

    /// Extract metadata summary
    fn extract_metadata_summary(&self, metadata: &std::collections::HashMap<String, MetadataValue>) -> Result<serde_json::Value> {
        let mut summary = json!({});
        
        // Count metadata entries by category
        let mut categories = std::collections::HashMap::new();
        for key in metadata.keys() {
            let category = key.split('.').next().unwrap_or("unknown");
            *categories.entry(category).or_insert(0) += 1;
        }
        
        summary["categories"] = json!(categories);
        summary["total_entries"] = json!(metadata.len());
        
        Ok(summary)
    }

    /// Calculate total model size
    fn calculate_total_size(&self, tensors: &[&TensorInfo]) -> u64 {
        tensors.iter().map(|t| t.data_size()).sum()
    }

    /// Convert MetadataValue to string
    fn metadata_value_to_string(&self, value: &MetadataValue) -> String {
        match value {
            MetadataValue::UInt8(v) => v.to_string(),
            MetadataValue::Int8(v) => v.to_string(),
            MetadataValue::UInt16(v) => v.to_string(),
            MetadataValue::Int16(v) => v.to_string(),
            MetadataValue::UInt32(v) => v.to_string(),
            MetadataValue::Int32(v) => v.to_string(),
            MetadataValue::Float32(v) => v.to_string(),
            MetadataValue::Bool(v) => v.to_string(),
            MetadataValue::String(v) => v.clone(),
            MetadataValue::Array(v) => format!("[array of {} items]", v.len()),
            MetadataValue::UInt64(v) => v.to_string(),
            MetadataValue::Int64(v) => v.to_string(),
            MetadataValue::Float64(v) => v.to_string(),
        }
    }

    /// Format quantization type for display
    fn format_quantization_type(&self, file_type: String) -> String {
        match file_type.as_str() {
            "0" => "F32".to_string(),
            "1" => "F16".to_string(),
            "2" => "Q4_0".to_string(),
            "3" => "Q4_1".to_string(),
            "6" => "Q5_0".to_string(),
            "7" => "Q5_1".to_string(),
            "8" => "Q8_0".to_string(),
            "9" => "Q8_1".to_string(),
            "10" => "Q2_K".to_string(),
            "11" => "Q3_K".to_string(),
            "12" => "Q4_K".to_string(),
            "13" => "Q5_K".to_string(),
            "14" => "Q6_K".to_string(),
            _ => format!("Unknown ({})", file_type),
        }
    }

    /// Validate model integrity
    fn validate_model(&self, gguf_file: &GGUFLoader, json_output: bool) -> Result<()> {
        let spinner = create_spinner("Validating model...");
        
        // Perform basic validation
        let mut issues = Vec::new();
        
        // Check header integrity
        let header = gguf_file.header();
        if header.version.0 < 2 {
            issues.push("GGUF version is outdated".to_string());
        }
        
        // Check tensor consistency
        let tensors = gguf_file.tensors();
        for (name, tensor) in tensors {
            if name.is_empty() {
                issues.push("Tensor with empty name found".to_string());
            }
            
            if tensor.dims.is_empty() {
                issues.push(format!("Tensor '{}' has no dimensions", name));
            }
            
            let calculated_size = tensor.data_size();
            if calculated_size == 0 {
                issues.push(format!("Tensor '{}' has zero size", name));
            }
        }
        
        // Check metadata consistency
        let metadata = &gguf_file.metadata().kv_pairs;
        if !metadata.contains_key("general.architecture") {
            issues.push("Missing required metadata: general.architecture".to_string());
        }
        
        spinner.finish_and_clear();
        
        if json_output {
            let validation = json!({
                "valid": issues.is_empty(),
                "issues": issues
            });
            print_output(&validation, true)?;
        } else {
            if issues.is_empty() {
                print_success("Model validation passed");
            } else {
                println!("{} Model validation failed with {} issues:", 
                    style("✗").red().bold(), issues.len());
                for (i, issue) in issues.iter().enumerate() {
                    println!("  {}. {}", i + 1, issue);
                }
            }
        }
        
        Ok(())
    }

    /// Print human-readable output
    fn print_human_readable(&self, info: &serde_json::Value, gguf_file: &GGUFLoader) -> Result<()> {
        // File information
        println!("{}", style("File Information").bold().cyan());
        println!("Format: {}", info["file_info"]["format"].as_str().unwrap_or("Unknown"));
        println!("Version: {}", info["file_info"]["version"].as_u64().unwrap_or(0));
        println!("Total Size: {}", format_bytes(info["file_info"]["total_size"].as_u64().unwrap_or(0)));
        println!("Tensor Count: {}", info["file_info"]["tensor_count"].as_u64().unwrap_or(0));
        println!();

        // Model information
        if let Some(model_info) = info.get("model") {
            println!("{}", style("Model Information").bold().cyan());
            if let Some(name) = model_info.get("name") {
                println!("Name: {}", name.as_str().unwrap_or("Unknown"));
            }
            if let Some(description) = model_info.get("description") {
                println!("Description: {}", description.as_str().unwrap_or("None"));
            }
            if let Some(author) = model_info.get("author") {
                println!("Author: {}", author.as_str().unwrap_or("Unknown"));
            }
            if let Some(version) = model_info.get("version") {
                println!("Version: {}", version.as_str().unwrap_or("Unknown"));
            }
            println!();
        }

        // Architecture information
        if let Some(arch_info) = info.get("architecture") {
            println!("{}", style("Architecture").bold().cyan());
            if let Some(arch_type) = arch_info.get("type") {
                println!("Type: {}", arch_type.as_str().unwrap_or("Unknown"));
            }
            if let Some(param_count) = arch_info.get("parameter_count") {
                println!("Parameters: {}", param_count.as_str().unwrap_or("Unknown"));
            }
            if let Some(context_length) = arch_info.get("context_length") {
                println!("Context Length: {}", context_length.as_str().unwrap_or("Unknown"));
            }
            if let Some(embedding_dim) = arch_info.get("embedding_dimension") {
                println!("Embedding Dimension: {}", embedding_dim.as_str().unwrap_or("Unknown"));
            }
            if let Some(heads) = arch_info.get("attention_heads") {
                println!("Attention Heads: {}", heads.as_str().unwrap_or("Unknown"));
            }
            if let Some(layers) = arch_info.get("layers") {
                println!("Layers: {}", layers.as_str().unwrap_or("Unknown"));
            }
            if let Some(quantization) = arch_info.get("quantization") {
                println!("Quantization: {}", quantization.as_str().unwrap_or("Unknown"));
            }
            println!();
        }

        // Tokenizer information
        if let Some(tokenizer_info) = info.get("tokenizer") {
            println!("{}", style("Tokenizer").bold().cyan());
            if let Some(model) = tokenizer_info.get("model") {
                println!("Model: {}", model.as_str().unwrap_or("Unknown"));
            }
            if let Some(vocab_size) = tokenizer_info.get("vocab_size") {
                println!("Vocabulary Size: {}", vocab_size.as_str().unwrap_or("Unknown"));
            }
            if let Some(special_tokens) = tokenizer_info.get("special_tokens") {
                println!("Special Tokens:");
                for (key, value) in special_tokens.as_object().unwrap() {
                    println!("  {}: {}", key, value.as_str().unwrap_or("Unknown"));
                }
            }
            println!();
        }

        // Tensor information
        if let Some(tensor_info) = info.get("tensors") {
            println!("{}", style("Tensors").bold().cyan());
            println!("Count: {}", tensor_info["count"].as_u64().unwrap_or(0));
            println!("Total Size: {:.2} MB", tensor_info["total_size_mb"].as_f64().unwrap_or(0.0));
            
            if let Some(type_dist) = tensor_info.get("type_distribution") {
                println!("Type Distribution:");
                for (dtype, count) in type_dist.as_object().unwrap() {
                    println!("  {}: {}", dtype, count.as_u64().unwrap_or(0));
                }
            }

            // Detailed tensor list if requested
            if self.tensors {
                println!();
                let tensors = gguf_file.tensors();
                let tensor_data: Vec<TensorSummary> = tensors.iter().take(10).map(|(name, t)| {
                    let size_mb = t.data_size() as f64 / (1024.0 * 1024.0);
                    TensorSummary {
                        name: name.clone(),
                        shape: format!("{:?}", t.dims),
                        dtype: format!("{:?}", t.ggml_type),
                        size_mb: format!("{:.2}", size_mb),
                    }
                }).collect();

                let table = Table::new(tensor_data).with(Style::modern()).to_string();
                println!("{}", table);
                
                if tensors.len() > 10 {
                    println!("... and {} more tensors", tensors.len() - 10);
                }
            }
            println!();
        }

        // Raw metadata if requested
        if self.raw {
            if let Some(raw_metadata) = info.get("raw_metadata") {
                println!("{}", style("Raw Metadata").bold().cyan());
                let metadata_entries: Vec<MetadataEntry> = raw_metadata.as_object()
                    .unwrap()
                    .iter()
                    .take(20)
                    .map(|(k, v)| MetadataEntry {
                        key: k.clone(),
                        value: v.as_str().unwrap_or("").to_string(),
                    })
                    .collect();

                let table = Table::new(metadata_entries).with(Style::modern()).to_string();
                println!("{}", table);
                
                if raw_metadata.as_object().unwrap().len() > 20 {
                    println!("... and {} more entries", raw_metadata.as_object().unwrap().len() - 20);
                }
            }
        } else if let Some(metadata_summary) = info.get("metadata") {
            println!("{}", style("Metadata Summary").bold().cyan());
            println!("Total Entries: {}", metadata_summary["total_entries"].as_u64().unwrap_or(0));
            if let Some(categories) = metadata_summary.get("categories") {
                println!("Categories:");
                for (category, count) in categories.as_object().unwrap() {
                    println!("  {}: {}", category, count.as_u64().unwrap_or(0));
                }
            }
        }

        Ok(())
    }
}

impl Default for InfoCommand {
    fn default() -> Self {
        Self {
            model: None,
            tensors: false,
            metadata_only: false,
            architecture: false,
            tokenizer: false,
            validate: false,
            raw: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_quantization_type() {
        let cmd = InfoCommand::default();
        assert_eq!(cmd.format_quantization_type("2".to_string()), "Q4_0");
        assert_eq!(cmd.format_quantization_type("1".to_string()), "F16");
        assert_eq!(cmd.format_quantization_type("999".to_string()), "Unknown (999)");
    }

    #[test]
    fn test_metadata_value_to_string() {
        let cmd = InfoCommand::default();
        assert_eq!(cmd.metadata_value_to_string(&MetadataValue::String("test".to_string())), "test");
        assert_eq!(cmd.metadata_value_to_string(&MetadataValue::UInt32(42)), "42");
        assert_eq!(cmd.metadata_value_to_string(&MetadataValue::Bool(true)), "true");
    }
}