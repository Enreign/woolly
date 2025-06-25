//! Example: Inspect a GGUF file and print its metadata and tensor information

use std::env;
use woolly_gguf::{GGUFLoader, Result};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        std::process::exit(1);
    }
    
    let path = &args[1];
    println!("Loading GGUF file: {}", path);
    
    // Load the GGUF file
    let loader = GGUFLoader::from_path(path)?;
    
    // Print header information
    let header = loader.header();
    println!("\nHeader Information:");
    println!("  Version: {}", header.version.0);
    println!("  Tensor count: {}", header.tensor_count);
    println!("  Metadata count: {}", header.metadata_kv_count);
    println!("  Alignment: {} bytes", loader.alignment());
    
    // Print metadata
    println!("\nMetadata:");
    if let Some(arch) = loader.architecture() {
        println!("  Architecture: {}", arch);
    }
    if let Some(name) = loader.model_name() {
        println!("  Model name: {}", name);
    }
    if let Some(qv) = loader.quantization_version() {
        println!("  Quantization version: {}", qv);
    }
    
    // Print tensor information
    println!("\nTensors:");
    let mut tensor_names: Vec<_> = loader.tensor_names();
    tensor_names.sort();
    
    for name in &tensor_names {
        if let Some(info) = loader.tensor_info(name) {
            println!("  {} - shape: {:?}, type: {:?}, size: {} bytes",
                name,
                info.shape(),
                info.ggml_type,
                info.data_size()
            );
        }
    }
    
    // Print summary
    println!("\nSummary:");
    println!("  Total tensors: {}", tensor_names.len());
    println!("  Total tensor data size: {} bytes ({:.2} MB)",
        loader.total_tensor_size(),
        loader.total_tensor_size() as f64 / (1024.0 * 1024.0)
    );
    println!("  File size: {} bytes ({:.2} MB)",
        loader.file_size(),
        loader.file_size() as f64 / (1024.0 * 1024.0)
    );
    
    Ok(())
}