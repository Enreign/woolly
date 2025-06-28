//! Simple tool to list all tensor names and their GGML data types in a GGUF file

use std::env;
use woolly_gguf::{GGUFLoader, Result};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        eprintln!("Example: {} models/granite-3.3-8b-instruct-Q4_K_M.gguf", args[0]);
        std::process::exit(1);
    }
    
    let path = &args[1];
    println!("Loading GGUF file: {}", path);
    
    // Load the GGUF file
    let loader = GGUFLoader::from_path(path)?;
    
    // Get and sort tensor names
    let mut tensor_names: Vec<_> = loader.tensor_names();
    tensor_names.sort();
    
    // Print header
    println!("\nTensors ({} total):", tensor_names.len());
    println!("{}", "=".repeat(80));
    println!("{:<50} {:<20}", "Tensor Name", "GGML Type");
    println!("{}", "-".repeat(80));
    
    // Print tensor information
    for name in &tensor_names {
        if let Some(tensor_info) = loader.tensor_info(name) {
            println!("{:<50} {:?}", name, tensor_info.ggml_type);
        }
    }
    
    println!("{}", "=".repeat(80));
    
    // Count tensors by type
    let mut type_counts = std::collections::HashMap::new();
    for name in &tensor_names {
        if let Some(tensor_info) = loader.tensor_info(name) {
            *type_counts.entry(format!("{:?}", tensor_info.ggml_type)).or_insert(0) += 1;
        }
    }
    
    // Print summary
    println!("\nTensor Type Summary:");
    println!("{}", "-".repeat(40));
    let mut type_entries: Vec<_> = type_counts.iter().collect();
    type_entries.sort_by_key(|(k, _)| k.as_str());
    for (dtype, count) in type_entries {
        println!("{:<20} {:<10}", dtype, count);
    }
    println!("{}", "-".repeat(40));
    println!("Total: {} tensors", tensor_names.len());
    
    Ok(())
}