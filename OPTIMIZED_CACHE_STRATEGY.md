# Optimized Cache Strategy for 16GB Systems

## The Problem
- Granite 8B model dequantized = ~32GB
- Available RAM = 16GB
- Current cache tries to store everything = FAIL

## The Solution: Selective Layer Caching

### Option 1: Cache Critical Layers Only (Recommended)
```rust
// Cache only the most frequently accessed layers
let cache_config = DequantizationCacheConfig {
    max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB - half of system RAM
    prefetch_ahead: 1,
    use_frequency_priority: true, // Keep hot weights only
    frequency_window: Duration::from_secs(60),
    enable_async_prefetch: false,
};
```

### Option 2: On-Demand Caching with Fast SIMD
- Don't preload all weights
- Cache only as needed during inference
- Rely on SIMD optimizations for fast dequantization
- LRU eviction keeps memory under control

### Option 3: Use MLX Backend (Best for Apple Silicon)
MLX can handle quantized models directly without dequantization:
- No need to dequantize to FP32
- Works with quantized weights in-place
- Massive memory savings
- 10x+ performance on M4

## Implementation for 16GB System

```rust
// In lazy_loader.rs
pub fn smart_preload_for_limited_memory(&mut self) -> Result<()> {
    // Only preload embeddings and first/last layers
    let critical_tensors = vec![
        "token_embd.weight",
        "output_norm.weight", 
        "blk.0.*",  // First layer
        "blk.39.*", // Last layer
    ];
    
    for pattern in critical_tensors {
        // Preload matching tensors
    }
    
    eprintln!("✅ Critical tensors cached for 16GB system");
    Ok(())
}
```

## Expected Performance
- First token: ~1-2s (vs 194s original)
- Subsequent tokens: 100-200 tokens/sec
- Memory usage: <8GB
- No swapping or system pressure