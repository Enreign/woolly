# Proposed Fix for Cache Persistence Issue

## The Problem

Cache entries are being lost between model preloading and inference, causing re-dequantization and performance variance.

## Proposed Solution

### Option 1: Verify Cache Persistence (Quick Fix)
Add verification after preloading to ensure cache entries persist:

```rust
// In lazy_loader.rs, after preload_all_weights()
pub fn verify_cache_persistence(&mut self) -> Result<()> {
    eprintln!("🔍 Verifying cache persistence...");
    
    // Check a few critical tensors
    let test_tensors = vec!["token_embd.weight", "blk.0.attn_norm.weight"];
    
    for tensor_name in test_tensors {
        if let Some(cached_data) = self.dequant_cache.get(tensor_name) {
            eprintln!("✅ Cache verified: {} is still cached", tensor_name);
        } else {
            eprintln!("❌ Cache lost: {} is NOT cached!", tensor_name);
        }
    }
    
    let stats = self.cache_stats();
    eprintln!("📊 Cache stats after verification - Hits: {}, Misses: {}", 
        stats.hits, stats.misses);
    
    Ok(())
}
```

### Option 2: Force Cache Retention (Medium Fix)
Modify the cache to prevent eviction of preloaded weights:

```rust
// In dequantization_cache.rs
pub struct DequantizationCache {
    // ... existing fields ...
    pinned_entries: Arc<RwLock<HashSet<String>>>, // New: track preloaded entries
}

// During preload, pin the entry
pub fn add_pinned(&self, key: String, data: Vec<f32>) -> Result<()> {
    // Add to cache without eviction possibility
    self.add_to_cache(key.clone(), data, size, duration)?;
    
    let mut pinned = self.pinned_entries.write().unwrap();
    pinned.insert(key);
    
    Ok(())
}
```

### Option 3: Simplify Architecture (Best Long-term Fix)
Replace complex cache with simpler HashMap that just stores all dequantized weights:

```rust
// New simple cache in lazy_loader.rs
pub struct SimpleWeightCache {
    weights: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl SimpleWeightCache {
    pub fn get(&self, name: &str) -> Option<Vec<f32>> {
        self.weights.read().unwrap().get(name).cloned()
    }
    
    pub fn insert(&self, name: String, data: Vec<f32>) {
        self.weights.write().unwrap().insert(name, data);
    }
}
```

## Implementation Steps

1. **Immediate**: Add verification logging to confirm cache is being lost
2. **Short-term**: Implement Option 1 to understand the issue better
3. **Medium-term**: If verification shows cache loss, implement Option 2
4. **Long-term**: Consider Option 3 for production robustness

## Expected Outcome

- Consistent 500+ tokens/sec performance
- No cache misses during inference for preloaded weights
- Predictable and fast inference latency