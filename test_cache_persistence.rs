// This would be a test to verify if the cache is working
// The issue is that during preload, cache reports 362 misses (correct)
// But during inference, it reports misses again (incorrect)

// The symptoms:
// 1. Cache instance is shared (strong_count: 364) ✓
// 2. Preload completes successfully ✓  
// 3. Cache shows correct stats after preload ✓
// 4. But inference gets cache misses ✗

// The only explanation: The cache HashMap is being cleared/reset
// between preload completion and inference start
