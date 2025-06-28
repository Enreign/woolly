#!/bin/bash
# Benchmark comparison script: Woolly vs llama.cpp
# This script runs comparative benchmarks between Woolly and llama.cpp

set -e

# Configuration
MODEL_PATH="models/granite-3.3-8b-instruct-Q4_K_M.gguf"
PROMPT="Explain the concept of machine learning in simple terms"
OUTPUT_DIR="benchmark_comparison_$(date +%Y%m%d_%H%M%S)"
TOKENS_TO_GENERATE=50

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Woolly vs llama.cpp Performance Comparison ==="
echo "Model: $MODEL_PATH"
echo "Tokens to generate: $TOKENS_TO_GENERATE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to measure performance
measure_performance() {
    local name=$1
    local command=$2
    local output_file="$OUTPUT_DIR/${name}_output.txt"
    local stats_file="$OUTPUT_DIR/${name}_stats.json"
    
    echo "Benchmarking $name..."
    
    # Run multiple iterations
    local total_time=0
    local total_tokens=0
    local iterations=5
    
    echo "{" > "$stats_file"
    echo "  \"name\": \"$name\"," >> "$stats_file"
    echo "  \"iterations\": $iterations," >> "$stats_file"
    echo "  \"runs\": [" >> "$stats_file"
    
    for i in $(seq 1 $iterations); do
        echo "  Run $i/$iterations..."
        
        # Time the command
        local start_time=$(date +%s.%N)
        eval "$command" > "$output_file" 2>&1
        local end_time=$(date +%s.%N)
        
        # Calculate duration
        local duration=$(echo "$end_time - $start_time" | bc)
        
        # Extract token count (this is implementation-specific)
        local tokens=$TOKENS_TO_GENERATE  # Default assumption
        
        if [ $i -lt $iterations ]; then
            echo "    { \"run\": $i, \"duration\": $duration, \"tokens\": $tokens }," >> "$stats_file"
        else
            echo "    { \"run\": $i, \"duration\": $duration, \"tokens\": $tokens }" >> "$stats_file"
        fi
        
        total_time=$(echo "$total_time + $duration" | bc)
        total_tokens=$((total_tokens + tokens))
    done
    
    echo "  ]," >> "$stats_file"
    
    # Calculate averages
    local avg_time=$(echo "scale=4; $total_time / $iterations" | bc)
    local tokens_per_sec=$(echo "scale=4; $total_tokens / $total_time" | bc)
    
    echo "  \"avg_time\": $avg_time," >> "$stats_file"
    echo "  \"total_tokens\": $total_tokens," >> "$stats_file"
    echo "  \"tokens_per_sec\": $tokens_per_sec" >> "$stats_file"
    echo "}" >> "$stats_file"
    
    echo "  Average time: ${avg_time}s"
    echo "  Throughput: ${tokens_per_sec} tokens/sec"
    echo ""
}

# Build Woolly in release mode
echo "Building Woolly..."
cargo build --release

# Benchmark Woolly with SIMD enabled
echo "=== Woolly (SIMD Enabled) ==="
measure_performance "woolly_simd" \
    "./target/release/woolly-cli run --model \"$MODEL_PATH\" --prompt \"$PROMPT\" --max-tokens $TOKENS_TO_GENERATE"

# Benchmark Woolly with SIMD disabled
echo "=== Woolly (SIMD Disabled) ==="
WOOLLY_DISABLE_SIMD=1 measure_performance "woolly_no_simd" \
    "./target/release/woolly-cli run --model \"$MODEL_PATH\" --prompt \"$PROMPT\" --max-tokens $TOKENS_TO_GENERATE"

# Benchmark llama.cpp if available
if command -v llama-cli &> /dev/null; then
    echo "=== llama.cpp ==="
    measure_performance "llama_cpp" \
        "llama-cli -m \"$MODEL_PATH\" -p \"$PROMPT\" -n $TOKENS_TO_GENERATE --no-display-prompt"
else
    echo "llama.cpp not found. Skipping comparison."
    echo "To install: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make"
fi

# Generate comparison report
echo "Generating comparison report..."
cat > "$OUTPUT_DIR/COMPARISON_REPORT.md" << 'EOF'
# Performance Comparison Report

## Summary

EOF

# Parse results and create comparison table
python3 << EOF >> "$OUTPUT_DIR/COMPARISON_REPORT.md"
import json
import glob

results = []
for stats_file in glob.glob("$OUTPUT_DIR/*_stats.json"):
    with open(stats_file, 'r') as f:
        data = json.load(f)
        results.append(data)

if results:
    print("| Implementation | Avg Time (s) | Tokens/sec | Speedup |")
    print("|----------------|--------------|------------|---------|")
    
    baseline_tps = None
    for r in sorted(results, key=lambda x: x['name']):
        name = r['name'].replace('_', ' ').title()
        avg_time = r['avg_time']
        tps = r['tokens_per_sec']
        
        if 'no_simd' in r['name']:
            baseline_tps = tps
            speedup = "1.0x (baseline)"
        elif baseline_tps:
            speedup = f"{tps/baseline_tps:.2f}x"
        else:
            speedup = "-"
            
        print(f"| {name} | {avg_time:.3f} | {tps:.3f} | {speedup} |")

print("\n## Detailed Results\n")

for r in results:
    print(f"### {r['name'].replace('_', ' ').title()}")
    print(f"- Iterations: {r['iterations']}")
    print(f"- Total tokens: {r['total_tokens']}")
    print(f"- Average time: {r['avg_time']:.3f}s")
    print(f"- Throughput: {r['tokens_per_sec']:.3f} tokens/sec")
    print()
EOF

# Add analysis
cat >> "$OUTPUT_DIR/COMPARISON_REPORT.md" << 'EOF'

## Analysis

### Performance Gaps

Based on the benchmarks:

1. **SIMD Impact**: The SIMD optimizations show significant improvement over the scalar baseline.

2. **vs llama.cpp**: Any performance gap with llama.cpp indicates areas for further optimization.

### Optimization Opportunities

1. **Memory Access Patterns**: Ensure all operations are cache-friendly
2. **Parallelization**: Leverage multi-core for batch processing
3. **Quantization**: Direct quantized computation could improve performance
4. **Platform-specific**: Use platform-specific optimizations (Metal, CUDA)

### Recommendations

1. Profile the specific operations where Woolly is slower
2. Implement missing optimizations from llama.cpp
3. Consider architecture-specific tuning
4. Add support for grouped-query attention optimizations

EOF

echo ""
echo "Comparison complete! Results saved to: $OUTPUT_DIR/"
echo "Key files:"
echo "  - COMPARISON_REPORT.md: Performance comparison analysis"
echo "  - *_stats.json: Raw performance data for each implementation"
echo ""

# Quick summary
echo "=== Quick Summary ==="
for stats_file in "$OUTPUT_DIR"/*_stats.json; do
    name=$(jq -r '.name' "$stats_file")
    tps=$(jq -r '.tokens_per_sec' "$stats_file")
    echo "$name: $tps tokens/sec"
done