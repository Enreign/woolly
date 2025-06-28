#!/bin/bash

# Comprehensive profiling script for Woolly
# This script profiles the Granite model inference to identify optimization opportunities

set -e

# Add cargo to PATH
export PATH="$HOME/.cargo/bin:$PATH"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BLUE='\033[0;34m'

echo "🔬 Woolly Performance Profiling Suite"
echo "===================================="
echo

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${YELLOW}⚠️  $1 not found. Installing...${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 is available${NC}"
        return 0
    fi
}

echo "📋 Checking required tools..."
echo "----------------------------"

# Check for basic tools
check_tool "cargo"
check_tool "rustc"

# Check for profiling tools
HAS_PERF=false
HAS_FLAMEGRAPH=false
HAS_INSTRUMENTS=false

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}ℹ️  macOS detected - checking for Instruments${NC}"
    if check_tool "instruments"; then
        HAS_INSTRUMENTS=true
    fi
    # Check for cargo-instruments
    if ! cargo instruments --version &> /dev/null; then
        echo -e "${YELLOW}Installing cargo-instruments...${NC}"
        cargo install cargo-instruments
    else
        echo -e "${GREEN}✓ cargo-instruments is available${NC}"
    fi
else
    # Linux - check for perf
    if check_tool "perf"; then
        HAS_PERF=true
    fi
fi

# Check for cargo-flamegraph
if ! cargo flamegraph --version &> /dev/null; then
    echo -e "${YELLOW}Installing cargo-flamegraph...${NC}"
    cargo install flamegraph
    HAS_FLAMEGRAPH=true
else
    echo -e "${GREEN}✓ cargo-flamegraph is available${NC}"
    HAS_FLAMEGRAPH=true
fi

# Check for samply (sampling profiler)
if ! samply --version &> /dev/null; then
    echo -e "${YELLOW}Installing samply...${NC}"
    cargo install samply
else
    echo -e "${GREEN}✓ samply is available${NC}"
fi

echo

# Create profile directory
PROFILE_DIR="profile_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_DIR"
echo -e "${BLUE}📁 Created profile directory: $PROFILE_DIR${NC}"
echo

# Build optimized binary with debug symbols
echo "🔨 Building optimized binary with debug symbols..."
echo "-------------------------------------------------"

# Create a custom profile for profiling
cat > Cargo.toml.profiling << 'EOF'
[profile.profiling]
inherits = "release"
debug = true
EOF

# Append to workspace Cargo.toml if not already present
if ! grep -q "\[profile.profiling\]" Cargo.toml; then
    cat Cargo.toml.profiling >> Cargo.toml
fi
rm Cargo.toml.profiling

# Build with profiling profile
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling
echo -e "${GREEN}✓ Build complete${NC}"
echo

# Create test program for profiling
echo "📝 Creating profiling test program..."
echo "-----------------------------------"

cat > "$PROFILE_DIR/profile_test.rs" << 'EOF'
use woolly_core::{Engine, Config, GenerationOptions};
use std::time::Instant;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Woolly profiling test");
    
    // Initialize engine
    let config = Config::default();
    let mut engine = Engine::new(config)?;
    
    // Load model
    let model_path = "models/granite-3.3-8b-instruct-Q4_K_M.gguf";
    println!("📥 Loading model: {}", model_path);
    let load_start = Instant::now();
    engine.load_model(model_path)?;
    let load_duration = load_start.elapsed();
    println!("✓ Model loaded in {:?}", load_duration);
    
    // Warm up
    println!("\n🔥 Warming up...");
    let warmup_options = GenerationOptions {
        max_tokens: 5,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        stream: false,
    };
    let _ = engine.generate("Hello", warmup_options.clone())?;
    
    // Profile different scenarios
    let mut results = Vec::new();
    
    // Test 1: Short generation
    println!("\n📊 Test 1: Short generation (10 tokens)");
    let start = Instant::now();
    let short_result = engine.generate("The future of AI is", GenerationOptions {
        max_tokens: 10,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        stream: false,
    })?;
    let duration = start.elapsed();
    let tokens = short_result.tokens_generated;
    let tokens_per_sec = tokens as f64 / duration.as_secs_f64();
    println!("  Generated {} tokens in {:?} ({:.2} tokens/sec)", tokens, duration, tokens_per_sec);
    results.push(("short_generation", duration, tokens, tokens_per_sec));
    
    // Test 2: Medium generation
    println!("\n📊 Test 2: Medium generation (50 tokens)");
    let start = Instant::now();
    let medium_result = engine.generate("Write a technical explanation of how", GenerationOptions {
        max_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        stream: false,
    })?;
    let duration = start.elapsed();
    let tokens = medium_result.tokens_generated;
    let tokens_per_sec = tokens as f64 / duration.as_secs_f64();
    println!("  Generated {} tokens in {:?} ({:.2} tokens/sec)", tokens, duration, tokens_per_sec);
    results.push(("medium_generation", duration, tokens, tokens_per_sec));
    
    // Test 3: Batch processing simulation
    println!("\n📊 Test 3: Batch processing (5 requests)");
    let batch_prompts = vec![
        "The weather today is",
        "In computer science,",
        "The best way to",
        "Machine learning is",
        "Technology has changed"
    ];
    let start = Instant::now();
    let mut total_tokens = 0;
    for prompt in &batch_prompts {
        let result = engine.generate(prompt, GenerationOptions {
            max_tokens: 20,
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stream: false,
        })?;
        total_tokens += result.tokens_generated;
    }
    let duration = start.elapsed();
    let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
    println!("  Generated {} tokens in {:?} ({:.2} tokens/sec)", total_tokens, duration, tokens_per_sec);
    results.push(("batch_processing", duration, total_tokens, tokens_per_sec));
    
    // Test 4: Memory-intensive generation
    println!("\n📊 Test 4: Memory-intensive generation (100 tokens)");
    let start = Instant::now();
    let long_result = engine.generate("Write a detailed story about", GenerationOptions {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        stream: false,
    })?;
    let duration = start.elapsed();
    let tokens = long_result.tokens_generated;
    let tokens_per_sec = tokens as f64 / duration.as_secs_f64();
    println!("  Generated {} tokens in {:?} ({:.2} tokens/sec)", tokens, duration, tokens_per_sec);
    results.push(("long_generation", duration, tokens, tokens_per_sec));
    
    // Write results to file
    let mut file = File::create("profile_results.csv")?;
    writeln!(file, "test,duration_ms,tokens,tokens_per_sec")?;
    for (test, duration, tokens, tps) in results {
        writeln!(file, "{},{},{},{:.2}", test, duration.as_millis(), tokens, tps)?;
    }
    
    println!("\n✅ Profiling test complete!");
    Ok(())
}
EOF

# Copy to examples directory for building
cp "$PROFILE_DIR/profile_test.rs" examples/profile_test.rs

echo -e "${GREEN}✓ Test program created${NC}"
echo

# Function to run profiling with different tools
run_profiling() {
    local tool=$1
    local output_prefix=$2
    
    echo -e "${BLUE}🔍 Running $tool profiling...${NC}"
    
    case $tool in
        "samply")
            samply record -o "$PROFILE_DIR/${output_prefix}_samply.json" \
                ./target/profiling/examples/profile_test
            echo -e "${GREEN}✓ Samply profile saved to ${output_prefix}_samply.json${NC}"
            ;;
            
        "instruments")
            if [[ "$HAS_INSTRUMENTS" == "true" ]]; then
                cargo instruments -t "Time Profiler" \
                    --profile profiling \
                    --example profile_test \
                    --out "$PROFILE_DIR/${output_prefix}_instruments.trace"
                echo -e "${GREEN}✓ Instruments trace saved${NC}"
            fi
            ;;
            
        "flamegraph")
            if [[ "$HAS_FLAMEGRAPH" == "true" ]]; then
                cd "$PROFILE_DIR"
                CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph \
                    --profile profiling \
                    --example profile_test \
                    --output "${output_prefix}_flamegraph.svg" \
                    -- 2>&1 | tee flamegraph.log
                cd ..
                echo -e "${GREEN}✓ Flamegraph saved to ${output_prefix}_flamegraph.svg${NC}"
            fi
            ;;
            
        "builtin")
            # Use Rust's built-in profiler output
            RUSTFLAGS="-C profile-generate=$PROFILE_DIR/pgo-data" \
                cargo build --profile profiling --example profile_test
            ./target/profiling/examples/profile_test
            echo -e "${GREEN}✓ PGO data collected${NC}"
            ;;
    esac
}

# Run basic benchmark first
echo "⏱️  Running baseline benchmark..."
echo "--------------------------------"
cd "$PROFILE_DIR"
/usr/bin/time -l ../target/profiling/examples/profile_test 2>&1 | tee baseline_benchmark.txt
cd ..
echo

# Extract and display memory statistics if available
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "💾 Memory Statistics:"
    grep "maximum resident set size" "$PROFILE_DIR/baseline_benchmark.txt" || true
    echo
fi

# Run different profiling tools
echo "🎯 Running profiling tools..."
echo "----------------------------"

# Always try samply as it works cross-platform
run_profiling "samply" "granite_inference"

# Platform-specific profiling
if [[ "$OSTYPE" == "darwin"* ]] && [[ "$HAS_INSTRUMENTS" == "true" ]]; then
    run_profiling "instruments" "granite_inference"
elif [[ "$HAS_PERF" == "true" ]]; then
    echo -e "${BLUE}🔍 Running perf profiling...${NC}"
    perf record -F 99 -g --call-graph dwarf \
        -o "$PROFILE_DIR/granite_inference_perf.data" \
        ./target/profiling/examples/profile_test
    perf report -g -i "$PROFILE_DIR/granite_inference_perf.data" \
        > "$PROFILE_DIR/perf_report.txt"
    echo -e "${GREEN}✓ Perf data saved${NC}"
fi

# Generate flamegraph if available
if [[ "$HAS_FLAMEGRAPH" == "true" ]]; then
    run_profiling "flamegraph" "granite_inference"
fi

echo

# Analyze hot functions using nm and addr2line
echo "🔥 Analyzing hot functions..."
echo "----------------------------"

# Create analysis script
cat > "$PROFILE_DIR/analyze_functions.sh" << 'EOF'
#!/bin/bash

echo "Top 20 largest functions by size:"
nm -S --size-sort target/profiling/examples/profile_test | tail -20 | while read size addr type name; do
    demangled=$(echo "$name" | c++filt)
    echo "$size $demangled"
done

echo -e "\nAnalyzing function symbols..."
nm target/profiling/examples/profile_test | grep -E "(matmul|gemm|dot|quantize|dequantize|attention|feedforward)" | c++filt | sort | uniq > hot_functions.txt
echo "Hot function candidates saved to hot_functions.txt"
EOF

chmod +x "$PROFILE_DIR/analyze_functions.sh"
cd "$PROFILE_DIR"
bash analyze_functions.sh
cd ..

echo

# Generate detailed performance report
echo "📊 Generating performance report..."
echo "---------------------------------"

cat > "$PROFILE_DIR/performance_report.md" << EOF
# Woolly Performance Profile Report
Generated: $(date)

## Test Configuration
- Model: Granite 3.3B 8-bit Instruct (Q4_K_M quantization)
- Platform: $(uname -sm)
- Rust version: $(rustc --version)

## Baseline Performance
$(cat "$PROFILE_DIR/baseline_benchmark.txt" | grep -E "(real|user|sys|maximum resident)")

## Test Results
$(cat "$PROFILE_DIR/profile_results.csv" 2>/dev/null || echo "No results file generated")

## Hot Function Analysis
$(head -20 "$PROFILE_DIR/hot_functions.txt" 2>/dev/null || echo "No hot functions identified")

## Profiling Artifacts
- Samply profile: granite_inference_samply.json
- Flamegraph: granite_inference_flamegraph.svg
- Baseline benchmark: baseline_benchmark.txt

## Key Observations
1. Matrix multiplication operations are likely the primary bottleneck
2. Memory allocation patterns may impact performance
3. Quantization/dequantization overhead needs investigation
4. Cache efficiency could be improved

## Recommended Optimizations
1. Implement SIMD-optimized matrix operations
2. Use memory pooling to reduce allocation overhead
3. Cache dequantized weights for frequently accessed layers
4. Optimize memory access patterns for better cache utilization
5. Consider parallelizing independent operations

EOF

echo -e "${GREEN}✓ Performance report generated${NC}"
echo

# Clean up
rm -f examples/profile_test.rs

echo "🎉 Profiling complete!"
echo "📁 Results saved in: $PROFILE_DIR"
echo
echo "📋 Next steps:"
echo "1. View the flamegraph: open $PROFILE_DIR/granite_inference_flamegraph.svg"
echo "2. Analyze with samply: samply load $PROFILE_DIR/granite_inference_samply.json"
echo "3. Read the performance report: $PROFILE_DIR/performance_report.md"
echo
echo "💡 To view hot spots in the code:"
echo "   - Use 'cargo asm' to inspect generated assembly"
echo "   - Use 'cargo expand' to see macro expansions"
echo "   - Profile specific functions with '#[inline(never)]' attribute"