#\!/bin/bash

echo "🔍 Direct SIMD Performance Test"
echo "==============================="
echo

# Test 1: SIMD Enabled
echo "Test 1: Running with SIMD enabled..."
echo "------------------------------------"
cd crates/woolly-tensor
export RUST_LOG=info
cargo bench --bench simd_benchmarks -- --warm-up-time 1 --measurement-time 5 dot_product/1024 2>&1 | tee ../../simd_enabled_bench.txt | grep -E "time:|found|faster|slower"

echo
echo "Test 2: Running with SIMD disabled..."
echo "-------------------------------------"
export WOOLLY_DISABLE_SIMD=1
cargo bench --bench simd_benchmarks -- --warm-up-time 1 --measurement-time 5 dot_product/1024 2>&1 | tee ../../simd_disabled_bench.txt | grep -E "time:|found|faster|slower"

echo
echo "📊 Checking results..."
cd ../..

# Extract timing information
SIMD_TIME=$(grep -oE "time:.*\[([0-9.]+) ([a-z]+)" simd_enabled_bench.txt | head -1 | awk '{print $2}')
NO_SIMD_TIME=$(grep -oE "time:.*\[([0-9.]+) ([a-z]+)" simd_disabled_bench.txt | head -1 | awk '{print $2}')

echo
echo "Results:"
echo "--------"
echo "SIMD Enabled:  $SIMD_TIME"
echo "SIMD Disabled: $NO_SIMD_TIME"

# Check if SIMD is faster or slower
if grep -q "Performance has regressed" simd_disabled_bench.txt; then
    echo
    echo "✅ Disabling SIMD improves performance\!"
elif grep -q "Performance has improved" simd_disabled_bench.txt; then
    echo
    echo "❌ Disabling SIMD makes it slower"
else
    echo
    echo "No clear performance difference detected"
fi
