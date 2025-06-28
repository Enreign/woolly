#!/bin/bash

echo "Analyzing GGUF Dequantization Progress..."
echo "========================================"
echo

# Count total dequantized tensors
total_dequantized=$(grep -c "Successfully dequantized" woolly-test.log)
echo "Total tensors dequantized so far: $total_dequantized"

# Count by quantization type
echo
echo "Dequantization by type:"
echo "- Q4_K: $(grep "Successfully dequantized.*Q4_K" woolly-test.log | wc -l)"
echo "- Q6_K: $(grep "Successfully dequantized.*Q6_K" woolly-test.log | wc -l)"
echo "- Q4_0: $(grep "Successfully dequantized.*Q4_0" woolly-test.log | wc -l)"
echo "- Q8_0: $(grep "Successfully dequantized.*Q8_0" woolly-test.log | wc -l)"

# Get latest layer being processed
latest_layer=$(grep "Successfully dequantized.*blk\." woolly-test.log | tail -1 | grep -oE "blk\.[0-9]+" | grep -oE "[0-9]+")
echo
echo "Latest layer being processed: Layer $latest_layer"

# Estimate progress (Granite 3.3 8B typically has 32 layers)
estimated_layers=32
if [ ! -z "$latest_layer" ]; then
    progress=$((latest_layer * 100 / estimated_layers))
    echo "Estimated progress: ~$progress% (assuming $estimated_layers layers)"
fi

# Check memory usage
echo
echo "Server memory usage:"
ps aux | grep woolly-server | grep -v grep | awk '{printf "PID: %s, CPU: %s%%, MEM: %s%%, VSZ: %s KB, RSS: %s KB\n", $2, $3, $4, $5, $6}'

# Show last few operations
echo
echo "Last 5 dequantization operations:"
grep "Successfully dequantized" woolly-test.log | tail -5