#!/bin/bash

# Fix CoreError usage in woolly-core using correct helper signatures
cd /Users/ssh/Documents/Code/ai-inference/woolly

# Fix simple message cases for different error types
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::model(\([^,)]*\))/CoreError::model("MODEL_ERROR", \1, "", "Check model configuration")/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::invalid_input(\([^,)]*\))/CoreError::invalid_input("INVALID_INPUT", \1, "Check input parameters")/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::cache(\([^,)]*\))/CoreError::cache("CACHE_ERROR", \1, "", "Check cache configuration")/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::tensor(\([^,)]*\))/CoreError::tensor("TENSOR_ERROR", \1, "", "Check tensor operations")/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::tokenizer(\([^,)]*\))/CoreError::tokenizer("TOKENIZER_ERROR", \1, "", "Check tokenizer configuration")/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::generation(\([^,)]*\))/CoreError::generation("GENERATION_ERROR", \1, "", "Check generation parameters")/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::context(\([^,)]*\))/CoreError::context("CONTEXT_ERROR", \1, "", "Check context window")/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::configuration(\([^,)]*\))/CoreError::configuration("CONFIG_ERROR", \1, "", "Check configuration")/g' {} \;

echo "Fixed CoreError usage with correct helper signatures"