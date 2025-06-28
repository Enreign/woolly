#!/bin/bash

# Fix CoreError usage in woolly-core
cd /Users/ssh/Documents/Code/ai-inference/woolly

# Fix Model errors
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::Model(\([^)]*\))/CoreError::model(\1)/g' {} \;

# Fix InvalidInput errors  
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::InvalidInput(\([^)]*\))/CoreError::invalid_input(\1)/g' {} \;

# Fix Cache errors
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::Cache(\([^)]*\))/CoreError::cache(\1)/g' {} \;

# Fix Tensor errors
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::Tensor(\([^)]*\))/CoreError::tensor(\1)/g' {} \;

# Fix Tokenizer errors
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::Tokenizer(\([^)]*\))/CoreError::tokenizer(\1)/g' {} \;

# Fix Generation errors
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::Generation(\([^)]*\))/CoreError::generation(\1)/g' {} \;

# Fix Context errors
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::Context(\([^)]*\))/CoreError::context(\1)/g' {} \;

# Fix Configuration errors
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/CoreError::Configuration(\([^)]*\))/CoreError::configuration(\1)/g' {} \;

echo "Fixed CoreError usage patterns"