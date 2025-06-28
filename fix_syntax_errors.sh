#!/bin/bash

cd /Users/ssh/Documents/Code/ai-inference/woolly

# Fix syntax errors caused by sed replacement issues
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/\.to_string(/\.to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/".to_string(,/",/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/".to_string()/"/g' {} \;

echo "Fixed syntax errors"