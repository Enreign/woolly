#!/bin/bash

cd /Users/ssh/Documents/Code/ai-inference/woolly

# Fix specific syntax issues one by one
find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/session_id.to_string())/session_id.to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/m.name().to_string())/m.name().to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/m.model_type().to_string())/m.model_type().to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/s.id().to_string())/s.id().to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/err.to_string())/err.to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/suggestion.to_string())/suggestion.to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/context").to_string())/context".to_string()/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/context"),/context",/g' {} \;

find crates/woolly-core/src -name "*.rs" -exec sed -i '' \
  's/suggestion"),/suggestion",/g' {} \;

echo "Fixed remaining syntax issues"