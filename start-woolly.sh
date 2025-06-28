#!/bin/bash

# Woolly Quick Start Script
# This script builds and starts Woolly server for Ole integration

set -e

echo "🚀 Woolly Quick Start"
echo "===================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Add Rust to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Rust is not installed${NC}"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check current directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Not in Woolly directory${NC}"
    echo "Please run this script from the woolly directory"
    exit 1
fi

# Build options
BUILD_MODE=${1:-release}
EXTRA_ARGS=""

if [ "$BUILD_MODE" = "debug" ]; then
    echo -e "${YELLOW}Building in debug mode...${NC}"
    BINARY_PATH="target/debug/woolly-server"
else
    echo -e "${YELLOW}Building in release mode...${NC}"
    EXTRA_ARGS="--release"
    BINARY_PATH="target/release/woolly-server"
fi

# Build Woolly
echo -e "\n${YELLOW}Building Woolly server...${NC}"
cargo build --bin woolly-server $EXTRA_ARGS

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build successful${NC}"

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo -e "\n${YELLOW}Creating models directory...${NC}"
    mkdir -p models
    echo -e "${GREEN}✅ Created ./models directory${NC}"
fi

# Check for config file
CONFIG_FILE="woolly-config.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "\n${YELLOW}Creating default configuration...${NC}"
    cat > $CONFIG_FILE << 'EOF'
# Woolly Server Configuration
bind = "0.0.0.0:8080"

[models]
models_dir = "./models"
default_model = ""
preload_models = []

[performance]
max_batch_size = 8
max_context_length = 4096
use_gpu = true
gpu_layers = 32

[mcp]
enabled = true
default_timeout = 30
max_concurrent_tools = 5

[auth]
allow_anonymous = true
jwt_secret = "change-me-in-production"
jwt_expiration = 3600
api_keys = []

[rate_limit]
requests_per_minute = 60
burst_size = 10

[logging]
level = "info"
format = "pretty"
EOF
    echo -e "${GREEN}✅ Created $CONFIG_FILE${NC}"
fi

# Set environment variables
export RUST_LOG=${RUST_LOG:-woolly_server=info,woolly_core=info}
export WOOLLY_CONFIG=$CONFIG_FILE

# Start server
echo -e "\n${GREEN}Starting Woolly server...${NC}"
echo "========================"
echo "Server URL: http://localhost:8080"
echo "Health check: http://localhost:8080/api/v1/health"
echo "Models API: http://localhost:8080/api/v1/models"
echo ""
echo "To use with Ole:"
echo "1. Open Ole desktop client"
echo "2. Go to Settings → Providers → Woolly"
echo "3. Ensure URL is set to http://localhost:8080"
echo "4. Click 'Test Connection'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo "========================"
echo ""

# Run the server
exec $BINARY_PATH