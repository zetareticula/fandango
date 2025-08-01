#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Fandango deployment...${NC}"

# Build the Rust backend
echo -e "${BLUE}🔨 Building Rust backend...${NC}"
cargo build --release

# Build the WebAssembly frontend
echo -e "${BLUE}🌐 Building WebAssembly frontend...${NC}"
cd web-ui

# Install wasm-bindgen if not already installed
if ! command -v wasm-bindgen &> /dev/null; then
    echo "Installing wasm-bindgen..."
    cargo install wasm-bindgen-cli
fi

# Build the frontend
trunk build --release

# Copy the built frontend to the server's static files
cd ..
mkdir -p static
cp -r web-ui/dist/* static/

echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo -e "${BLUE}🚀 Starting Fandango server...${NC}"

# Run the server
cargo run --bin server --release
