# Fandango

## Overview

Fandango is an advanced, open-source platform designed to optimize large language model (LLM) inference through dynamic precision scaling and quantization. Built with Rust and Python, it leverages a Kubernetes-native architecture to deliver scalable, efficient performance. The platform now includes a high-performance quantization server for efficient model compression and inference acceleration.

## Key Features

- **Dynamic Precision Scaling**: Adjusts KVCache quantization levels based on attention locality and entropy heuristics, supporting 4-bit precision for low-load scenarios, 8-bit for high concurrency, and float16 for edge cases.
- **Sparse and Shared KVCache Deduplication**: Reduces memory usage by deduplicating KVCache, enhancing efficiency for batched and long-context inference.
- **Serverless Multi-Model Scheduler**: Enables on-demand loading and unloading of models, with caching of quantized layers to optimize performance and resource utilization.
- **High-Performance Quantization Server**: Real-time model quantization and inference with WebSocket support for efficient model serving.

## Quick Start

### Prerequisites

- Rust (latest stable version)
- Node.js and npm (for web UI development)
- wasm-pack (`cargo install wasm-pack`)
- trunk (`cargo install trunk`)
- wasm-opt (optional, for optimizing WebAssembly output)

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zetareticula/fandango.git
   cd fandango
   ```

2. **Run the Quantization Server**:
   ```bash
   # Navigate to the quantization server
   cd quantization_server
   
   # Build and run the server
   cargo run
   ```
   The server will start on http://localhost:8080

3. **Run the Main Application**:
   ```bash
   # Build in release mode
   cargo build --release
   
   # Run the main server
   ./target/release/fandango
   ```

3. **Build and serve the web UI**:
   ```bash
   cd web-ui
   trunk serve
   ```
   The web UI will be available at http://localhost:8080

### Production Deployment

1. **Build the application**:
   ```bash
   # Build the Rust backend
   cargo build --release
   
   # Build the web UI
   cd web-ui
   trunk build --release
   cd ..
   ```

2. **Run the production server**:
   ```bash
   # Set environment variables
   export RUST_LOG=info
   export PORT=8080
   
   # Run the server
   ./target/release/fandango
   ```

3. **Access the application**:
   - Web UI: http://your-server-ip:8080
   - API: http://your-server-ip:8080/api
   - WebSocket: ws://your-server-ip:8080/ws

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   # Build the main application
   docker build -t fandango .
   
   # Or build just the quantization server
   cd quantization_server
   docker build -t fandango-quantization .
   ```

2. **Run the containers**:
   ```bash
   # Main application
   docker run -d -p 8000:8000 --name fandango fandango
   
   # Quantization server
   docker run -d -p 8080:8080 --name fandango-quantization fandango-quantization
   ```

## Development

### Project Structure

- `src/` - Rust source code
  - `bin/` - Binary targets including the quantization server
  - `lib.rs` - Library root
  - `web/` - Web server implementation
  - `core/` - Core functionality
- `quantization_server/` - Standalone quantization server
  - `src/` - Server implementation
  - `Cargo.toml` - Server dependencies
- `web-ui/` - Web frontend (Yew + WebAssembly)
  - `src/` - Frontend source code
  - `static/` - Static assets

### Quantization Server API

The quantization server provides the following endpoints:

- `GET /health` - Health check endpoint
- `POST /api/quantize` - Quantize a model
- `POST /api/infer/{model_name}` - Run inference with a quantized model

See [QUANTIZATION_SERVER.md](QUANTIZATION_SERVER.md) for detailed API documentation.

### Building for WebAssembly

```bash
cd web-ui
wasm-pack build --target web --out-name wasm --out-dir ./static
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
