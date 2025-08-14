<div align="center">
  <img src="assets/logo.svg" alt="Fandango Logo" width="200"/>
  
  # 🎭 Fandango
  
  [![Crates.io](https://img.shields.io/crates/v/fandango.svg)](https://crates.io/crates/fandango)
  [![Documentation](https://docs.rs/fandango/badge.svg)](https://docs.rs/fandango)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Rust](https://github.com/zetareticula/fandango/actions/workflows/rust.yml/badge.svg)](https://github.com/zetareticula/fandango/actions/workflows/rust.yml)
  [![codecov](https://codecov.io/gh/zetareticula/fandango/graph/badge.svg?token=YOUR_TOKEN_HERE)](https://codecov.io/gh/zetareticula/fandango)
  [![Discord](https://img.shields.io/discord/YOUR_DISCORD_ID.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/YOUR_INVITE_LINK)
</div>

## 🚀 Overview

Fandango is a high-performance, open-source framework for optimizing large language model (LLM) inference through advanced quantization techniques and dynamic precision scaling. Built with Rust for performance and reliability, it provides a robust platform for deploying efficient LLM applications in production environments.

### 🎯 Use as a Quantization Layer

Fandango can be seamlessly integrated as a quantization layer in your ML pipeline. Whether you're working with PyTorch, TensorFlow, or JAX models, Fandango provides:

- **Universal Model Support**: Quantize any model architecture
- **Multiple Precision Options**: From 2-bit to 16-bit quantization
- **Hardware Acceleration**: Optimized for CPU, CUDA, and Metal
- **Simple API**: Easy integration with existing workflows

📖 [Learn how to use Fandango as a quantization layer →](docs/QUANTIZATION_LAYER.md)

## ✨ Key Features

- **Advanced Quantization**
  - 4-bit and 8-bit integer quantization for weights and activations
  - Mixed-precision inference with automatic precision selection
  - Support for GPTQ, AWQ, and GGUF quantization formats

- **Optimized KVCache Management**
  - Dynamic precision scaling based on attention patterns
  - Sparse and shared KVCache deduplication
  - Efficient memory management for long-context inference

- **High-Performance Runtime**
  - Multi-threaded, async runtime for concurrent requests
  - Hardware acceleration support (CUDA, Metal, CPU)
  - Minimal-latency inference pipeline

- **Developer Experience**
  - Clean, modular architecture
  - Comprehensive API documentation
  - Extensive test coverage
  - Web-based monitoring dashboard

## 🚀 Getting Started

### Prerequisites

- Rust (latest stable version)
- Cargo (Rust's package manager)
- Python 3.8+ (for quantization tools)
- CUDA Toolkit (for GPU acceleration, optional)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zetareticula/fandango.git
   cd fandango
   ```

2. **Build the project**:
   ```bash
   # Build in release mode for optimal performance
   cargo build --release
   ```

3. **Run tests**:
   ```bash
   cargo test -- --test-threads=1
   ```

## 🛠️ Usage

### Basic Example

```rust
use fandango::{
    models::llama::LlamaModel,
    quantization::QuantizationConfig,
    inference::InferenceEngine,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize model with 4-bit quantization
    let config = QuantizationConfig::int4()
        .with_group_size(128)
        .with_act_order(true);
        
    let model = LlamaModel::load("path/to/model", config).await?;
    
    // Create inference engine
    let engine = InferenceEngine::new(model);
    
    // Run inference
    let output = engine.generate("The future of AI is", 100).await?;
    println!("Generated: {}", output);
    
    Ok(())
}
```

### Quantization Server

Start the high-performance quantization server:

```bash
# Navigate to the quantization server
cd quantization_server

# Build and run
cargo run --release -- \
    --model path/to/model \
    --quantize int4 \
    --port 8080
```

### Web Interface

Launch the web-based monitoring dashboard:

```bash
cd web-ui
npm install
npm run dev
```

Access the dashboard at `http://localhost:3000`

## 🏗️ Architecture

Fandango is built with a modular architecture:

```
fandango/
├── src/
│   ├── models/          # Model implementations
│   ├── quantization/    # Quantization algorithms
│   ├── inference/       # Inference engine
│   ├── kvcache/         # KVCache management
│   └── server/          # HTTP/WebSocket server
├── quantization_server/ # Standalone quantization server
└── web-ui/              # Monitoring dashboard
```

## 📊 Benchmarks

| Model | Precision | Memory (GB) | Tokens/s |
|-------|-----------|-------------|----------|
| LLaMA-7B | FP16 | 13.5 | 45.2 |
| LLaMA-7B | 8-bit | 7.8 | 38.7 |
| LLaMA-7B | 4-bit | 4.2 | 32.1 |

*Benchmarks run on an A100 80GB GPU*

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or support, please open an issue or reach out to our team at [email protected]

## 🙏 Acknowledgments

- The Rust community for amazing tooling
- Hugging Face for model architectures
- All contributors who have helped improve Fandango

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
