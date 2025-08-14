# Using Fandango as a Quantization Layer

Fandango provides a powerful and flexible quantization layer that can be integrated into any Rust-based machine learning pipeline. This document explains how to use Fandango as a standalone quantization service or as an embedded library.

## Table of Contents
- [Quick Start](#quick-start)
- [Integration Methods](#integration-methods)
  - [As a Library](#as-a-library)
  - [As a Service](#as-a-service)
- [Quantization Formats](#quantization-formats)
- [Advanced Configuration](#advanced-configuration)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Add Fandango to Your Project

Add this to your `Cargo.toml`:

```toml
[dependencies]
fandango = { git = "https://github.com/zetareticula/fandango.git" }
```

### Basic Usage

```rust
use fandango::{
    quantization::{QuantizationConfig, Quantizer},
    tensor::Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample tensor
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_slice(&data, &[2, 3])?;
    
    // Configure quantization
    let config = QuantizationConfig::int8()
        .with_group_size(64)
        .with_act_order(true);
    
    // Create quantizer
    let quantizer = Quantizer::new(config);
    
    // Quantize the tensor
    let quantized = quantizer.quantize(&tensor)?;
    
    // Dequantize back to f32
    let dequantized = quantizer.dequantize(&quantized)?;
    
    Ok(())
}
```

## Integration Methods

### As a Library

Fandango can be directly embedded in your Rust project:

1. **Add the dependency** as shown in the Quick Start
2. **Import the quantization module**
3. **Configure and use the quantizer**

### As a Service

For non-Rust applications, use the Fandango Quantization Server:

1. **Start the server**:
   ```bash
   cargo run --release --bin quantization_server -- --port 8080
   ```

2. **Make HTTP requests**:
   ```bash
   # Quantize a model
   curl -X POST http://localhost:8080/api/quantize \
     -H "Content-Type: application/json" \
     -d '{"model_path": "path/to/model", "config": {"quant_type": "int4"}}'
   ```

## Quantization Formats

Fandango supports multiple quantization formats:

| Format | Bits | Memory Savings | Accuracy | Speed |
|--------|------|----------------|----------|-------|
| FP16   | 16   | 2x             | Best     | Fast  |
| INT8   | 8    | 4x             | Great    | Faster|
| INT4   | 4    | 8x             | Good     | Fastest|
| Mixed  | 2-16 | Variable       | Best     | Fast  |

## Advanced Configuration

### Quantization Config

```rust
let config = QuantizationConfig::int4()
    .with_group_size(128)         // Group size for quantization
    .with_act_order(true)         // Activation order preservation
    .with_use_cuda(true)          // Enable CUDA acceleration
    .with_use_mkl(true)           // Enable MKL optimizations
    .with_verbose(true);          // Enable verbose logging
```

### Custom Quantization

Implement your own quantization strategy:

```rust
use fandango::quantization::{QuantizationStrategy, QuantizedTensor};

struct MyQuantizer;

impl QuantizationStrategy for MyQuantizer {
    fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        // Custom quantization logic
        todo!()
    }
    
    fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Tensor> {
        // Custom dequantization logic
        todo!()
    }
}
```

## Performance Considerations

1. **Hardware Acceleration**:
   - Enable CUDA for NVIDIA GPUs
   - Use MKL for Intel CPUs
   - Enable Metal for Apple Silicon

2. **Memory Usage**:
   - Larger group sizes reduce memory overhead
   - Consider batch size when processing multiple tensors

3. **Threading**:
   - Fandango automatically uses all available CPU cores
   - Set `RAYON_NUM_THREADS` to control thread count

## Troubleshooting

### Common Issues

1. **CUDA Errors**:
   - Ensure CUDA toolkit is installed
   - Check driver compatibility

2. **Performance Issues**:
   - Enable `--release` builds
   - Check CPU/GPU utilization
   - Monitor memory usage

3. **Quantization Artifacts**:
   - Try different group sizes
   - Adjust activation order
   - Consider mixed-precision

## Getting Help

For support, open an issue on [GitHub](https://github.com/zetareticula/fandango/issues) or join our [Discord](https://discord.gg/your-invite-link).
