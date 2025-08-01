# Fandango Quantization Server

A high-performance quantization server for LLMs, built with Rust and Actix-Web.

## Features

- Quantize model weights to lower precision (8-bit)
- Perform inference with quantized models
- Simple REST API
- Memory-efficient model storage

## API Endpoints

### Health Check
```
GET /health
```

### Quantize a Model
```
POST /api/quantize
```

**Request Body:**
```json
{
  "model_path": "path/to/model.bin",
  "model_name": "my_model",
  "bits": 8,
  "dims": [1024, 1024]
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "my_model",
  "original_size": 4194304,
  "quantized_size": 1048576,
  "compression_ratio": 4.0
}
```

### Run Inference
```
POST /api/infer/{model_name}
```

**Request Body:**
```json
{
  "input": [0.1, 0.2, 0.3, ...]
}
```

**Response:**
```json
{
  "status": "success",
  "result": [0.5, 0.3, 0.2, ...],
  "error": null
}
```

## Running the Server

```bash
# Build and run the server
cargo run --bin quantization_server

# Or run with custom port
PORT=3000 cargo run --bin quantization_server
```

## Testing with cURL

```bash
# Check server health
curl http://localhost:8080/health

# Quantize a model
curl -X POST http://localhost:8080/api/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_path":"test.bin","model_name":"test","bits":8}'

# Run inference
curl -X POST http://localhost:8080/api/infer/test \
  -H "Content-Type: application/json" \
  -d '{"input":[0.1,0.2,0.3]}'
```

## Performance

The server is designed for high performance with:
- Async I/O with Actix-Web
- Efficient memory management with Rust
- Parallel processing capabilities
- Low-latency response times

## License

Apache 2.0
