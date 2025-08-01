# Fandango Quantization Server

A high-performance, production-ready quantization server for large language models, built with Rust and Actix-Web. This server provides efficient model quantization and inference capabilities with minimal latency and memory footprint.

## Features

- 🚀 **High-Performance**: Built with Rust for maximum performance and safety
- ⚡ **Low Latency**: Optimized for real-time model serving
- 🔒 **Thread-Safe**: Concurrent request handling with async/await
- 📊 **Metrics**: Built-in performance monitoring
- 🔄 **WebSocket Support**: Real-time updates and streaming
- 🐳 **Docker Ready**: Containerized deployment

## Table of Contents

- [API Reference](#api-reference)
  - [Health Check](#health-check)
  - [Quantize Model](#quantize-a-model)
  - [Run Inference](#run-inference)
  - [List Models](#list-models)
  - [Delete Model](#delete-model)
- [Error Handling](#error-handling)
- [WebSocket API](#websocket-api)
- [Performance Tuning](#performance-tuning)
- [Deployment](#deployment)
- [Development](#development)
- [License](#license)

## API Reference

All API endpoints are prefixed with `/api/v1`. The server uses standard HTTP status codes and JSON for request/response bodies.

### Base URL
```
http://localhost:8080/api/v1
```

### Authentication

All endpoints require an API key in the `X-API-Key` header:
```
X-API-Key: your_api_key_here
```

### Health Check

Check if the server is running and healthy.

```http
GET /health
```

#### Response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 1234.56,
  "models_loaded": 5,
  "memory_usage_mb": 256.78
}
```

### Quantize a Model

Quantize a model and store it in memory for inference.

```http
POST /api/v1/quantize
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | string | Yes | Path to the model file (local or URL) |
| `model_name` | string | Yes | Unique name to identify the quantized model |
| `bits` | integer | No | Number of bits for quantization (default: 8) |
| `dims` | array | No | Dimensions of the weight tensor (default: [1024, 1024]) |
| `quant_method` | string | No | Quantization method: 'uniform' or 'dynamic' (default: 'uniform') |
| `group_size` | integer | No | Group size for group-wise quantization (default: 64) |

```json
{
  "model_path": "models/llama2-7b.bin",
  "model_name": "llama2-7b-int8",
  "bits": 8,
  "dims": [4096, 4096],
  "quant_method": "uniform",
  "group_size": 128
}
```

#### Response

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `model_name` | string | Name of the quantized model |
| `original_size` | integer | Size of original model in bytes |
| `quantized_size` | integer | Size of quantized model in bytes |
| `compression_ratio` | float | Compression ratio (original_size / quantized_size) |
| `quantization_time_ms` | float | Time taken for quantization in milliseconds |

```json
{
  "status": "success",
  "model_name": "llama2-7b-int8",
  "original_size": 13800000000,
  "quantized_size": 3450000000,
  "compression_ratio": 4.0,
  "quantization_time_ms": 1250.32
}
```

### Run Inference

Run inference using a quantized model.

```http
POST /api/v1/infer/{model_name}
```

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | Yes | Name of the quantized model |

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | array | Yes | Input tensor as a flat array |
| `temperature` | float | No | Sampling temperature (default: 1.0) |
| `max_tokens` | integer | No | Maximum number of tokens to generate (default: 50) |
| `stream` | boolean | No | Whether to stream the response (default: false) |

```json
{
  "input": [0.1, 0.2, 0.3, ...],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

#### Response (Non-Streaming)

```json
{
  "status": "success",
  "model": "llama2-7b-int8",
  "output": [0.5, 0.3, 0.2, ...],
  "inference_time_ms": 45.67,
  "tokens_generated": 42
}
```

#### Response (Streaming)

When `stream: true`, the response is sent as Server-Sent Events (SSE):

```
event: token
data: {"token": "Hello", "id": 1}

event: token
data: {"token": " world", "id": 2}

event: done
data: {"status": "success", "tokens_generated": 2}
```

### List Models

List all loaded quantized models.

```http
GET /api/v1/models
```

#### Response

```json
{
  "status": "success",
  "models": [
    {
      "name": "llama2-7b-int8",
      "size_bytes": 3450000000,
      "quantization_bits": 8,
      "created_at": "2025-07-31T17:30:00Z"
    },
    {
      "name": "mistral-7b-int4",
      "size_bytes": 2000000000,
      "quantization_bits": 4,
      "created_at": "2025-07-30T10:15:00Z"
    }
  ]
}
```

### Delete Model

Remove a quantized model from memory.

```http
DELETE /api/v1/models/{model_name}
```

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | Yes | Name of the model to delete |

#### Response

```json
{
  "status": "success",
  "message": "Model 'llama2-7b-int8' deleted successfully",
  "freed_memory_bytes": 3450000000
}
```

## Error Handling

All error responses follow this format:

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_INPUT",
    "message": "Invalid input tensor dimensions",
    "details": {
      "expected": [1024, 1024],
      "got": [1024, 1025]
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTH_REQUIRED` | 401 | Missing or invalid API key |
| `INVALID_INPUT` | 400 | Invalid request parameters |
| `MODEL_NOT_FOUND` | 404 | Specified model not found |
| `QUANTIZATION_ERROR` | 500 | Error during quantization |
| `INFERENCE_ERROR` | 500 | Error during inference |
| `OUT_OF_MEMORY` | 507 | Not enough memory to load model |

## WebSocket API

For real-time interaction, the server provides WebSocket endpoints:

```
ws://localhost:8080/api/v1/ws
```

### Messages

#### Quantize Model
```json
{
  "type": "quantize",
  "id": "req_123",
  "model_path": "models/llama2-7b.bin",
  "model_name": "llama2-7b-int8",
  "bits": 8
}
```

#### Stream Tokens
```json
{
  "type": "generate",
  "id": "req_456",
  "model_name": "llama2-7b-int8",
  "prompt": "Hello, how are you?",
  "max_tokens": 100
}
```

## Performance Tuning

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `WORKERS` | num_cpus | Number of worker threads |
| `MAX_MODEL_SIZE_MB` | 8192 | Maximum model size in MB |
| `LOG_LEVEL` | info | Logging level (error, warn, info, debug, trace) |
| `CACHE_DIR` | ./cache | Directory for cached models |

### Benchmark Results

| Model | Precision | Batch Size | Throughput (tokens/s) | Latency (ms/token) |
|-------|-----------|------------|----------------------|-------------------|
| LLaMA-7B | FP16 | 1 | 45.2 | 22.1 |
| LLaMA-7B | INT8 | 1 | 78.5 | 12.7 |
| LLaMA-7B | INT4 | 1 | 112.3 | 8.9 |
| LLaMA-7B | INT8 | 8 | 210.4 | 38.0 |

## Deployment

### Docker

```bash
# Build the image
docker build -t fandango-quantization .

# Run with custom config
docker run -d \
  -p 8080:8080 \
  -e PORT=8080 \
  -e WORKERS=4 \
  -e MAX_MODEL_SIZE_MB=16384 \
  fandango-quantization
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fandango-quantization
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fandango-quantization
  template:
    metadata:
      labels:
        app: fandango-quantization
    spec:
      containers:
      - name: server
        image: fandango-quantization:latest
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080"
        - name: WORKERS
          value: "4"
        resources:
          limits:
            cpu: "4"
            memory: "16Gi"
          requests:
            cpu: "2"
            memory: "12Gi"
```

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/zetareticula/fandango.git
cd fandango/quantization_server

# Build in release mode
cargo build --release

# Run with debug logging
RUST_LOG=debug ./target/release/quantization_server
```

### Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test -- --ignored

# Run benchmarks
cargo bench
```

## License

```
Copyright 2025 Zeta Reticula

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
