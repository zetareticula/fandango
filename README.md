# Fandango


## Overview

Fandango is an advanced, open-source platform designed to optimize large language model (LLM) inference through dynamic precision scaling. Built with Rust and Python, it leverages a Kubernetes-native architecture to deliver scalable, efficient performance. This project extends the existing directory structure and codebase to integrate sophisticated features, ensuring seamless compatibility with its core components.

## Key Features

- **Dynamic Precision Scaling**: Adjusts KVCache quantization levels based on attention locality and entropy heuristics, supporting 4-bit precision for low-load scenarios, 8-bit for high concurrency, and float16 for edge cases.
- **Sparse and Shared KVCache Deduplication**: Reduces memory usage by deduplicating KVCache, enhancing efficiency for batched and long-context inference.
- **Serverless Multi-Model Scheduler**: Enables on-demand loading and unloading of models, with caching of quantized layers to optimize performance and resource utilization.

Fandango’s design ensures robust integration and scalability, making it ideal for modern AI workloads. Explore the documentation and contribute to its development!



