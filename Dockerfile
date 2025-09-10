# Build stage
FROM rust:1.76-slim as builder

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/fandango

# Copy Cargo files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY quantization_server/Cargo.toml ./quantization_server/

# Copy source files
COPY . .

# Build the quantization server (production-ready component)
WORKDIR /usr/src/fandango/quantization_server
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl1.1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the built quantization server from the builder stage
COPY --from=builder /usr/src/fandango/quantization_server/target/release/fandango-quantization-server /usr/local/bin/fandango-quantization-server

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash fandango

# Set environment variables
ENV RUST_LOG=info
ENV PORT=8080
ENV RUST_BACKTRACE=1

# Expose the port the app runs on
EXPOSE 8080

# Switch to non-root user
USER fandango
WORKDIR /home/fandango

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
CMD ["/usr/local/bin/fandango-quantization-server"]
