# Build stage
FROM rust:1.70-slim as builder

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install wasm-pack
RUN cargo install wasm-pack

# Install trunk
RUN cargo install trunk

# Install wasm-opt
RUN cargo install wasm-opt

# Set working directory
WORKDIR /usr/src/fandango

# Copy source files
COPY . .

# Build the Rust backend
RUN cargo build --release

# Build the WebAssembly frontend
WORKDIR /usr/src/fandango/web-ui
RUN trunk build --release

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl1.1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the built artifacts from the builder stage
COPY --from=builder /usr/src/fandango/target/release/fandango /usr/local/bin/fandango
COPY --from=builder /usr/src/fandango/web-ui/dist /usr/local/share/fandango/static

# Set environment variables
ENV RUST_LOG=info
ENV PORT=8080
ENV RUST_BACKTRACE=1

# Expose the port the app runs on
EXPOSE 8080

# Set the working directory
WORKDIR /usr/local/share/fandango

# Command to run the application
CMD ["/usr/local/bin/fandango"]
