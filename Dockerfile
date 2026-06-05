# Powrush MMO Production Dockerfile v14.11
# PATSAGi Council + Ra-Thor Thunder approved
# Multi-stage build for minimal secure image
# One container = TCP (7777) + WebSocket (7778) + Beautiful Browser Client (8080)

# ==================== BUILDER ====================
FROM rust:1.80-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock* ./
COPY powrush/Cargo.toml ./powrush/

# Create dummy main to cache dependencies
RUN mkdir -p powrush/src && echo 'fn main(){}' > powrush/src/main.rs || true
RUN cargo build --release --features server -p powrush || true

# Now copy real source
COPY . .

# Build the production binary
RUN cargo build --release --features server -p powrush

# ==================== RUNTIME ====================
FROM debian:bookworm-slim

WORKDIR /app

# Runtime dependencies (minimal)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 powrush

# Copy the compiled binary
COPY --from=builder /app/target/release/powrush-server /app/powrush-server

# Copy the beautiful self-contained browser client
COPY --from=builder /app/powrush/web /app/web

# Create directories for logs/config (writable by non-root)
RUN mkdir -p /app/logs && chown -R powrush:powrush /app

USER powrush

# Environment defaults (override in docker-compose or k8s)
ENV POWRUSH_TCP_PORT=7777 \
    POWRUSH_WS_PORT=7778 \
    POWRUSH_HTTP_PORT=8080 \
    POWRUSH_TICK_RATE_MS=100 \
    RUST_LOG=info

EXPOSE 7777 7778 8080

# Healthcheck (simple TCP on game port)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD nc -z localhost 7777 || exit 1

# Run the unified production server
ENTRYPOINT ["./powrush-server"]
