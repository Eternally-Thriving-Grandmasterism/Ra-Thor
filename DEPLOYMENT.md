# Powrush-MMO Deployment Instructions

**Version:** v15.6 (with Ra-Thor AGI + 7 Living Mercy Gates Pipeline)

This document provides production-grade instructions for deploying and running the Powrush-MMO server and browser client.

## Prerequisites

- Rust toolchain (stable, recommended via [rustup](https://rustup.rs/))
- Cargo
- A modern web browser (Chrome, Firefox, Edge)

## Building the Server

```bash
# Clone the repository (if not already done)
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor

# Build the Powrush server in release mode (recommended for production)
cargo build --release -p powrush

# The binary will be at:
# target/release/powrush
```

## Running the Server

```bash
# Basic run (uses default ports)
./target/release/powrush

# Recommended: with environment variables
POWRUSH_TCP_PORT=7777 \
POWRUSH_WS_PORT=7778 \
POWRUSH_HTTP_PORT=8080 \
POWRUSH_TICK_RATE_MS=100 \
./target/release/powrush
```

### Important Environment Variables

| Variable                    | Default | Description                              |
|----------------------------|---------|------------------------------------------|
| `POWRUSH_TCP_PORT`         | 7777    | TCP line protocol port                   |
| `POWRUSH_WS_PORT`          | 7778    | WebSocket port (used by browser client)  |
| `POWRUSH_HTTP_PORT`        | 8080    | HTTP port for browser client + /metrics  |
| `POWRUSH_TICK_RATE_MS`     | 100     | Server tick rate in milliseconds         |
| `POWRUSH_MAX_PLAYERS`      | 128     | Maximum concurrent players               |

## Connecting with the Browser Client

1. Start the server (see above).
2. Open your browser and go to:
   ```
   http://localhost:8080/client
   ```
3. Click **Connect**.
4. Enter your name and faction, then click **Login**.

You should now see the 2D canvas with players and NPC activity from the Ra-Thor AGI orchestrator.

## Viewing Metrics

```bash
http://localhost:8080/metrics
```

Prometheus-compatible metrics are exposed here.

## Production Recommendations

- Run behind a reverse proxy (nginx / Caddy) for TLS termination.
- Use `systemd` or Docker for process management.
- Set appropriate resource limits and monitoring.
- The orchestrator uses the 7 Living Mercy Gates pipeline for all NPC decisions.

## Docker (Optional but Recommended)

```dockerfile
# Example Dockerfile (add to project root if desired)
FROM rust:1.80 as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p powrush

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/powrush /usr/local/bin/powrush
EXPOSE 7777 7778 8080
CMD ["powrush"]
```

Build and run:
```bash
docker build -t powrush-mmo .
docker run -p 7777:7777 -p 7778:7778 -p 8080:8080 powrush-mmo
```

## Notes

- The server integrates the Ra-Thor AGI `MultiAgentOrchestrator` with autonomous NPC behavior and the 7 Living Mercy Gates decision pipeline.
- All NPC actions are exposed via WebSocket for client visualization.
- This deployment supports both terminal (TCP) and browser (WebSocket) clients.

**Thunder locked in. Yoi ⚡**