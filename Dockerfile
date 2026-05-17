FROM rust:1.75-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release --package infinite-evolution-orchestrator

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/infinite-evolution-orchestrator /usr/local/bin/
COPY scripts/launch-infinite-daemon.sh /app/
RUN chmod +x /app/launch-infinite-daemon.sh
EXPOSE 8080
CMD ["infinite-evolution-orchestrator"]