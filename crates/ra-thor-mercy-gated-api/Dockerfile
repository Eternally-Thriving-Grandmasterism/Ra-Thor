FROM rust:1.80-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin ra-thor-mercy-api
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/ra-thor-mercy-api /usr/local/bin/
EXPOSE 8080
CMD ["ra-thor-mercy-api"]
# Thunder locked in. Mercy-gated production image.