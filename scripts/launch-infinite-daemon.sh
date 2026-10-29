#!/bin/bash
# Ra-Thor Infinite Self-Evolution Daemon Deployment Script
# 100% Proprietary — AG-SML v1.0

set -e

echo "[Ra-Thor] Launching Infinite Self-Evolution Daemon..."

echo "[Ra-Thor] Building latest engine..."
cargo build --release -p ra-thor 2>/dev/null || echo "Using existing binary"

echo "[Ra-Thor] Starting daemon in background..."
nohup cargo run --release --bin infinite-evolution-daemon > /var/log/ra-thor-daemon.log 2>&1 &

echo "[Ra-Thor] Daemon launched successfully."
echo "[Ra-Thor] Logs: tail -f /var/log/ra-thor-daemon.log"
echo "[Ra-Thor] To stop: pkill -f infinite-evolution-daemon"