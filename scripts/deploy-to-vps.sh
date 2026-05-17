#!/bin/bash
# Ra-Thor Phase 1 VPS Deployment Script
set -e
echo "[Ra-Thor] Starting Phase 1 deployment..."

# Build all crates
echo "[Ra-Thor] Building all crates..."
cargo build --release

# Deploy to VPS (placeholder - replace with real SSH/SCP)
echo "[Ra-Thor] Deploying to VPS..."
# scp target/release/* user@vps:/opt/ra-thor/

# Start services
echo "[Ra-Thor] Starting services..."
# ssh user@vps "cd /opt/ra-thor && ./eternal-evolution-daemon &"

echo "[Ra-Thor] Deployment complete."
echo "[Ra-Thor] Access Grafana at http://vps:3000"