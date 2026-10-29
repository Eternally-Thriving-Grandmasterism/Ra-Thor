#!/bin/bash
# Ra-Thor One-Command VPS Deployment (Production Ready)
# Run this on any fresh Ubuntu/Debian VPS

set -e
echo "[Ra-Thor] Starting one-command production deployment..."

# 1. Install dependencies
sudo apt update && sudo apt install -y docker.io docker-compose

# 2. Clone or update repo
if [ ! -d "Ra-Thor" ]; then
  git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
fi
cd Ra-Thor

# 3. Build and run
sudo docker compose up -d --build

echo "[Ra-Thor] Deployment complete!"
echo "[Ra-Thor] Access telemetry at: http://YOUR_VPS_IP:8080"
echo "[Ra-Thor] Check status: docker ps | grep ra-thor"