#!/bin/bash
# wasm-build.sh — one-click build helper
echo "⚡ Building Rust/WASM 1048576D lattice engine..."
cd "$(dirname "$0")"
wasm-pack build --target web --out-dir pkg src/lattice --out-name 1048576d_wzw_engine
echo "✅ WASM ready! Refresh ra-thor-standalone-demo.html"
