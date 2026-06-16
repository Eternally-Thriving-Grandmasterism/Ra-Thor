#!/usr/bin/env bash
set -euo pipefail

# Build script for Lean Mercy Threshold → WASM
# Produces mercy_threshold.wasm that can be loaded by mercy-threshold-wasm crate

LEAN_DIR="lean/mercy-threshold"
OUT_DIR="target/wasm"
WASM_NAME="mercy_threshold.wasm"

echo "[Ra-Thor] Building Mercy Threshold WASM from Lean..."

# Ensure lake is available
if ! command -v lake &> /dev/null; then
    echo "Error: lake not found. Please install Lean 4 toolchain."
    exit 1
fi

mkdir -p "$OUT_DIR"

pushd "$LEAN_DIR" > /dev/null

# Clean previous build
lake clean || true

# Build the Lean package
lake build

# For Lean 4 WASM export we use the standard approach:
# Either lean --backend=wasm or lake with wasm target.
# Here we assume a configured lake target or use lean directly.
# In production this would be:
# lean --backend=wasm --o "$OUT_DIR/$WASM_NAME" MercyThreshold.lean

# For now we create a placeholder that the wasmtime harness can use
# until full lean4wasm integration is wired.
echo "[Warning] Using placeholder WASM for now. Full lean4wasm integration pending."

echo 'module.exports = {};' > "$OUT_DIR/$WASM_NAME.placeholder.js"  # placeholder

# In real setup, the actual .wasm would be generated here.
# For CI we will use a minimal valid WASM or build from Rust side as fallback.

echo "[Ra-Thor] WASM build step completed (Lean side)."

echo "Output would be at: $OUT_DIR/$WASM_NAME"

popd > /dev/null

echo "Done."