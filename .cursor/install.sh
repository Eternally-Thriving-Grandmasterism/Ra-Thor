#!/usr/bin/env bash
# Ra-Thor Cloud Agent install — idempotent dependency/toolchain refresh.
# The Core Tier-1 gate needs a Rust toolchain with edition2024 support
# (>= 1.85). The default base image ships an older stable, so pin the
# latest stable via rustup before building.
set -euo pipefail

# Tier-1 default workspace members (see root Cargo.toml + core-tier1-ci.yml).
TIER1_PACKAGES=(
  lattice-conductor-v14
  reality-thriving-transfer
  kardashev-orchestration
  ra-thor-one-organism
  github-connector
  gpu-compute-pipeline
  quantum-swarm
  sovereign-recovery
  ra-thor-monorepo-intelligence
  mercy_tolc_operator_algebra
  fractal-mercy-ledger-adapter
  mercy-security
)

rustup toolchain install stable --profile minimal --no-self-update
rustup default stable
rustup component add rustfmt clippy

# Expand into "-p <pkg>" pairs for cargo.
PKG_ARGS=()
for pkg in "${TIER1_PACKAGES[@]}"; do
  PKG_ARGS+=(-p "$pkg")
done

# Warm the build cache for the Core Tier-1 gate so agents start fast.
cargo build "${PKG_ARGS[@]}"

# Warm the ONE Organism web demo (the runnable HTTP app).
cargo build -p ra-thor-one-organism --example one_organism_web_demo --features web-demo
