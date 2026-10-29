# Wasm + Candle Experiment

**Status:** Isolated experiment (not yet integrated into main self-evolution loops)

## Purpose

This crate explores the feasibility of compiling and running [Candle](https://github.com/huggingface/candle) (a pure Rust ML framework) inside WebAssembly.

The goal is to evaluate whether Wasm can serve as a portable, sandboxed runtime for parts of Rathor.ai's self-evolution reasoning in the future.

## Current Scope (Phase 1)

- Basic tensor operations that compile and run both natively and in Wasm.
- Foundation for future model loading and inference inside Wasm.

## How to Build & Test

### Native
```bash
cargo test
```

### WebAssembly
```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown
```

## Next Steps (Planned)

- Load a small transformer model inside Wasm.
- Compare performance vs native Candle.
- Explore integration with `run_self_evolution_loop()` via feature flags.

## Safety Note

This experiment is intentionally isolated. It will only be integrated into the main cosmic loops after thorough evaluation.