# llama.cpp GGUF Experiment

**Status:** Isolated experiment (not yet integrated into main self-evolution loops)

## Purpose

This crate provides reliable GGUF model loading and inference using **llama.cpp** via Rust bindings.

It serves as the **primary path** for loading and running large language models in Rathor.ai until pure-Rust alternatives (such as Candle) reach comparable maturity for GGUF support.

## Goals

- Provide stable, high-performance GGUF model loading
- Support GPU acceleration (CUDA, Metal, Vulkan)
- Keep integration clean and configurable
- Serve as the foundation for actual reasoning inside the self-evolution loops

## Current Scope (Phase 1)

- Basic model loading from GGUF files
- Session creation
- Foundation for text generation

## Planned Next Steps

- Add text generation / completion functions
- Add proper error handling and configuration
- Integrate with `run_self_evolution_loop()` via feature flags
- Performance benchmarking

## Notes

- This experiment runs in parallel with the `wasm-ml-candle` exploration.
- New developments in Rust ML / Wasm will be reviewed cyclically, but we will not switch core paths without strong justification.

## Safety & Control

Model loading and inference will remain under the control of the self-evolution loop, respecting mercy-gating and TOLC evaluation principles.