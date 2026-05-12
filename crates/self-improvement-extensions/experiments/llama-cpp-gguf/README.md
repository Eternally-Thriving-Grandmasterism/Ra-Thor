# llama.cpp GGUF Experiment

**Status:** Isolated experiment (not yet integrated into main self-evolution loops)

## Purpose

This crate provides reliable GGUF model loading and text generation using **llama.cpp** via Rust bindings.

It serves as the **primary path** for loading and running large language models in Rathor.ai until pure-Rust alternatives (such as Candle) reach comparable maturity for GGUF support.

## Current Capabilities (Phase 1)

- Load GGUF models from disk
- Create inference sessions
- Generate text from a prompt with configurable sampling parameters

## Usage Example

```rust
use llama_cpp_gguf::{load_gguf_model, generate_text, ModelConfig, GenerationConfig};

let config = ModelConfig {
    model_path: "models/phi-2.Q4_K_M.gguf".to_string(),
    context_size: 4096,
    gpu_layers: 99,
};

let model = load_gguf_model(&config).expect("Failed to load model");

let gen_config = GenerationConfig {
    max_tokens: 200,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
};

let output = generate_text(&model, "Explain the concept of self-evolution in AI.", &gen_config)
    .expect("Generation failed");

println!("Generated text:\n{}", output);
```

## Sampling Parameters

- `temperature`: Controls randomness (higher = more creative)
- `top_p`: Nucleus sampling
- `top_k`: Top-K sampling
- `repeat_penalty`: Reduces repetition
- `max_tokens`: Maximum number of tokens to generate

## Next Steps (Planned)

- Add streaming generation support
- Improve error handling and configuration
- Add support for system prompts / chat templates
- Integrate with `run_self_evolution_loop()` via feature flags

## Notes

- This experiment runs in parallel with the `wasm-ml-candle` exploration.
- New developments in Rust ML / Wasm will be reviewed cyclically, but we will not switch core paths without strong justification.

## Safety & Control

Model loading and inference remain under the control of the self-evolution loop, respecting mercy-gating and TOLC evaluation principles.