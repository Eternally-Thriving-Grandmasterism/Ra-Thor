# llama.cpp GGUF Experiment

**Status:** Isolated experiment (not yet integrated into main self-evolution loops)

## Purpose

This crate provides reliable GGUF model loading and text generation using **llama.cpp** via Rust bindings.

It serves as the **primary path** for loading and running large language models in Rathor.ai until pure-Rust alternatives (such as Candle) reach comparable maturity for GGUF support.

## Current Capabilities

- Load GGUF models from disk
- Create inference sessions
- Generate text from raw prompts
- Generate text from chat messages (with system prompt support)
- Streaming generation via callback

## Usage Examples

### Basic Text Generation

```rust
use llama_cpp_gguf::{generate_text, load_gguf_model, GenerationConfig, ModelConfig};

let model = load_gguf_model(&ModelConfig {
    model_path: "models/phi-2.Q4_K_M.gguf".to_string(),
    ..Default::default()
}).unwrap();

let output = generate_text(&model, "Explain self-evolution.", &GenerationConfig::default()).unwrap();
println!("{}", output);
```

### Chat with System Prompt

```rust
use llama_cpp_gguf::{generate_chat, ChatMessage, GenerationConfig, ModelConfig, load_gguf_model};

let model = load_gguf_model(&ModelConfig {
    model_path: "models/llama-3-8b.Q4_K_M.gguf".to_string(),
    ..Default::default()
}).unwrap();

let messages = vec![
    ChatMessage {
        role: "system".to_string(),
        content: "You are a helpful self-improvement agent.".to_string(),
    },
    ChatMessage {
        role: "user".to_string(),
        content: "How can I improve my daily routine?".to_string(),
    },
];

let output = generate_chat(&model, &messages, &GenerationConfig::default()).unwrap();
println!("{}", output);
```

### Streaming Generation

```rust
use llama_cpp_gguf::{generate_text_stream, GenerationConfig, ModelConfig, load_gguf_model};

let model = load_gguf_model(&ModelConfig { model_path: "...".to_string(), ..Default::default() }).unwrap();

generate_text_stream(&model, "Tell me a story.", &GenerationConfig::default(), |token| {
    print!("{}", token);
}).unwrap();
```

## Sampling Parameters

- `temperature`, `top_p`, `top_k`, `repeat_penalty`, `max_tokens`

## Next Steps

- Better chat template detection
- Tool calling / function calling support
- Integration into `run_self_evolution_loop()`

## Notes

This experiment runs in parallel with the `wasm-ml-candle` exploration. We review new developments cyclically but stay focused on stable progress.