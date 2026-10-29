# llama-cpp-gguf

**Status:** Isolated Experiment  
**Location:** `crates/self-improvement-extensions/experiments/llama-cpp-gguf/`  
**Primary Purpose:** GGUF model loading and inference using llama.cpp

---

## Overview

This crate provides reliable GGUF model loading and inference using **llama.cpp** via Rust bindings. It serves as the **primary inference path** for running large language models in Rathor.ai until pure-Rust alternatives (such as Candle) reach comparable maturity in GGUF support and tool calling.

The crate is intentionally kept isolated until it is reviewed and approved for deeper integration into the self-evolution loops.

---

## Current Features

- Load GGUF models from disk
- Text generation (synchronous + async streaming)
- Chat generation with system prompt support
- Tool / function calling support (prompt-based + JSON parsing)

---

## Installation & Setup

This crate is an **internal experiment** and is not published on crates.io.

### Adding the Dependency

Add the following to your `Cargo.toml`:

```toml
[dependencies]
llama-cpp-gguf = { path = "crates/self-improvement-extensions/experiments/llama-cpp-gguf" }
```

### Building llama.cpp (Required)

`llama-cpp-rs` requires the llama.cpp library to be built. You have two options:

#### Option 1: Automatic Build (Recommended)

The crate will automatically download and build llama.cpp when you run:

```bash
cargo build
```

This is the **simplest and most common approach** for most users and development environments.

#### Option 2: Use a Prebuilt / System llama.cpp

If you already have llama.cpp built or installed system-wide, set the environment variable:

```bash
export LLAMA_CPP_PATH=/path/to/your/llama.cpp
```

Then build normally:

```bash
cargo build
```

### GPU Acceleration (Optional but Recommended)

Enable GPU backends using Cargo features:

```toml
[dependencies]
llama-cpp-gguf = { path = "...", features = ["cuda"] }
```

Supported features:
- `cuda` — NVIDIA GPUs (recommended when available)
- `metal` — Apple Silicon
- `vulkan` — Cross-platform GPU support

---

## Public API Summary

| Function                        | Description                              | Async |
|--------------------------------|------------------------------------------|-------|
| `load_gguf_model`              | Load a GGUF model                        | No    |
| `generate_text`                | Generate text from a prompt              | No    |
| `generate_chat`                | Chat generation with system prompt       | No    |
| `generate_chat_with_tools`     | Chat with optional tool calling          | No    |
| `generate_text_stream_async`   | Async streaming text generation          | Yes   |
| `try_parse_tool_call`          | Parse tool calls from model output       | No    |

---

## Recommended Configuration

For self-evolution workloads, we recommend:

- `gpu_layers: 99` (offload as much as possible)
- Quantization: **Q4_K_M** (default) or **Q5_K_M** (higher quality)
- Enable `flash_attn` when supported by the model
- `context_size`: 4096–8192 (depending on available VRAM)

---

## Usage Examples

See the examples in `examples/basic_loop_integration.rs` and the function documentation in `src/lib.rs`.

---

## Design Notes

- Tool calling is implemented via prompt engineering + JSON parsing for broad compatibility.
- Async streaming uses `tokio::task::spawn_blocking` + `mpsc` channels.
- The crate remains isolated until reviewed and approved for integration into `run_self_evolution_loop()`.

---

## Alignment

This experiment supports Rathor.ai’s goal of building reliable, mercy-gated, self-evolving intelligence by providing a stable and performant foundation for local LLM inference.