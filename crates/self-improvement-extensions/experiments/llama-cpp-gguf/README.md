# llama.cpp GGUF Experiment

**Status:** Isolated experiment

## Current Features

- GGUF model loading
- Text generation (sync + async streaming)
- Chat with system prompt support
- Tool/function calling support

## Async Streaming Example

```rust
use llama_cpp_gguf::{generate_text_stream_async, ModelConfig, GenerationConfig, load_gguf_model};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let model = load_gguf_model(&ModelConfig {
        model_path: "models/phi-2.Q4_K_M.gguf".to_string(),
        ..Default::default()
    }).unwrap();

    let mut receiver = generate_text_stream_async(
        model,
        "Explain self-evolution in simple terms.".to_string(),
        GenerationConfig::default(),
    ).await.unwrap();

    while let Some(token) = receiver.recv().await {
        if token.starts_with("[ERROR]") {
            eprintln!("{}", token);
            break;
        }
        print!("{}", token);
    }
}
```