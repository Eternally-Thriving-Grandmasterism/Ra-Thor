/// Experimental module: llama.cpp GGUF model loading
///
/// This is an isolated experiment and is **not yet integrated** into the main
/// self-evolution cosmic loops.
///
/// Purpose: Provide reliable GGUF model loading and inference using llama.cpp
/// as the primary path until pure-Rust alternatives (like Candle) mature.

use llama_cpp_rs::{LlamaModel, LlamaParams, SessionParams};

/// Configuration for loading a GGUF model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: String,
    pub context_size: u32,
    pub gpu_layers: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            context_size: 4096,
            gpu_layers: 99, // Use as many GPU layers as possible by default
        }
    }
}

/// Load a GGUF model with the given configuration
pub fn load_gguf_model(config: &ModelConfig) -> Result<LlamaModel, String> {
    if config.model_path.is_empty() {
        return Err("Model path cannot be empty".to_string());
    }

    let model = LlamaModel::load_from_file(
        &config.model_path,
        LlamaParams {
            n_gpu_layers: config.gpu_layers,
            ..Default::default()
        },
    )
    .map_err(|e| format!("Failed to load GGUF model: {}", e))?;

    Ok(model)
}

/// Create a new inference session
pub fn create_session(model: &LlamaModel, config: &ModelConfig) -> Result<llama_cpp_rs::Session, String> {
    let session = model
        .create_session(SessionParams {
            n_ctx: config.context_size,
            ..Default::default()
        })
        .map_err(|e| format!("Failed to create session: {}", e))?;

    Ok(session)
}