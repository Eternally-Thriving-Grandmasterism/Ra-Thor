/// Experimental module: llama.cpp GGUF model loading and text generation
///
/// This is an isolated experiment and is **not yet integrated** into the main
/// self-evolution cosmic loops.
///
/// Primary path for GGUF model loading until pure-Rust alternatives mature.

use llama_cpp_rs::{LlamaModel, LlamaParams, SamplingParams, SessionParams};
use std::fmt;

/// Custom error type for llama.cpp operations
#[derive(Debug, Clone)]
pub enum LlamaError {
    InvalidConfig(String),
    ModelLoadError(String),
    SessionError(String),
    TokenizationError(String),
    GenerationError(String),
    Unknown(String),
}

impl fmt::Display for LlamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlamaError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            LlamaError::ModelLoadError(msg) => write!(f, "Failed to load model: {}", msg),
            LlamaError::SessionError(msg) => write!(f, "Session error: {}", msg),
            LlamaError::TokenizationError(msg) => write!(f, "Tokenization error: {}", msg),
            LlamaError::GenerationError(msg) => write!(f, "Generation error: {}", msg),
            LlamaError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl std::error::Error for LlamaError {}

/// Configuration for loading and running a GGUF model
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
            gpu_layers: 99,
        }
    }
}

/// Sampling parameters for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

/// Load a GGUF model
pub fn load_gguf_model(config: &ModelConfig) -> Result<LlamaModel, LlamaError> {
    if config.model_path.is_empty() {
        return Err(LlamaError::InvalidConfig("Model path cannot be empty".to_string()));
    }

    let model = LlamaModel::load_from_file(
        &config.model_path,
        LlamaParams {
            n_gpu_layers: config.gpu_layers,
            ..Default::default()
        },
    )
    .map_err(|e| LlamaError::ModelLoadError(e.to_string()))?;

    Ok(model)
}

/// Generate text from a prompt (non-streaming)
pub fn generate_text(
    model: &LlamaModel,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String, LlamaError> {
    let session = model
        .create_session(SessionParams {
            n_ctx: config.context_size,
            ..Default::default()
        })
        .map_err(|e| LlamaError::SessionError(e.to_string()))?;

    let mut output = String::new();

    let sampling = SamplingParams {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        repeat_penalty: config.repeat_penalty,
        ..Default::default()
    };

    let tokens = model
        .tokenize(prompt, true)
        .map_err(|e| LlamaError::TokenizationError(e.to_string()))?;

    session
        .feed_tokens(&tokens)
        .map_err(|e| LlamaError::SessionError(e.to_string()))?;

    for _ in 0..config.max_tokens {
        let token = session
            .sample(&sampling)
            .map_err(|e| LlamaError::GenerationError(e.to_string()))?;

        if model.is_eos(token) {
            break;
        }

        let piece = model
            .detokenize(&[token])
            .map_err(|e| LlamaError::GenerationError(e.to_string()))?;

        output.push_str(&piece);

        session
            .feed_tokens(&[token])
            .map_err(|e| LlamaError::SessionError(e.to_string()))?;
    }

    Ok(output)
}

/// Streaming text generation using a callback
pub fn generate_text_stream<F>(
    model: &LlamaModel,
    prompt: &str,
    config: &GenerationConfig,
    mut on_token: F,
) -> Result<(), LlamaError>
where
    F: FnMut(&str),
{
    let session = model
        .create_session(SessionParams {
            n_ctx: config.context_size,
            ..Default::default()
        })
        .map_err(|e| LlamaError::SessionError(e.to_string()))?;

    let sampling = SamplingParams {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        repeat_penalty: config.repeat_penalty,
        ..Default::default()
    };

    let tokens = model
        .tokenize(prompt, true)
        .map_err(|e| LlamaError::TokenizationError(e.to_string()))?;

    session
        .feed_tokens(&tokens)
        .map_err(|e| LlamaError::SessionError(e.to_string()))?;

    for _ in 0..config.max_tokens {
        let token = session
            .sample(&sampling)
            .map_err(|e| LlamaError::GenerationError(e.to_string()))?;

        if model.is_eos(token) {
            break;
        }

        let piece = model
            .detokenize(&[token])
            .map_err(|e| LlamaError::GenerationError(e.to_string()))?;

        on_token(&piece);

        session
            .feed_tokens(&[token])
            .map_err(|e| LlamaError::SessionError(e.to_string()))?;
    }

    Ok(())
}