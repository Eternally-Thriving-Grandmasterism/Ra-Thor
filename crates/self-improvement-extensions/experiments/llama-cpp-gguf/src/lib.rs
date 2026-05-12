/// Experimental module: llama.cpp GGUF model loading and text generation
///
/// This is an isolated experiment and is **not yet integrated** into the main
/// self-evolution cosmic loops.
///
/// Primary path for GGUF model loading until pure-Rust alternatives mature.

use llama_cpp_rs::{LlamaModel, LlamaParams, SamplingParams, SessionParams};

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

/// Generate text from a prompt using the loaded model
pub fn generate_text(
    model: &LlamaModel,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String, String> {
    let session = model
        .create_session(SessionParams {
            n_ctx: 4096,
            ..Default::default()
        })
        .map_err(|e| format!("Failed to create session: {}", e))?;

    let mut output = String::new();

    let sampling = SamplingParams {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        repeat_penalty: config.repeat_penalty,
        ..Default::default()
    };

    // Tokenize prompt
    let tokens = model
        .tokenize(prompt, true)
        .map_err(|e| format!("Tokenization failed: {}", e))?;

    // Feed prompt
    session
        .feed_tokens(&tokens)
        .map_err(|e| format!("Failed to feed prompt: {}", e))?;

    // Generate tokens
    for _ in 0..config.max_tokens {
        let token = session
            .sample(&sampling)
            .map_err(|e| format!("Sampling failed: {}", e))?;

        if model.is_eos(token) {
            break;
        }

        let piece = model
            .detokenize(&[token])
            .map_err(|e| format!("Detokenization failed: {}", e))?;

        output.push_str(&piece);

        session
            .feed_tokens(&[token])
            .map_err(|e| format!("Failed to feed token: {}", e))?;
    }

    Ok(output)
}