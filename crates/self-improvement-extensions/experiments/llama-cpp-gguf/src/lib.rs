/// Experimental module: llama.cpp GGUF model loading and text generation
///
/// This is an isolated experiment and is **not yet integrated** into the main
/// self-evolution cosmic loops.

use llama_cpp_rs::{LlamaModel, LlamaParams, SamplingParams, SessionParams};
use std::fmt;

#[derive(Debug, Clone)]
pub enum LlamaError {
    InvalidConfig(String),
    ModelLoadError(String),
    SessionError(String),
    TokenizationError(String),
    GenerationError(String),
    ToolCallError(String),
    Unknown(String),
}

impl fmt::Display for LlamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlamaError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            LlamaError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            LlamaError::SessionError(msg) => write!(f, "Session error: {}", msg),
            LlamaError::TokenizationError(msg) => write!(f, "Tokenization error: {}", msg),
            LlamaError::GenerationError(msg) => write!(f, "Generation error: {}", msg),
            LlamaError::ToolCallError(msg) => write!(f, "Tool call error: {}", msg),
            LlamaError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl std::error::Error for LlamaError {}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: String,
    pub context_size: u32,
    pub gpu_layers: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self { model_path: String::new(), context_size: 4096, gpu_layers: 99 }
    }
}

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
        Self { max_tokens: 256, temperature: 0.7, top_p: 0.9, top_k: 40, repeat_penalty: 1.1 }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

pub fn load_gguf_model(config: &ModelConfig) -> Result<LlamaModel, LlamaError> {
    if config.model_path.is_empty() {
        return Err(LlamaError::InvalidConfig("Model path empty".to_string()));
    }
    LlamaModel::load_from_file(&config.model_path, LlamaParams { n_gpu_layers: config.gpu_layers, ..Default::default() })
        .map_err(|e| LlamaError::ModelLoadError(e.to_string()))
}

pub fn generate_text(model: &LlamaModel, prompt: &str, config: &GenerationConfig) -> Result<String, LlamaError> {
    let session = model.create_session(SessionParams { n_ctx: config.context_size, ..Default::default() })
        .map_err(|e| LlamaError::SessionError(e.to_string()))?;

    let sampling = SamplingParams {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        repeat_penalty: config.repeat_penalty,
        ..Default::default()
    };

    let tokens = model.tokenize(prompt, true).map_err(|e| LlamaError::TokenizationError(e.to_string()))?;
    session.feed_tokens(&tokens).map_err(|e| LlamaError::SessionError(e.to_string()))?;

    let mut output = String::new();
    for _ in 0..config.max_tokens {
        let token = session.sample(&sampling).map_err(|e| LlamaError::GenerationError(e.to_string()))?;
        if model.is_eos(token) { break; }
        let piece = model.detokenize(&[token]).map_err(|e| LlamaError::GenerationError(e.to_string()))?;
        output.push_str(&piece);
        session.feed_tokens(&[token]).map_err(|e| LlamaError::SessionError(e.to_string()))?;
    }
    Ok(output)
}

/// Generate chat completion with optional tool calling support
pub fn generate_chat_with_tools(
    model: &LlamaModel,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    config: &GenerationConfig,
) -> Result<String, LlamaError> {
    let mut prompt = String::new();

    if let Some(tools) = tools {
        prompt.push_str("You can use tools. When using a tool, respond ONLY with valid JSON: {\"tool_call\": {\"name\": \"...\", \"arguments\": {...}}}\n\nAvailable tools:\n");
        for t in tools {
            prompt.push_str(&format!("- {}: {}\n", t.name, t.description));
        }
        prompt.push_str("\n");
    }

    for m in messages {
        prompt.push_str(&format!("<|{}|>\n{}\n", m.role, m.content));
    }
    prompt.push_str("<|assistant|>\n");

    generate_text(model, &prompt, config)
}

/// Try to extract a tool call from model output
pub fn try_parse_tool_call(text: &str) -> Option<ToolCall> {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(tc) = json.get("tool_call") {
            if let (Some(name), Some(args)) = (tc.get("name"), tc.get("arguments")) {
                return Some(ToolCall {
                    name: name.as_str().unwrap_or("").to_string(),
                    arguments: args.clone(),
                });
            }
        }
    }
    None
}