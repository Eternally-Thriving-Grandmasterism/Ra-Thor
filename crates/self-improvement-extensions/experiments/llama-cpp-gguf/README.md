# llama.cpp GGUF Experiment

**Status:** Isolated experiment

## Current Features

- GGUF model loading
- Text generation (streaming + non-streaming)
- Chat messages with system prompt support
- **Basic tool/function calling support**

## Tool Calling Example

```rust
use llama_cpp_gguf::{generate_chat_with_tools, try_parse_tool_call, ChatMessage, Tool, GenerationConfig, ModelConfig, load_gguf_model};

let tools = vec![Tool {
    name: "create_github_issue".to_string(),
    description: "Create a GitHub issue".to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "body": { "type": "string" }
        },
        "required": ["title", "body"]
    }),
}];

let messages = vec![ChatMessage {
    role: "user".to_string(),
    content: "Create an issue about improving the self-evolution loop".to_string(),
}];

let response = generate_chat_with_tools(&model, &messages, Some(&tools), &GenerationConfig::default()).unwrap();

if let Some(tool_call) = try_parse_tool_call(&response) {
    println!("Model wants to call tool: {} with args: {}", tool_call.name, tool_call.arguments);
} else {
    println!("Model responded normally: {}", response);
}
```

## Notes

Tool calling is implemented via prompt engineering + JSON parsing. Native tool calling (for models that support it) can be added later.