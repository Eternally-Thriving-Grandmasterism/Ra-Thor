// monorepo-intelligence/src/lsp_symbol_resolver.rs
// Ra-Thor Monorepo Intelligence — LSP Symbol Resolution Integration v14.91 SYMBIOSIS
// Pluggable resolver for precise semantic symbol info via Language Server Protocol
// Complements tree-sitter chunking with compiler-level understanding (accurate kinds, signatures, definitions, references)
// TOLC 8 Living Mercy Gates | PATSAGi Councils | ONE Organism (Ra-Thor ↔ Grok) symbiosis
// Production-grade: fallback to improved syntactic, full LSP for semantic analysis

use crate::index_types::Symbol;

/// Trait for pluggable symbol resolution (syntactic or LSP-backed)
pub trait SymbolResolver {
    fn resolve_symbols(&self, content: &str, language: &str, file_path: &str) -> Vec<Symbol>;
}

/// Simple syntactic resolver — improved baseline (LSP-ready)
/// Handles more Rust constructs than original extract_symbols_simple
/// Can be replaced by LspSymbolResolver for full semantic power
pub struct SimpleSymbolResolver;

impl SymbolResolver for SimpleSymbolResolver {
    fn resolve_symbols(&self, content: &str, language: &str, _file_path: &str) -> Vec<Symbol> {
        extract_symbols_improved(content, language)
    }
}

/// LSP-backed resolver for production semantic symbol resolution
/// Uses Language Server Protocol (e.g. rust-analyzer for Rust, typescript-language-server for JS)
/// Provides: precise SymbolKind, range locations, container names, documentation, type info
/// Full integration: spawn server via stdio, initialize, didOpen, textDocument/documentSymbol or workspace/symbol
pub struct LspSymbolResolver {
    // server_command: String, // e.g. "rust-analyzer"
    // TODO: Add LSP client fields (process handle, jsonrpc transport, capabilities)
}

impl LspSymbolResolver {
    pub fn new(_server_command: &str) -> Self {
        // Example full integration steps:
        // 1. use std::process::{Command, Stdio};
        // 2. spawn Command::new(server_command).stdin(Stdio::piped()).stdout(Stdio::piped())
        // 3. Use jsonrpc_core for requests, lsp_types for structs
        // 4. Send initialize, initialized, textDocument/didOpen with content
        // 5. Request textDocument/documentSymbol -> Vec<SymbolInformation>
        // 6. Map to our Symbol (kind from SymbolKind enum, location from range)
        // Dependencies to add to Cargo.toml / workspace:
        //   lsp-types = "0.95"
        //   jsonrpc-core = "18"
        //   tokio = { version = "1", features = ["full"] }  // for async if needed
        //   For transport: may need lsp-server or custom stdio reader
        Self {}
    }

    /// Example: resolve symbols for a file (stub until full client)
    pub fn resolve_file(&self, content: &str, language: &str, file_path: &str) -> Vec<Symbol> {
        // In full impl: send LSP requests, parse response
        // For now: fallback with note
        eprintln!("// LspSymbolResolver stub for {} — using improved syntactic. Full LSP pending.", file_path);
        extract_symbols_improved(content, language)
    }
}

impl SymbolResolver for LspSymbolResolver {
    fn resolve_symbols(&self, content: &str, language: &str, file_path: &str) -> Vec<Symbol> {
        self.resolve_file(content, language, file_path)
    }
}

/// Improved extraction logic (LSP-like: more kinds, better signatures, line numbers)
/// Used by Simple and as fallback for Lsp
/// Extend this or replace entirely when full LSP client is wired
fn extract_symbols_improved(content: &str, language: &str) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    if language == "rust" || language == "rs" {
        for (line_num, line) in content.lines().enumerate() {
            let t = line.trim_start(); // preserve indent for context but trim for prefix
            let trimmed = t.trim();

            // Functions (pub fn, fn, async fn)
            if let Some(rest) = trimmed.strip_prefix("pub fn ").or_else(|| trimmed.strip_prefix("fn ")).or_else(|| trimmed.strip_prefix("async fn ")) {
                if let Some(end) = rest.find('(') {
                    let name = rest[..end].trim().to_string();
                    if !name.is_empty() {
                        symbols.push(Symbol {
                            name,
                            kind: "function".to_string(),
                            line_start: line_num,
                            line_end: line_num + 1,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                }
            }
            // Structs
            else if let Some(rest) = trimmed.strip_prefix("pub struct ").or_else(|| trimmed.strip_prefix("struct ")) {
                if let Some(end) = rest.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    let name = rest[..end].trim().to_string();
                    if !name.is_empty() {
                        symbols.push(Symbol {
                            name,
                            kind: "struct".to_string(),
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                } else if !rest.trim().is_empty() {
                    symbols.push(Symbol {
                        name: rest.trim().to_string(),
                        kind: "struct".to_string(),
                        line_start: line_num,
                        line_end: line_num,
                        signature: Some(trimmed.to_string()),
                    });
                }
            }
            // Enums
            else if let Some(rest) = trimmed.strip_prefix("pub enum ").or_else(|| trimmed.strip_prefix("enum ")) {
                if let Some(end) = rest.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    let name = rest[..end].trim().to_string();
                    if !name.is_empty() {
                        symbols.push(Symbol {
                            name,
                            kind: "enum".to_string(),
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                }
            }
            // Traits
            else if let Some(rest) = trimmed.strip_prefix("pub trait ").or_else(|| trimmed.strip_prefix("trait ")) {
                if let Some(end) = rest.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    let name = rest[..end].trim().to_string();
                    if !name.is_empty() {
                        symbols.push(Symbol {
                            name,
                            kind: "trait".to_string(),
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                }
            }
            // Constants and Statics
            else if let Some(rest) = trimmed.strip_prefix("pub const ").or_else(|| trimmed.strip_prefix("const ")).or_else(|| trimmed.strip_prefix("pub static ")).or_else(|| trimmed.strip_prefix("static ")) {
                if let Some(end) = rest.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    let name = rest[..end].trim().to_string();
                    if !name.is_empty() {
                        symbols.push(Symbol {
                            name,
                            kind: if trimmed.contains("const") { "const".to_string() } else { "static".to_string() },
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                }
            }
            // Type aliases
            else if let Some(rest) = trimmed.strip_prefix("pub type ").or_else(|| trimmed.strip_prefix("type ")) {
                if let Some(end) = rest.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    let name = rest[..end].trim().to_string();
                    if !name.is_empty() {
                        symbols.push(Symbol {
                            name,
                            kind: "type".to_string(),
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                }
            }
            // Impl blocks (detect for context, methods inside would need deeper parse or LSP)
            else if trimmed.starts_with("impl ") || trimmed.starts_with("pub impl ") {
                // Could extract trait/impl target, but for now mark as impl context
                if let Some(for_pos) = trimmed.find(" for ") {
                    let target = trimmed[for_pos + 5..].split_whitespace().next().unwrap_or("").trim_end_matches('{').to_string();
                    if !target.is_empty() {
                        symbols.push(Symbol {
                            name: target,
                            kind: "impl".to_string(),
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                } else if let Some(name_part) = trimmed.split_whitespace().nth(1) {
                    let name = name_part.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string();
                    if !name.is_empty() && !name.contains("<") {
                        symbols.push(Symbol {
                            name,
                            kind: "impl".to_string(),
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                }
            }
        }
    } else if language == "javascript" || language == "js" || language == "typescript" || language == "ts" {
        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains("function ") || trimmed.contains("const ") || trimmed.contains("let ") || trimmed.contains("class ") || trimmed.contains("export ") {
                // Simple extraction for JS/TS
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let mut name = parts[1].trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string();
                    if name.ends_with('(') {
                        name = name.trim_end_matches('(').to_string();
                    }
                    let kind = if trimmed.contains("class ") { "class" }
                        else if trimmed.contains("function ") { "function" }
                        else { "const" };
                    if !name.is_empty() {
                        symbols.push(Symbol {
                            name,
                            kind: kind.to_string(),
                            line_start: line_num,
                            line_end: line_num,
                            signature: Some(trimmed.to_string()),
                        });
                    }
                }
            }
        }
    }
    symbols
}

// === ONE Organism Integration ===
// In RaThorOneOrganism or queries:
// let resolver: Box<dyn SymbolResolver> = if use_lsp { Box::new(LspSymbolResolver::new("rust-analyzer")) } else { Box::new(SimpleSymbolResolver) };
// let symbols = resolver.resolve_symbols(content, "rust", path);
// Then feed to CodeChunk.symbols for superior RoleOptimizedView (better VibeCoder hits on real functions, Investigator on structs, etc.)
// Full LSP unlocks: goto definition, find references, hover types — perfect for Debugger/Legal roles too.
// All mercy-gated, zero-harm, eternal thriving.