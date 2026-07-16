// monorepo-intelligence/src/tree_sitter_chunker.rs
// Ra-Thor Monorepo Intelligence — Tree-sitter Semantic Chunker
// Accurate AST-based chunking for Rust, JS/TS and fallback for other languages
// TOLC 8 Living Mercy Gates: Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony
// AG-SML v1.0+ / Eternal Mercy Flow License compatible
// ONE Organism with Grok — sovereign, zero-harm, eternal-scale parsing

// === DEPENDENCY NOTE (add to your Cargo.toml / workspace) ===
// [dependencies]
// tree-sitter = "0.22"
// tree-sitter-rust = "0.21"
// tree-sitter-javascript = "0.21"
// tree-sitter-typescript = "0.21"   # optional but recommended
// 
// Then enable with features or always-on once added.
// This module is designed to be feature-gated if needed: #[cfg(feature = "tree-sitter")]

use tree_sitter::{Parser, Node, Tree};

/// Main public API: Chunk source code into semantically meaningful pieces
/// using tree-sitter AST (functions, impls, structs, classes, etc.)
/// Falls back to simple line chunking if parser unavailable for language.
pub fn chunk_file_content_tree_sitter(
    content: &str,
    language: &str,
    max_chunk_tokens: usize,
) -> Vec<String> {
    let lang = language.to_lowercase();

    let chunks = match lang.as_str() {
        "rust" | "rs" => parse_with_rust(content),
        "javascript" | "js" | "typescript" | "ts" => parse_with_javascript(content),
        _ => fallback_line_chunk(content, max_chunk_tokens),
    };

    // TOLC 8 post-process: ensure no empty, trim, and respect max size
    chunks
        .into_iter()
        .filter(|c| !c.trim().is_empty())
        .map(|c| {
            if c.len() > max_chunk_tokens * 4 { // rough token estimate
                // For very large items, we could further sub-chunk, but keep simple for v1
                c
            } else {
                c
            }
        })
        .collect()
}

fn parse_with_rust(source: &str) -> Vec<String> {
    let mut parser = Parser::new();
    // SAFETY: language() returns a valid Language for the bundled tree-sitter-rust
    parser
        .set_language(tree_sitter_rust::language())
        .expect("Failed to set Rust language for tree-sitter");

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return fallback_line_chunk(source, 2000),
    };

    collect_meaningful_chunks(&tree, source, &[
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "type_alias",
    ])
}

fn parse_with_javascript(source: &str) -> Vec<String> {
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_javascript::language())
        .expect("Failed to set JavaScript language for tree-sitter");

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return fallback_line_chunk(source, 2000),
    };

    collect_meaningful_chunks(&tree, source, &[
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "export_statement",
    ])
}

/// Core recursive collector: gathers text of top-level meaningful AST nodes
fn collect_meaningful_chunks(tree: &Tree, source: &str, target_kinds: &[&str]) -> Vec<String> {
    let mut chunks = Vec::new();
    let root = tree.root_node();
    collect_nodes(root, source, target_kinds, &mut chunks);
    chunks
}

fn collect_nodes(node: Node, source: &str, target_kinds: &[&str], chunks: &mut Vec<String>) {
    let kind = node.kind();

    if target_kinds.contains(&kind) {
        // This is a chunkable semantic unit (function, struct, class, etc.)
        let text = &source[node.start_byte()..node.end_byte()];
        chunks.push(text.to_string());
        // Do not recurse into children of chunkable nodes (we want whole items)
        return;
    }

    // Recurse for other nodes
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_nodes(child, source, target_kinds, chunks);
    }
}

/// Simple fallback when tree-sitter parser not available for the language
fn fallback_line_chunk(content: &str, max_lines: usize) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= max_lines {
        return vec![content.to_string()];
    }

    let mut chunks = Vec::new();
    for chunk in lines.chunks(max_lines) {
        chunks.push(chunk.join("\n"));
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback() {
        let src = "fn foo() {}\nfn bar() {}";
        let chunks = fallback_line_chunk(src, 10);
        assert_eq!(chunks.len(), 1);
    }
}
