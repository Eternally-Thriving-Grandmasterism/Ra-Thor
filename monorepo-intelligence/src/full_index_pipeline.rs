// monorepo-intelligence/src/full_index_pipeline.rs
// Ra-Thor Monorepo Intelligence — Full Incremental Indexing Pipeline
// Tree-sitter powered semantic chunking + symbol extraction
// TOLC 8 Living Mercy Gates | PATSAGi aligned | ONE Organism

use crate::index_types::{CodeChunk, FileIndexEntry, MonorepoIndex, Symbol};
use crate::paginated_monorepo_parser::{walk_tree_paginated, chunk_file_content};
use std::collections::HashMap;

/// Configuration for the indexing pipeline
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub owner: String,
    pub repo: String,
    pub include_paths: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub max_files_per_run: usize,
    pub chunk_max_tokens: usize,
    pub languages: Vec<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            owner: "Eternally-Thriving-Grandmasterism".to_string(),
            repo: "Ra-Thor".to_string(),
            include_paths: vec!["src/".to_string(), "core/".to_string(), "crates/".to_string(), "monorepo-intelligence/".to_string()],
            exclude_patterns: vec!["target/".to_string(), "node_modules/".to_string(), ".git/".to_string()],
            max_files_per_run: 500,
            chunk_max_tokens: 8000,
            languages: vec!["rust".to_string(), "javascript".to_string(), "js".to_string()],
        }
    }
}

/// Main entry point — build or incrementally update the monorepo index
pub fn build_or_update_index(
    previous_index: Option<MonorepoIndex>,
    config: &IndexConfig,
) -> Result<MonorepoIndex, String> {
    let last_sha = previous_index
        .as_ref()
        .map(|i| i.last_tree_sha.as_str())
        .unwrap_or("main");

    let tree_result = walk_tree_paginated(
        &config.owner,
        &config.repo,
        None,
        1,
        100,
        Some(last_sha),
    )?;

    let mut index = previous_index.unwrap_or_else(|| MonorepoIndex::new(&tree_result.last_tree_sha));
    let mut processed = 0;

    for entry in tree_result.entries {
        if processed >= config.max_files_per_run { break; }

        let path = &entry.path;
        if !should_index_path(path, config) { continue; }

        let language = detect_language(path);
        if !config.languages.contains(&language) { continue; }

        // TODO (next step): Replace stub with real content fetch via github_connector.rs
        // Compare entry.sha with previous index entry to skip unchanged files
        let content = format!(
            "// TODO: Fetch real content for {} via GitHub connector\n// Current SHA: {}",
            path, entry.sha
        );

        let chunks_text = chunk_file_content(&content, &language, config.chunk_max_tokens);

        let mut file_symbols: Vec<Symbol> = Vec::new();
        let mut code_chunks: Vec<CodeChunk> = Vec::new();

        for (i, chunk_text) in chunks_text.iter().enumerate() {
            let symbols = extract_symbols_simple(chunk_text, &language, i);
            file_symbols.extend(symbols.clone());

            code_chunks.push(CodeChunk {
                content: chunk_text.clone(),
                start_line: 1,
                end_line: chunk_text.lines().count(),
                symbols,
                chunk_type: if language == "rust" { "rust_item".to_string() } else { "js_item".to_string() },
            });
        }

        let file_entry = FileIndexEntry {
            path: path.clone(),
            sha: entry.sha.clone(),
            language,
            size_bytes: entry.size.unwrap_or(0),
            chunks: code_chunks,
            symbol_count: file_symbols.len(),
            last_indexed_at: chrono::Utc::now().to_rfc3339(),
        };

        index.files.insert(path.clone(), file_entry);
        processed += 1;
    }

    index.indexed_file_count = index.files.len();
    index.total_chunks = index.files.values().map(|f| f.chunks.len()).sum();
    index.total_symbols = index.files.values().map(|f| f.symbol_count).sum();
    index.last_tree_sha = tree_result.last_tree_sha;
    index.update_timestamp();

    Ok(index)
}

fn should_index_path(path: &str, config: &IndexConfig) -> bool {
    let included = config.include_paths.iter().any(|p| path.starts_with(p));
    let excluded = config.exclude_patterns.iter().any(|p| path.contains(p));
    included && !excluded
}

fn detect_language(path: &str) -> String {
    if path.ends_with(".rs") { "rust".to_string() }
    else if path.ends_with(".js") || path.ends_with(".mjs") { "javascript".to_string() }
    else if path.ends_with(".ts") { "typescript".to_string() }
    else { "other".to_string() }
}

/// Lightweight symbol extraction (v1). Upgrade later with real tree-sitter queries.
fn extract_symbols_simple(content: &str, language: &str, _chunk_index: usize) -> Vec<Symbol> {
    let mut symbols = Vec::new();

    if language == "rust" {
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(name) = trimmed.strip_prefix("pub fn ") {
                if let Some(end) = name.find('(') {
                    symbols.push(Symbol {
                        name: name[..end].to_string(),
                        kind: "function".to_string(),
                        line_start: 0,
                        line_end: 0,
                        signature: Some(trimmed.to_string()),
                    });
                }
            }
            if let Some(name) = trimmed.strip_prefix("pub struct ") {
                if let Some(end) = name.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    symbols.push(Symbol {
                        name: name[..end].to_string(),
                        kind: "struct".to_string(),
                        line_start: 0,
                        line_end: 0,
                        signature: None,
                    });
                }
            }
        }
    }
    // Add JS/TS extraction as needed
    symbols
}
