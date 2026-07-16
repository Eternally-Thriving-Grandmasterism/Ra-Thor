// monorepo-intelligence/src/full_index_pipeline.rs
// Ra-Thor Monorepo Intelligence — Full Incremental Indexing Pipeline v1.1
// Tree-sitter semantic chunking + pluggable content fetching
// TOLC 8 Living Mercy Gates | PATSAGi aligned | ONE Organism

use crate::index_types::{CodeChunk, FileIndexEntry, MonorepoIndex, Symbol};
use crate::paginated_monorepo_parser::chunk_file_content;
use std::collections::HashMap;

/// Trait for fetching file content (allows real GitHub connector or local FS)
pub trait ContentFetcher {
    fn fetch(&self, path: &str, sha: &str) -> Result<String, String>;
}

/// Simple closure-based fetcher for flexibility
pub struct FnContentFetcher<F>(pub F)
where
    F: Fn(&str, &str) -> Result<String, String>;

impl<F> ContentFetcher for FnContentFetcher<F>
where
    F: Fn(&str, &str) -> Result<String, String>,
{
    fn fetch(&self, path: &str, sha: &str) -> Result<String, String> {
        (self.0)(path, sha)
    }
}

/// Default stub fetcher (for testing / skeleton runs)
pub struct StubContentFetcher;

impl ContentFetcher for StubContentFetcher {
    fn fetch(&self, path: &str, sha: &str) -> Result<String, String> {
        Ok(format!(
            "// STUB CONTENT for {} (sha: {})\n// Replace with real fetch via github_connector",
            path, sha
        ))
    }
}

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
            include_paths: vec!["src/".into(), "core/".into(), "crates/".into(), "monorepo-intelligence/".into()],
            exclude_patterns: vec!["target/".into(), ".git/".into()],
            max_files_per_run: 300,
            chunk_max_tokens: 6000,
            languages: vec!["rust".into(), "javascript".into()],
        }
    }
}

/// Main pipeline entrypoint — now with real content fetching support
pub fn build_or_update_index<F: ContentFetcher>(
    previous_index: Option<MonorepoIndex>,
    config: &IndexConfig,
    fetcher: &F,
) -> Result<MonorepoIndex, String> {
    // For now we still use the paginated walker stub.
    // In next iteration we can pass real tree walking too.
    let last_sha = previous_index
        .as_ref()
        .map(|i| i.last_tree_sha.clone())
        .unwrap_or_else(|| "main".to_string());

    let mut index = previous_index.unwrap_or_else(|| MonorepoIndex::new(&last_sha));
    let mut processed = 0;

    // TODO: Replace with real paginated tree walk + SHA diffing
    // For demonstration we simulate a few files
    let sample_files = vec![
        ("src/main.rs", "abc123", "rust"),
        ("core/lattice.rs", "def456", "rust"),
    ];

    for (path, sha, lang) in sample_files {
        if processed >= config.max_files_per_run { break; }
        if !should_index_path(path, config) { continue; }

        let content = fetcher.fetch(path, sha)?;

        let chunks_text = chunk_file_content(&content, lang, config.chunk_max_tokens);

        let mut file_symbols = Vec::new();
        let mut code_chunks = Vec::new();

        for (i, chunk_text) in chunks_text.iter().enumerate() {
            let symbols = extract_symbols_simple(chunk_text, lang, i);
            file_symbols.extend(symbols.clone());

            code_chunks.push(CodeChunk {
                content: chunk_text.clone(),
                start_line: 1,
                end_line: chunk_text.lines().count(),
                symbols,
                chunk_type: if lang == "rust" { "rust_item".into() } else { "js_item".into() },
            });
        }

        let entry = FileIndexEntry {
            path: path.to_string(),
            sha: sha.to_string(),
            language: lang.to_string(),
            size_bytes: content.len() as u64,
            chunks: code_chunks,
            symbol_count: file_symbols.len(),
            last_indexed_at: chrono::Utc::now().to_rfc3339(),
        };

        index.files.insert(path.to_string(), entry);
        processed += 1;
    }

    index.indexed_file_count = index.files.len();
    index.total_chunks = index.files.values().map(|f| f.chunks.len()).sum();
    index.total_symbols = index.files.values().map(|f| f.symbol_count).sum();
    index.update_timestamp();

    Ok(index)
}

fn should_index_path(path: &str, config: &IndexConfig) -> bool {
    let included = config.include_paths.iter().any(|p| path.starts_with(p));
    let excluded = config.exclude_patterns.iter().any(|p| path.contains(p));
    included && !excluded
}

fn extract_symbols_simple(content: &str, language: &str, _chunk_index: usize) -> Vec<Symbol> {
    // (same lightweight implementation as before — can be upgraded with real tree-sitter queries)
    let mut symbols = Vec::new();
    if language == "rust" {
        for line in content.lines() {
            let t = line.trim();
            if let Some(name) = t.strip_prefix("pub fn ") {
                if let Some(end) = name.find('(') {
                    symbols.push(Symbol {
                        name: name[..end].to_string(),
                        kind: "function".to_string(),
                        line_start: 0,
                        line_end: 0,
                        signature: Some(t.to_string()),
                    });
                }
            }
        }
    }
    symbols
}
