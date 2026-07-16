// monorepo-intelligence/src/index_types.rs
// Ra-Thor Monorepo Intelligence — Core Index Data Types
// TOLC 8 Living Mercy Gates aligned | AG-SML v1.0+ compatible
// ONE Organism | Sovereign, queryable, incremental monorepo knowledge

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single extracted symbol (function, struct, trait, impl block, class, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub kind: String,           // "function", "struct", "impl", "trait", "class", "method", etc.
    pub line_start: usize,
    pub line_end: usize,
    pub signature: Option<String>,
}

/// A semantically chunked piece of code with its extracted symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub symbols: Vec<Symbol>,
    pub chunk_type: String,     // "function", "impl_block", "module", "class", "fallback"
}

/// Full index entry for one file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileIndexEntry {
    pub path: String,
    pub sha: String,
    pub language: String,
    pub size_bytes: u64,
    pub chunks: Vec<CodeChunk>,
    pub symbol_count: usize,
    pub last_indexed_at: String, // ISO timestamp
}

/// The complete, serializable monorepo index
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MonorepoIndex {
    pub version: String,
    pub last_tree_sha: String,
    pub indexed_file_count: usize,
    pub total_symbols: usize,
    pub total_chunks: usize,
    pub files: HashMap<String, FileIndexEntry>, // key = path
    pub mercy_valence: f64,
    pub created_at: String,
    pub updated_at: String,
}

impl MonorepoIndex {
    pub fn new(last_tree_sha: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            version: "1.0.0".to_string(),
            last_tree_sha: last_tree_sha.to_string(),
            indexed_file_count: 0,
            total_symbols: 0,
            total_chunks: 0,
            files: HashMap::new(),
            mercy_valence: 0.999999,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn update_timestamp(&mut self) {
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}
