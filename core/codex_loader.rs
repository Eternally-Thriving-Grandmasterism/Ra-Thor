// core/codex_loader.rs
// Codex Loader — The eyes and memory of the Omnimaster Root Core
// Scans /docs folder, loads codices with full FENCA + Mercy + Global Cache integration
// Enables true self-review, idea recycling, and eternal innovation to the nth degree

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::audit_logger::AuditLogger;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CodexMetadata {
    pub filename: String,
    pub title: String,
    pub last_modified: u64,
    pub mercy_relevance: f64,
}

pub struct CodexLoader;

impl CodexLoader {
    /// Scan /docs folder and return list of codex filenames (used by SelfReviewLoop)
    pub async fn scan_docs_folder() -> Vec<String> {
        let mut codices = vec![];

        if let Ok(entries) = fs::read_dir("docs") {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |ext| ext == "md") {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        codices.push(name.to_string());
                    }
                }
            }
        }
        codices
    }

    /// Load a single codex with full FENCA + Mercy + caching
    pub async fn load_codex(filename: &str) -> Option<String> {
        let cache_key = GlobalCache::make_key("codex_full", &json!({"filename": filename}));

        if let Some(cached) = GlobalCache::get(&cache_key) {
            if let Ok(content) = serde_json::from_value::<String>(cached) {
                return Some(content);
            }
        }

        let path = format!("docs/{}", filename);
        let content = fs::read_to_string(&path).ok()?;

        // FENCA verification
        let fenca_result = FENCA::verify_codex_content(&content).await;
        if !fenca_result.is_verified() {
            return None;
        }

        // Mercy Engine check
        let mercy_scores = MercyEngine::evaluate_codex_content(&content);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return None;
        }

        // Cache with adaptive TTL
        let ttl = GlobalCache::adaptive_ttl(86400 * 30, fenca_result.fidelity(), valence, 255); // 30 days for codices
        GlobalCache::set(&cache_key, serde_json::to_value(&content).unwrap(), ttl, 255, fenca_result.fidelity(), valence);

        // Audit the load
        let _ = AuditLogger::log(
            "root", None, "codex_loaded", filename, true,
            fenca_result.fidelity(), valence, vec![],
            serde_json::json!({"size_bytes": content.len()}),
        ).await;

        Some(content)
    }

    /// Parse basic metadata from codex (used by IdeaRecycler)
    pub fn parse_metadata(content: &str) -> CodexMetadata {
        let title = content.lines().next().unwrap_or("Untitled Codex").trim().to_string();
        let last_modified = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        CodexMetadata {
            filename: "unknown".to_string(), // filled by caller
            title,
            last_modified,
            mercy_relevance: 0.95, // default high relevance for Ra-Thor codices
        }
    }
}
