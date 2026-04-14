// core/lazy_codex_loader.rs
// Lazy-Loading Codex Loader — On-demand loading of codices from /docs with FENCA, Mercy Engine, Global Cache, and AuditLogger integration
// This makes the Root Core truly efficient and self-evolving without loading everything at startup

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::audit_logger::AuditLogger;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LoadedCodex {
    pub filename: String,
    pub content: String,
    pub parsed_metadata: Value,
    pub last_loaded: u64,
}

pub struct LazyCodexLoader;

impl LazyCodexLoader {
    /// Load a specific codex on-demand (lazy)
    pub async fn load_codex(filename: &str) -> Result<LoadedCodex, crate::master_kernel::KernelResult> {

        let cache_key = GlobalCache::make_key("codex", &json!({"filename": filename}));

        // 1. Check Global Cache first (fast path)
        if let Some(cached) = GlobalCache::get(&cache_key) {
            return serde_json::from_value(cached).map_err(|_| crate::master_kernel::KernelResult {
                status: "codex_deserialize_error".to_string(),
                ..Default::default()
            });
        }

        // 2. FENCA verification before loading from disk
        let fenca_result = FENCA::verify_codex_filename(filename).await;
        if !fenca_result.is_verified() {
            return Err(fenca_result.gentle_reroute());
        }

        // 3. Mercy Engine check
        let mercy_scores = MercyEngine::evaluate_codex_filename(filename);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return Err(MercyEngine::gentle_reroute_with_preservation_filename(filename, &mercy_scores));
        }

        // 4. Actually load the codex from /docs
        let content = CodexFileReader::read_from_docs(filename).await
            .map_err(|_| crate::master_kernel::KernelResult {
                status: "codex_read_error".to_string(),
                ..Default::default()
            })?;

        // 5. Parse basic metadata
        let parsed_metadata = CodexParser::parse_metadata(&content);

        let loaded = LoadedCodex {
            filename: filename.to_string(),
            content,
            parsed_metadata,
            last_loaded: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // 6. Cache the loaded codex with adaptive TTL
        let ttl = GlobalCache::adaptive_ttl(7200, fenca_result.fidelity(), valence, 200); // 2 hours default
        GlobalCache::set(&cache_key, serde_json::to_value(&loaded).unwrap(), ttl, 200, fenca_result.fidelity(), valence);

        // 7. Audit log the lazy load
        let _ = AuditLogger::log(
            "root",
            None,
            "lazy_codex_load",
            filename,
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            serde_json::json!({"size_bytes": loaded.content.len()}),
        ).await;

        Ok(loaded)
    }

    /// Clear a specific codex from cache (for refresh)
    pub fn clear_cache(filename: &str) {
        let cache_key = GlobalCache::make_key("codex", &json!({"filename": filename}));
        GlobalCache::clear(Some(&cache_key));
    }
}
