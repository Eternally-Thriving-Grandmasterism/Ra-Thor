// core/rebac_graph_storage.rs
// ReBAC Graph Storage — tenant-isolated, FENCA-verified, mercy-gated, IndexedDB-backed relationship graph for fast traversal

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::audit_logger::AuditLogger;
use crate::indexed_db_persistence::IndexedDBPersistence;
use crate::parallel_ghz_worker::ParallelGHZWorker;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Relationship {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub tenant_id: String,
    pub mercy_level: u8,
    pub created_at: u64,
    pub expires_at: Option<u64>,
}

pub struct ReBACGraphStorage;

impl ReBACGraphStorage {
    const STORE_NAME: &'static str = "rebac_graph";

    /// Store a new relationship (FENCA + Mercy + Audit gated)
    pub async fn store(relationship: Relationship) -> Result<(), crate::master_kernel::KernelResult> {
        let tenant_id = &relationship.tenant_id;

        // 1. FENCA — primordial truth
        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request */, tenant_id);
        if !fenca_result.is_verified() {
            return Err(fenca_result.gentle_reroute());
        }

        // 2. Mercy Engine
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return Err(MercyEngine::gentle_reroute_with_preservation(/* dummy request */, &mercy_scores));
        }

        // 3. Persistent IndexedDB save
        IndexedDBPersistence::save(tenant_id, &format!("rebac:{}", serde_json::to_string(&relationship).unwrap()), &relationship).await?;

        // 4. Audit log
        let _ = AuditLogger::log(
            tenant_id,
            None,
            "rebac_store",
            &format!("{}->{}:{}", relationship.subject, relationship.relation, relationship.object),
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            serde_json::json!({"mercy_level": relationship.mercy_level}),
        ).await;

        // 5. Global Cache for fast lookup
        let cache_key = GlobalCache::make_key_with_tenant("rebac_rel", &serde_json::json!(&relationship), Some(tenant_id));
        let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, relationship.mercy_level);
        GlobalCache::set(&cache_key, serde_json::to_value(&relationship).unwrap(), ttl, relationship.mercy_level as u8, fenca_result.fidelity(), valence);

        Ok(())
    }

    /// Load all relationships for a subject (parallel + cached)
    pub async fn load_for_subject(subject: &str, tenant_id: &str) -> Vec<Relationship> {
        let cache_key = GlobalCache::make_key_with_tenant("rebac_subject", &json!({"subject": subject}), Some(tenant_id));

        if let Some(cached) = GlobalCache::get(&cache_key) {
            return serde_json::from_value(cached).unwrap_or_default();
        }

        // Parallel load using GHZ Worker
        let relationships = ParallelGHZWorker::parallel_load_relationships(subject, tenant_id);

        // Cache result
        let fenca_fidelity = 1.0; // placeholder for cached load
        let valence = 1.0;
        let ttl = GlobalCache::adaptive_ttl(3600, fenca_fidelity, valence, 200);
        GlobalCache::set(&cache_key, serde_json::to_value(&relationships).unwrap(), ttl, 200, fenca_fidelity, valence);

        relationships
    }

    /// Fast graph traversal for ReBAC queries (used by ReBAC::traverse)
    pub fn traverse(subject: &str, requested_relation: &str, object: &str, tenant_id: &str) -> bool {
        let relationships = /* load_for_subject would be called in real code */;
        // Parallel BFS using GHZ Worker for large graphs
        ParallelGHZWorker::parallel_graph_traverse(subject, requested_relation, object, tenant_id, 1.0, 1.0)
    }
}
