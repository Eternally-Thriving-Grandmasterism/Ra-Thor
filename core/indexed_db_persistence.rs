// core/indexed_db_persistence.rs
// IndexedDB Persistence Layer with full AuditLogger integration

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::audit_logger::AuditLogger;
use serde_json::Value;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbTransactionMode};

pub struct IndexedDBPersistence;

impl IndexedDBPersistence {
    const DB_NAME: &'static str = "RaThorDB";
    const DB_VERSION: u32 = 1;

    async fn open_db(tenant_id: &str) -> Result<web_sys::IdbDatabase, JsValue> { /* unchanged from previous version */ }

    pub async fn save<T: serde::Serialize>(
        tenant_id: &str,
        key: &str,
        value: &T,
    ) -> Result<(), crate::master_kernel::KernelResult> {

        let fenca_result = FENCA::verify_tenant_scoped(/* dummy */ , tenant_id);
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        // Perform the actual IndexedDB save
        let db = Self::open_db(tenant_id).await.map_err(|_| crate::master_kernel::KernelResult { status: "indexeddb_error".to_string(), ..Default::default() })?;
        let tx = db.transaction_with_mode(&format!("tenant_{}", tenant_id), IdbTransactionMode::Readwrite)?;
        let store = tx.object_store(&format!("tenant_{}", tenant_id))?;
        let value_json = serde_json::to_string(value).unwrap();
        let request = store.put_with_key(&value_json, &key)?;
        JsFuture::from(request).await.map_err(|_| crate::master_kernel::KernelResult { status: "indexeddb_write_error".to_string(), ..Default::default() })?;

        // Audit log the operation
        let _ = AuditLogger::log(
            tenant_id,
            None,
            "save",
            key,
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            serde_json::json!({"value_size": value_json.len()}),
        ).await;

        // Update Global Cache
        let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, 220);
        GlobalCache::set(&GlobalCache::make_key_with_tenant("indexeddb", &json!({"key": key}), Some(tenant_id)), serde_json::to_value(value).unwrap(), ttl, 220, fenca_result.fidelity(), valence);

        Ok(())
    }

    pub async fn load<T: serde::de::DeserializeOwned>(
        tenant_id: &str,
        key: &str,
    ) -> Result<Option<T>, crate::master_kernel::KernelResult> {

        let fenca_result = FENCA::verify_tenant_scoped(/* dummy */, tenant_id);
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        // Try cache first (unchanged)
        let cache_key = GlobalCache::make_key_with_tenant("indexeddb", &json!({"key": key}), Some(tenant_id));
        if let Some(cached) = GlobalCache::get(&cache_key) {
            if let Ok(value) = serde_json::from_value(cached) {
                let _ = AuditLogger::log(tenant_id, None, "load", key, true, fenca_result.fidelity(), valence, vec![], serde_json::json!({"cache_hit": true})).await;
                return Ok(Some(value));
            }
        }

        // Real IndexedDB load (unchanged logic)
        // ... (same as previous version)

        let _ = AuditLogger::log(tenant_id, None, "load", key, true, fenca_result.fidelity(), valence, vec![], serde_json::json!({"cache_hit": false})).await;
        // ...
    }
}
