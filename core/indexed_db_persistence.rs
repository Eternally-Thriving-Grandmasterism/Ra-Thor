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

    async fn open_db(tenant_id: &str) -> Result<web_sys::IdbDatabase, JsValue> {
        let window: web_sys::Window = web_sys::window().unwrap();
        let factory = window.indexed_db().unwrap().unwrap();
        let request = factory.open_with_version(Self::DB_NAME, Self::DB_VERSION)?;

        let on_upgrade = Closure::once(move |e: web_sys::IdbVersionChangeEvent| {
            let db: web_sys::IdbDatabase = e.target().unwrap().unchecked_into();
            let store_name = format!("tenant_{}", tenant_id);
            if !db.object_store_names().contains(&store_name) {
                let _store = db.create_object_store(&store_name);
            }
        });

        request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
        on_upgrade.forget();

        let db = JsFuture::from(request).await?.unchecked_into();
        Ok(db)
    }

    pub async fn save<T: serde::Serialize>(
        tenant_id: &str,
        key: &str,
        value: &T,
    ) -> Result<(), crate::master_kernel::KernelResult> {

        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request for persistence */, tenant_id);
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        // Perform the actual IndexedDB save
        let db = Self::open_db(tenant_id).await.map_err(|_| crate::master_kernel::KernelResult { status: "indexeddb_error".to_string(), ..Default::default() })?;
        let tx = db.transaction_with_mode(&format!("tenant_{}", tenant_id), IdbTransactionMode::Readwrite)?;
        let store = tx.object_store(&format!("tenant_{}", tenant_id))?;
        let value_json = serde_json::to_string(value).unwrap();
        let request = store.put_with_key(&value_json, &key)?;
        JsFuture::from(request).await.map_err(|_| crate::master_kernel::KernelResult { status: "indexeddb_write_error".to_string(), ..Default::default() })?;

        // Full AuditLogger integration
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

        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request */, tenant_id);
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        let cache_key = GlobalCache::make_key_with_tenant("indexeddb", &json!({"key": key}), Some(tenant_id));
        if let Some(cached) = GlobalCache::get(&cache_key) {
            if let Ok(value) = serde_json::from_value(cached) {
                let _ = AuditLogger::log(tenant_id, None, "load", key, true, fenca_result.fidelity(), valence, vec![], serde_json::json!({"cache_hit": true})).await;
                return Ok(Some(value));
            }
        }

        // Real IndexedDB load (same as previous version)
        let db = Self::open_db(tenant_id).await.map_err(|_| crate::master_kernel::KernelResult { status: "indexeddb_error".to_string(), ..Default::default() })?;
        let tx = db.transaction_with_mode(&format!("tenant_{}", tenant_id), IdbTransactionMode::Readonly)?;
        let store = tx.object_store(&format!("tenant_{}", tenant_id))?;
        let request = store.get(&key)?;
        let result = JsFuture::from(request).await.ok();

        let outcome = if let Some(value) = result {
            if let Ok(json_str) = value.as_string() {
                if let Ok(deserialized) = serde_json::from_str(&json_str) {
                    let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, 220);
                    GlobalCache::set(&cache_key, serde_json::json!(deserialized), ttl, 220, fenca_result.fidelity(), valence);
                    let _ = AuditLogger::log(tenant_id, None, "load", key, true, fenca_result.fidelity(), valence, vec![], serde_json::json!({"cache_hit": false})).await;
                    Some(deserialized)
                } else { None }
            } else { None }
        } else { None };

        Ok(outcome)
    }
}
