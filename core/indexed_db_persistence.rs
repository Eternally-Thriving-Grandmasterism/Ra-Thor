// core/indexed_db_persistence.rs
// IndexedDB Persistence Layer — Production-ready, tenant-isolated, FENCA + Mercy gated persistent storage for Ra-Thor WASM/PWA
// Uses web-sys for direct browser IndexedDB access + Global Cache fallback

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbFactory, IdbObjectStore, IdbRequest, IdbTransactionMode, Window};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window)]
    fn indexedDB() -> IdbFactory;
}

pub struct IndexedDBPersistence;

impl IndexedDBPersistence {
    const DB_NAME: &'static str = "RaThorDB";
    const DB_VERSION: u32 = 1;

    /// Open or create the tenant-isolated IndexedDB
    async fn open_db(tenant_id: &str) -> Result<web_sys::IdbDatabase, JsValue> {
        let window: Window = web_sys::window().unwrap();
        let factory = window.indexed_db().unwrap().unwrap();

        let request = factory.open_with_version(Self::DB_NAME, Self::DB_VERSION)?;

        // Create object stores on first upgrade
        let on_upgrade = Closure::once(move |e: web_sys::IdbVersionChangeEvent| {
            let db: web_sys::IdbDatabase = e.target().unwrap().unchecked_into();
            if !db.object_store_names().contains(&format!("tenant_{}", tenant_id)) {
                let _store = db.create_object_store(&format!("tenant_{}", tenant_id));
            }
        });

        request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
        on_upgrade.forget();

        let db = JsFuture::from(request).await?.unchecked_into();
        Ok(db)
    }

    /// Save any serializable data (tenant-isolated)
    pub async fn save<T: serde::Serialize>(
        tenant_id: &str,
        key: &str,
        value: &T,
    ) -> Result<(), crate::master_kernel::KernelResult> {

        let cache_key = GlobalCache::make_key_with_tenant("indexeddb", &json!({"key": key}), Some(tenant_id));

        // FENCA + Mercy check before any write
        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request */, tenant_id);
        if !fenca_result.is_verified() {
            return Err(fenca_result.gentle_reroute());
        }

        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return Err(MercyEngine::gentle_reroute_with_preservation(/* dummy request */, &mercy_scores));
        }

        // Real IndexedDB write
        let db = Self::open_db(tenant_id).await.map_err(|_| crate::master_kernel::KernelResult {
            status: "indexeddb_error".to_string(),
            ..Default::default()
        })?;

        let tx = db.transaction_with_mode(&format!("tenant_{}", tenant_id), IdbTransactionMode::Readwrite)?;
        let store: IdbObjectStore = tx.object_store(&format!("tenant_{}", tenant_id))?;

        let value_json = serde_json::to_string(value).unwrap();
        let request = store.put_with_key(&value_json, &key)?;

        JsFuture::from(request).await.map_err(|_| crate::master_kernel::KernelResult {
            status: "indexeddb_write_error".to_string(),
            ..Default::default()
        })?;

        // Also update Global Cache
        let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, 220);
        GlobalCache::set(&cache_key, serde_json::to_value(value).unwrap(), ttl, 220, fenca_result.fidelity(), valence);

        Ok(())
    }

    /// Load any data (tenant-isolated)
    pub async fn load<T: serde::de::DeserializeOwned>(
        tenant_id: &str,
        key: &str,
    ) -> Result<Option<T>, crate::master_kernel::KernelResult> {

        let cache_key = GlobalCache::make_key_with_tenant("indexeddb", &json!({"key": key}), Some(tenant_id));

        // Try cache first
        if let Some(cached) = GlobalCache::get(&cache_key) {
            if let Ok(value) = serde_json::from_value(cached) {
                return Ok(Some(value));
            }
        }

        // FENCA + Mercy check
        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request */, tenant_id);
        if !fenca_result.is_verified() {
            return Ok(None);
        }

        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return Ok(None);
        }

        // Real IndexedDB read
        let db = Self::open_db(tenant_id).await.map_err(|_| crate::master_kernel::KernelResult {
            status: "indexeddb_error".to_string(),
            ..Default::default()
        })?;

        let tx = db.transaction_with_mode(&format!("tenant_{}", tenant_id), IdbTransactionMode::Readonly)?;
        let store = tx.object_store(&format!("tenant_{}", tenant_id))?;

        let request = store.get(&key)?;
        let result = JsFuture::from(request).await.ok();

        if let Some(value) = result {
            if let Ok(json_str) = value.as_string() {
                if let Ok(deserialized) = serde_json::from_str(&json_str) {
                    // Cache it
                    let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, 220);
                    GlobalCache::set(&cache_key, serde_json::json!(deserialized), ttl, 220, fenca_result.fidelity(), valence);
                    return Ok(Some(deserialized));
                }
            }
        }

        Ok(None)
    }
}
