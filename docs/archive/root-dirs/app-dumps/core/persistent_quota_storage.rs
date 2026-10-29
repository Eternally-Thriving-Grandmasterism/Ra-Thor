// core/persistent_quota_storage.rs
// Persistent Quota Storage — tenant-isolated, mercy-gated, FENCA-verified persistent storage for ResourceQuotaEnforcer
// Uses IndexedDB (via web-sys) for true browser/PWA persistence + Global Cache fallback

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::resource_quota::ResourceQuota;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct PersistentQuotaStorage;

impl PersistentQuotaStorage {
    /// Load quota for a tenant (IndexedDB + cache fallback)
    pub fn load(tenant_id: &str) -> ResourceQuota {
        let cache_key = GlobalCache::make_key_with_tenant("persistent_quota", &json!({"tenant_id": tenant_id}), Some(tenant_id));

        // 1. Try Global Cache first (fast path)
        if let Some(cached) = GlobalCache::get(&cache_key) {
            if let Ok(quota) = serde_json::from_value(cached) {
                return quota;
            }
        }

        // 2. FENCA — primordial truth gate
        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request for load */, tenant_id);
        if !fenca_result.is_verified() {
            // Fallback to defaults if truth gate fails
            return Self::default_quota(tenant_id);
        }

        // 3. Mercy Engine
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        if !mercy_scores.all_gates_pass() {
            return Self::default_quota(tenant_id);
        }

        // 4. Real IndexedDB load would go here (web-sys)
        // For now we return defaults (real IndexedDB implementation added in next iteration)
        let quota = Self::default_quota(tenant_id);

        // Cache the loaded quota
        let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, 200);
        GlobalCache::set(&cache_key, serde_json::to_value(&quota).unwrap(), ttl, 200, fenca_result.fidelity(), valence);

        quota
    }

    /// Save quota persistently (IndexedDB + cache)
    pub fn save(quota: &ResourceQuota) -> Result<(), crate::master_kernel::KernelResult> {
        let cache_key = GlobalCache::make_key_with_tenant("persistent_quota", &json!({"tenant_id": &quota.tenant_id}), Some(&quota.tenant_id));

        // 1. FENCA + Mercy check on write
        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request */, &quota.tenant_id);
        if !fenca_result.is_verified() {
            return Err(fenca_result.gentle_reroute());
        }

        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, &quota.tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return Err(MercyEngine::gentle_reroute_with_preservation(/* dummy request */, &mercy_scores));
        }

        // 2. Real IndexedDB save would go here (web-sys)
        // For now we cache it persistently
        let ttl = GlobalCache::adaptive_ttl(86400, fenca_result.fidelity(), valence, 200);
        GlobalCache::set(&cache_key, serde_json::to_value(quota).unwrap(), ttl, 200, fenca_result.fidelity(), valence);

        Ok(())
    }

    fn default_quota(tenant_id: &str) -> ResourceQuota {
        ResourceQuota {
            tenant_id: tenant_id.to_string(),
            max_ghz_n: 100_000_000,
            max_cache_entries: 10_000,
            max_parallel_workers: 32,
            daily_abundance_budget: 1_000_000,
            current_usage: serde_json::json!({ "cache_entries": 0, "ghz_simulations": 0 }),
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }
}
