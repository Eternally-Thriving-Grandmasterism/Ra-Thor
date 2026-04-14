// core/resource_quota.rs
// Resource Quota Enforcement — mercy-aware, tenant-isolated resource limits for enterprise use
// Fully integrated with Master Sovereign Kernel, FENCA, Mercy Engine, Global Cache, and Adaptive TTL

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResourceQuota {
    pub tenant_id: String,
    pub max_ghz_n: usize,
    pub max_cache_entries: usize,
    pub max_parallel_workers: u32,
    pub daily_abundance_budget: u64,
    pub current_usage: Value,
    pub last_updated: u64,
}

pub struct ResourceQuotaEnforcer;

impl ResourceQuotaEnforcer {
    /// Main enforcement check — called in multi-user orchestrator
    pub fn enforce(tenant_id: &str, request: &crate::master_kernel::RequestPayload) -> Result<(), crate::master_kernel::KernelResult> {

        let cache_key = GlobalCache::make_key_with_tenant("quota", &request.data, Some(tenant_id));

        // 1. Global Cache hit with adaptive TTL
        if let Some(cached) = GlobalCache::get(&cache_key) {
            let quota: ResourceQuota = serde_json::from_value(cached).unwrap_or_default();
            if Self::within_limits(&quota, request) {
                return Ok(());
            }
        }

        // 2. FENCA — primordial truth gate (tenant-scoped)
        let fenca_result = FENCA::verify_tenant_scoped(request, tenant_id);
        if !fenca_result.is_verified() {
            return Err(fenca_result.gentle_reroute());
        }

        // 3. Mercy Engine (abundance + non-harm)
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(request, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        if !mercy_scores.all_gates_pass() {
            return Err(MercyEngine::gentle_reroute_with_preservation(request, &mercy_scores));
        }

        // 4. Load current quota and check limits
        let mut quota = Self::load(tenant_id);
        if !Self::within_limits(&quota, request) {
            return Err(MercyEngine::gentle_reroute("Resource quota exceeded — mercy reroute to abundant lower-cost path"));
        }

        // 5. Update usage and cache with adaptive TTL
        quota.update_usage(request);
        let ttl = GlobalCache::adaptive_ttl(3600, fenca_result.fidelity(), valence, 180);
        GlobalCache::set(&cache_key, serde_json::to_value(&quota).unwrap(), ttl, 180, fenca_result.fidelity(), valence);

        Ok(())
    }

    fn within_limits(quota: &ResourceQuota, request: &crate::master_kernel::RequestPayload) -> bool {
        request.estimated_ghz_n <= quota.max_ghz_n &&
        quota.current_usage["cache_entries"].as_u64().unwrap_or(0) < quota.max_cache_entries as u64
    }

    fn load(tenant_id: &str) -> ResourceQuota {
        // In production this would load from persistent tenant-isolated storage
        // For now we return defaults (real storage will be added in next step)
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

    fn update_usage(&mut self, request: &crate::master_kernel::RequestPayload) {
        // Update usage counters
        let mut usage = self.current_usage.clone();
        if let Some(entries) = usage["cache_entries"].as_u64() {
            usage["cache_entries"] = serde_json::json!(entries + 1);
        }
        self.current_usage = usage;
        self.last_updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }
}
