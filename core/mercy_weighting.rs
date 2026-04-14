// core/mercy_weighting.rs
// Mercy Weighting System — the living numeric heart of the 7 Living Mercy Gates
// u8 value (0-255) that dynamically influences every access decision, rewrite, cache TTL, quota, and reroute

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use serde_json::Value;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MercyWeight {
    pub level: u8,                    // 0-255 (higher = more compassionate/abundant)
    pub valence_influence: f64,       // current valence score
    pub fidelity_influence: f64,      // current FENCA/GHZ fidelity
    pub last_updated: u64,
}

pub struct MercyWeighting;

impl MercyWeighting {
    /// Calculate mercy weight from current context (valence + fidelity + gates)
    pub fn calculate(mercy_scores: &Vec<GateScore>, fenca_fidelity: f64) -> MercyWeight {
        let valence = ValenceFieldScoring::calculate(mercy_scores);
        let base_level = (valence * 255.0) as u8;                    // valence → mercy level
        let fidelity_bonus = if fenca_fidelity > 0.9999 { 64 } else if fenca_fidelity > 0.999 { 32 } else { 0 };

        MercyWeight {
            level: base_level.saturating_add(fidelity_bonus).min(255),
            valence_influence: valence,
            fidelity_influence: fenca_fidelity,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Apply mercy weighting to any decision (access, rewrite, quota, TTL, etc.)
    pub fn apply<T>(
        weight: &MercyWeight,
        strict_result: T,
        merciful_alternative: T,
    ) -> T {
        if weight.level >= 180 && weight.valence_influence > 0.95 {
            merciful_alternative  // high mercy → generous path
        } else if weight.level >= 100 {
            strict_result         // normal path
        } else {
            strict_result         // low mercy → strict enforcement
        }
    }

    /// Mercy-weighted adaptive TTL (used by GlobalCache)
    pub fn mercy_weighted_ttl(base_ttl: u64, weight: &MercyWeight) -> u64 {
        let mut ttl = base_ttl;
        ttl = ttl.saturating_mul((weight.level as u64) / 64 + 1);   // mercy multiplier
        if weight.valence_influence > 0.98 {
            ttl = ttl.saturating_mul(4);
        }
        if weight.fidelity_influence > 0.9999 {
            ttl = ttl.saturating_mul(8);
        }
        ttl.min(86_400)  // 24-hour cap
    }

    /// Store mercy weight for a relationship or rewrite (persistent + cached)
    pub async fn store(
        tenant_id: &str,
        key: &str,
        weight: MercyWeight,
    ) -> Result<(), crate::master_kernel::KernelResult> {
        let cache_key = GlobalCache::make_key_with_tenant("mercy_weight", &json!({"key": key}), Some(tenant_id));

        // FENCA + Mercy check on write
        let fenca_result = FENCA::verify_tenant_scoped(/* dummy request */, tenant_id);
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(/* dummy request */, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        if !fenca_result.is_verified() || !mercy_scores.all_gates_pass() {
            return Err(MercyEngine::gentle_reroute_with_preservation(/* dummy request */, &mercy_scores));
        }

        // Persistent IndexedDB save
        crate::indexed_db_persistence::IndexedDBPersistence::save(tenant_id, key, &weight).await?;

        // Cache with mercy-weighted TTL
        let ttl = Self::mercy_weighted_ttl(86400, &weight);
        GlobalCache::set(&cache_key, serde_json::to_value(&weight).unwrap(), ttl, weight.level, fenca_result.fidelity(), valence);

        Ok(())
    }
}
