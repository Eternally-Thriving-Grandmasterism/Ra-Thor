// crates/cache/lib.rs
// Ra-Thor Cache Crate — Dedicated home for GlobalCache and Adaptive TTL
// Fully cross-pollinated with kernel, innovation generator, self-review loop,
// VQC, biomimetic, mercy, and the entire Omnimaster lattice

pub mod global_cache;

// Public re-exports for clean workspace usage
pub use global_cache::GlobalCache;

// Core adaptive TTL constants for the living lattice
pub const DEFAULT_CACHE_TTL: u64 = 86400;        // 24 hours
pub const CODEX_CACHE_TTL: u64 = 86400 * 30;     // 30 days for codices
pub const INNOVATION_CACHE_TTL: u64 = 86400 * 7; // 7 days for innovations

// Convenience function used across the entire Omnimaster lattice
pub fn adaptive_ttl(base_ttl: u64, fidelity: f64, valence: f64, mercy_weight: u8) -> u64 {
    GlobalCache::adaptive_ttl(base_ttl, fidelity, valence, mercy_weight)
}

// Cross-pollination hook — notifies innovation and self-review systems when cache changes
pub async fn trigger_cache_innovation(key: &str, valence: f64, mercy_weight: u8) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("Global Cache update detected for key: {} with valence {:.2}", key, valence)],
        &vec![], // mercy_scores populated by caller
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}
