// crates/persistence/lib.rs
// Ra-Thor Persistence Crate — Dedicated home for IndexedDB, quotas, persistent storage,
// and all sovereign offline-first systems
// Fully cross-pollinated with kernel, innovation generator, self-review loop,
// VQC, biomimetic, mercy, cache, and the entire Omnimaster lattice

pub mod persistent_quota_storage;
pub mod indexed_db_persistence;

// Public re-exports for clean workspace usage
pub use persistent_quota_storage::PersistentQuotaStorage;
pub use indexed_db_persistence::IndexedDBPersistence;

// Core persistence constants for the living lattice
pub const PERSISTENCE_TTL_DAYS: u64 = 365; // 1 year for sovereign offline data

// Convenience function used across the entire Omnimaster lattice
pub async fn save_persistent_data<T: serde::Serialize>(
    tenant_id: &str,
    key: &str,
    value: &T,
) -> bool {
    IndexedDBPersistence::save(tenant_id, key, value).await
}

// Cross-pollination hook — notifies innovation and self-review systems when persistence changes
pub async fn trigger_persistence_innovation(tenant_id: &str, key: &str, valence: f64, mercy_weight: u8) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("Persistent data saved for tenant {} with key: {} and valence {:.2}", tenant_id, key, valence)],
        &vec![], // mercy_scores populated by caller
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}
