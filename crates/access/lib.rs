// crates/access/lib.rs
// Ra-Thor Access Crate — Dedicated home for ReBAC, RBAC, ABAC, tenant isolation,
// relationship graph traversal, and all sovereign access control systems
// Fully cross-pollinated with kernel, quantum, mercy, biomimetic, persistence,
// cache, orchestration, innovation generator, self-review loop, and the entire Omnimaster lattice

pub mod rebac_graph_storage;
pub mod rbac;
pub mod abac;
pub mod hybrid_access;

// Public re-exports for clean workspace usage
pub use rebac_graph_storage::ReBACGraphStorage;
pub use rbac::RBAC;
pub use abac::ABAC;
pub use hybrid_access::HybridAccess;

// Core access constants for the living lattice
pub const TENANT_ISOLATION_PREFIX: &str = "tenant_";

// Convenience function used across the entire Omnimaster lattice
pub async fn check_access(request: crate::master_kernel::RequestPayload) -> crate::master_kernel::KernelResult {
    HybridAccess::check(&request).await
}

// Cross-pollination hook — notifies innovation and self-review systems when access decisions are made
pub async fn trigger_access_innovation(mercy_scores: &[crate::mercy_engine::GateScore], mercy_weight: u8, decision: &str) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("Access control decision '{}' made with valence {:.2}", decision, crate::valence_field_scoring::ValenceFieldScoring::calculate(mercy_scores))],
        mercy_scores,
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}
