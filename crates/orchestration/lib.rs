// crates/orchestration/lib.rs
// Ra-Thor Orchestration Crate — Dedicated home for multi-user orchestration,
// RootCoreOrchestrator coordination, and all lattice-level delegation logic
// Fully cross-pollinated with kernel, quantum, mercy, biomimetic, persistence,
// cache, innovation generator, self-review loop, and the entire Omnimaster lattice

pub mod multi_user_orchestrator;

// Public re-exports for clean workspace usage
pub use multi_user_orchestrator::MultiUserOrchestrator;

// Convenience entry point used across the entire Omnimaster lattice
pub async fn orchestrate_multi_user(request: crate::master_kernel::RequestPayload) -> crate::master_kernel::KernelResult {
    crate::root_core_orchestrator::RootCoreOrchestrator::orchestrate(request).await
}

// Cross-pollination hooks
pub async fn trigger_orchestration_innovation(mercy_scores: &[crate::mercy_engine::GateScore], mercy_weight: u8) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("Orchestration layer activated with valence {:.2}", crate::valence_field_scoring::ValenceFieldScoring::calculate(mercy_scores))],
        mercy_scores,
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}

// Eternal self-optimization trigger for orchestration layer
pub async fn trigger_eternal_orchestration_optimization() {
    crate::self_review_loop::SelfReviewLoop::run().await;
}
