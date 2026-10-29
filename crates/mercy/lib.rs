// crates/mercy/lib.rs
// Ra-Thor Mercy Crate — Dedicated home for Mercy Engine, Valence Scoring,
// 7 Living Mercy Gates, and MercyWeighting
// Fully cross-pollinated with kernel, quantum, biomimetic, innovation generator, and the full lattice

pub mod mercy_engine;
pub mod valence_field_scoring;
pub mod mercy_weighting;

// Public re-exports for clean workspace usage
pub use mercy_engine::MercyEngine;
pub use valence_field_scoring::ValenceFieldScoring;
pub use mercy_weighting::MercyWeighting;

// Core mercy gate definitions (7 Living Gates)
pub const MERCY_GATES: [&str; 7] = [
    "Truth", "Non-Harm", "Abundance", "Sovereignty",
    "Harmony", "Joy", "Peace"
];

// Convenience function for the entire Omnimaster lattice
pub async fn evaluate_mercy_deep(request: &crate::master_kernel::RequestPayload) -> (f64, Vec<crate::mercy_engine::GateScore>) {
    let scores = MercyEngine::evaluate_deep_with_tenant(request, &request.tenant_id);
    let valence = ValenceFieldScoring::calculate(&scores);
    (valence, scores)
}

// Cross-pollination hook back to kernel and innovation systems
pub async fn trigger_mercy_innovation(mercy_scores: &[crate::mercy_engine::GateScore], mercy_weight: u8) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("Mercy Engine evaluation with valence {:.2}", ValenceFieldScoring::calculate(mercy_scores))],
        mercy_scores,
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}
