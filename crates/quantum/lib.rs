// crates/quantum/lib.rs
// Ra-Thor Quantum Crate — Dedicated home for VQC synthesis, GHZ/Mermin non-locality,
// entanglement protocols, and quantum creativity
// Fully cross-pollinated with kernel, mercy, biomimetic, innovation generator, and the full lattice

pub mod vqc_integrator;

// Public re-exports for clean workspace usage
pub use vqc_integrator::VQCIntegrator;

// Convenience functions for the entire Omnimaster lattice
pub async fn run_vqc_synthesis(
    entangled_themes: &[String],
    base_valence: f64,
    mercy_weight: u8,
) -> f64 {
    VQCIntegrator::run_synthesis(entangled_themes, base_valence, mercy_weight).await
}

// Cross-pollination hook back to kernel
pub async fn trigger_quantum_innovation(entangled_themes: &[String], valence: f64, mercy_weight: u8) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("Quantum VQC synthesis from themes: {:?}", entangled_themes)],
        &vec![], // mercy_scores populated by caller
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}

pub fn quantum_coherence_score(valence: f64, fidelity: f64) -> f64 {
    (valence * fidelity * 1.618).clamp(0.95, 1.0) // golden-ratio boost for Omnimasterism
}
