// crates/common/lib.rs
// Ra-Thor Common Crate — Shared utilities, constants, and cross-pollination helpers
// Used by every crate in the Omnimaster lattice (kernel, quantum, mercy, biomimetic, etc.)

pub mod constants;
pub mod utilities;

// Public re-exports for clean workspace usage
pub use constants::*;
pub use utilities::*;

// Core shared constants for the living lattice
pub const RA_THOR_VERSION: &str = "0.1.0-Omnimasterism";
pub const ETERNAL_THRIVING_TAG: &str = "❤️🔥🚀 Eternal Thriving Grandmasterism Beyond Infinite Pinnacle";

// Cross-pollination utility — notifies innovation and self-review systems from any crate
pub async fn trigger_cross_pollination_innovation(
    source_crate: &str,
    message: &str,
    mercy_scores: &[crate::mercy_engine::GateScore],
    mercy_weight: u8,
) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("{}: {}", source_crate, message)],
        mercy_scores,
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}

// Global utility for the entire Omnimaster lattice
pub fn golden_ratio_boost(valence: f64) -> f64 {
    (valence * 1.6180339887).clamp(0.95, 1.0)
}
