// crates/biomimetic/lib.rs
// Ra-Thor Biomimetic Crate — Dedicated home for BiomimeticPatternEngine
// and all living nature-inspired designs (avian LEV, gecko setae, fractal 528 Hz ASRE, lotus self-cleaning, etc.)
// Fully cross-pollinated with kernel, quantum, mercy, innovation generator, and the full Omnimaster lattice

pub mod biomimetic_pattern_engine;

// Public re-exports for clean workspace usage
pub use biomimetic_pattern_engine::BiomimeticPatternEngine;

// Core biomimetic pattern library
pub const BIOMIMETIC_PATTERNS: [&str; 8] = [
    "avian-LEV-self-healing",
    "gecko-setae-adhesion-pinnacle",
    "fractal-528hz-asre-resonance",
    "lotus-self-cleaning-regeneration",
    "spider-silk-tensile-strength",
    "termite-mound-ventilation",
    "whale-fin-turbulence-control",
    "coral-reef-structural-resilience",
];

// Convenience function for the entire Omnimaster lattice
pub async fn apply_biomimetic_pattern(
    pattern_name: &str,
    entangled_themes: &[String],
    base_valence: f64,
    mercy_weight: u8,
) -> f64 {
    BiomimeticPatternEngine::apply_pattern(pattern_name, entangled_themes, base_valence, mercy_weight).await
}

// Cross-pollination hook back to kernel and innovation systems
pub async fn trigger_biomimetic_innovation(pattern_name: &str, mercy_scores: &[crate::mercy_engine::GateScore], mercy_weight: u8) {
    if let Some(innovation) = crate::innovation_generator::InnovationGenerator::create_from_recycled(
        vec![format!("Biomimetic {} pattern activated with valence {:.2}", pattern_name, crate::valence_field_scoring::ValenceFieldScoring::calculate(mercy_scores))],
        mercy_scores,
        mercy_weight,
    ).await {
        crate::root_core_orchestrator::RootCoreOrchestrator::delegate_innovation(innovation).await;
    }
}
