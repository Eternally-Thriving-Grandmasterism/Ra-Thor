// crates/biomimetic/lib.rs
// Ra-Thor Biomimetic Crate — Dedicated home for BiomimeticPatternEngine
// + full multi-algorithm Biomimetic Optimization Engine (SwarmIntelligence)
// Fully cross-pollinated with kernel, quantum, mercy, innovation generator, and the full Omnimaster lattice

// Note: The authoritative pattern engine lives in core/biomimetic_pattern_engine.rs.
// SwarmIntelligence (multi-algorithm optimizer) lives in this crate.

pub use crate::biomimetic_pattern_engine::BiomimeticPatternEngine;
pub use crate::swarm_intelligence::{
    SwarmIntelligence, BiomimeticAlgorithm, BiomimeticOptimizationResult,
};

// Core biomimetic pattern library (names only — full structured catalog lives in core)
pub const BIOMIMETIC_PATTERNS: [&str; 8] = [
    "avian-LEV-self-healing",
    "gecko-setae-adhesion-pinnacle",
    "fractal-528hz-asre-resonance",
    "lotus-self-cleaning-regeneration",
    "mycelial-network-intelligence",
    "spider-silk-tensile-strength",
    "termite-mound-ventilation",
    "whale-fin-turbulence-control",
];

/// Convenience function for the entire Omnimaster lattice
pub async fn apply_biomimetic_pattern(
    pattern_name: &str,
    entangled_themes: &[String],
    base_valence: f64,
    mercy_weight: u8,
) -> f64 {
    BiomimeticPatternEngine::apply_pattern(pattern_name, entangled_themes, base_valence, mercy_weight).await
}

/// Select the best living pattern then apply it
pub async fn select_and_apply_biomimetic(
    entangled_themes: &[String],
    base_valence: f64,
    mercy_weight: u8,
) -> f64 {
    let (_pattern, coherence) =
        BiomimeticPatternEngine::select_and_apply(entangled_themes, base_valence, mercy_weight).await;
    coherence
}

/// Run a full biomimetic optimization with the chosen algorithm
pub async fn run_biomimetic_optimization(
    algorithm: BiomimeticAlgorithm,
    agents: usize,
    environment: &[String],
    base_valence: f64,
    mercy_weight: u8,
) -> BiomimeticOptimizationResult {
    SwarmIntelligence::run_optimization(algorithm, agents, environment, base_valence, mercy_weight).await
}
