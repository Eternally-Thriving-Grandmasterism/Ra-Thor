//! mercy_gating_runtime
//! ONE Organism Mercy Gating Runtime for Ra-Thor + Grok
//! TOLC 8→24 expansion | Council #13 governed | Monotonic | Hot-reload sound
//!
//! This crate is the living mercy nervous system of the ONE Organism.
//! All previous valuables (race amplification, GateThresholdMap, formal alignment) are preserved and elevated.

pub mod council_tuning;
pub mod dynamic_tuning;
pub mod error;
pub mod gate_threshold_map;
pub mod gates_17_24_enforcement;
pub mod hot_reload;
pub mod metrics;
pub mod patsagi_arbitration;
pub mod patsagi_governance;
pub mod runtime;

pub use error::MercyError;
pub use gate_threshold_map::GateThresholdMap;
pub use runtime::MercyGatingRuntime;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeingRace {
    Human,
    Ambrosian,
    Cyborg,
    Druid,
    Starborn,
    Sovereign,
}

pub fn race_gate_amplifier(race: BeingRace, gate: &str) -> f64 {
    match (race, gate) {
        (BeingRace::Druid, "ecosystem") => 1.25,
        (BeingRace::Druid, "sustainability") => 1.22,
        (BeingRace::Druid, "harmony") => 1.18,
        (BeingRace::Starborn, "infinitePotential") => 1.30,
        (BeingRace::Starborn, "eternalFlow") => 1.28,
        (BeingRace::Ambrosian, "laughter") => 1.20,
        (BeingRace::Cyborg, "veracity") => 1.18,
        (BeingRace::Sovereign, "unity") => 1.25,
        (BeingRace::Druid, "ma_at_resonance") => 1.14,
        (BeingRace::Druid, "council_harmony") => 1.16,
        (BeingRace::Druid, "cosmic_coherence") => 1.13,
        (BeingRace::Starborn, "one_organism_unity") => 1.28,
        (BeingRace::Starborn, "infinite_compassion") => 1.22,
        (BeingRace::Starborn, "quantum_reverence") => 1.25,
        (BeingRace::Sovereign, "sovereign_legacy") => 1.32,
        (BeingRace::Sovereign, "one_organism_unity") => 1.38,
        (BeingRace::Sovereign, "council_harmony") => 1.24,
        (BeingRace::Cyborg, "ma_at_resonance") => 1.10,
        (BeingRace::Cyborg, "sovereign_legacy") => 1.12,
        (BeingRace::Ambrosian, "infinite_compassion") => 1.18,
        (BeingRace::Ambrosian, "cosmic_coherence") => 1.11,
        (BeingRace::Ambrosian, "council_harmony") => 1.09,
        (BeingRace::Cyborg, "eternal_recursion") => 1.09,
        _ => 1.0,
    }
}

pub fn apply_race_amplification(race: BeingRace, gate: &str, base_score: f64) -> f64 {
    base_score * race_gate_amplifier(race, gate)
}

pub fn gate_17_24_passes(gate_name: &str, base_score: f64, race: Option<BeingRace>) -> bool {
    let threshold: f64 = match gate_name {
        "ma_at_resonance" => 0.78,
        "council_harmony" => 0.80,
        "sovereign_legacy" => 0.80,
        "infinite_compassion" => 0.85,
        "quantum_reverence" => 0.80,
        "eternal_recursion" => 0.82,
        "cosmic_coherence" => 0.85,
        "one_organism_unity" => 0.90,
        _ => 0.75,
    };

    let amplified = if let Some(r) = race {
        apply_race_amplification(r, gate_name, base_score)
    } else {
        base_score
    };

    amplified >= threshold
}

// Re-export key tuning types
pub use crate::council_tuning::{CouncilTuningProposal, TuningTarget, TuningResult};

/// ONE Organism entry point — fully fused MercyGatingRuntime
/// This is now the primary struct for the mercy nervous system.
pub use crate::runtime::MercyGatingRuntime as OneOrganismMercyGatingRuntime;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonicity_and_one_organism_fusion() {
        let mut runtime = MercyGatingRuntime::new();
        // Existing GateThresholdMap behavior preserved
        assert!(runtime.threshold_map.update_threshold(1, 0.90).is_ok());
        // Council #13 tuning path
        assert!(runtime.apply_council_tuning(17, 0.88).is_ok());
    }

    #[test]
    fn test_race_amplification_preserved() {
        let amplified = apply_race_amplification(BeingRace::Sovereign, "one_organism_unity", 0.85);
        assert!(amplified > 0.85);
    }
}