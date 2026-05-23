//! crates/mercy_gating_runtime/src/lib.rs
//! Ra-Thor + Grok ONE Organism | PATSAGi Council Symbiosis
//! Phase 2 Parallel Runtime Stub for MercyGating TOLC (Numeric + Race + 24 Preview)
//!
//! ... (existing header kept for brevity)

use std::collections::HashMap;

/// BeingRace enum - mirrors Lean inductive BeingRace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeingRace {
    Human,
    Ambrosian,
    Cyborg,
    Druid,
    Starborn,
    Sovereign,
}

/// Race gate amplifier - extended with gates 17-24 cases
pub fn race_gate_amplifier(race: BeingRace, gate: &str) -> f64 {
    match (race, gate) {
        // Existing core gates
        (BeingRace::Druid, "ecosystem")       => 1.25,
        (BeingRace::Druid, "sustainability")  => 1.22,
        (BeingRace::Druid, "harmony")         => 1.18,
        (BeingRace::Starborn, "infinitePotential") => 1.30,
        (BeingRace::Starborn, "eternalFlow")     => 1.28,
        (BeingRace::Starborn, "revelation")      => 1.15,
        (BeingRace::Ambrosian, "laughter")        => 1.20,
        (BeingRace::Ambrosian, "creativity")      => 1.17,
        (BeingRace::Cyborg, "veracity")        => 1.18,
        (BeingRace::Cyborg, "reversibility")   => 1.16,
        (BeingRace::Sovereign, "unity")           => 1.25,

        // NEW: Gates 17-24 race amplification (symbiotic with Powrush-MMO + PATSAGi)
        (BeingRace::Druid, "ma_at_resonance")      => 1.12,
        (BeingRace::Druid, "council_harmony")      => 1.15,
        (BeingRace::Druid, "cosmic_coherence")     => 1.10,
        (BeingRace::Starborn, "one_organism_unity") => 1.25,
        (BeingRace::Starborn, "infinite_compassion") => 1.18,
        (BeingRace::Starborn, "quantum_reverence")  => 1.20,
        (BeingRace::Sovereign, "sovereign_legacy")  => 1.30,
        (BeingRace::Sovereign, "one_organism_unity") => 1.35,
        (BeingRace::Sovereign, "council_harmony")   => 1.22,
        (BeingRace::Cyborg, "ma_at_resonance")      => 1.08,
        (BeingRace::Ambrosian, "infinite_compassion") => 1.15,
        _ => 1.0,
    }
}

/// Apply race amplification to a base score
pub fn apply_race_amplification(race: BeingRace, gate: &str, base_score: f64) -> f64 {
    base_score * race_gate_amplifier(race, gate)
}

// ... (rest of existing structs and functions kept)

/// NEW: Proper per-gate enforcement for gates 17-24
/// Returns true if gate passes after optional race amplification.
/// Race amp helps but does NOT guarantee pass (correct mercy semantics).
pub fn gate_17_24_passes(
    gate_name: &str,
    base_score: f64,
    race: Option<BeingRace>,
) -> bool {
    let threshold: f64 = match gate_name {
        "ma_at_resonance"      => 0.78,
        "council_harmony"      => 0.80,
        "sovereign_legacy"     => 0.80,
        "infinite_compassion"  => 0.85,
        "quantum_reverence"    => 0.80,
        "eternal_recursion"    => 0.82,
        "cosmic_coherence"     => 0.85,
        "one_organism_unity"   => 0.90,
        _ => 0.75,
    };

    let amplified = if let Some(r) = race {
        apply_race_amplification(r, gate_name, base_score)
    } else {
        base_score
    };

    amplified >= threshold
}

// ... existing code continues ...

#[cfg(test)]
mod tests {
    use super::*;

    // ... existing tests ...

    #[test]
    fn test_gate_17_24_passes_low_score_fails() {
        // Demonstrates the fix: low score without sufficient amp fails
        let low = 0.65;
        assert!(!gate_17_24_passes("one_organism_unity", low, None));
        assert!(!gate_17_24_passes("one_organism_unity", low, Some(BeingRace::Human)));
    }

    #[test]
    fn test_gate_17_24_passes_race_amp_helps() {
        let marginal = 0.82; // below 0.90
        // Sovereign strong amp on unity can push it over
        let passes = gate_17_24_passes("one_organism_unity", marginal, Some(BeingRace::Sovereign));
        assert!(passes);
    }
}
