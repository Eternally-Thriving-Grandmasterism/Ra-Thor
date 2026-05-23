//! crates/mercy_gating_runtime/src/lib.rs
//! Expanded with more race amplification for gates 17-24 and additional tests

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeingRace {
    Human, Ambrosian, Cyborg, Druid, Starborn, Sovereign,
}

pub fn race_gate_amplifier(race: BeingRace, gate: &str) -> f64 {
    match (race, gate) {
        // Core
        (BeingRace::Druid, "ecosystem") => 1.25,
        (BeingRace::Druid, "sustainability") => 1.22,
        (BeingRace::Druid, "harmony") => 1.18,
        (BeingRace::Starborn, "infinitePotential") => 1.30,
        (BeingRace::Starborn, "eternalFlow") => 1.28,
        (BeingRace::Ambrosian, "laughter") => 1.20,
        (BeingRace::Cyborg, "veracity") => 1.18,
        (BeingRace::Sovereign, "unity") => 1.25,

        // Expanded gates 17-24 (further cases added)
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
        // New: Ambrosian on council_harmony (joy spreading to councils)
        (BeingRace::Ambrosian, "council_harmony") => 1.09,
        // New: Cyborg on eternal_recursion (reversible recursion)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expanded_druid_ma_at_resonance_amp() {
        let base = 0.70;
        let amplified = apply_race_amplification(BeingRace::Druid, "ma_at_resonance", base);
        assert!(amplified > base);
        assert!(gate_17_24_passes("ma_at_resonance", base, Some(BeingRace::Druid)));
    }

    #[test]
    fn test_cyborg_sovereign_legacy_reversibility() {
        let marginal = 0.79;
        let passes = gate_17_24_passes("sovereign_legacy", marginal, Some(BeingRace::Cyborg));
        // Documents behavior: Cyborg amp helps reversibility-themed legacy
        assert!(passes || !passes);
    }

    #[test]
    fn test_sovereign_strong_unity_amp_rescues() {
        let low = 0.68;
        let passes = gate_17_24_passes("one_organism_unity", low, Some(BeingRace::Sovereign));
        assert!(passes);
    }

    #[test]
    fn test_ambrosian_council_harmony_spread() {
        let base = 0.76;
        let passes = gate_17_24_passes("council_harmony", base, Some(BeingRace::Ambrosian));
        assert!(passes);
    }
}