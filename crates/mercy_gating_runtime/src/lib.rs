//! Race amplifiers and per-gate comparisons.
//! `gate_17_24_passes` compares a score with the source-literal thresholds in that function.
//! Those thresholds are not measured results.
//! `src/error.rs` does not exist. `MercyError` is not defined.
//! These files are not modules: `dynamic_tuning.rs`, `gates_17_24_enforcement.rs`,
//! `hot_reload.rs`, `metrics.rs`, `patsagi_arbitration.rs`, `patsagi_governance.rs`, `runtime.rs`.
//! `src/bin/metrics_server.rs` is not a binary target.
//! Contact: info@Rathor.ai

pub mod council_tuning;
pub mod gate_threshold_map;

pub use council_tuning::{CouncilTuningProposal, MercyGatingRuntime, TuningResult, TuningTarget};
pub use gate_threshold_map::GateThresholdMap;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sovereign_unity_amplifier_is_the_source_literal_1_38() {
        assert_eq!(
            race_gate_amplifier(BeingRace::Sovereign, "one_organism_unity"),
            1.38
        );
        assert_eq!(
            race_gate_amplifier(BeingRace::Human, "one_organism_unity"),
            1.0
        );
        assert_eq!(
            apply_race_amplification(BeingRace::Sovereign, "one_organism_unity", 0.85),
            0.85 * 1.38
        );
    }

    #[test]
    fn unity_gate_passes_at_the_source_literal_0_90_and_fails_below_it() {
        assert!(gate_17_24_passes("one_organism_unity", 0.90, None));
        assert!(!gate_17_24_passes("one_organism_unity", 0.899, None));
        assert_eq!(
            apply_race_amplification(BeingRace::Druid, "ecosystem", 2.0),
            2.5
        );
    }

    #[test]
    fn maat_proposal_below_650_is_stored_as_650() {
        let mut runtime = MercyGatingRuntime::new();
        assert_eq!(runtime.ma_at_threshold, 717.0);
        let proposal = CouncilTuningProposal {
            council_id: 99,
            target: TuningTarget::MaAtThreshold,
            new_value: 100.0,
            justification: "lower attempt".to_string(),
            proposed_at_turn: 1,
        };
        let result = runtime.apply_council_tuning(&proposal);
        assert!(result.success);
        assert_eq!(result.previous_value, 717.0);
        assert_eq!(result.new_value, 650.0);
        assert_eq!(runtime.ma_at_threshold, 650.0);
    }

    #[test]
    fn gate_threshold_map_rejects_a_decrease_and_keeps_0_80() {
        let mut map = GateThresholdMap::new();
        assert_eq!(map.get_threshold("council_harmony"), 0.80);
        assert!(map.set_threshold("one_organism_unity", 0.91).is_ok());
        assert_eq!(map.get_threshold("one_organism_unity"), 0.91);
        assert!(map.set_threshold("one_organism_unity", 0.89).is_err());
        assert_eq!(map.get_threshold("one_organism_unity"), 0.91);
        assert_eq!(map.get_threshold("council_harmony"), 0.80);
        assert!(map.would_pass("one_organism_unity", 0.1, 0.91));
        assert!(!map.would_pass("one_organism_unity", 0.99, 0.90));
    }
}
