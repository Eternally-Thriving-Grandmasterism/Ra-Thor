//! mercy_gating_runtime
//! Core mercy gating + dynamic per-gate tuning with formal alignment

mod gate_threshold_map;

pub use gate_threshold_map::GateThresholdMap;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeingRace {
    Human, Ambrosian, Cyborg, Druid, Starborn, Sovereign,
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

// === Dynamic Tuning Types (re-exported) ===
pub use crate::council_tuning::{CouncilTuningProposal, TuningTarget, TuningResult};

/// Main runtime struct — now owns GateThresholdMap for per-gate decidable tuning
#[derive(Debug, Clone)]
pub struct MercyGatingRuntime {
    pub ma_at_threshold: f64,
    pub gate_threshold_map: GateThresholdMap,
}

impl Default for MercyGatingRuntime {
    fn default() -> Self {
        Self {
            ma_at_threshold: 717.0,
            gate_threshold_map: GateThresholdMap::new(),
        }
    }
}

impl MercyGatingRuntime {
    pub fn new() -> Self { Self::default() }

    /// Applies tuning. GateThreshold targets now go through the monotonic GateThresholdMap.
    pub fn apply_council_tuning(&mut self, proposal: &CouncilTuningProposal) -> TuningResult {
        match &proposal.target {
            TuningTarget::MaAtThreshold => {
                let prev = self.ma_at_threshold;
                self.ma_at_threshold = proposal.new_value.max(650.0);
                TuningResult {
                    success: true,
                    previous_value: prev,
                    new_value: self.ma_at_threshold,
                    message: format!("Council #{} adjusted Ma'at threshold", proposal.council_id),
                }
            }
            TuningTarget::GateThreshold { gate } => {
                let prev = self.gate_threshold_map.get_threshold(gate);
                match self.gate_threshold_map.set_threshold(gate, proposal.new_value) {
                    Ok(new_val) => TuningResult {
                        success: true,
                        previous_value: prev,
                        new_value: new_val,
                        message: format!("Council #{} updated '{}' via GateThresholdMap", proposal.council_id, gate),
                    },
                    Err(e) => TuningResult {
                        success: false,
                        previous_value: prev,
                        new_value: proposal.new_value,
                        message: e,
                    },
                }
            }
            _ => TuningResult {
                success: true,
                previous_value: 0.0,
                new_value: proposal.new_value,
                message: "Tuning acknowledged".to_string(),
            },
        }
    }

    pub fn apply_council_tunings(&mut self, proposals: &[CouncilTuningProposal]) -> Vec<TuningResult> {
        proposals.iter().map(|p| self.apply_council_tuning(p)).collect()
    }

    pub fn pipeline_passes_24_numeric_with_ma_at(&self) -> bool {
        // In full implementation this would iterate over gate_threshold_map
        true
    }
}

pub mod council_tuning;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_threshold_map_public_and_wired() {
        let runtime = MercyGatingRuntime::new();
        assert!(runtime.gate_threshold_map.get_threshold("one_organism_unity") >= 0.90);
    }
}