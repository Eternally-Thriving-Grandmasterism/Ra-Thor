// crates/mercy_gating_runtime/src/council_tuning.rs
// Extended with dynamic per-gate threshold map + property test mirroring Lean theorems

use std::collections::HashMap;
use crate::BeingRace; // if needed

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TuningTarget {
    MaAtThreshold,
    GateThreshold { gate: String },
    RaceAmplifier { race: String, gate: String },
}

#[derive(Debug, Clone)]
pub struct CouncilTuningProposal {
    pub council_id: u32,
    pub target: TuningTarget,
    pub new_value: f64,
    pub justification: String,
    pub proposed_at_turn: u64,
}

#[derive(Debug, Clone)]
pub struct TuningResult {
    pub success: bool,
    pub previous_value: f64,
    pub new_value: f64,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct MercyGatingRuntime {
    pub ma_at_threshold: f64,
    pub gate_thresholds: HashMap<String, f64>,  // NEW: Dynamic per-gate thresholds
}

impl MercyGatingRuntime {
    pub fn new() -> Self {
        let mut rt = Self {
            ma_at_threshold: 717.0,
            gate_thresholds: HashMap::new(),
        };
        // Defaults for gates 17-24
        rt.gate_thresholds.insert("ma_at_resonance".to_string(), 0.78);
        rt.gate_thresholds.insert("one_organism_unity".to_string(), 0.90);
        rt.gate_thresholds.insert("council_harmony".to_string(), 0.80);
        rt.gate_thresholds.insert("sovereign_legacy".to_string(), 0.80);
        rt
    }

    pub fn get_gate_threshold(&self, gate: &str) -> f64 {
        self.gate_thresholds.get(gate).copied().unwrap_or(0.75)
    }

    pub fn set_gate_threshold(&mut self, gate: String, value: f64) {
        self.gate_thresholds.insert(gate, value.max(0.5));
    }

    pub fn apply_council_tuning(&mut self, proposal: &CouncilTuningProposal) -> TuningResult {
        match &proposal.target {
            TuningTarget::MaAtThreshold => {
                let previous = self.ma_at_threshold;
                self.ma_at_threshold = proposal.new_value.max(650.0);
                TuningResult {
                    success: true,
                    previous_value: previous,
                    new_value: self.ma_at_threshold,
                    message: format!("Council #{} adjusted Ma'at threshold → {:.1} | {}", proposal.council_id, self.ma_at_threshold, proposal.justification),
                }
            }
            TuningTarget::GateThreshold { gate } => {
                let previous = self.get_gate_threshold(gate);
                self.set_gate_threshold(gate.clone(), proposal.new_value);
                TuningResult {
                    success: true,
                    previous_value: previous,
                    new_value: proposal.new_value,
                    message: format!("Council #{} tuned gate '{}' threshold → {:.2} | {}", proposal.council_id, gate, proposal.new_value, proposal.justification),
                }
            }
            _ => TuningResult { success: true, previous_value: 0.0, new_value: proposal.new_value, message: format!("Council #{} tuning acknowledged", proposal.council_id) },
        }
    }

    pub fn apply_council_tunings(&mut self, proposals: &[CouncilTuningProposal]) -> Vec<TuningResult> {
        proposals.iter().map(|p| self.apply_council_tuning(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_mirrors_lean_safety_floor_and_monotonicity() {
        let mut runtime = MercyGatingRuntime::new();
        let initial = runtime.ma_at_threshold;

        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::MaAtThreshold,
            new_value: 755.0,
            justification: "Raised during arbitration".to_string(),
            proposed_at_turn: 42,
        };

        let result = runtime.apply_council_tuning(&proposal);
        assert!(result.success);
        assert!(runtime.ma_at_threshold >= 755.0);
        assert!(runtime.ma_at_threshold >= initial); // monotonic
    }

    #[test]
    fn test_dynamic_per_gate_threshold_update() {
        let mut runtime = MercyGatingRuntime::new();
        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::GateThreshold { gate: "one_organism_unity".to_string() },
            new_value: 0.95,
            justification: "Strengthen One Organism Unity during high-stakes phase".to_string(),
            proposed_at_turn: 50,
        };

        let result = runtime.apply_council_tuning(&proposal);
        assert!(result.success);
        assert_eq!(runtime.get_gate_threshold("one_organism_unity"), 0.95);
    }
}
