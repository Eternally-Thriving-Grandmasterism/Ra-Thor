// Council tuning.
// apply_council_tuning refuses a lower new_value and keeps the stored value.
// An equal or higher new_value is stored.
// The comparison is `new_value < current`, the same test as GateThresholdMap::set_threshold.
// set_gate_threshold still stores value.max(0.5). 0.5 is a source literal, not a measured floor.
// That method can replace a higher stored value. apply_council_tuning does not call it.

use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub struct MercyGatingRuntime {
    pub ma_at_threshold: f64,
    pub gate_thresholds: HashMap<String, f64>,
}

impl Default for MercyGatingRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl MercyGatingRuntime {
    /// `ma_at_threshold` starts at the source literal `717.0`.
    /// Gate defaults are the source literals written in this function, not measured results.
    pub fn new() -> Self {
        let mut rt = Self {
            ma_at_threshold: 717.0,
            gate_thresholds: HashMap::new(),
        };
        // Defaults for gates 17-24
        rt.gate_thresholds
            .insert("ma_at_resonance".to_string(), 0.78);
        rt.gate_thresholds
            .insert("one_organism_unity".to_string(), 0.90);
        rt.gate_thresholds
            .insert("council_harmony".to_string(), 0.80);
        rt.gate_thresholds
            .insert("sovereign_legacy".to_string(), 0.80);
        rt
    }

    pub fn get_gate_threshold(&self, gate: &str) -> f64 {
        self.gate_thresholds.get(gate).copied().unwrap_or(0.75)
    }

    /// Stores `value.max(0.5)`. `0.5` is a source literal, not a measured floor.
    /// This can replace a higher stored threshold.
    /// `apply_council_tuning` does not call this method.
    pub fn set_gate_threshold(&mut self, gate: String, value: f64) {
        self.gate_thresholds.insert(gate, value.max(0.5));
    }

    /// Refuses `new_value` when it is below the stored value. The stored value stays.
    /// An equal or higher `new_value` is stored. `success` is false only on a refusal.
    /// `MaAtThreshold` compares with `ma_at_threshold`.
    /// `GateThreshold` compares with `get_gate_threshold` (a missing name reports `0.75`).
    /// `0.75` is that function's source literal, not a measured floor.
    /// `RaceAmplifier` does not change a threshold. `success` is true for that target.
    pub fn apply_council_tuning(&mut self, proposal: &CouncilTuningProposal) -> TuningResult {
        match &proposal.target {
            TuningTarget::MaAtThreshold => {
                let previous = self.ma_at_threshold;
                if proposal.new_value < previous {
                    return TuningResult {
                        success: false,
                        previous_value: previous,
                        new_value: previous,
                        message: format!(
                            "Ma'at threshold can only increase or stay (current: {previous:.1})"
                        ),
                    };
                }
                self.ma_at_threshold = proposal.new_value;
                TuningResult {
                    success: true,
                    previous_value: previous,
                    new_value: self.ma_at_threshold,
                    message: format!(
                        "Council #{} adjusted Ma'at threshold → {:.1} | {}",
                        proposal.council_id, self.ma_at_threshold, proposal.justification
                    ),
                }
            }
            TuningTarget::GateThreshold { gate } => {
                let previous = self.get_gate_threshold(gate);
                if proposal.new_value < previous {
                    return TuningResult {
                        success: false,
                        previous_value: previous,
                        new_value: previous,
                        message: format!(
                            "Threshold for '{gate}' can only increase or stay (current: {previous:.2})"
                        ),
                    };
                }
                self.gate_thresholds
                    .insert(gate.clone(), proposal.new_value);
                TuningResult {
                    success: true,
                    previous_value: previous,
                    new_value: proposal.new_value,
                    message: format!(
                        "Council #{} tuned gate '{}' threshold → {:.2} | {}",
                        proposal.council_id, gate, proposal.new_value, proposal.justification
                    ),
                }
            }
            _ => TuningResult {
                success: true,
                previous_value: 0.0,
                new_value: proposal.new_value,
                message: format!("Council #{} tuning acknowledged", proposal.council_id),
            },
        }
    }

    pub fn apply_council_tunings(
        &mut self,
        proposals: &[CouncilTuningProposal],
    ) -> Vec<TuningResult> {
        proposals
            .iter()
            .map(|p| self.apply_council_tuning(p))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maat_raise_to_755_stores_755() {
        let mut runtime = MercyGatingRuntime::new();
        assert_eq!(runtime.ma_at_threshold, 717.0);

        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::MaAtThreshold,
            new_value: 755.0,
            justification: "Raised during arbitration".to_string(),
            proposed_at_turn: 42,
        };

        let result = runtime.apply_council_tuning(&proposal);
        assert!(result.success);
        assert_eq!(result.previous_value, 717.0);
        assert_eq!(result.new_value, 755.0);
        assert_eq!(runtime.ma_at_threshold, 755.0);
    }

    #[test]
    fn test_dynamic_per_gate_threshold_update() {
        let mut runtime = MercyGatingRuntime::new();
        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::GateThreshold {
                gate: "one_organism_unity".to_string(),
            },
            new_value: 0.95,
            justification: "Strengthen One Organism Unity during high-stakes phase".to_string(),
            proposed_at_turn: 50,
        };

        let result = runtime.apply_council_tuning(&proposal);
        assert!(result.success);
        assert_eq!(runtime.get_gate_threshold("one_organism_unity"), 0.95);
    }

    #[test]
    fn lower_ma_at_proposal_is_rejected_and_717_stays() {
        let mut runtime = MercyGatingRuntime::new();
        let proposal = CouncilTuningProposal {
            council_id: 99,
            target: TuningTarget::MaAtThreshold,
            new_value: 100.0,
            justification: "lower attempt".to_string(),
            proposed_at_turn: 1,
        };
        let result = runtime.apply_council_tuning(&proposal);
        assert!(!result.success);
        assert_eq!(result.previous_value, 717.0);
        assert_eq!(result.new_value, 717.0);
        assert_eq!(runtime.ma_at_threshold, 717.0);
    }

    #[test]
    fn equal_ma_at_proposal_is_accepted() {
        let mut runtime = MercyGatingRuntime::new();
        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::MaAtThreshold,
            new_value: 717.0,
            justification: "same value".to_string(),
            proposed_at_turn: 2,
        };
        let result = runtime.apply_council_tuning(&proposal);
        assert!(result.success);
        assert_eq!(result.previous_value, 717.0);
        assert_eq!(result.new_value, 717.0);
        assert_eq!(runtime.ma_at_threshold, 717.0);
    }

    #[test]
    fn lower_named_threshold_is_rejected_and_the_old_value_stays() {
        let mut runtime = MercyGatingRuntime::new();
        assert_eq!(runtime.get_gate_threshold("one_organism_unity"), 0.90);
        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::GateThreshold {
                gate: "one_organism_unity".to_string(),
            },
            new_value: 0.50,
            justification: "lower attempt".to_string(),
            proposed_at_turn: 3,
        };
        let result = runtime.apply_council_tuning(&proposal);
        assert!(!result.success);
        assert_eq!(result.previous_value, 0.90);
        assert_eq!(result.new_value, 0.90);
        assert_eq!(runtime.get_gate_threshold("one_organism_unity"), 0.90);
        assert_eq!(runtime.get_gate_threshold("council_harmony"), 0.80);
    }

    #[test]
    fn higher_named_threshold_is_accepted() {
        let mut runtime = MercyGatingRuntime::new();
        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::GateThreshold {
                gate: "council_harmony".to_string(),
            },
            new_value: 0.81,
            justification: "raise".to_string(),
            proposed_at_turn: 4,
        };
        let result = runtime.apply_council_tuning(&proposal);
        assert!(result.success);
        assert_eq!(result.previous_value, 0.80);
        assert_eq!(result.new_value, 0.81);
        assert_eq!(runtime.get_gate_threshold("council_harmony"), 0.81);
        assert_eq!(runtime.get_gate_threshold("one_organism_unity"), 0.90);
    }
}
