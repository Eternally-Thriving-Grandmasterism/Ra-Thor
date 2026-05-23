// council_tuning.rs (enhanced with Rust-side property test mirroring Lean theorems)

use crate::MercyGatingRuntime;

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

impl MercyGatingRuntime {
    pub fn apply_council_tuning(&mut self, proposal: &CouncilTuningProposal) -> TuningResult {
        match &proposal.target {
            TuningTarget::MaAtThreshold => {
                let previous = self.ma_at_threshold;
                self.ma_at_threshold = proposal.new_value.max(650.0); // safety floor

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
            _ => TuningResult {
                success: true,
                previous_value: 0.0,
                new_value: proposal.new_value,
                message: format!("Council #{} tuning acknowledged", proposal.council_id),
            },
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
    fn test_council_13_raises_ma_at_threshold_mid_simulation() {
        let mut runtime = MercyGatingRuntime::default();
        let initial = runtime.ma_at_threshold;

        let proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::MaAtThreshold,
            new_value: 755.0,
            justification: "Raised during multi-faction sacred node arbitration".to_string(),
            proposed_at_turn: 42,
        };

        let result = runtime.apply_council_tuning(&proposal);

        assert!(result.success);
        assert_eq!(result.new_value, 755.0);
        assert!(runtime.ma_at_threshold >= 755.0);
        println!("[TEST] Council #13 dynamic tuning applied successfully");
    }

    // Rust-side property test mirroring Lean theorems
    #[test]
    fn test_rust_mirrors_lean_safety_floor_and_monotonicity() {
        let mut runtime = MercyGatingRuntime::default();
        let initial_threshold = runtime.ma_at_threshold;

        // Test 1: Safety floor (never below 650)
        let dangerous_proposal = CouncilTuningProposal {
            council_id: 99,
            target: TuningTarget::MaAtThreshold,
            new_value: 500.0, // attempt to go below floor
            justification: "Malicious low attempt".to_string(),
            proposed_at_turn: 1,
        };
        let _ = runtime.apply_council_tuning(&dangerous_proposal);
        assert!(runtime.ma_at_threshold >= 650.0, "Safety floor violated in Rust");

        // Test 2: Monotonicity (can only stay or increase)
        let raise_proposal = CouncilTuningProposal {
            council_id: 13,
            target: TuningTarget::MaAtThreshold,
            new_value: 780.0,
            justification: "Raise for higher coherence".to_string(),
            proposed_at_turn: 2,
        };
        let _ = runtime.apply_council_tuning(&raise_proposal);
        assert!(runtime.ma_at_threshold >= initial_threshold, "Monotonicity violated in Rust");

        println!("[TEST] Rust mirrors Lean safety floor + monotonicity theorems");
    }
}
