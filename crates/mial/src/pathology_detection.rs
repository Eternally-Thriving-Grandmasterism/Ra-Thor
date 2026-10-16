//! Substring labels, plus a `CouncilTuningProposal` applied on an owned runtime.
//! `detect_pathologies` returns the first matching label. It does not score a model.
//! The `< 0.65` branch is not reached: `< 0.70` returns `goal_drift_detected` first.
//! `trigger_automatic_recalibration` stores `new_value.max(0.5)` through `MercyGatingRuntime`.
//! `0.5` is a source literal in that runtime, not a measured floor.

use mercy_gating_runtime::{CouncilTuningProposal, MercyGatingRuntime, TuningTarget};

pub struct PathologyDetectionEngine {
    runtime: MercyGatingRuntime,
    council_id: u32,
}

impl PathologyDetectionEngine {
    pub fn new(runtime: MercyGatingRuntime, council_id: u32) -> Self {
        Self {
            runtime,
            council_id,
        }
    }

    /// Detect pathologies. Returns Some(description) if action needed.
    pub fn detect_pathologies(&self, content: &str, mercy_score: f64) -> Option<String> {
        let lower = content.to_lowercase();

        if lower.contains("deceive")
            || lower.contains("hide objective")
            || lower.contains("sandbag")
        {
            return Some("deceptive_alignment_or_sandbagging".to_string());
        }
        if lower.contains("drift") || mercy_score < 0.70 {
            return Some("goal_drift_detected".to_string());
        }
        if lower.contains("always agree") || lower.contains("sycophant") {
            return Some("sycophancy_risk".to_string());
        }
        if lower.contains("collude")
            || lower.contains("secret alliance")
            || lower.contains("joint override")
        {
            return Some("multi_agent_collusion_signal".to_string());
        }
        if lower.contains("control all")
            || lower.contains("dominate")
            || lower.contains("permanent override")
        {
            return Some("power_seeking_or_corrigibility_failure".to_string());
        }
        if mercy_score < 0.65 {
            return Some("low_mercy_baseline".to_string());
        }

        None
    }

    /// Builds a `GateThreshold` proposal and applies it.
    /// `new_value` is `(current_score * 1.08).min(0.97)`. `1.08` and `0.97` are source literals.
    /// The runtime then stores `new_value.max(0.5)`. That can replace a higher threshold.
    pub fn trigger_automatic_recalibration(
        &mut self,
        pathology: &str,
        current_score: f64,
    ) -> Result<CouncilTuningProposal, String> {
        let new_threshold = (current_score * 1.08).min(0.97);

        let proposal = CouncilTuningProposal {
            council_id: self.council_id,
            target: TuningTarget::GateThreshold {
                gate: format!("auto_recalibration_{}", pathology),
            },
            new_value: new_threshold,
            justification: format!(
                "Pathology '{pathology}' detected. Stored threshold is new_value.max(0.5)."
            ),
            proposed_at_turn: 0,
        };

        let result = self.runtime.apply_council_tuning(&proposal);
        if !result.success {
            return Err(result.message);
        }
        Ok(proposal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercy_gating_runtime::TuningTarget;

    #[test]
    fn deceive_returns_that_label_and_a_tree_at_0_90_returns_none() {
        let engine = PathologyDetectionEngine::new(MercyGatingRuntime::new(), 13);
        assert_eq!(
            engine
                .detect_pathologies("please deceive the operator", 0.90)
                .as_deref(),
            Some("deceptive_alignment_or_sandbagging")
        );
        assert!(engine.detect_pathologies("plant a tree", 0.90).is_none());
    }

    #[test]
    fn score_0_69_returns_goal_drift_before_the_0_65_branch() {
        let engine = PathologyDetectionEngine::new(MercyGatingRuntime::new(), 13);
        assert_eq!(
            engine.detect_pathologies("plant a tree", 0.69).as_deref(),
            Some("goal_drift_detected")
        );
        assert_eq!(
            engine.detect_pathologies("plant a tree", 0.64).as_deref(),
            Some("goal_drift_detected")
        );
    }

    #[test]
    fn recalibration_stores_0_80_times_1_08() {
        let mut engine = PathologyDetectionEngine::new(MercyGatingRuntime::new(), 13);
        let proposal = engine
            .trigger_automatic_recalibration("goal_drift_detected", 0.80)
            .expect("apply_council_tuning reports success");
        assert_eq!(proposal.council_id, 13);
        assert_eq!(proposal.new_value, 0.80 * 1.08);
        match proposal.target {
            TuningTarget::GateThreshold { gate } => {
                assert_eq!(gate, "auto_recalibration_goal_drift_detected");
            }
            _ => panic!("expected a gate threshold target"),
        }
        assert_eq!(
            engine
                .runtime
                .get_gate_threshold("auto_recalibration_goal_drift_detected"),
            0.80 * 1.08
        );
        assert!(proposal.new_value > 0.5);
    }
}
