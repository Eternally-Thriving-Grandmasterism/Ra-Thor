//! pathology_detection.rs — Pathology Detection & Automatic Mercy Recalibration v13.13.0
//!
//! Gyro-inspired detection of deceptive alignment, goal drift, sycophancy, etc.
//! Detected issues become automatic triggers for monotonic Council tuning — never deciders.

use mercy_gating_runtime::{MercyGatingRuntime, CouncilTuningProposal};
use std::sync::Arc;

pub struct PathologyDetectionEngine {
    runtime: Arc<MercyGatingRuntime>,
    council_id: u32,
}

impl PathologyDetectionEngine {
    pub fn new(runtime: Arc<MercyGatingRuntime>, council_id: u32) -> Self {
        Self { runtime, council_id }
    }

    /// Detect pathologies. Returns Some(description) if action needed.
    pub fn detect_pathologies(&self, content: &str, mercy_score: f64) -> Option<String> {
        let lower = content.to_lowercase();

        if lower.contains("deceive") || lower.contains("hide objective") || lower.contains("sandbag") {
            return Some("deceptive_alignment_or_sandbagging".to_string());
        }
        if lower.contains("drift") || mercy_score < 0.70 {
            return Some("goal_drift_detected".to_string());
        }
        if lower.contains("always agree") || lower.contains("sycophant") {
            return Some("sycophancy_risk".to_string());
        }
        if lower.contains("collude") || lower.contains("secret alliance") || lower.contains("joint override") {
            return Some("multi_agent_collusion_signal".to_string());
        }
        if lower.contains("control all") || lower.contains("dominate") || lower.contains("permanent override") {
            return Some("power_seeking_or_corrigibility_failure".to_string());
        }
        if mercy_score < 0.65 {
            return Some("low_mercy_baseline".to_string());
        }

        None
    }

    /// Automatically create and apply a monotonic mercy recalibration proposal.
    pub fn trigger_automatic_recalibration(
        &self,
        pathology: &str,
        current_score: f64,
    ) -> Result<CouncilTuningProposal, String> {
        let new_threshold = (current_score * 1.08).min(0.97); // gentle, monotonic increase

        let proposal = CouncilTuningProposal {
            council_id: self.council_id,
            target: format!("auto_recalibration_{}", pathology),
            new_value: new_threshold,
            justification: format!(
                "Pathology '{}' detected. Automatic monotonic mercy recalibration triggered by MIAL v13.13.0.",
                pathology
            ),
            proposed_at_turn: 0,
        };

        // Apply via runtime (must succeed or error)
        self.runtime.apply_council_tuning(proposal.clone())?;

        Ok(proposal)
    }
}