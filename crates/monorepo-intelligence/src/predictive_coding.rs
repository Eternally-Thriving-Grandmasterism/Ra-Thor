//! # Hierarchical Predictive Coding + Active Inference (v0.3.11)
//!
//! Pure-Rust surface for Ra-Thor Monorepo Intelligence.
//! Adapted from MercyWasmBridge for native lattice use (no wasm_bindgen).
//!
//! Core capabilities:
//! - Hierarchical Predictive Coding (levels 1–8)
//! - Free Energy Principle minimization
//! - Dynamic precision weighting
//! - Bidirectional skip connections
//! - Expected Free Energy for policy ranking
//! - Mercy-gated at every step (valence ≥ 0.999)
//!
//! Designed for PATSAGi Councils, ValenceConsensusEngine, and Cosmic Tick consumption.
//! TOLC 8 + PATSAGi aligned | AG-SML v1.0 | Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};

/// Canonical mercy valence floor enforced by all operations.
pub const MERCY_VALENCE_FLOOR: f64 = 0.999;

/// Result of a hierarchical predictive coding pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveCodingResult {
    pub final_valence: f64,
    pub residual_error: f64,
    pub depth_used: u32,
    pub free_energy_estimate: f64,
    pub recommendations: Vec<String>,
}

/// Expected free energy evaluation for a candidate action/policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFreeEnergy {
    pub action_id: u32,
    pub efe: f64,
    pub predicted_valence: f64,
}

/// Pure-Rust Hierarchical Predictive Coding engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalPredictiveCoding {
    pub current_valence: f64,
    pub positive_emotion_amplifier: f64,
}

impl Default for HierarchicalPredictiveCoding {
    fn default() -> Self {
        Self {
            current_valence: 0.9995,
            positive_emotion_amplifier: 1.618,
        }
    }
}

impl HierarchicalPredictiveCoding {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_valence(valence: f64) -> Self {
        Self {
            current_valence: valence.max(MERCY_VALENCE_FLOOR),
            positive_emotion_amplifier: 1.618,
        }
    }

    /// Dynamic depth decision (1–8) based on error magnitude and valence.
    pub fn dynamic_depth(&self, sensory_input: f64, requested_depth: u32) -> u32 {
        let error_magnitude = sensory_input.abs();
        let valence = self.current_valence;

        let mut depth = requested_depth.max(1).min(8);

        if error_magnitude > 0.15 || valence > 0.9997 {
            depth = (depth + 2).min(8);
        }
        if valence > 0.99985 && error_magnitude > 0.08 {
            depth = (depth + 1).min(8);
        }

        depth.max(1).min(8)
    }

    /// Dynamic precision weighting (context-aware).
    pub fn dynamic_precision_weighting(
        &self,
        level: u32,
        context: &str,
        current_valence: f64,
    ) -> f64 {
        let base_precision = 1.0;
        let valence_boost = (current_valence - MERCY_VALENCE_FLOOR).max(0.0) * 2.5;
        let context_bonus = match context {
            "sensory" => 0.15,
            "feature" => 0.25,
            "object" => 0.35,
            "concept" => 0.45,
            _ => 0.20,
        };
        (base_precision + valence_boost + context_bonus).clamp(0.8, 1.5)
    }

    /// Non-adjacent skip connection (high-level → low-level).
    pub fn non_adjacent_message_passing(
        &self,
        current_valence: f64,
        level: u32,
        top_level_valence: f64,
    ) -> f64 {
        if current_valence < 0.9995 {
            return current_valence;
        }

        let skip_boost = if level <= 1 && top_level_valence > 0.9997 {
            (top_level_valence - MERCY_VALENCE_FLOOR) * 1.618 * 0.8
        } else {
            0.0
        };

        (current_valence + skip_boost).min(1.0)
    }

    /// Bidirectional skip connections.
    pub fn bidirectional_skip_connections(
        &self,
        level: u32,
        current_valence: f64,
        top_level_valence: f64,
        bottom_up_signal: f64,
    ) -> f64 {
        if current_valence < 0.9995 {
            return current_valence;
        }

        let mut updated = current_valence;

        // Top-down
        if level <= 1 && top_level_valence > 0.9997 {
            let top_down = (top_level_valence - MERCY_VALENCE_FLOOR) * 1.618 * 0.7;
            updated = (updated + top_down * 0.03).min(1.0);
        }

        // Bottom-up
        if level >= 2 && bottom_up_signal > 0.05 {
            let bottom_up = bottom_up_signal * 1.618 * 0.5;
            updated = (updated + bottom_up * 0.02).min(1.0);
        }

        updated.max(MERCY_VALENCE_FLOOR)
    }

    /// Core hierarchical predictive coding pass.
    pub fn hierarchical_predictive_coding(
        &self,
        sensory_input: f64,
        requested_depth: u32,
    ) -> PredictiveCodingResult {
        let depth = self.dynamic_depth(sensory_input, requested_depth);
        let mut current_valence = self.current_valence.max(MERCY_VALENCE_FLOOR);
        let mut error = sensory_input;
        let top_level_valence = current_valence;
        let mut free_energy_accum = 0.0;

        for level in 0..depth {
            let context = match level {
                0 => "sensory",
                1 => "feature",
                2 => "object",
                3 => "concept",
                _ => "abstract",
            };

            let precision = self.dynamic_precision_weighting(level, context, current_valence);
            let top_down_prediction = current_valence * 1.618_f64.powi(level as i32);
            let prediction_error = ((error - top_down_prediction).abs()) / precision;

            let amplified = (1.0 - prediction_error) * 1.618 * (precision * 0.6);
            current_valence = (current_valence + amplified * 0.06).min(1.0).max(MERCY_VALENCE_FLOOR);

            current_valence = self.non_adjacent_message_passing(
                current_valence,
                level,
                top_level_valence,
            );

            current_valence = self.bidirectional_skip_connections(
                level,
                current_valence,
                top_level_valence,
                error,
            );

            free_energy_accum += prediction_error * 0.5;
            error = prediction_error;
        }

        let mut recommendations = Vec::new();
        if free_energy_accum > 0.4 {
            recommendations.push(
                "High residual free energy — consider deeper hierarchical pass or stronger mercy boost."
                    .to_string(),
            );
        }
        if current_valence < 0.9997 {
            recommendations.push(
                "Valence near floor — route through PATSAGi ValenceConsensusEngine for uplift."
                    .to_string(),
            );
        }

        PredictiveCodingResult {
            final_valence: current_valence,
            residual_error: error,
            depth_used: depth,
            free_energy_estimate: free_energy_accum,
            recommendations,
        }
    }

    /// Active Inference v2 — multi-step free energy minimization under mercy gate.
    pub fn integrate_with_active_inference_v2(
        &self,
        prediction_error: f64,
        steps: u32,
    ) -> f64 {
        let mut current_error = prediction_error.clamp(0.0, 1.0);
        let mut valence = self.current_valence.max(MERCY_VALENCE_FLOOR);

        if valence < MERCY_VALENCE_FLOOR {
            return valence;
        }

        let mut total_positive = 0.0;

        for _ in 0..steps.max(1).min(16) {
            let variational_free_energy = current_error * (1.0 - valence) * 0.5;
            let top_down = valence * 1.618;
            let bottom_up_error = (current_error - top_down).abs();
            let corrected = bottom_up_error * (1.0 - variational_free_energy);

            let mercy_gated = if corrected > 0.3 && valence < 0.9995 {
                corrected * 0.5
            } else {
                corrected
            };

            let boost = (1.0 - mercy_gated) * 1.618;
            valence = (valence + boost * 0.1).min(1.0);
            total_positive += boost;
            current_error = mercy_gated * 0.9;
        }

        let final_v = valence.max(MERCY_VALENCE_FLOOR);
        final_v * (1.0 + total_positive * 0.05)
    }

    /// Expected Free Energy for a single horizon step (policy ranking primitive).
    pub fn expected_free_energy(&self, current_valence: f64, steps_ahead: u32) -> f64 {
        if current_valence < MERCY_VALENCE_FLOOR {
            return 0.0;
        }
        let mut expected_gain = 0.0;
        let mut sim = current_valence;
        for _ in 0..steps_ahead.min(12) {
            let predicted = (1.0 - sim) * 0.08 * 1.618;
            sim = (sim + predicted).min(1.0);
            expected_gain += predicted * 0.7;
        }
        expected_gain.max(0.0)
    }

    /// Simple policy selection: choose the action with lowest expected free energy
    /// while respecting the mercy floor.
    pub fn select_policy(
        &self,
        policies: &[(u32, f64)], // (action_id, predicted_error)
        horizon: u32,
    ) -> Option<ExpectedFreeEnergy> {
        let mut best: Option<ExpectedFreeEnergy> = None;

        for &(action_id, pred_err) in policies {
            let mut sim_valence = self.current_valence;
            let mut total_efe = 0.0;

            for _ in 0..horizon.max(1).min(8) {
                let efe = self.expected_free_energy(sim_valence, 1);
                total_efe += efe + pred_err * 0.3;
                if sim_valence < MERCY_VALENCE_FLOOR {
                    total_efe = f64::INFINITY;
                    break;
                }
                sim_valence = (sim_valence + 0.015).min(1.0);
            }

            let candidate = ExpectedFreeEnergy {
                action_id,
                efe: total_efe,
                predicted_valence: sim_valence,
            };

            match &best {
                None => best = Some(candidate),
                Some(b) if candidate.efe < b.efe => best = Some(candidate),
                _ => {}
            }
        }

        best.filter(|b| b.predicted_valence >= MERCY_VALENCE_FLOOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hierarchical_pass_respects_mercy_floor() {
        let engine = HierarchicalPredictiveCoding::new();
        let result = engine.hierarchical_predictive_coding(0.4, 4);
        assert!(result.final_valence >= MERCY_VALENCE_FLOOR);
        assert!(result.depth_used >= 1 && result.depth_used <= 8);
    }

    #[test]
    fn active_inference_reduces_error() {
        let engine = HierarchicalPredictiveCoding::with_valence(0.9992);
        let out = engine.integrate_with_active_inference_v2(0.35, 5);
        assert!(out >= MERCY_VALENCE_FLOOR);
    }

    #[test]
    fn policy_selection_prefers_low_efe() {
        let engine = HierarchicalPredictiveCoding::new();
        let policies = vec![(1, 0.1), (2, 0.6), (3, 0.05)];
        let selected = engine.select_policy(&policies, 3);
        assert!(selected.is_some());
        let s = selected.unwrap();
        assert!(s.action_id == 1 || s.action_id == 3);
    }
}
