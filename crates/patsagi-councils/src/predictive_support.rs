//! # Predictive Support for PATSAGi Councils (v14.15.9)
//!
//! Thin, pure-Rust surface that allows PATSAGi Councils, ValenceConsensusEngine,
//! and RaThorFeedbackLoop to consume hierarchical predictive coding / active inference
//! signals without creating a circular dependency on monorepo-intelligence.
//!
//! The full HierarchicalPredictiveCoding engine lives in monorepo-intelligence.
//! This module provides the reception + mercy-gated uplift path for council use.
//!
//! TOLC 8 + PATSAGi aligned | AG-SML v1.0 | Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Canonical floor — must stay in lockstep with monorepo-intelligence.
pub const MERCY_VALENCE_FLOOR: f64 = 0.999;

/// Errors that can occur in predictive support operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum PredictiveSupportError {
    #[error("input contains NaN or infinite value: {0}")]
    InvalidNumeric(String),

    #[error("no candidate policies provided for ranking")]
    EmptyCandidates,

    #[error("horizon must be ≥ 1, got {0}")]
    InvalidHorizon(u32),
}

/// Lightweight result that councils can accept from any predictive coding engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveSignal {
    pub final_valence: f64,
    pub residual_error: f64,
    pub depth_used: u32,
    pub free_energy_estimate: f64,
    pub source: String,
}

impl PredictiveSignal {
    pub fn new(
        final_valence: f64,
        residual_error: f64,
        depth_used: u32,
        free_energy_estimate: f64,
        source: impl Into<String>,
    ) -> Result<Self, PredictiveSupportError> {
        if !final_valence.is_finite() {
            return Err(PredictiveSupportError::InvalidNumeric(
                "final_valence".to_string(),
            ));
        }
        if !residual_error.is_finite() {
            return Err(PredictiveSupportError::InvalidNumeric(
                "residual_error".to_string(),
            ));
        }
        if !free_energy_estimate.is_finite() {
            return Err(PredictiveSupportError::InvalidNumeric(
                "free_energy_estimate".to_string(),
            ));
        }

        Ok(Self {
            final_valence: final_valence.max(MERCY_VALENCE_FLOOR),
            residual_error,
            depth_used,
            free_energy_estimate,
            source: source.into(),
        })
    }

    /// Whether this signal is strong enough to influence a council vote or feedback cycle.
    pub fn is_actionable(&self) -> bool {
        self.final_valence >= 0.9993 && self.residual_error < 0.45
    }

    /// Soft uplift factor that can be applied to a council's mercy_valence.
    pub fn mercy_uplift(&self) -> f64 {
        if !self.is_actionable() {
            return 0.0;
        }
        let base = (self.final_valence - MERCY_VALENCE_FLOOR) * 0.8;
        let error_penalty = self.residual_error * 0.15;
        (base - error_penalty).clamp(0.0, 0.04)
    }
}

/// Simple free-energy informed policy ranking helper for councils.
#[derive(Debug, Clone)]
pub struct CouncilPolicyHint {
    pub action_label: String,
    pub expected_free_energy: f64,
    pub predicted_valence: f64,
}

/// Rank a set of candidate council actions by expected free energy (lower is better)
/// while enforcing the mercy floor.
pub fn rank_policies_by_efe(
    candidates: &[(String, f64, f64)], // (label, predicted_error, current_valence)
    horizon: u32,
) -> Result<Vec<CouncilPolicyHint>, PredictiveSupportError> {
    if candidates.is_empty() {
        return Err(PredictiveSupportError::EmptyCandidates);
    }
    if horizon < 1 {
        return Err(PredictiveSupportError::InvalidHorizon(horizon));
    }

    let mut ranked = Vec::new();

    for (label, pred_err, valence) in candidates {
        if !pred_err.is_finite() || !valence.is_finite() {
            return Err(PredictiveSupportError::InvalidNumeric(format!(
                "candidate '{}'",
                label
            )));
        }

        let mut sim = (*valence).max(MERCY_VALENCE_FLOOR);
        let mut total_efe = 0.0;

        for _ in 0..horizon.min(6) {
            let step_efe = (1.0 - sim) * 0.08 * 1.618 + pred_err * 0.25;
            total_efe += step_efe;
            if sim < MERCY_VALENCE_FLOOR {
                total_efe = f64::INFINITY;
                break;
            }
            sim = (sim + 0.012).min(1.0);
        }

        ranked.push(CouncilPolicyHint {
            action_label: label.clone(),
            expected_free_energy: total_efe,
            predicted_valence: sim,
        });
    }

    ranked.sort_by(|a, b| {
        a.expected_free_energy
            .partial_cmp(&b.expected_free_energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_respects_floor() {
        let s = PredictiveSignal::new(0.97, 0.2, 3, 0.1, "test").unwrap();
        assert!(s.final_valence >= MERCY_VALENCE_FLOOR);
    }

    #[test]
    fn actionable_requires_strength() {
        let weak = PredictiveSignal::new(0.9991, 0.6, 2, 0.5, "weak").unwrap();
        assert!(!weak.is_actionable());

        let strong = PredictiveSignal::new(0.9996, 0.2, 4, 0.15, "strong").unwrap();
        assert!(strong.is_actionable());
        assert!(strong.mercy_uplift() > 0.0);
    }

    #[test]
    fn rejects_nan_signal() {
        let err = PredictiveSignal::new(f64::NAN, 0.1, 2, 0.05, "bad");
        assert!(matches!(err, Err(PredictiveSupportError::InvalidNumeric(_))));
    }

    #[test]
    fn rejects_empty_candidates() {
        let err = rank_policies_by_efe(&[], 3);
        assert!(matches!(err, Err(PredictiveSupportError::EmptyCandidates)));
    }
}
