//! mwpo.rs — Mercy-Weighted Preference Optimization (MWPO) v13.13.0
//!
//! Extends DPO/RLAIF concepts but weights and filters trajectories
//! EXCLUSIVELY by runtime mercy scores from MercyGatingRuntime.
//! Never allows mercy score to decrease. Monotonic strengthening enforced.

use mercy_gating_runtime::{MercyGatingRuntime, BeingRace};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MercyTrajectory {
    pub content: String,
    pub mercy_score: f64,
    pub race: BeingRace,
    pub delta: f64,
}

pub struct MercyWeightedPreferenceOptimization {
    runtime: Arc<MercyGatingRuntime>,
    min_mercy_threshold: f64,
}

impl MercyWeightedPreferenceOptimization {
    pub fn new(runtime: Arc<MercyGatingRuntime>) -> Self {
        Self {
            runtime,
            min_mercy_threshold: 0.78,
        }
    }

    pub fn filter_trajectory(&self, content: &str, race: BeingRace) -> Option<MercyTrajectory> {
        if let Ok(score) = self.runtime.evaluate_proposal(content, Some(race.clone())) {
            if score >= self.min_mercy_threshold {
                Some(MercyTrajectory {
                    content: content.to_string(),
                    mercy_score: score,
                    race,
                    delta: 0.0,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn weight_and_optimize(&self, proposal: &str, current_mercy_score: f64, race: BeingRace) -> Result<String, String> {
        let new_score = self.runtime.evaluate_proposal(proposal, Some(race.clone()))?;

        if new_score < current_mercy_score {
            return Err("MWPO rejected: new trajectory has lower mercy score.".to_string());
        }

        let optimized = format!(
            "[MWPO v13.13.0 | mercy_score={:.4} | race={:?}] {}",
            new_score, race, proposal
        );

        Ok(optimized)
    }

    pub fn compute_mercy_weighted_signal(&self, trajectory: &MercyTrajectory) -> f64 {
        trajectory.mercy_score * 1.2 + trajectory.delta.max(0.0) * 0.8
    }
}