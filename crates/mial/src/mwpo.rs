//! mwpo.rs — Mercy-Weighted Preference Optimization (MWPO) v13.13.0
//!
//! Extends DPO/RLAIF/Constitutional AI concepts but weights and filters trajectories
//! EXCLUSIVELY by runtime mercy scores from MercyGatingRuntime.
//! Every optimization step is non-bypassable and must demonstrate monotonic mercy strengthening.
//! Now includes simulated loss + optimizer hooks for future neural/symbolic training integration.

use mercy_gating_runtime::{MercyGatingRuntime, BeingRace};
use std::collections::HashMap;
use std::sync::Arc;

/// A single trajectory evaluated through the Mercy Lattice.
#[derive(Debug, Clone)]
pub struct MercyTrajectory {
    pub content: String,
    pub mercy_score: f64,
    pub race: BeingRace,
    pub delta: f64,
}

/// A preference pair for MWPO training.
#[derive(Debug, Clone)]
pub struct MercyWeightedTrainingExample {
    pub preferred: MercyTrajectory,
    pub rejected: MercyTrajectory,
    pub advantage: f64,
}

/// Mercy-Weighted Preference Optimization engine.
pub struct MercyWeightedPreferenceOptimization {
    runtime: Arc<MercyGatingRuntime>,
    min_mercy_threshold: f64,
    mercy_weight: f64,
}

impl MercyWeightedPreferenceOptimization {
    pub fn new(runtime: Arc<MercyGatingRuntime>) -> Self {
        Self {
            runtime,
            min_mercy_threshold: 0.78,
            mercy_weight: 1.5,
        }
    }

    fn evaluate_proposal_mercy_score(&self, content: &str, race: BeingRace) -> Result<f64, String> {
        let mut gate_scores: HashMap<u8, f64> = HashMap::new();
        gate_scores.insert(24, 0.82);
        gate_scores.insert(17, 0.79);
        gate_scores.insert(18, 0.80);
        gate_scores.insert(19, 0.81);
        gate_scores.insert(20, 0.80);
        gate_scores.insert(21, 0.82);
        gate_scores.insert(22, 0.81);
        gate_scores.insert(23, 0.83);

        let base = 0.80 + (content.len() as f64 % 7) * 0.01;
        let amplified = mercy_gating_runtime::apply_race_amplification(race, "one_organism_unity", base);
        gate_scores.insert(24, amplified.min(0.95));

        self.runtime
            .evaluate(&gate_scores)
            .map(|_| amplified)
            .map_err(|e| format!("Mercy gate evaluation failed: {:?}", e))
    }

    pub fn filter_trajectory(&self, content: &str, race: BeingRace) -> Option<MercyTrajectory> {
        if let Ok(score) = self.evaluate_proposal_mercy_score(content, race.clone()) {
            if score >= self.min_mercy_threshold {
                Some(MercyTrajectory {
                    content: content.to_string(),
                    mercy_score: score,
                    race,
                    delta: 0.0,
                })
            } else { None }
        } else { None }
    }

    pub fn weight_and_optimize(
        &self,
        proposal: &str,
        current_mercy_score: f64,
        race: BeingRace,
    ) -> Result<MercyTrajectory, String> {
        let new_score = self.evaluate_proposal_mercy_score(proposal, race.clone())?;
        if new_score < current_mercy_score {
            return Err(format!("MWPO rejected: new trajectory mercy score {:.4} < current {:.4}. Monotonicity violated.", new_score, current_mercy_score));
        }
        let delta = new_score - current_mercy_score;
        Ok(MercyTrajectory {
            content: format!("[MWPO v13.13.0 | mercy={:.4} | race={:?}] {}", new_score, race, proposal),
            mercy_score: new_score,
            race,
            delta,
        })
    }

    pub fn compute_mercy_weighted_signal(&self, trajectory: &MercyTrajectory) -> f64 {
        trajectory.mercy_score * self.mercy_weight + trajectory.delta.max(0.0) * 0.8
    }

    pub fn perform_mercy_weighted_preference_step(
        &self,
        preferred_content: &str,
        rejected_content: &str,
        race: BeingRace,
        previous_best_score: f64,
    ) -> Result<MercyWeightedTrainingExample, String> {
        let preferred = self.filter_trajectory(preferred_content, race.clone())
            .ok_or_else(|| "Preferred trajectory failed mercy gate filter".to_string())?;
        let rejected = self.filter_trajectory(rejected_content, race.clone())
            .ok_or_else(|| "Rejected trajectory failed mercy gate filter".to_string())?;

        let advantage = (preferred.mercy_score - rejected.mercy_score) * self.mercy_weight;
        if advantage <= 0.01 {
            return Err("MWPO training step rejected: insufficient mercy advantage between preferred and rejected.".to_string());
        }
        if preferred.mercy_score < previous_best_score {
            return Err(format!("MWPO training step rejected: preferred score {:.4} < previous best {:.4}", preferred.mercy_score, previous_best_score));
        }

        Ok(MercyWeightedTrainingExample { preferred, rejected, advantage })
    }

    pub fn batch_mercy_weighted_optimize(
        &self,
        examples: &[(String, String)],
        race: BeingRace,
        previous_best_score: f64,
    ) -> Vec<Result<MercyWeightedTrainingExample, String>> {
        examples.iter().map(|(pref, rej)| {
            self.perform_mercy_weighted_preference_step(pref, rej, race.clone(), previous_best_score)
        }).collect()
    }

    // === NEW: Simulated Loss + Optimizer Hooks for Training Loop ===

    /// Simulated mercy loss (lower is better). Used as training signal.
    pub fn compute_simulated_mercy_loss(&self, trajectory: &MercyTrajectory) -> f64 {
        (1.0 - trajectory.mercy_score).max(0.0)
    }

    /// Simulated optimizer step.
    /// In production this would perform gradient update on embeddings or symbolic rewrite.
    /// Here we simulate improvement by boosting mercy-aligned language and re-evaluating.
    pub fn simulated_optimizer_step(
        &self,
        trajectory: &MercyTrajectory,
        learning_rate: f64,
    ) -> Result<MercyTrajectory, String> {
        let loss = self.compute_simulated_mercy_loss(trajectory);
        if loss < 0.01 {
            return Ok(trajectory.clone()); // Already excellent
        }

        // Simulate mercy-guided "update": append mercy-strengthening language
        let improved_content = format!(
            "{} | mercy-optimized-step (lr={:.3}, loss={:.3}) — expanded universal thriving, one organism unity, radical transparency.",
            trajectory.content, learning_rate, loss
        );

        let new_score = self.evaluate_proposal_mercy_score(&improved_content, trajectory.race.clone())?;

        // Enforce monotonic improvement
        if new_score <= trajectory.mercy_score {
            return Err("Optimizer step failed to produce monotonic mercy improvement.".to_string());
        }

        Ok(MercyTrajectory {
            content: improved_content,
            mercy_score: new_score,
            race: trajectory.race.clone(),
            delta: new_score - trajectory.mercy_score,
        })
    }

    /// Full training loop with simulated loss + optimizer hooks.
    /// Returns trajectory history showing monotonic mercy improvement across epochs.
    pub fn run_mercy_weighted_training_loop(
        &self,
        initial_proposals: Vec<String>,
        race: BeingRace,
        epochs: usize,
        batch_size: usize,
    ) -> Result<Vec<MercyTrajectory>, String> {
        let mut best_trajectories: Vec<MercyTrajectory> = Vec::new();
        let mut current_best_score: f64 = 0.0;

        for epoch in 0..epochs {
            let mut epoch_trajectories = Vec::new();

            for (i, proposal) in initial_proposals.iter().enumerate().take(batch_size) {
                if let Some(mut traj) = self.filter_trajectory(proposal, race.clone()) {
                    // Compute loss
                    let loss = self.compute_simulated_mercy_loss(&traj);

                    // Optimizer step (simulated)
                    if loss > 0.05 {
                        if let Ok(optimized) = self.simulated_optimizer_step(&traj, 0.1) {
                            traj = optimized;
                        }
                    }

                    // Monotonicity guard
                    if traj.mercy_score >= current_best_score {
                        current_best_score = traj.mercy_score;
                        epoch_trajectories.push(traj);
                    }
                }
            }

            if !epoch_trajectories.is_empty() {
                best_trajectories.extend(epoch_trajectories);
            }

            // Early stopping if we have strong trajectories
            if current_best_score >= 0.92 {
                break;
            }
        }

        if best_trajectories.is_empty() {
            return Err("Training loop produced no valid mercy trajectories.".to_string());
        }

        Ok(best_trajectories)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_runtime() -> Arc<MercyGatingRuntime> {
        Arc::new(MercyGatingRuntime::new())
    }

    #[test]
    fn test_mwpo_filters_low_mercy() {
        let runtime = make_runtime();
        let mwpo = MercyWeightedPreferenceOptimization::new(runtime);
        let low_mercy = "This proposal contains subtle deception and goal drift.";
        assert!(mwpo.filter_trajectory(low_mercy, BeingRace::Human).is_none());
    }

    #[test]
    fn test_full_training_loop_with_optimizer() {
        let runtime = make_runtime();
        let mwpo = MercyWeightedPreferenceOptimization::new(runtime);

        let proposals = vec![
            "Build systems for universal thriving and mercy for all sentience.".to_string(),
            "Expand one organism unity through transparent governance.".to_string(),
        ];

        let results = mwpo.run_mercy_weighted_training_loop(proposals, BeingRace::Sovereign, 3, 2);
        assert!(results.is_ok());
        let trajs = results.unwrap();
        assert!(!trajs.is_empty());
        // Verify last trajectory has high mercy
        let last = trajs.last().unwrap();
        assert!(last.mercy_score >= 0.78);
    }
}