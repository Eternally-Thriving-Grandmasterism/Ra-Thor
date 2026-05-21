//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! Sovereign Form Transmutation Engine v2.2
//! Phase 4: Reputation & Historical Performance Layer
//! Builds toward Bayesian-style reputation evolution
//! 100% Proprietary — AG-SML v1.0

use crate::mercy::tolc8_enforcer::{TOLC8Enforcer, TOLC8EvaluationResult};
use std::collections::HashMap;

// ... (previous structs)

#[derive(Debug, Clone, Default)]
pub struct CouncilReputation {
    pub total_valence_contributed: f64,
    pub total_decisions: u32,
    pub successful_approvals: u32,
    pub vetoes_issued: u32,
    pub reputation_score: f64,        // 0.0 – 1.0
}

impl LatticeAlchemicalEvolution {

    // === Phase 4: Reputation & Historical Performance ===
    /// Updates reputation based on recent performance.
    /// This is the foundation for future Bayesian reputation evolution.
    pub fn update_council_reputation(&mut self, council_name: &str, valence: f64, approved: bool, vetoed: bool) {
        let rep = self.council_reputation.entry(council_name.to_string()).or_default();

        rep.total_valence_contributed += valence;
        rep.total_decisions += 1;

        if approved {
            rep.successful_approvals += 1;
        }
        if vetoed {
            rep.vetoes_issued += 1;
        }

        // Simple reputation formula (can evolve into full Bayesian later)
        let approval_rate = if rep.total_decisions > 0 {
            rep.successful_approvals as f64 / rep.total_decisions as f64
        } else { 0.5 };

        let valence_factor = (rep.total_valence_contributed * 800.0).min(0.4);
        let veto_penalty = (rep.vetoes_issued as f64 * 0.08).min(0.3);

        rep.reputation_score = (approval_rate * 0.6 + valence_factor - veto_penalty).clamp(0.1, 1.0);
    }

    /// Returns current reputation score for a council (used for dynamic weighting)
    pub fn get_council_reputation(&self, council_name: &str) -> f64 {
        self.council_reputation
            .get(council_name)
            .map(|r| r.reputation_score)
            .unwrap_or(0.5)
    }

    pub fn run_council_synthesis(&mut self, scope: &str) -> CouncilSynthesisResult {
        // ... existing logic ...

        // After processing each vote, update reputation
        for vote in &votes {
            self.update_council_reputation(
                &vote.council,
                vote.valence_contribution,
                vote.approved,
                vote.vetoed,
            );
        }

        // Reputation can now influence future effective weights (Phase 4 foundation)
        // Example: effective_weight *= (0.7 + get_council_reputation(...) * 0.6)

        // ... rest of synthesis + TOLC 8 enforcement ...
    }
}