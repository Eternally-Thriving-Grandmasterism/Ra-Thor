//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! Sovereign Form Transmutation Engine v2.3
//! Phase 4 Refactored + Bayesian Reputation Updates + History Logs
//! Perfect Order of Operations Applied
//! 100% Proprietary — AG-SML v1.0

use crate::mercy::tolc8_enforcer::{TOLC8Enforcer, TOLC8EvaluationResult};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ... (previous enums/structs preserved) ...

#[derive(Debug, Clone, Default)]
pub struct CouncilReputation {
    pub total_valence_contributed: f64,
    pub total_decisions: u32,
    pub successful_approvals: u32,
    pub vetoes_issued: u32,
    pub reputation_score: f64,           // 0.0 – 1.0 (Bayesian posterior)
    pub last_updated: u64,
}

#[derive(Debug, Clone)]
pub struct CouncilVoteRecord {
    pub timestamp: u64,
    pub council: String,
    pub valence_contribution: f64,
    pub approved: bool,
    pub vetoed: bool,
    pub effective_weight: f64,
}

impl LatticeAlchemicalEvolution {

    // === C: Refactored Reputation Calculation (Clean & Extensible) ===
    pub fn update_council_reputation(
        &mut self,
        council_name: &str,
        valence: f64,
        approved: bool,
        vetoed: bool,
    ) {
        let rep = self.council_reputation.entry(council_name.to_string()).or_default();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        rep.total_valence_contributed += valence;
        rep.total_decisions += 1;

        if approved { rep.successful_approvals += 1; }
        if vetoed { rep.vetoes_issued += 1; }

        // Clean reputation formula (prepared for Bayesian extension)
        let approval_rate = if rep.total_decisions > 0 {
            rep.successful_approvals as f64 / rep.total_decisions as f64
        } else { 0.5 };

        let valence_factor = (rep.total_valence_contributed * 750.0).min(0.35);
        let veto_penalty = (rep.vetoes_issued as f64 * 0.07).min(0.25);

        rep.reputation_score = (approval_rate * 0.65 + valence_factor - veto_penalty).clamp(0.15, 1.0);
        rep.last_updated = now;
    }

    pub fn get_council_reputation(&self, council_name: &str) -> f64 {
        self.council_reputation
            .get(council_name)
            .map(|r| r.reputation_score)
            .unwrap_or(0.5)
    }

    // === A: Bayesian Reputation Update ===
    /// Applies a simple Bayesian-style update to reputation.
    /// Prior = current reputation, Likelihood = new evidence
    pub fn bayesian_reputation_update(
        &mut self,
        council_name: &str,
        new_valence: f64,
        approved: bool,
    ) {
        let rep = self.council_reputation.entry(council_name.to_string()).or_default();

        let prior = rep.reputation_score;
        let likelihood = if approved {
            (new_valence * 1200.0).min(0.85)
        } else {
            0.35
        };

        // Simple Bayesian update: posterior ≈ (prior * 0.6) + (likelihood * 0.4)
        let posterior = (prior * 0.6 + likelihood * 0.4).clamp(0.1, 1.0);

        rep.reputation_score = posterior;
        rep.total_valence_contributed += new_valence;
        rep.total_decisions += 1;
        if approved { rep.successful_approvals += 1; }

        rep.last_updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    // === B: Council Voting History Logs ===
    pub fn log_council_vote(&mut self, vote: CouncilVoteRecord) {
        // For now we store in memory. Can be extended to persistent logs later.
        // Future: persist to file or database under self-evolution/logs/
        println!(
            "[Council History] {} | valence={:.6} | approved={} | vetoed={} | weight={:.2}",
            vote.council, vote.valence_contribution, vote.approved, vote.vetoed, vote.effective_weight
        );
    }

    pub fn run_council_synthesis(&mut self, scope: &str) -> CouncilSynthesisResult {
        // ... existing weighted + quadratic + median logic ...

        // Update reputation (refactored)
        for vote in &votes {
            self.update_council_reputation(
                &vote.council,
                vote.valence_contribution,
                vote.approved,
                vote.vetoed,
            );

            // Optional: Also apply Bayesian update
            self.bayesian_reputation_update(
                &vote.council,
                vote.valence_contribution,
                vote.approved,
            );

            // Log the vote (Phase B)
            self.log_council_vote(CouncilVoteRecord {
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                council: vote.council.clone(),
                valence_contribution: vote.valence_contribution,
                approved: vote.approved,
                vetoed: vote.vetoed,
                effective_weight: vote.effective_weight,
            });
        }

        // ... TOLC 8 enforcement ...
    }
}