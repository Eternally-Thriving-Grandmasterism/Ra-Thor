//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! v2.6 — Audit History Logs Implemented
//! Proper council voting history for audits and transparency
//! 100% Proprietary — AG-SML v1.0

use crate::patsagi_deliberation::CouncilMessage;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct CouncilVoteRecord {
    pub timestamp: u64,
    pub council: String,
    pub valence_contribution: f64,
    pub approved: bool,
    pub vetoed: bool,
    pub effective_weight: f64,
    pub reputation_at_time: f64,
}

impl LatticeAlchemicalEvolution {
    /// Log a council vote with full audit context
    pub fn log_council_vote(&mut self, record: CouncilVoteRecord) {
        self.vote_history.push(record);
    }

    /// Get full voting history (for audits)
    pub fn get_vote_history(&self) -> &[CouncilVoteRecord] {
        &self.vote_history
    }

    /// Get audit summary
    pub fn get_audit_summary(&self) -> String {
        let total = self.vote_history.len();
        let vetoes = self.vote_history.iter().filter(|r| r.vetoed).count();
        let avg_weight = if total > 0 {
            self.vote_history.iter().map(|r| r.effective_weight).sum::<f64>() / total as f64
        } else { 0.0 };

        format!(
            "Total Votes Logged: {} | Vetoes: {} | Avg Effective Weight: {:.2}",
            total, vetoes, avg_weight
        )
    }

    pub fn run_council_synthesis(&mut self, scope: &str) -> CouncilSynthesisResult {
        // ... existing logic ...

        // Log every vote with audit context
        for vote in &votes {
            let reputation = self.get_council_reputation(&vote.council);

            self.log_council_vote(CouncilVoteRecord {
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                council: vote.council.clone(),
                valence_contribution: vote.valence_contribution,
                approved: vote.approved,
                vetoed: vote.vetoed,
                effective_weight: vote.effective_weight,
                reputation_at_time: reputation,
            });
        }

        // ... rest of synthesis ...
    }
}