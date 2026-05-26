//! Mercy-Weighted Quadratic Voting — Phase 14.1 Governance Primitive
//! Explicit dedicated module with full auditability and mercy alignment.

use crate::mercy::MercyScore; // Assuming mercy module exists or will be wired

/// Represents a single vote in the mercy-weighted quadratic system.
#[derive(Debug, Clone)]
pub struct MercyWeightedVote {
    pub voter_id: String,
    pub proposal_id: String,
    pub raw_power: f64,           // Base voting power
    pub mercy_alignment: f64,     // 0.0 - 1.0 from 7 Living Mercy Gates evaluation
    pub conviction_multiplier: f64, // From conviction staking
}

impl MercyWeightedVote {
    pub fn effective_power(&self) -> f64 {
        // Quadratic voting: cost ~ power^2, but weighted by mercy
        let mercy_weight = self.mercy_alignment.max(0.1); // Minimum mercy floor
        (self.raw_power * mercy_weight * self.conviction_multiplier).sqrt()
    }

    pub fn audit_log(&self) -> String {
        format!(
            "[AUDIT] Vote by {} on {} | Raw: {:.2} | Mercy: {:.2} | Conviction: {:.2} | Effective: {:.2}",
            self.voter_id, self.proposal_id, self.raw_power, self.mercy_alignment, self.conviction_multiplier, self.effective_power()
        )
    }
}

/// Tallies mercy-weighted quadratic votes with full audit trail.
pub fn tally_mercy_weighted_quadratic_votes(votes: &[MercyWeightedVote]) -> (f64, Vec<String>) {
    let mut total = 0.0;
    let mut audit_trail = Vec::new();

    for vote in votes {
        let power = vote.effective_power();
        total += power;
        audit_trail.push(vote.audit_log());
    }

    (total, audit_trail)
}

/// Checks if a proposal passes with mercy-weighted quadratic voting.
pub fn proposal_passes_mercy_quadratic(
    votes: &[MercyWeightedVote],
    threshold: f64,
) -> (bool, Vec<String>) {
    let (total_power, audit) = tally_mercy_weighted_quadratic_votes(votes);
    let passes = total_power >= threshold;
    (passes, audit)
}