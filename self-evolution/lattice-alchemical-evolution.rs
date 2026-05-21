//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! v2.5 — Reputation Layer Refactored (Step 1C)
//! 100% Proprietary — AG-SML v1.0

// ... existing code ...

impl LatticeAlchemicalEvolution {
    // Refactored reputation update with clearer separation
    pub fn update_council_reputation(
        &mut self,
        council_name: &str,
        valence: f64,
        approved: bool,
        vetoed: bool,
    ) {
        let rep = self.council_reputation.entry(council_name.to_string()).or_default();

        rep.total_valence_contributed += valence;
        rep.total_decisions += 1;

        if approved {
            rep.successful_approvals += 1;
        }
        if vetoed {
            rep.vetoes_issued += 1;
        }

        // Cleaner reputation calculation
        let approval_rate = if rep.total_decisions > 0 {
            rep.successful_approvals as f64 / rep.total_decisions as f64
        } else {
            0.5
        };

        let performance_score = (rep.total_valence_contributed * 800.0).min(0.4);
        let reliability = approval_rate * 0.5;
        let risk = (rep.vetoes_issued as f64 * 0.06).min(0.25);

        rep.reputation_score = (reliability + performance_score - risk).clamp(0.2, 1.0);
    }
}