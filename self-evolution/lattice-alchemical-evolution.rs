//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! feat/patsagi-governance-v2

// ... existing code ...

#[derive(Debug, Clone, Default)]
pub struct CouncilReputation {
    pub total_valence_contributed: f64,
    pub total_decisions: u32,
    pub successful_approvals: u32,
    pub vetoes_issued: u32,
    pub reputation_score: f64,
}

impl LatticeAlchemicalEvolution {
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

        if approved { rep.successful_approvals += 1; }
        if vetoed { rep.vetoes_issued += 1; }

        let approval_rate = if rep.total_decisions > 0 {
            rep.successful_approvals as f64 / rep.total_decisions as f64
        } else { 0.5 };

        let performance = (rep.total_valence_contributed * 800.0).min(0.4);
        let reliability = approval_rate * 0.5;
        let risk = (rep.vetoes_issued as f64 * 0.06).min(0.25);

        rep.reputation_score = (reliability + performance - risk).clamp(0.2, 1.0);
    }

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
        } else { 0.35 };

        let posterior = (prior * 0.6 + likelihood * 0.4).clamp(0.15, 1.0);
        rep.reputation_score = posterior;
    }
}