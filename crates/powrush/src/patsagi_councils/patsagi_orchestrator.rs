//! # PATSAGi Orchestrator — Advanced Consensus Engine
//!
//! Production-grade consensus with:
//! - Coherence-weighted voting
//! - Temporal decay
//! - Byzantine-resistant threshold (66%)
//! - Lattice modulation ready

use crate::patsagi_councils::{
    abundance_council, harmony_council, joy_council, radical_love_council, truth_council,
    CouncilDecision, PATSAGiConsensus,
};
use crate::mercy::MercyGateStatus;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct PATSAGiOrchestrator {
    decision_history: Vec<(u64, PATSAGiConsensus)>,
}

impl PATSAGiOrchestrator {
    pub fn new() -> Self {
        Self {
            decision_history: Vec::new(),
        }
    }

    pub async fn run_full_consensus(
        &mut self,
        action_description: &str,
        context: &str,
        mercy_status: &MercyGateStatus,
    ) -> PATSAGiConsensus {
        let mut decisions = Vec::new();

        // 1. Radical Love Veto (absolute)
        let love = radical_love_council::evaluate(action_description, context);
        decisions.push(love.clone());

        if !love.approved {
            let consensus = PATSAGiConsensus {
                decisions,
                overall_approved: false,
                final_weight: 0.0,
            };
            self.record_consensus(consensus.clone());
            return consensus;
        }

        // 2. Run other councils
        decisions.push(abundance_council::evaluate(action_description, context));
        decisions.push(truth_council::evaluate(action_description, context));
        decisions.push(harmony_council::evaluate(action_description, context));
        decisions.push(joy_council::evaluate(action_description, context));

        // 3. Coherence-weighted voting + temporal decay
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for decision in &decisions {
            let age = if let Some((ts, _)) = self.decision_history.last() {
                now.saturating_sub(*ts) as f32
            } else {
                0.0
            };
            let decay = (-age as f32 / 3600.0).exp(); // 1-hour half-life
            let effective_weight = decision.weight * decay;

            weighted_sum += if decision.approved { effective_weight } else { 0.0 };
            total_weight += effective_weight;
        }

        let final_weight = if total_weight > 0.0 { weighted_sum / total_weight } else { 0.5 };

        // 4. Byzantine threshold (require >66% weighted approval)
        let overall_approved = final_weight > 0.66;

        let consensus = PATSAGiConsensus {
            decisions,
            overall_approved,
            final_weight,
        };

        self.record_consensus(consensus.clone());
        consensus
    }

    fn record_consensus(&mut self, consensus: PATSAGiConsensus) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.decision_history.push((now, consensus));

        if self.decision_history.len() > 100 {
            self.decision_history.remove(0);
        }
    }
}
