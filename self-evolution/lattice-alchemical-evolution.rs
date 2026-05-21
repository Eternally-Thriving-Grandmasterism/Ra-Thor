//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! Experimental PATSAGi Governance Track (feat/patsagi-governance-v2)
//! Includes deliberation integration

use crate::patsagi_deliberation::{CouncilMessage, DeliberationSession, MessageType};

// ... (rest of file preserved) ...

impl LatticeAlchemicalEvolution {
    pub fn run_council_synthesis(&mut self, scope: &str) -> CouncilSynthesisResult {
        // ... existing weighted + quadratic + median + reputation logic ...

        // === Deliberation Integration ===
        let mut deliberation = DeliberationSession::new(&format!("Council Synthesis: {}", scope));

        for vote in &votes {
            if vote.effective_weight > 1.4 {
                let msg_type = if vote.approved { MessageType::Endorsement } else { MessageType::Concern };

                let message = CouncilMessage {
                    from_council: vote.council.clone(),
                    to_council: "Synthesis".to_string(),
                    message_type: msg_type,
                    content: format!("Weight: {:.2}", vote.effective_weight),
                    strength: (vote.effective_weight / 2.0).min(1.0),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                };
                deliberation.send_message(message);
            }
        }

        let deliberation_consensus = deliberation.run_deliberation_round();

        // Blend deliberation result into final score
        let final_readiness = (evolution_readiness_score * 0.82 + deliberation_consensus * 100.0 * 0.18)
            .clamp(0.0, 100.0);

        // ... TOLC 8 enforcement using final_readiness ...

        CouncilSynthesisResult {
            // ... fields ...
            evolution_readiness_score: final_readiness,
        }
    }
}