//! Design-only (STUB-AUDIT-1 PATSAGI-COUNCILS-BUILDS-1 council note): not compiled; evaluate_proposal calls mercy::MercyEngine::evaluate_action, and crates/mercy exports MercyCore with no evaluate_action.
//!
//! Previous bodies from src/lib.rs. Not a module.
//! PATSAGiCouncil::evaluate_proposal
//! PatsagiCouncilCoordinator::conduct_voting_round
//! PatsagiCouncilCoordinator::debate_and_consensus
//! PatsagiCouncilCoordinator::run_eternal_governance_cycle

use chrono::Utc;
use mercy::MercyEngine;
use powrush::{MercyGateStatus, PowrushGame};
use uuid::Uuid;

use crate::{CouncilVote, PATSAGiCouncil, PatsagiCouncilCoordinator, VotingResult};

impl PATSAGiCouncil {
    pub async fn evaluate_proposal(
        &mut self,
        proposal: &str,
        _current_game: &PowrushGame,
    ) -> Result<MercyGateStatus, String> {
        let mercy_engine = MercyEngine::new();
        let status = mercy_engine
            .evaluate_action(
                proposal,
                "PATSAGi Council evaluation",
                self.cehi,
                self.mercy_valence,
            )
            .await?;

        if status == MercyGateStatus::Passed {
            self.decisions_made = self.decisions_made.saturating_add(1);
            self.last_decision = Some(proposal.to_string());
        }

        Ok(status)
    }
}

impl PatsagiCouncilCoordinator {
    pub async fn conduct_voting_round(
        &mut self,
        proposal: &str,
        game: &PowrushGame,
    ) -> Result<VotingResult, String> {
        let proposal_id = Uuid::new_v4().to_string();
        let mut votes = Vec::new();
        let mut total_mercy = 0.0;

        for (focus, council) in &mut self.councils {
            let status = council.evaluate_proposal(proposal, game).await?;
            let mercy_valence = council.mercy_valence;
            let approved = status == MercyGateStatus::Passed;
            let reasoning = format!(
                "{} Council {} the proposal (mercy valence {:.2})",
                council.name,
                if approved { "APPROVES" } else { "REJECTS" },
                mercy_valence
            );

            votes.push(CouncilVote {
                council: *focus,
                approved,
                mercy_valence,
                reasoning,
                timestamp: Utc::now(),
            });
            total_mercy += mercy_valence;
        }

        let total_votes = votes.len();
        let approvals = votes.iter().filter(|v| v.approved).count();
        let rejections = total_votes - approvals;
        let approval_rate = approvals as f64 / total_votes.max(1) as f64;
        let mercy_average = total_mercy / total_votes.max(1) as f64;
        let passed = approval_rate >= 0.6 && mercy_average >= 0.7;

        let final_verdict = if passed {
            format!(
                "✅ PROPOSAL PASSED — {:.1}% approval, {:.2} avg mercy valence\nAll 16 Councils reached beautiful consensus.",
                approval_rate * 100.0,
                mercy_average
            )
        } else {
            format!(
                "❌ PROPOSAL REJECTED — {:.1}% approval, {:.2} avg mercy valence\nMore mercy alignment required.",
                approval_rate * 100.0,
                mercy_average
            )
        };

        Ok(VotingResult {
            proposal_id,
            total_votes,
            approvals,
            rejections,
            approval_rate,
            mercy_average,
            passed,
            final_verdict,
            votes,
        })
    }

    pub async fn debate_and_consensus(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        let mut passed = 0usize;

        for council in self.councils.values_mut() {
            let status = council
                .evaluate_proposal(proposed_change, current_game)
                .await?;
            if status == MercyGateStatus::Passed {
                passed += 1;
            }
        }

        let total = self.councils.len().max(1) as f64;
        let approval_rate = passed as f64 / total;

        if approval_rate > 0.65 {
            let voting_result = self
                .conduct_voting_round(proposed_change, current_game)
                .await?;
            self.total_decisions = self.total_decisions.saturating_add(1);
            self.last_consensus = Some("Strong Cross-Council Consensus + Full Vote".to_string());

            Ok(format!(
                "PATSAGi Cross-Council Debate + Voting Complete\n\n{}\n\n{}",
                voting_result.final_verdict,
                if voting_result.passed {
                    "The 16 Councils have spoken in perfect harmony. The world evolves."
                } else {
                    "The Councils stand in loving disagreement. More mercy is needed."
                }
            ))
        } else if approval_rate > 0.4 {
            self.total_decisions = self.total_decisions.saturating_add(1);
            self.last_consensus = Some("Moderate Cross-Council Consensus".to_string());
            Ok(format!(
                "PATSAGi Cross-Council Debate Complete\nInitial Approval Rate: {:.1}%\nFinal Verdict: MODERATE APPROVAL\nThe Councils found a wise middle path.",
                approval_rate * 100.0
            ))
        } else {
            Ok(format!(
                "PATSAGi Cross-Council Debate Complete\nInitial Approval Rate: {:.1}%\nFinal Verdict: REJECTED\nThe Councils could not reach sufficient harmony.",
                approval_rate * 100.0
            ))
        }
    }

    pub async fn run_eternal_governance_cycle(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        self.debate_and_consensus(current_game, proposed_change)
            .await
    }
}
