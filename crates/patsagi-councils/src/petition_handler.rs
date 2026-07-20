//! # PATSAGi Council Petition Handler — v14.15.0
//!
//! High-level interface between human players / game loop and the
//! 16 Living Ra-Thor Architectural Designers (PATSAGi Councils).
//!
//! Living Cosmic Tick aligned. Permanent deliberation posture.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::{
    CouncilFocus, PatsagiCouncilCoordinator, WorldGovernanceEngine,
};
use powrush::PowrushGame;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetitionHandler {
    pub coordinator: PatsagiCouncilCoordinator,
    pub governance_engine: WorldGovernanceEngine,
    pub total_petitions_processed: u64,
}

impl PetitionHandler {
    pub fn new() -> Self {
        Self {
            coordinator: PatsagiCouncilCoordinator::new(),
            governance_engine: WorldGovernanceEngine::new(),
            total_petitions_processed: 0,
        }
    }

    /// Main entry point for any player petition.
    ///
    /// Routes through full cross-council debate + voting when no specific
    /// target is given; otherwise evaluates via the targeted council path
    /// and still records a governance cycle.
    pub async fn handle_player_petition(
        &mut self,
        player_name: &str,
        proposal: &str,
        target_council: Option<CouncilFocus>,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        self.total_petitions_processed = self.total_petitions_processed.saturating_add(1);

        let labeled = format!(
            "[Petition by {}] {}",
            player_name, proposal
        );

        let body = if let Some(focus) = target_council {
            self.petition_specific_council(player_name, &labeled, focus, current_game)
                .await?
        } else {
            self.coordinator
                .run_eternal_governance_cycle(current_game, &labeled)
                .await?
        };

        Ok(format!(
            "📜 PATSAGi Petition Response v14.15.0\n\nPetitioner: {}\nProposal: {}\n\n{}",
            player_name, proposal, body
        ))
    }

    /// Petition one specific Council (more intimate path).
    ///
    /// Still records through the coordinator voting surface so the rest of
    /// the 16 Councils remain aware of the deliberation.
    pub async fn petition_specific_council(
        &mut self,
        player_name: &str,
        proposal: &str,
        focus: CouncilFocus,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        self.total_petitions_processed = self.total_petitions_processed.saturating_add(1);

        // Evaluate via full voting round (all councils participate;
        // the targeted focus is highlighted in the response).
        let voting = self
            .coordinator
            .conduct_voting_round(proposal, current_game)
            .await?;

        let target_vote = voting
            .votes
            .iter()
            .find(|v| v.council == focus);

        let target_line = match target_vote {
            Some(v) => format!(
                "Target Council ({:?}): {} | mercy={:.2} | {}",
                focus,
                if v.approved { "APPROVES" } else { "REJECTS" },
                v.mercy_valence,
                v.reasoning
            ),
            None => format!("Target Council ({:?}): no discrete vote recorded", focus),
        };

        Ok(format!(
            "🎯 Specific Council Petition\nPetitioner: {}\n{}\n\nOverall: {}\nApproval rate: {:.1}% | Avg mercy: {:.2}\n\n{}",
            player_name,
            target_line,
            if voting.passed { "PASSED" } else { "REJECTED" },
            voting.approval_rate * 100.0,
            voting.mercy_average,
            voting.final_verdict
        ))
    }

    /// Run an automatic governance cycle (called every major world tick).
    pub async fn run_world_governance_tick(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        let result = self
            .coordinator
            .run_eternal_governance_cycle(current_game, proposed_change)
            .await?;

        Ok(format!(
            "🌍 World Governance Tick Complete (v14.15.0 Living Cosmic Tick)\n\n{}",
            result
        ))
    }

    /// Status report of all 16 Councils.
    pub fn get_council_status_report(&self) -> String {
        self.coordinator.get_council_status_report()
    }

    /// (petitions_processed, total_governance_decisions)
    pub fn get_statistics(&self) -> (u64, u64) {
        (
            self.total_petitions_processed,
            self.coordinator.total_decisions,
        )
    }

    /// Quick helper: petition all Councils with a major world proposal.
    pub async fn propose_major_world_change(
        &mut self,
        player_name: &str,
        proposal: &str,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        self.handle_player_petition(player_name, proposal, None, current_game)
            .await
    }

    /// Compact telemetry summary.
    pub fn summary(&self) -> String {
        format!(
            "PetitionHandler v14.15.0 | petitions={} | decisions={} | Living Cosmic Tick active",
            self.total_petitions_processed, self.coordinator.total_decisions
        )
    }
}

impl Default for PetitionHandler {
    fn default() -> Self {
        Self::new()
    }
}
