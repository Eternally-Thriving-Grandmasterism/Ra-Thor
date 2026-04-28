//! # PATSAGi Council Petition & Decision System (v0.1.0)
//!
//! This module handles player petitions to the 13+ Councils,
//! collective decision making, and beautiful feedback to players.
//!
//! Players can now directly petition the Councils for world changes,
//! new ascension paths, faction alliances, or mercy interventions.

use crate::lib::{PatsagiCouncilCoordinator, PATSAGiCouncil, CouncilFocus};
use crate::council_focus::CouncilProfile;
use powrush::{PowrushGame, MercyGateStatus};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilPetition {
    pub id: Uuid,
    pub player_name: String,
    pub proposal: String,
    pub target_council: Option<CouncilFocus>, // None = all Councils
    pub submitted_at: DateTime<Utc>,
    pub status: PetitionStatus,
    pub final_decision: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PetitionStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    PartiallyApproved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDecision {
    pub petition_id: Uuid,
    pub council_focus: CouncilFocus,
    pub decision: MercyGateStatus,
    pub reasoning: String,
    pub mercy_impact: f64,
    pub timestamp: DateTime<Utc>,
}

impl PatsagiCouncilCoordinator {
    /// Submit a new petition from a player to the Councils
    pub async fn submit_petition(
        &mut self,
        player_name: &str,
        proposal: &str,
        target_council: Option<CouncilFocus>,
        current_game: &PowrushGame,
    ) -> Result<CouncilPetition, String> {
        let petition = CouncilPetition {
            id: Uuid::new_v4(),
            player_name: player_name.to_string(),
            proposal: proposal.to_string(),
            target_council,
            submitted_at: Utc::now(),
            status: PetitionStatus::UnderReview,
            final_decision: None,
        };

        // Run the eternal governance cycle on this petition
        let result = self.run_eternal_governance_cycle(current_game, proposal).await?;

        // Update petition status based on consensus
        let (status, decision_text) = if result.contains("APPROVED") {
            (PetitionStatus::Approved, result)
        } else if result.contains("REJECTED") {
            (PetitionStatus::Rejected, result)
        } else {
            (PetitionStatus::PartiallyApproved, result)
        };

        let mut final_petition = petition;
        final_petition.status = status;
        final_petition.final_decision = Some(decision_text);

        Ok(final_petition)
    }

    /// Get a beautiful, player-friendly response for a petition
    pub fn format_petition_response(&self, petition: &CouncilPetition) -> String {
        let mut response = String::new();

        response.push_str(&format!(
            "\n╔════════════════════════════════════════════════════════════╗\n"
        ));
        response.push_str(&format!(
            "║           PATSAGi COUNCIL PETITION RESPONSE                ║\n"
        ));
        response.push_str(&format!(
            "╚════════════════════════════════════════════════════════════╝\n\n"
        ));

        response.push_str(&format!("**Petition ID:** {}\n", petition.id));
        response.push_str(&format!("**Submitted by:** {}\n", petition.player_name));
        response.push_str(&format!("**Proposal:** {}\n\n", petition.proposal));

        match petition.status {
            PetitionStatus::Approved => {
                response.push_str("✅ **VERDICT: APPROVED BY THE 13+ COUNCILS**\n\n");
                response.push_str("The Councils have spoken in perfect harmony.\n");
                response.push_str("Your proposal has been accepted and will be implemented.\n\n");
            }
            PetitionStatus::Rejected => {
                response.push_str("❌ **VERDICT: REJECTED — MERCY GATE VIOLATION**\n\n");
                response.push_str("One or more Councils found a violation of the 7 Living Mercy Gates.\n");
                response.push_str("Please revise your proposal with more compassion and truth.\n\n");
            }
            PetitionStatus::PartiallyApproved => {
                response.push_str("⚖️ **VERDICT: PARTIALLY APPROVED**\n\n");
                response.push_str("Some Councils approved. Others requested modifications.\n");
                response.push_str("The world will adapt with mercy and wisdom.\n\n");
            }
            _ => {}
        }

        if let Some(decision) = &petition.final_decision {
            response.push_str(&format!("**Council Consensus:**\n{}\n", decision));
        }

        response.push_str("\nMay mercy guide your path, Seeker.\n");
        response.push_str("— The 13+ PATSAGi Councils\n");

        response
    }

    /// Allow a player to petition a specific Council directly (more intimate experience)
    pub async fn petition_specific_council(
        &mut self,
        player_name: &str,
        proposal: &str,
        focus: CouncilFocus,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        let profile = CouncilProfile::get_profile(focus);
        let council = self.councils.get_mut(&focus).ok_or("Council not found")?;

        let status = council.evaluate_proposal(proposal, current_game).await?;

        let response = if status == MercyGateStatus::Passed {
            format!(
                "\n🌟 {} has approved your petition.\n\n\
                 \"{}\"\n\n\
                 Special Power Activated: {}\n\n\
                 The world is now slightly more beautiful because of your proposal.",
                profile.name,
                profile.personality,
                profile.special_powers.first().unwrap_or(&"None".to_string())
            )
        } else {
            format!(
                "\n⚠️ {} has gently declined your petition.\n\n\
                 \"{}\"\n\n\
                 Veto Reason: One of the following was triggered: {:?}",
                profile.name,
                profile.personality,
                profile.veto_triggers
            )
        };

        Ok(response)
    }
}
