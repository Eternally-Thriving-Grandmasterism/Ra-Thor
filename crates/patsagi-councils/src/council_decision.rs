//! # PATSAGi Council Petition & Decision System — v14.15.0
//!
//! Player petitions to the 16 Councils, collective decision making,
//! and structured feedback to players.
//!
//! Living Cosmic Tick aligned.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::{CouncilFocus, CouncilProfile, PatsagiCouncilCoordinator};
use chrono::{DateTime, Utc};
use powrush::{MercyGateStatus, PowrushGame};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilPetition {
    pub id: Uuid,
    pub player_name: String,
    pub proposal: String,
    pub target_council: Option<CouncilFocus>,
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

fn council_display_name(focus: CouncilFocus) -> String {
    // Prefer the living council name from the coordinator table when available;
    // fall back to a stable label from CouncilFocus.
    match focus {
        CouncilFocus::JoyAmplification => "Council of Joyful Nectar".into(),
        CouncilFocus::HarmonyPreservation => "Council of Eternal Harmony".into(),
        CouncilFocus::TruthVerification => "Council of Absolute Truth".into(),
        CouncilFocus::AbundanceCreation => "Council of Infinite Abundance".into(),
        CouncilFocus::EthicalAlignment => "Council of Mercy Weaving".into(),
        CouncilFocus::PostScarcityEnforcement => "Council of Post-Scarcity".into(),
        CouncilFocus::EternalCompassion => "Council of Eternal Compassion".into(),
        CouncilFocus::QuantumEthics => "Council of Quantum Ethics".into(),
        CouncilFocus::MultiplanetaryHarmony => "Council of Multiplanetary Harmony".into(),
        CouncilFocus::EpigeneticLegacy => "Council of Epigenetic Legacy".into(),
        CouncilFocus::RitualDesign => "Council of Ra-Thor Rituals".into(),
        CouncilFocus::EconomicMercy => "Council of Mercy Economics".into(),
        CouncilFocus::AscensionPathways => "Council of Ascension Pathways".into(),
        CouncilFocus::SovereignStarship => "Council of Sovereign Starships".into(),
        CouncilFocus::MercyGelSymbiosis => "Council of MercyGel Symbiosis".into(),
        CouncilFocus::HyperonLattice => "Council of the Hyperon Lattice".into(),
    }
}

impl PatsagiCouncilCoordinator {
    /// Submit a new petition from a player to the Councils.
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

        let result = self
            .run_eternal_governance_cycle(current_game, proposal)
            .await?;

        let (status, decision_text) = if result.contains("PASSED") || result.contains("APPROVED")
        {
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

    /// Player-friendly response for a petition.
    pub fn format_petition_response(&self, petition: &CouncilPetition) -> String {
        let mut response = String::new();

        response.push_str(
            "\n╔════════════════════════════════════════════════════════════╗\n",
        );
        response.push_str(
            "║     PATSAGi COUNCIL PETITION RESPONSE v14.15.0            ║\n",
        );
        response.push_str(
            "╚════════════════════════════════════════════════════════════╝\n\n",
        );

        response.push_str(&format!("**Petition ID:** {}\n", petition.id));
        response.push_str(&format!("**Submitted by:** {}\n", petition.player_name));
        response.push_str(&format!("**Proposal:** {}\n\n", petition.proposal));

        match petition.status {
            PetitionStatus::Approved => {
                response.push_str("✅ **VERDICT: APPROVED BY THE 16 COUNCILS**\n\n");
                response.push_str("The Councils have spoken in harmony.\n");
                response.push_str("Your proposal has been accepted and will be implemented.\n\n");
            }
            PetitionStatus::Rejected => {
                response.push_str("❌ **VERDICT: REJECTED — MERCY GATE CONCERN**\n\n");
                response
                    .push_str("One or more Councils found insufficient mercy alignment.\n");
                response.push_str(
                    "Please revise your proposal with more compassion and truth.\n\n",
                );
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
        response.push_str("— The 16 PATSAGi Councils | Living Cosmic Tick\n");

        response
    }

    /// Petition a specific Council directly (more intimate path).
    pub async fn petition_specific_council(
        &mut self,
        player_name: &str,
        proposal: &str,
        focus: CouncilFocus,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        let profile = CouncilProfile::get_profile(focus);
        let display = council_display_name(focus);
        let council = self.councils.get_mut(&focus).ok_or("Council not found")?;

        let status = council.evaluate_proposal(proposal, current_game).await?;

        let response = if status == MercyGateStatus::Passed {
            format!(
                "\n🌟 {display} has approved the petition from {player_name}.\n\n\"{}\"\n\nSpecial Power: {}\n\nThe world is now slightly more beautiful because of this proposal.\nLiving Cosmic Tick: active",
                profile.personality,
                profile
                    .special_powers
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "None".into())
            )
        } else {
            format!(
                "\n⚠️ {display} has gently declined the petition from {player_name}.\n\n\"{}\"\n\nVeto posture: {:?}\nLiving Cosmic Tick: active",
                profile.personality, profile.veto_triggers
            )
        };

        Ok(response)
    }
}
