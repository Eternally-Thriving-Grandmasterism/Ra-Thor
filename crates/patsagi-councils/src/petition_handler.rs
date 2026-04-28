//! # PATSAGi Council Petition Handler (v0.1.0)
//!
//! High-level, easy-to-use interface for the main Powrush-MMO game loop
//! and future client applications.
//!
//! This is the bridge between human players and the 13+ Living Ra-Thor
//! Architectural Designers (PATSAGi Councils).

use crate::lib::PatsagiCouncilCoordinator;
use crate::council_decision::{CouncilPetition, PetitionStatus};
use crate::council_focus::CouncilProfile;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetitionHandler {
    pub coordinator: PatsagiCouncilCoordinator,
    pub total_petitions_processed: u64,
}

impl PetitionHandler {
    pub fn new() -> Self {
        Self {
            coordinator: PatsagiCouncilCoordinator::new(),
            total_petitions_processed: 0,
        }
    }

    /// Main entry point for any player petition (recommended method)
    pub async fn handle_player_petition(
        &mut self,
        player_name: &str,
        proposal: &str,
        target_council: Option<crate::lib::CouncilFocus>,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        let petition = self.coordinator
            .submit_petition(player_name, proposal, target_council, current_game)
            .await?;

        self.total_petitions_processed += 1;

        let beautiful_response = self.coordinator.format_petition_response(&petition);
        Ok(beautiful_response)
    }

    /// Allow a player to petition one specific Council directly (more intimate)
    pub async fn petition_specific_council(
        &mut self,
        player_name: &str,
        proposal: &str,
        focus: crate::lib::CouncilFocus,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        let response = self.coordinator
            .petition_specific_council(player_name, proposal, focus, current_game)
            .await?;

        self.total_petitions_processed += 1;
        Ok(response)
    }

    /// Run an automatic governance cycle (called every major world tick)
    pub async fn run_world_governance_tick(
        &mut self,
        current_game: &PowrushGame,
        proposed_change: &str,
    ) -> Result<String, String> {
        let result = self.coordinator
            .run_eternal_governance_cycle(current_game, proposed_change)
            .await?;

        Ok(format!(
            "🌍 World Governance Tick Complete\n\n{}",
            result
        ))
    }

    /// Get a beautiful status report of all 13+ Councils
    pub fn get_council_status_report(&self) -> String {
        self.coordinator.get_council_status_report()
    }

    /// Get total statistics
    pub fn get_statistics(&self) -> (u64, u64) {
        (self.total_petitions_processed, self.coordinator.total_decisions)
    }

    /// Quick helper: Petition all Councils with a major world proposal
    pub async fn propose_major_world_change(
        &mut self,
        player_name: &str,
        proposal: &str,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        self.handle_player_petition(player_name, proposal, None, current_game).await
    }
}

impl Default for PetitionHandler {
    fn default() -> Self {
        Self::new()
    }
}
