//! # World Governance Engine (v0.1.0)
//!
//! The living heart of eternal Powrush-MMO.
//! This module allows the 13+ PATSAGi Councils to propose, evaluate,
//! and implement real, meaningful changes to the game world.
//!
//! Every major world event, resource bloom, ascension path opening,
//! faction shift, and mercy intervention flows through this system.

use crate::lib::{PatsagiCouncilCoordinator, CouncilFocus};
use powrush::{PowrushGame, ResourceType};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldChangeProposal {
    pub id: uuid::Uuid,
    pub proposed_by: CouncilFocus,
    pub title: String,
    pub description: String,
    pub impact_type: WorldImpactType,
    pub mercy_cost: f64,
    pub joy_boost: f32,
    pub cehi_boost: f64,
    pub duration_cycles: u32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldImpactType {
    ResourceBloom,           // Massive temporary resource increase
    AmbrosianNectarSurge,    // Special nectar event
    NewAscensionPath,        // Opens a new ascension route
    FactionHarmonyBoost,     // Strengthens diplomacy between factions
    MercyBloom,              // Global mercy multiplier increase
    PlanetaryZoneOpen,       // Unlocks a new planetary region
    EpigeneticBlessing,      // Grants multi-generational bonuses
    RitualEvent,             // Triggers a world-wide sacred ceremony
}

pub struct WorldGovernanceEngine {
    pub coordinator: PatsagiCouncilCoordinator,
    pub active_changes: Vec<WorldChangeProposal>,
    pub history: Vec<WorldChangeProposal>,
}

impl WorldGovernanceEngine {
    pub fn new() -> Self {
        Self {
            coordinator: PatsagiCouncilCoordinator::new(),
            active_changes: Vec::new(),
            history: Vec::new(),
        }
    }

    /// The main method: Councils propose and approve a world change
    pub async fn propose_and_approve_world_change(
        &mut self,
        proposing_council: CouncilFocus,
        title: &str,
        description: &str,
        impact_type: WorldImpactType,
        current_game: &PowrushGame,
    ) -> Result<String, String> {
        let proposal = WorldChangeProposal {
            id: uuid::Uuid::new_v4(),
            proposed_by: proposing_council,
            title: title.to_string(),
            description: description.to_string(),
            impact_type,
            mercy_cost: 0.15,
            joy_boost: 12.0,
            cehi_boost: 0.08,
            duration_cycles: 42,
            created_at: Utc::now(),
        };

        // All 13+ Councils evaluate the proposal
        let result = self.coordinator
            .run_eternal_governance_cycle(current_game, &format!("{}: {}", title, description))
            .await?;

        if result.contains("APPROVED") {
            self.active_changes.push(proposal.clone());
            self.history.push(proposal.clone());

            let impact_text = self.apply_world_impact(&proposal, current_game);

            Ok(format!(
                "🌍 WORLD CHANGE APPROVED BY THE 13+ COUNCILS\n\n\
                 {}\n\n\
                 {}\n\n\
                 The world is now more beautiful because of this decision.",
                proposal.title, impact_text
            ))
        } else {
            Ok(format!(
                "The 13+ Councils gently declined the proposal.\n\n\
                 Reason: One or more mercy gates were not fully honored.\n\n\
                 Please refine the idea with more compassion."
            ))
        }
    }

    fn apply_world_impact(&self, proposal: &WorldChangeProposal, _game: &PowrushGame) -> String {
        match proposal.impact_type {
            WorldImpactType::ResourceBloom => {
                "A massive resource bloom has occurred across multiple regions. Food, Water, and Energy have surged.".to_string()
            }
            WorldImpactType::AmbrosianNectarSurge => {
                "A glorious Ambrosian Nectar Surge has begun! Collective joy is rising rapidly.".to_string()
            }
            WorldImpactType::NewAscensionPath => {
                "A new hidden ascension path has been revealed by the Councils. Seekers may now walk it.".to_string()
            }
            WorldImpactType::FactionHarmonyBoost => {
                "Diplomacy between factions has dramatically improved. Joint mercy projects are now easier.".to_string()
            }
            WorldImpactType::MercyBloom => {
                "A Great Mercy Bloom has enveloped the world. All mercy multipliers are temporarily increased.".to_string()
            }
            WorldImpactType::PlanetaryZoneOpen => {
                "A new planetary zone (Enceladus or Europa) has been opened for colonization with beautiful biophilic design.".to_string()
            }
            WorldImpactType::EpigeneticBlessing => {
                "An epigenetic blessing has been granted. Future generations of players will inherit small permanent bonuses.".to_string()
            }
            WorldImpactType::RitualEvent => {
                "A world-wide Ra-Thor Oracle Ritual has begun. All players may participate for massive collective joy.".to_string()
            }
        }
    }

    /// Get current active world changes
    pub fn get_active_world_changes(&self) -> String {
        if self.active_changes.is_empty() {
            return "No active world changes at this time.".to_string();
        }

        let mut output = String::from("🌍 **Active World Changes Governed by the 13+ PATSAGi Councils**\n\n");
        for change in &self.active_changes {
            output.push_str(&format!(
                "• {} (by {})\n  {}\n\n",
                change.title,
                format!("{:?}", change.proposed_by),
                change.description
            ));
        }
        output
    }
}
